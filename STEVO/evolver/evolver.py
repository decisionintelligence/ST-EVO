import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GCNConv


class Evolver(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, depth,
                 num_sampling_steps=100, max_rounds=3, diffusion_batch_mul=4):
        super().__init__()
        self.gcn = GCN(in_channels, hidden_channels, out_channels)
        self.flow_match = FlowMatch(out_channels, in_channels, out_channels, depth, in_channels, num_sampling_steps)
        self.hidden_size = hidden_channels
        self.max_rounds = max_rounds
        self.diffusion_batch_mul = diffusion_batch_mul

        self.round_embeddings = sinusoidal_position_embedding(
            batch_size=1, num_heads=1,
            max_len=self.max_rounds, output_dim=in_channels).squeeze(0).squeeze(0)

    def forward(self, node_features, adj_mx, cond: torch.Tensor = None,
                anchor: torch.Tensor = None, round: int = 0, eps=1e2):
        prototype = self.gcn(node_features, adj_mx)

        round_embedding = self.round_embeddings[round]
        cond += round_embedding

        loss = None
        if self.training:
            if anchor is not None:
                parallel_anchor = anchor.repeat(self.diffusion_batch_mul, 1)
                parallel_cond = cond.repeat(self.diffusion_batch_mul, 1)
                parallel_proto = prototype.repeat(self.diffusion_batch_mul, 1)
                predict_v = self.flow_match(target=parallel_anchor, z=parallel_cond, prototype=parallel_proto)
                predict_v = predict_v.reshape(self.diffusion_batch_mul, -1, predict_v.shape[-1])
                target_v = parallel_anchor.reshape(self.diffusion_batch_mul, -1, parallel_anchor.shape[-1])
                loss = ((predict_v @ predict_v.transpose(-2, -1) - target_v @ target_v.transpose(-2, -1)) ** 2)
                value_mask = loss < eps
                loss = loss[value_mask].sum() / max(1, value_mask.sum())
            else:
                loss = None

            outputs = self.flow_match.sample(z=cond, prototype=prototype, num_samples=100).mean(1)
        else:
            outputs = self.flow_match.sample(z=cond, prototype=prototype, num_samples=100).mean(1)

        return loss, outputs


def sinusoidal_position_embedding(
        batch_size: int,
        num_heads: int,
        max_len: int,
        output_dim: int,
        factor: float = 1.0
) -> torch.Tensor:
    """
    Generate sinusoidal position embeddings.

    Args:
        batch_size (int): Batch size.
        num_heads (int): Number of attention heads.
        max_len (int): Maximum sequence length.
        output_dim (int): Output dimension.
        device (torch.device): Device type.
        factor (float, optional): Scaling factor. Defaults to 1.0.

    Returns:
        torch.Tensor: Sinusoidal position embedding tensor with shape (bs, head, max_len, output_dim).
    """
    # Generate position indices
    position = torch.arange(0, max_len * factor, 1 / factor, dtype=torch.float).unsqueeze(-1)
    # Generate frequency indices
    ids = torch.arange(0, output_dim // 2, dtype=torch.float)
    theta = torch.pow(10000, -2 * ids / output_dim)

    # Calculate position embeddings
    embeddings = position * theta
    embeddings = torch.stack([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)

    # Expand dimensions to match batch size and number of attention heads
    embeddings = embeddings.repeat((batch_size, num_heads, *([1] * len(embeddings.shape))))
    embeddings = torch.reshape(embeddings, (batch_size, num_heads, -1, output_dim))

    # If the factor is greater than 1, perform interpolation
    if factor > 1.0:
        interpolation_indices = torch.linspace(0, embeddings.shape[2] - 1, max_len).long()
        embeddings = embeddings[:, :, interpolation_indices, :]

    return embeddings


class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, in_channels)

        self.mlp = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, out_channels)
        )

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        logs = F.log_softmax(x, dim=1)

        return self.mlp(logs)


class FlowMatch(nn.Module):
    """Flow Loss"""

    def __init__(self, target_channels, z_channels, out_channels, depth, width, num_sampling_steps):
        super(FlowMatch, self).__init__()
        self.in_channels = target_channels
        self.net = SimpleMLPAdaLN(
            in_channels=target_channels,
            model_channels=width,
            out_channels=out_channels,
            z_channels=z_channels,
            num_res_blocks=depth
        )
        self.num_sampling_steps = num_sampling_steps

    def forward(self, target, z, prototype=None):
        noise = torch.randn_like(target)
        t = torch.rand(target.shape[0], device=target.device)

        if prototype is not None:
            noised_target = t[:, None] * target + (1 - t[:, None]) * (prototype + noise)
        else:
            noised_target = t[:, None] * target + (1 - t[:, None]) * noise

        predict_v = self.net(noised_target, t * 1000, z)

        return predict_v

    def sample(self, z, prototype=None, num_samples=1):
        z = z.repeat(num_samples, 1)
        noise = torch.randn(z.shape[0], self.in_channels).to(z.device)
        if prototype is not None:
            prototype = prototype.repeat(num_samples, 1)
            start_point = noise + prototype
            x = noise + prototype
        else:
            start_point = noise
            x = noise
        dt = 1.0 / self.num_sampling_steps
        for i in range(self.num_sampling_steps):
            t = (torch.ones((x.shape[0])) * i /
                 self.num_sampling_steps).to(x.device)
            pred = self.net(x, t * 1000, z)
            x = x + (pred - start_point) * dt

        x = x.reshape(num_samples, -1, self.in_channels).transpose(0, 1)
        return x


def modulate(x, shift, scale):
    return x * (1 + scale) + shift


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0,
                                                 end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class ResBlock(nn.Module):
    """
    A residual block that can optionally change the number of channels.
    :param channels: the number of input channels.
    """

    def __init__(
            self,
            channels
    ):
        super().__init__()
        self.channels = channels

        self.in_ln = nn.LayerNorm(channels, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(channels, channels, bias=True),
            nn.SiLU(),
            nn.Linear(channels, channels, bias=True),
        )

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(channels, 3 * channels, bias=True)
        )

    def forward(self, x, y):
        shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(
            y).chunk(3, dim=-1)
        h = modulate(self.in_ln(x), shift_mlp, scale_mlp)
        h = self.mlp(h)
        return x + gate_mlp * h


class FinalLayer(nn.Module):
    """
    The final layer adopted from DiT.
    """

    def __init__(self, model_channels, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(
            model_channels, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(model_channels, out_channels, bias=False)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(model_channels, 2 * model_channels, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = modulate(self.norm_final(x), shift, scale)
        o = self.linear(x)
        return o


class SimpleMLPAdaLN(nn.Module):
    """
    The MLP for Diffusion Loss.
    :param in_channels: channels in the input Tensor.
    :param model_channels: base channel count for the model.
    :param out_channels: channels in the output Tensor.
    :param z_channels: channels in the condition.
    :param num_res_blocks: number of residual blocks per downsample.
    """

    def __init__(
            self,
            in_channels,
            model_channels,
            out_channels,
            z_channels,
            num_res_blocks,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks

        self.time_embed = TimestepEmbedder(model_channels)
        self.cond_embed = nn.Linear(z_channels, model_channels)

        self.input_proj = nn.Linear(in_channels, model_channels)

        res_blocks = []
        for i in range(num_res_blocks):
            res_blocks.append(ResBlock(
                model_channels,
            ))

        self.res_blocks = nn.ModuleList(res_blocks)
        self.final_layer = FinalLayer(model_channels, out_channels)

        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Initialize timestep embedding MLP
        nn.init.normal_(self.time_embed.mlp[0].weight, std=0.02)
        nn.init.normal_(self.time_embed.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers
        for block in self.res_blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)

    def forward(self, x, t, c):
        """
        Apply the model to an input batch.
        :param x: an [N x C] Tensor of inputs.
        :param t: a 1-D batch of timesteps.
        :param c: conditioning from AR transformer.
        :return: an [N x C] Tensor of outputs.
        """
        x = self.input_proj(x)
        t = self.time_embed(t)
        c = self.cond_embed(c)
        y = t + c

        for block in self.res_blocks:
            x = block(x, y)

        return self.final_layer(x, y)
