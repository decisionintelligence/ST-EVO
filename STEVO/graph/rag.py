from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import torch


@dataclass
class STMemoryItem:
    """
    Data structure for a single memory unit.
    Stores node embeddings instead of explicit graph structures.
    """
    prompt_embedding: torch.Tensor  # The vector representation of the query/prompt (Key)
    node_features: List[torch.Tensor]  # List of [N, H] tensors (Value). T steps of node states.
    token_cost: int  # Computational cost (used for retention policy)
    uncertainty: float  # Entropy/Uncertainty (used for retention policy)
    access_count: int = 0  # How many times this memory has been retrieved
    insert_order: int = 0  # logical timestamp for insertion


class STTensorRAG:
    def __init__(self, capacity: int = 100):
        self.capacity = capacity
        self.buffer: List[STMemoryItem] = []
        self.global_insert_counter = 0

    def __len__(self):
        return len(self.buffer)

    def _calculate_retention_score(self, item: STMemoryItem) -> float:
        """
        Calculates the value of keeping an item in memory.

        New Policy:
        - Low Cost -> Keep (Prefer lightweight items to save space/overhead)
        - Low Uncertainty (Value closer to 0) -> Keep (Prefer high confidence/reliable knowledge)
        - High Access -> Keep (Useful knowledge)
        """
        epsilon = 1e-6

        # 1. Handle Uncertainty
        # Assumption: Input is negative (e.g., -10 vs -0.1).
        # -10 (Lower raw value) -> High Uncertainty -> Bad -> abs(-10) = 10 (Large penalty)
        # -0.1 (Higher raw value, closer to 0) -> Low Uncertainty -> Good -> abs(-0.1) = 0.1 (Small penalty)
        # We use the absolute value so that higher uncertainty results in a larger denominator.
        uncertainty_magnitude = abs(item.uncertainty)

        # 2. Construct Penalty Factor
        # Since we want to keep items with Low Cost and Low Uncertainty,
        # we multiply them to form the penalty (denominator).
        penalty = (item.token_cost * uncertainty_magnitude) + epsilon

        # 3. Calculate Base Value (Inverse Relationship)
        # The smaller the penalty (Low Cost & Low Uncertainty), the higher the base value.
        base_value = 1.0 / penalty

        # 4. Popularity Weighting
        # Logarithmic smoothing to favor frequently accessed items while preventing
        # the "rich-get-richer" effect.
        popularity_factor = 1 + np.log(item.access_count + 1)

        return base_value * popularity_factor

    def _evict_and_insert(self, new_item: STMemoryItem):
        """
        Handles memory overflow by evicting the least valuable item.
        Implements a Value-based Eviction Policy.
        """
        # 1. Calculate score for the new candidate
        new_score = self._calculate_retention_score(new_item)

        # 2. Calculate scores for all existing items
        current_scores = [self._calculate_retention_score(item) for item in self.buffer]

        # 3. Identify the "weakest" link
        min_score_idx = np.argmin(current_scores)
        min_score = current_scores[min_score_idx]

        # 4. Decision: Replace only if the new item is more valuable
        if new_score > min_score:
            print(f"    [Eviction] Swapping idx {min_score_idx} (Score: {min_score:.2f}) "
                  f"with new item (Score: {new_score:.2f})")
            self.buffer[min_score_idx] = new_item
        else:
            print(
                f"    [Drop] New item score ({new_score:.2f}) is lower than the worst existing item ({min_score:.2f}).")

    def add_memory(self, embedding: torch.Tensor,
                   node_features: List[torch.Tensor],
                   token_cost: int,
                   uncertainty: float):
        """
        Insert a new memory item.

        Args:
            embedding: The key vector for retrieval.
            node_features: A list of tensors with shape [N, Hidden_Dim].
                           (The graph can be reconstructed later via Feature @ Feature.T).
            token_cost: Computational cost.
            uncertainty: Uncertainty metric.
        """
        # Detach gradient graphs to save memory and prevent leaks
        embedding = embedding.detach()

        # Process node features (detach and ensure list format)
        processed_features = []
        for feat in node_features:
            # Ensure it is detached.
            # Note: We keep them on their original device (e.g., GPU) for fast retrieval/computation later.
            processed_features.append(feat.detach())

        self.global_insert_counter += 1
        new_item = STMemoryItem(
            prompt_embedding=embedding,
            node_features=processed_features,
            token_cost=token_cost,
            uncertainty=uncertainty,
            access_count=0,
            insert_order=self.global_insert_counter
        )

        if len(self.buffer) < self.capacity:
            # Cold start phase: just append
            self.buffer.append(new_item)
        else:
            # Full capacity phase: trigger eviction policy
            self._evict_and_insert(new_item)

    def query(self, query_vec: torch.Tensor, top_k: int = 3) -> List[Tuple[STMemoryItem, float]]:
        """
        Vector-based retrieval using PyTorch.
        Returns the top-k most similar memory items.
        """
        if not self.buffer:
            return []

        # 1. Stack keys from buffer -> (Batch_Size, Embed_Dim)
        # Assumption: All embeddings are on the same device.
        keys = torch.stack([item.prompt_embedding for item in self.buffer])

        # 2. Compute Similarity
        # Using Matrix Multiplication (equivalent to Cosine Sim if vectors are normalized)
        # Shape: (Batch_Size, Embed_Dim) @ (Embed_Dim) -> (Batch_Size)
        scores = torch.matmul(keys, query_vec)

        # 3. Get Top-K
        actual_k = min(top_k, len(self.buffer))
        top_scores, top_indices = torch.topk(scores, k=actual_k)

        results = []
        # Check if we are past the cold-start phase to enable popularity counting
        is_warmed_up = len(self.buffer) >= self.capacity

        # Convert to CPU/List for Python loop handling
        top_indices = top_indices.tolist()
        top_scores = top_scores.tolist()

        for idx, score in zip(top_indices, top_scores):
            item = self.buffer[idx]

            # Update access count only if system is warmed up
            # This prevents "First-Mover Advantage" bias
            if is_warmed_up:
                item.access_count += 1

            results.append((item, score))

        return results


# ===========================
# Validation Code
# ===========================
if __name__ == "__main__":
    # Mock data generator: Creates [Num_Nodes, Hidden_Dim] tensors
    def mock_node_tensor(num_nodes=6, hidden_dim=768):
        # Simulating features for 6 nodes
        return [torch.randn(num_nodes, hidden_dim)]


    # Initialize RAG with small capacity for testing
    rag = STTensorRAG(capacity=3)


    # Mock embeddings (random vectors for testing)
    def mock_embed():
        return torch.randn(384)  # Example embedding dim


    print("=== Adding Node Feature Tensors ===")

    # 1. High cost, Low uncertainty -> High Value
    rag.add_memory(mock_embed(), mock_node_tensor(), token_cost=200, uncertainty=0.2)

    # 2. Medium cost
    rag.add_memory(mock_embed(), mock_node_tensor(), token_cost=150, uncertainty=0.3)

    # 3. High cost, Low uncertainty (Buffer is now full)
    rag.add_memory(mock_embed(), mock_node_tensor(), token_cost=300, uncertainty=0.1)

    # Verify storage structure
    first_item = rag.buffer[0]
    print(f"Stored Feature Shape: {first_item.node_features[0].shape}")
    # Expected: torch.Size([6, 768])

    print("\n=== Performing Query ===")
    query_vector = mock_embed()
    results = rag.query(query_vector, top_k=2)

    for item, score in results:
        print(f"Hit Score: {score:.4f} | Cost: {item.token_cost} | Uncertainty: {item.uncertainty}")
        # Validate that we can reconstruct a graph structure (Adjacency ~ Feature @ Feature.T)
        features = item.node_features[0]
        reconstructed_adj = torch.matmul(features, features.t())
        print(f"Reconstructed Adj Shape: {reconstructed_adj.shape}")
        # Expected: torch.Size([6, 6])
