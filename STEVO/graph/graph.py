import asyncio
from abc import ABC
from typing import Any, List, Optional, Dict, Tuple

import numpy as np
import shortuuid
import torch
from torch_geometric.utils import dense_to_sparse

from STEVO.agents.agent_registry import AgentRegistry
from STEVO.evolver.evolver import Evolver
from STEVO.graph.node import Node
from STEVO.graph.rag import STTensorRAG
from STEVO.llm.profile_embedding import get_sentence_embedding
from STEVO.prompt.prompt_set_registry import PromptSetRegistry


class Graph(ABC):
    """
    A framework for managing and executing a network of nodes using a language model.

    This class enables the creation of a graph structure for processing and analyzing data. Each node
    in the graph can perform specific operations, allowing for complex data processing workflows.
    The graph supports integration with language models, making it suitable for tasks that require
    natural language processing capabilities.

    The communication of the node depends on the node.spatial_predecessors and node.spatial_successors.
    
    Attributes:
        domain (str): The domain for which this graph is used.
        llm_name (str): The name of the llm that used for processing within the nodes.
        nodes (dict): A collection of nodes, each identified by a unique UUID.

    Methods:
        build_graph(): Method to be implemented for constructing the graph structure.
        add_node(node): Adds a new node to the graph with a unique identifier.
        run(inputs, num_steps=10, single_agent=False): Executes the graph for a specified number of steps, processing provided inputs.
    """

    def __init__(self,
                 domain: str,
                 llm_name: Optional[str],
                 agent_names: List[str],
                 decision_method: str,
                 optimized_spatial: bool = False,
                 initial_spatial_probability: float = 0.5,
                 fixed_spatial_masks: List[List[int]] = None,
                 optimized_temporal: bool = False,
                 initial_temporal_probability: float = 0.5,
                 fixed_temporal_masks: List[List[int]] = None,
                 node_kwargs: List[Dict] = None,
                 ):

        if fixed_spatial_masks is None:
            fixed_spatial_masks = [[1 if i != j else 0 for j in range(len(agent_names))] for i in
                                   range(len(agent_names))]
        if fixed_temporal_masks is None:
            fixed_temporal_masks = [[1 for j in range(len(agent_names))] for i in range(len(agent_names))]
        fixed_spatial_masks = torch.tensor(fixed_spatial_masks).view(-1)
        fixed_temporal_masks = torch.tensor(fixed_temporal_masks).view(-1)
        assert len(fixed_spatial_masks) == len(agent_names) * len(
            agent_names), "The fixed_spatial_masks doesn't match the number of agents"
        assert len(fixed_temporal_masks) == len(agent_names) * len(
            agent_names), "The fixed_temporal_masks doesn't match the number of agents"

        self.id: str = shortuuid.ShortUUID().random(length=4)
        self.domain: str = domain
        self.llm_name: str = llm_name
        self.agent_names: List[str] = agent_names
        self.optimized_spatial = optimized_spatial
        self.optimized_temporal = optimized_temporal
        self.decision_node: Node = AgentRegistry.get(decision_method,
                                                     **{"domain": self.domain, "llm_name": self.llm_name})
        self.nodes: Dict[str, Node] = {}
        self.potential_spatial_edges: List[List[str, str]] = []
        self.potential_temporal_edges: List[List[str, str]] = []
        self.node_kwargs = node_kwargs if node_kwargs is not None else [{} for _ in agent_names]

        self.init_nodes()  # add nodes to the self.nodes
        self.init_potential_edges()  # add potential edges to the self.potential_spatial/temporal_edges

        self.prompt_set = PromptSetRegistry.get(domain)
        self.role_adj_matrix = self.construct_adj_matrix()
        self.features = self.construct_features()
        self.evolver = Evolver(self.features.size(1), 16, 16, 3, 20)
        self.rag = STTensorRAG(capacity=20)
        init_spatial_logit = torch.log(torch.tensor(
            initial_spatial_probability / (1 - initial_spatial_probability))) if optimized_spatial else 10.0
        # self.spatial_logits = torch.nn.Parameter(torch.ones(len(self.potential_spatial_edges), requires_grad=optimized_spatial) * init_spatial_logit,
        #                                          requires_grad=optimized_spatial) # trainable edge logits
        self.spatial_masks = torch.nn.Parameter(fixed_spatial_masks, requires_grad=False)  # fixed edge masks

        init_temporal_logit = torch.log(torch.tensor(
            initial_temporal_probability / (1 - initial_temporal_probability))) if optimized_temporal else 10.0
        self.temporal_logits = torch.nn.Parameter(
            torch.ones(len(self.potential_temporal_edges), requires_grad=optimized_temporal) * init_temporal_logit,
            requires_grad=optimized_temporal)  # trainable edge logits
        self.temporal_masks = torch.nn.Parameter(fixed_temporal_masks, requires_grad=False)  # fixed edge masks

    def construct_adj_matrix(self):
        role_connect: List[Tuple[str, str]] = self.prompt_set.get_role_connection()
        num_nodes = self.num_nodes
        role_adj = torch.zeros((num_nodes, num_nodes))
        role_2_id = {}

        for edge in role_connect:
            in_role, out_role = edge
            role_2_id[in_role] = []
            role_2_id[out_role] = []
        for i, node_id in enumerate(self.nodes):
            role = self.nodes[node_id].role
            role_2_id[role].append(i)

        for edge in role_connect:
            in_role, out_role = edge
            in_ids = role_2_id[in_role]
            out_ids = role_2_id[out_role]
            for in_id in in_ids:
                for out_id in out_ids:
                    role_adj[in_id][out_id] = 1

        edge_index, edge_weight = dense_to_sparse(role_adj)
        return edge_index

    def construct_features(self):
        features = []
        for node_id in self.nodes:
            role = self.nodes[node_id].role
            profile = self.prompt_set.get_description(role)
            feature = get_sentence_embedding(profile)
            features.append(feature)
        features = torch.tensor(np.array(features))
        return features

    def construct_new_features(self, query):
        query_embedding = torch.tensor(get_sentence_embedding(query))
        task_features = query_embedding
        return task_features

    @property
    def stability(self):
        p = 0.2
        alpha = 0.5
        beta = 0.5
        all_nodes_entropy = []

        for node_id in self.nodes:
            e_list = self.nodes[node_id].entropy
            if not e_list: continue

            all_nodes_entropy.extend(e_list)

        sorted_e = sorted(all_nodes_entropy, reverse=True)
        top_n = max(1, int(len(sorted_e) * p))
        top_p = sorted_e[:top_n]

        top_mean = np.mean(top_p)
        top_var = np.var(top_p)

        ui_score = alpha * top_mean + beta / (1 + top_var)
        return ui_score

    @property
    def num_tokens(self):
        num_tokens = 0
        for node_id in self.nodes:
            node = self.nodes[node_id]
            num_tokens += node.num_completion_tokens
        return num_tokens

    @property
    def spatial_adj_matrix(self):
        matrix = np.zeros((len(self.nodes), len(self.nodes)))
        for i, node1_id in enumerate(self.nodes):
            for j, node2_id in enumerate(self.nodes):
                if self.nodes[node2_id] in self.nodes[node1_id].spatial_successors:
                    matrix[i, j] = 1
        return matrix

    @property
    def temporal_adj_matrix(self):
        matrix = np.zeros((len(self.nodes), len(self.nodes)))
        for i, node1_id in enumerate(self.nodes):
            for j, node2_id in enumerate(self.nodes):
                if self.nodes[node2_id] in self.nodes[node1_id].temporal_successors:
                    matrix[i, j] = 1
        return matrix

    @property
    def num_edges(self):
        num_edges = 0
        for node in self.nodes.values():
            num_edges += len(node.spatial_successors)
        return num_edges

    @property
    def num_nodes(self):
        return len(self.nodes)

    def find_node(self, id: str):
        if id in self.nodes.keys():
            return self.nodes[id]
        raise Exception(f"Node not found: {id} among "
                        f"{[node.id for node in self.nodes.values()]}")

    def add_node(self, node: Node):
        node_id = node.id if node.id is not None else shortuuid.ShortUUID().random(length=4)
        while node_id in self.nodes:
            node_id = shortuuid.ShortUUID().random(length=4)
        node.id = node_id
        self.nodes[node_id] = node
        return node

    def init_nodes(self):
        """
        Creates and adds new nodes to the graph.
        """
        for agent_name, kwargs in zip(self.agent_names, self.node_kwargs):
            if agent_name in AgentRegistry.registry:
                kwargs["domain"] = self.domain
                kwargs["llm_name"] = self.llm_name
                agent_instance = AgentRegistry.get(agent_name, **kwargs)
                self.add_node(agent_instance)

    def init_potential_edges(self):
        """
        Creates and potential edges to the graph.
        """
        for node1_id in self.nodes.keys():
            for node2_id in self.nodes.keys():
                self.potential_spatial_edges.append([node1_id, node2_id])
                self.potential_temporal_edges.append([node1_id, node2_id])

    def clear_spatial_connection(self):
        """
        Clear all the spatial connection of the nodes in the graph.
        """
        for node_id in self.nodes.keys():
            self.nodes[node_id].spatial_predecessors = []
            self.nodes[node_id].spatial_successors = []
        self.decision_node.spatial_predecessors = []
        self.decision_node.spatial_successors = []

    def clear_temporal_connection(self):
        """
        Clear all the temporal connection of the nodes in the graph.
        """
        for node_id in self.nodes.keys():
            self.nodes[node_id].temporal_predecessors = []
            self.nodes[node_id].temporal_successors = []

    def connect_decision_node(self):
        for node_id in self.nodes.keys():
            self.nodes[node_id].add_successor(self.decision_node)

    def construct_spatial_connection(self, temperature: float = 1.0,
                                     threshold: float = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Constructs spatial edges based on logits and masks.
        Returns the sum of log probabilities and the constructed adjacency matrix.
        """
        self.clear_spatial_connection()
        log_probs = [torch.tensor(0.0, requires_grad=self.optimized_spatial)]

        # [New] 1. Initialize Adjacency Matrix
        num_nodes = len(self.nodes)
        # Use the same device as logits to ensure compatibility
        device = self.spatial_logits.device if hasattr(self, 'spatial_logits') else 'cpu'
        adj_matrix = torch.zeros((num_nodes, num_nodes), device=device)

        # [New] 2. Map Node ID to Index to ensure matrix alignment
        # This assumes the order of nodes in self.nodes is consistent across calls
        node_id_to_idx = {node_id: i for i, node_id in enumerate(self.nodes.keys())}

        for potential_connection, edge_logit, edge_mask in zip(self.potential_spatial_edges, self.spatial_logits,
                                                               self.spatial_masks):
            out_node_id = potential_connection[0]
            in_node_id = potential_connection[1]
            out_node: Node = self.find_node(out_node_id)
            in_node: Node = self.find_node(in_node_id)

            # Get matrix indices
            out_idx = node_id_to_idx[out_node_id]
            in_idx = node_id_to_idx[in_node_id]

            if edge_mask == 0.0:
                continue
            elif edge_mask == 1.0 and self.optimized_spatial == False:
                if not self.check_cycle(in_node, {out_node}):
                    out_node.add_successor(in_node, 'spatial')
                    # [New] Update Matrix for fixed edges
                    adj_matrix[out_idx, in_idx] = 1.0
                continue

            # Logic for optimized/learnable edges
            if not self.check_cycle(in_node, {out_node}):
                edge_prob = torch.sigmoid(edge_logit / temperature)
                if threshold:
                    # gradient break | inference
                    edge_prob = torch.tensor(1 if edge_prob > threshold else 0)

                # gradient keep | training
                if torch.rand(1) < edge_prob:
                    out_node.add_successor(in_node, 'spatial')
                    log_probs.append(torch.log(edge_prob))
                    # [New] Update Matrix for sampled edges
                    adj_matrix[out_idx, in_idx] = 1.0
                else:
                    log_probs.append(torch.log(1 - edge_prob))

        adj_matrix, _ = dense_to_sparse(adj_matrix)

        # Return tuple: (Total Log Prob, Adjacency Matrix)
        return torch.sum(torch.stack(log_probs)), adj_matrix

    #
    # def construct_spatial_connection(self, temperature: float = 1.0,
    #                                  threshold: float = None, ):  # temperature must >= 1.0
    #     self.clear_spatial_connection()
    #     log_probs = [torch.tensor(0.0, requires_grad=self.optimized_spatial)]
    #
    #     for potential_connection, edge_logit, edge_mask in zip(self.potential_spatial_edges, self.spatial_logits,
    #                                                            self.spatial_masks):
    #         out_node: Node = self.find_node(potential_connection[0])
    #         in_node: Node = self.find_node(potential_connection[1])
    #         if edge_mask == 0.0:
    #             continue
    #         elif edge_mask == 1.0 and self.optimized_spatial == False:
    #             if not self.check_cycle(in_node, {out_node}):
    #                 out_node.add_successor(in_node, 'spatial')
    #             continue
    #         if not self.check_cycle(in_node, {out_node}):
    #             edge_prob = torch.sigmoid(edge_logit / temperature)
    #             if threshold:
    #                 # gradient break | inference
    #                 edge_prob = torch.tensor(1 if edge_prob > threshold else 0)
    #             # gradient keep | training
    #             if torch.rand(1) < edge_prob:
    #                 out_node.add_successor(in_node, 'spatial')
    #                 log_probs.append(torch.log(edge_prob))
    #             else:
    #                 log_probs.append(torch.log(1 - edge_prob))
    #
    #     return torch.sum(torch.stack(log_probs))

    def construct_temporal_connection(self, round: int = 0, temperature: float = 1.0,
                                      threshold: float = None, ):  # temperature must >= 1.0
        self.clear_temporal_connection()
        log_probs = [torch.tensor(0.0, requires_grad=self.optimized_temporal)]
        if round == 0:
            return torch.sum(torch.stack(log_probs))
        for potential_connection, edge_logit, edge_mask in zip(self.potential_temporal_edges, self.temporal_logits,
                                                               self.temporal_masks):
            out_node: Node = self.find_node(potential_connection[0])
            in_node: Node = self.find_node(potential_connection[1])
            if edge_mask == 0.0:
                continue
            elif edge_mask == 1.0 and self.optimized_temporal == False:
                if not self.check_cycle(in_node, {out_node}):
                    out_node.add_successor(in_node, 'temporal')
                continue

            edge_prob = torch.sigmoid(edge_logit / temperature)
            if threshold:
                edge_prob = torch.tensor(1 if edge_prob > threshold else 0)
            if torch.rand(1) < edge_prob:
                out_node.add_successor(in_node, 'temporal')
                log_probs.append(torch.log(edge_prob))
            else:
                log_probs.append(torch.log(1 - edge_prob))

        return torch.sum(torch.stack(log_probs))

    # def run(self, inputs: Any,
    #         num_rounds: int = 3,
    #         max_tries: int = 3,
    #         max_time: int = 600, ) -> List[Any]:
    #     # inputs:{'task':"xxx"}
    #     log_probs = 0
    #     for round in range(num_rounds):
    #         log_probs += self.construct_spatial_connection()
    #         log_probs += self.construct_temporal_connection(round)
    #
    #         in_degree = {node_id: len(node.spatial_predecessors) for node_id, node in self.nodes.items()}
    #         zero_in_degree_queue = [node_id for node_id, deg in in_degree.items() if deg == 0]
    #
    #         while zero_in_degree_queue:
    #             current_node_id = zero_in_degree_queue.pop(0)
    #             tries = 0
    #             while tries < max_tries:
    #                 try:
    #                     self.nodes[current_node_id].execute(inputs)  # output is saved in the node.outputs
    #                     break
    #                 except Exception as e:
    #                     print(f"Error during execution of node {current_node_id}: {e}")
    #                 tries += 1
    #             for successor in self.nodes[current_node_id].spatial_successors:
    #                 if successor.id not in self.nodes.keys():
    #                     continue
    #                 in_degree[successor.id] -= 1
    #                 if in_degree[successor.id] == 0:
    #                     zero_in_degree_queue.append(successor.id)
    #
    #         self.update_memory()
    #
    #     self.connect_decision_node()
    #     self.decision_node.execute(inputs)
    #     final_answers = self.decision_node.outputs
    #     if len(final_answers) == 0:
    #         final_answers.append("No answer of the decision node")
    #
    #     return final_answers, log_probs

    async def arun(self, input: Dict[str, str],
                   num_rounds: int = 3,
                   max_tries: int = 3,
                   max_time: int = 600) -> List[Any]:
        # inputs:{'task':"xxx"}
        log_probs = 0
        task_feature = self.construct_new_features(input['task'])
        regular_loss = 0
        adj_mx = self.role_adj_matrix

        if len(self.rag) == 0:
            anchor = None
            anchors = None
        else:
            STensors, _ = self.rag.query(task_feature, 1)[0]
            anchors = STensors.node_features

        potential_traj = [task_feature]
        for round in range(num_rounds):
            if anchors is not None:
                anchor = anchors[round]

            cond = task_feature.unsqueeze(0).repeat((self.num_nodes, 1))

            loss, logits = self.evolver(node_features=self.features, cond=cond, adj_mx=adj_mx,
                                        round=round, anchor=anchor)

            if loss is not None:
                regular_loss += loss

            self.spatial_logits = logits @ logits.t()
            self.spatial_logits = min_max_norm(torch.flatten(self.spatial_logits))

            spatial_log_prob, adj_mx = self.construct_spatial_connection()
            potential_traj.append(logits)

            log_probs += spatial_log_prob
            log_probs += self.construct_temporal_connection(round)

            in_degree = {node_id: len(node.spatial_predecessors) for node_id, node in self.nodes.items()}
            zero_in_degree_queue = [node_id for node_id, deg in in_degree.items() if deg == 0]

            while zero_in_degree_queue:
                current_node_id = zero_in_degree_queue.pop(0)
                tries = 0
                while tries < max_tries:
                    try:
                        await asyncio.wait_for(self.nodes[current_node_id].async_execute(input),
                                               timeout=max_time)  # output is saved in the node.outputs
                        break
                    except Exception as e:
                        print(f"Error during execution of node {current_node_id}: {e}")
                    tries += 1
                for successor in self.nodes[current_node_id].spatial_successors:
                    if successor.id not in self.nodes.keys():
                        continue
                    in_degree[successor.id] -= 1
                    if in_degree[successor.id] == 0:
                        zero_in_degree_queue.append(successor.id)

            self.update_memory()

        stability = self.stability
        potential_traj.extend([self.num_tokens, stability])
        self.connect_decision_node()
        await self.decision_node.async_execute(input)
        final_answers = self.decision_node.outputs
        if len(final_answers) == 0:
            final_answers.append("No answer of the decision node")

        regular_loss /= num_rounds
        return final_answers, log_probs, regular_loss, potential_traj

    def update_memory(self):
        for id, node in self.nodes.items():
            node.update_memory()

    def check_cycle(self, new_node, target_nodes):
        if new_node in target_nodes:
            return True
        for successor in new_node.spatial_successors:
            if self.check_cycle(successor, target_nodes):
                return True
        return False

    def update_masks(self, pruning_rate: float) -> torch.Tensor:
        if self.optimized_spatial:
            num_edges = (self.spatial_masks > 0).sum()
            num_masks = (self.spatial_masks == 0).sum()
            prune_num_edges = torch.round(num_edges * pruning_rate) if torch.round(num_edges * pruning_rate) > 0 else 1
            _edge_logits = self.spatial_logits.clone()
            min_edge_logit = _edge_logits.min()
            _edge_logits[self.spatial_masks == 0] = min_edge_logit - 1.0
            sorted_edges_idx = torch.argsort(_edge_logits)
            prune_idx = sorted_edges_idx[:int(prune_num_edges + num_masks)]
            self.spatial_masks[prune_idx] = 0

        if self.optimized_temporal:
            num_edges = (self.temporal_masks > 0).sum()
            num_masks = (self.temporal_masks == 0).sum()
            prune_num_edges = torch.round(num_edges * pruning_rate) if torch.round(num_edges * pruning_rate) > 0 else 1
            _edge_logits = self.temporal_logits.clone()
            min_edge_logit = _edge_logits.min()
            _edge_logits[self.temporal_masks == 0] = min_edge_logit - 1.0
            sorted_edges_idx = torch.argsort(_edge_logits)
            prune_idx = sorted_edges_idx[:int(prune_num_edges + num_masks)]
            self.temporal_masks[prune_idx] = 0
        return self.spatial_masks, self.temporal_masks


def min_max_norm(tensor: torch.Tensor):
    min_val = tensor.min()
    max_val = tensor.max()
    normalized_0_to_1 = (tensor - min_val) / (max_val - min_val)
    normalized_minus1_to_1 = normalized_0_to_1 * 2 - 1
    return normalized_minus1_to_1
