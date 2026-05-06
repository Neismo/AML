from datasets import load_dataset
import networkx as nx
from networkx.algorithms.graph_hashing import weisfeiler_lehman_graph_hash
import numpy as np
import random

class GraphDataset:
    def __init__(self):
        self.dataset = load_dataset("graphs-datasets/MUTAG", split="train")

        self.num_nodes = [graph["num_nodes"] for graph in self.dataset]
        self.R = {n: self.compute_r(n) for n in self.num_nodes}  # link probability per node count

        self.networkx_graphs = []  # precompute networkx graphs for all original dataset entries
        for i in range(len(self.dataset)):
            self.networkx_graphs.append(self.get_original_graph_as_networkx(i))

        self.hashes = []
        for graph in self.networkx_graphs:
            self.hashes.append(weisfeiler_lehman_graph_hash(graph, iterations=20))

    def sample_num_nodes(self):
        # Sample the number of nodes from the empirical distribution of the dataset
        return random.choice(self.num_nodes)
    
    def compute_r(self, num_nodes: int):
        # Compute the empirical link probability R for a given number of nodes by averaging over the dataset
        matching_graphs = [graph for graph in self.dataset if graph["num_nodes"] == num_nodes]

        if not matching_graphs:
            raise ValueError(f"No training graphs found with num_nodes={num_nodes}")

        max_possible_edges = num_nodes * (num_nodes - 1) / 2
        if max_possible_edges == 0:
            return 0.0

        densities = []
        for graph in matching_graphs:
            src_nodes, dst_nodes = graph["edge_index"]

            # Treat the graph as undirected and avoid double-counting reciprocal edges.
            unique_edges = {
                (min(src, dst), max(src, dst))
                for src, dst in zip(src_nodes, dst_nodes)
                if src != dst
            }
            densities.append(len(unique_edges) / max_possible_edges)

        return sum(densities) / len(densities)

    def sample_graph(self, num_nodes: int | None = None, return_networkx: bool = True):
        num_nodes = num_nodes or self.sample_num_nodes()
        r = self.R[num_nodes]

        # Sample edges based on the computed link probability R
        adjacency_matrix = np.zeros((num_nodes, num_nodes))
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):  # Avoid self-loops and double-counting
                if random.random() < r:
                    adjacency_matrix[i, j] = 1
                    adjacency_matrix[j, i] = 1
        if return_networkx:
            G = nx.from_numpy_array(adjacency_matrix)
            return G
        
        return num_nodes, adjacency_matrix
    
    def get_original_graph_as_networkx(self, index: int):
        """
        Fetches a graph from the original MUTAG dataset by index 
        and converts it into a NetworkX graph.
        """
        graph_data = self.dataset[index]
        G = nx.Graph()
        
        # Add the nodes
        G.add_nodes_from(range(graph_data["num_nodes"]))
        
        # Add the edges
        src_nodes, dst_nodes = graph_data["edge_index"]
        edges = zip(src_nodes, dst_nodes)
        G.add_edges_from(edges)
        
        #if "node_feat" in graph_data:
        #    for i, feat in enumerate(graph_data["node_feat"]):
        #        G.nodes[i]["feature"] = feat
                
        return G

    def __getitem__(self, key):
        if key > len(self.dataset) or key < 0:
            raise IndexError(f"Index {key} is out of bounds for dataset of size {len(self.dataset)}")
        return self.get_original_graph_as_networkx(key), self.dataset[key]

GC = GraphDataset()