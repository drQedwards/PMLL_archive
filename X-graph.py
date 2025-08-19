# core/xgraph/memory_graph.py
import torch
import networkx as nx
from typing import Tuple, List, Optional
import numpy as np

class XGraphMemory:
    """X-Graph memory structure for efficient data routing"""
    
    def __init__(self, dimensions: Tuple[int, ...], compression_ratio: float = 0.8960):
        self.dimensions = dimensions
        self.compression_ratio = compression_ratio
        self.graph = nx.DiGraph()
        self._build_x_graph()
        
    def _build_x_graph(self):
        """Build the X-shaped graph topology"""
        center = tuple(d // 2 for d in self.dimensions)
        
        # Create X pattern with diagonal connections
        for i in range(self.dimensions[0]):
            for j in range(self.dimensions[1]):
                node_id = f"x_{i}_{j}"
                self.graph.add_node(node_id, 
                                  position=(i, j),
                                  memory=torch.zeros(768))  # Hidden dimension
                
                # Add X-pattern edges
                if i > 0 and j > 0:
                    self.graph.add_edge(f"x_{i-1}_{j-1}", node_id)
                if i > 0 and j < self.dimensions[1] - 1:
                    self.graph.add_edge(f"x_{i-1}_{j+1}", node_id)
    
    def route_data(self, input_tensor: torch.Tensor, path: List[str]) -> torch.Tensor:
        """Route data through specified path in x-graph"""
        current = input_tensor
        
        for node_id in path:
            if node_id in self.graph:
                node_data = self.graph.nodes[node_id]
                # Apply compression
                current = self._compress(current, self.compression_ratio)
                # Store in node memory
                node_data['memory'] = current
                # Apply transformation based on neighbors
                current = self._transform_with_neighbors(current, node_id)
        
        return current
    
    def _compress(self, tensor: torch.Tensor, ratio: float) -> torch.Tensor:
        """Apply compression using the 8960 ratio mentioned"""
        original_shape = tensor.shape
        compressed_size = int(np.prod(original_shape) * ratio)
        
        # Flatten, compress, and reshape
        flat = tensor.flatten()
        indices = torch.topk(flat.abs(), compressed_size).indices
        compressed = torch.zeros_like(flat)
        compressed[indices] = flat[indices]
        
        return compressed.reshape(original_shape)
    
    def _transform_with_neighbors(self, tensor: torch.Tensor, node_id: str) -> torch.Tensor:
        """Transform tensor based on neighbor information"""
        neighbors = list(self.graph.neighbors(node_id))
        if not neighbors:
            return tensor
        
        neighbor_memories = []
        for neighbor in neighbors[:4]:  # Limit to 4 neighbors for efficiency
            neighbor_memories.append(self.graph.nodes[neighbor]['memory'])
        
        if neighbor_memories:
            neighbor_stack = torch.stack(neighbor_memories)
            # Weighted average with current tensor
            return 0.7 * tensor + 0.3 * neighbor_stack.mean(dim=0)
        
        return tensor
