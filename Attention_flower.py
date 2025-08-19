# core/xgraph/attention.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple

class AttentionFlower(nn.Module):
    """Attention Flower pattern for multi-head attention with petal structure"""
    
    def __init__(self, num_petals: int = 8, hidden_dim: int = 768, dropout: float = 0.1):
        super().__init__()
        self.num_petals = num_petals
        self.hidden_dim = hidden_dim
        self.petal_dim = hidden_dim // num_petals
        
        # Create petal-specific projections
        self.petal_projections = nn.ModuleList([
            nn.Linear(hidden_dim, self.petal_dim * 3)  # Q, K, V for each petal
            for _ in range(num_petals)
        ])
        
        self.output_projection = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
        # Learnable petal interaction weights (the flower center)
        self.petal_interactions = nn.Parameter(
            torch.randn(num_petals, num_petals) / math.sqrt(num_petals)
        )
        
    async def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass through attention flower"""
        batch_size, seq_len, _ = x.shape
        
        petal_outputs = []
        
        for petal_idx in range(self.num_petals):
            # Project through petal
            petal_qkv = self.petal_projections[petal_idx](x)
            q, k, v = petal_qkv.chunk(3, dim=-1)
            
            # Scaled dot-product attention
            scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.petal_dim)
            
            if mask is not None:
                scores = scores.masked_fill(mask == 0, -1e9)
            
            attn_weights = F.softmax(scores, dim=-1)
            attn_weights = self.dropout(attn_weights)
            
            petal_output = torch.matmul(attn_weights, v)
            petal_outputs.append(petal_output)
        
        # Stack and apply petal interactions (the flower pattern)
        petal_stack = torch.stack(petal_outputs, dim=1)  # [batch, petals, seq, dim]
        
        # Apply learned petal interactions
        interaction_weights = F.softmax(self.petal_interactions, dim=-1)
        interacted = torch.einsum('bpsd,pq->bqsd', petal_stack, interaction_weights)
        
        # Combine petals
        combined = interacted.reshape(batch_size, seq_len, self.hidden_dim)
        
        # Output projection
        output = self.output_projection(combined)
        
        return output
