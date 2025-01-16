import torch
import torch.nn as nn

class MemoryRetriever(nn.Module):
    def __init__(self, memory_dim: int = 256, num_heads: int = 4):
        super().__init__()
        self.attention = nn.MultiheadAttention(memory_dim, num_heads)
        self.output_layer = nn.Linear(memory_dim, memory_dim)
        self.norm = nn.LayerNorm(memory_dim)
    
    def forward(self, 
                query: torch.Tensor,
                memory_bank: torch.Tensor,
                memory_mask: torch.Tensor = None) -> torch.Tensor:
        attended_memory, _ = self.attention(
            query.unsqueeze(0),
            memory_bank,
            memory_bank,
            key_padding_mask=memory_mask
        )
        output = self.output_layer(attended_memory.squeeze(0))
        return self.norm(output + query)
