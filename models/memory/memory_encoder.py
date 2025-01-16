import torch
import torch.nn as nn
from transformers import BertModel

class MemoryEncoder(nn.Module):
    def __init__(self, memory_dim: int = 256):
        super().__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.memory_projection = nn.Linear(768, memory_dim)
        self.memory_norm = nn.LayerNorm(memory_dim)
    
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        memory_vector = self.memory_projection(pooled_output)
        return self.memory_norm(memory_vector)
