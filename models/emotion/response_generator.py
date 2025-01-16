import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel, GPT2Config

class EmotionalResponseGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = GPT2Config.from_pretrained('gpt2')
        self.config.emotion_embedding_dim = 8
        self.gpt2 = GPT2LMHeadModel(self.config)
        self.emotion_embeddings = nn.Embedding(8, self.config.emotion_embedding_dim)
        self.emotion_projection = nn.Linear(
            self.config.emotion_embedding_dim,
            self.config.hidden_size
        )
    
    def forward(self, 
                input_ids: torch.Tensor,
                attention_mask: torch.Tensor,
                emotion_ids: torch.Tensor) -> torch.Tensor:
        emotion_embeds = self.emotion_embeddings(emotion_ids)
        emotion_hidden = self.emotion_projection(emotion_embeds)
        
        outputs = self.gpt2(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=None
        )
        
        hidden_states = outputs.hidden_states
        hidden_states += emotion_hidden.unsqueeze(1).expand(-1, hidden_states.size(1), -1)
        
        return self.gpt2.lm_head(hidden_states)
