from typing import Dict, List, Optional, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertConfig

class PersonalityBaseModel(nn.Module):
    def __init__(
        self,
        num_traits: int = 32,
        hidden_size: int = 768,
        num_layers: int = 6,
        num_attention_heads: int = 12,
        dropout_prob: float = 0.1
    ):
        super().__init__()
        
        # BERT for text understanding
        self.bert_config = BertConfig(
            hidden_size=hidden_size,
            num_hidden_layers=num_layers,
            num_attention_heads=num_attention_heads,
            intermediate_size=hidden_size * 4,
            hidden_dropout_prob=dropout_prob,
            attention_probs_dropout_prob=dropout_prob
        )
        self.bert = BertModel(self.bert_config)
        
        # Personality trait modeling
        self.trait_embeddings = nn.Parameter(
            torch.randn(num_traits, hidden_size)
        )
        
        # Personality dynamics
        self.dynamics = nn.ModuleDict({
            'evolution': nn.GRU(
                hidden_size,
                hidden_size,
                num_layers=2,
                batch_first=True
            ),
            'trait_attention': nn.MultiheadAttention(
                hidden_size,
                num_attention_heads
            ),
            'context_processor': nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=hidden_size,
                    nhead=num_attention_heads
                ),
                num_layers=2
            )
        })
        
        # Output layers
        self.trait_predictor = nn.Linear(hidden_size, num_traits)
        self.personality_projection = nn.Linear(hidden_size, hidden_size)
        
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        context_embeddings: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        # Process input text
        bert_outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        hidden_states = bert_outputs.last_hidden_state
        
        # Process context if provided
        if context_embeddings is not None:
            context_processed = self.dynamics['context_processor'](
                context_embeddings
            )
            hidden_states = torch.cat([hidden_states, context_processed], dim=1)
        
        # Trait attention mechanism
        trait_attention, _ = self.dynamics['trait_attention'](
            hidden_states,
            self.trait_embeddings.unsqueeze(0).expand(
                hidden_states.size(0), -1, -1
            ),
            self.trait_embeddings.unsqueeze(0).expand(
                hidden_states.size(0), -1, -1
            )
        )
        
        # Evolution through GRU
        evolved_states, _ = self.dynamics['evolution'](
            trait_attention
        )
        
        # Generate outputs
        personality_embedding = self.personality_projection(
            evolved_states[:, -1, :]
        )
        trait_predictions = torch.sigmoid(
            self.trait_predictor(personality_embedding)
        )
        
        return {
            'personality_embedding': personality_embedding,
            'trait_predictions': trait_predictions,
            'evolved_states': evolved_states
        }
