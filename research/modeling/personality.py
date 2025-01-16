from typing import Dict, List, Optional, Any
import torch
import torch.nn as nn
import numpy as np
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class PersonalityModelConfig:
    embedding_dim: int = 256
    num_layers: int = 4
    num_heads: int = 8
    dropout: float = 0.1

class PersonalityModel(nn.Module):
    """Advanced personality modeling system."""
    
    def __init__(self, config: PersonalityModelConfig):
        super().__init__()
        self.config = config
        
        # Trait embedding
        self.trait_embedding = nn.Linear(config.embedding_dim, config.embedding_dim)
        
        # Personality encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.embedding_dim,
            nhead=config.num_heads,
            dropout=config.dropout
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.num_layers
        )
        
        # Decision projector
        self.decision_proj = nn.Sequential(
            nn.Linear(config.embedding_dim, config.embedding_dim * 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.embedding_dim * 2, config.embedding_dim)
        )
