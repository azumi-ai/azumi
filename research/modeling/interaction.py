from typing import Dict, List, Optional, Any
import torch
import torch.nn as nn
import numpy as np
from dataclasses import dataclass

@dataclass
class InteractionModelConfig:
    hidden_size: int = 256
    num_layers: int = 3
    dropout: float = 0.1

class InteractionModel(nn.Module):
    """Advanced interaction modeling system."""
    
    def __init__(self, config: InteractionModelConfig):
        super().__init__()
        self.config = config
        
        # Interaction encoder
        self.encoder = nn.LSTM(
            input_size=config.hidden_size,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            dropout=config.dropout,
            batch_first=True
        )
        
        # Interaction predictor
        self.predictor = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size * 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size * 2, config.hidden_size)
        )
