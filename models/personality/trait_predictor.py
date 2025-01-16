import torch
import torch.nn as nn
from .base_model import PersonalityBaseModel

class TraitPredictor(PersonalityBaseModel):
    def __init__(self, num_traits: int = 5):
        super().__init__(
            input_dim=768,  # BERT embedding dimension
            hidden_dim=256,
            output_dim=num_traits
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        traits = super().forward(x)
        return self.sigmoid(traits)
