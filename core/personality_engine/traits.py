from typing import Dict, List, Optional, Any
import torch
import torch.nn as nn
import numpy as np
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class TraitConfig:
    base_traits: List[str]
    trait_ranges: Dict[str, tuple]
    evolution_rate: float = 0.1
    stability_factor: float = 0.7
    interaction_strength: float = 0.3
    min_trait_value: float = 0.0
    max_trait_value: float = 1.0

class TraitManager:
    def __init__(self, config: TraitConfig):
        self.config = config
        self.traits = {
            trait: np.random.uniform(
                *self.config.trait_ranges.get(
                    trait, 
                    (self.config.min_trait_value, self.config.max_trait_value)
                )
            )
            for trait in config.base_traits
        }
        self.trait_history = []
        self.interaction_matrix = self._initialize_interaction_matrix()
        
    def _initialize_interaction_matrix(self) -> np.ndarray:
        n_traits = len(self.config.base_traits)
        matrix = np.zeros((n_traits, n_traits))
        
        for i in range(n_traits):
            for j in range(n_traits):
                if i != j:
                    # Initialize with small random interactions
                    matrix[i, j] = np.random.normal(
                        0,
                        self.config.interaction_strength
                    )
        
        return matrix
    
    def evolve_traits(
        self,
        context: Dict[str, float],
        emotional_state: Optional[Dict[str, float]] = None
    ) -> Dict[str, float]:
        """Evolve traits based on context and emotional state."""
        
        # Store current state
        self.trait_history.append(self.traits.copy())
        
        # Calculate trait influences
        influences = self._calculate_influences(context, emotional_state)
        
        # Update traits
        new_traits = {}
        for i, trait in enumerate(self.config.base_traits):
            current_value = self.traits[trait]
            
            # Calculate new value considering:
            # 1. Current value stability
            # 2. Context influence
            # 3. Trait interactions
            # 4. Random variation
            
            stability_component = current_value * self.config.stability_factor
            influence_component = influences[trait] * (1 - self.config.stability_factor)
            interaction_component = self._calculate_trait_interactions(trait)
            random_component = np.random.normal(0, 0.01)
            
            new_value = (
                stability_component +
                influence_component +
                interaction_component +
                random_component
            )
            
            # Ensure value stays within bounds
            new_traits[trait] = np.clip(
                new_value,
                self.config.min_trait_value,
                self.config.max_trait_value
            )
        
        self.traits = new_traits
        return self.traits
    
    def _calculate_influences(
        self,
        context: Dict[str, float],
        emotional_state: Optional[Dict[str, float]] = None
    ) -> Dict[str, float]:
        """Calculate external influences on traits."""
        
        influences = {}
        for trait in self.config.base_traits:
            # Base influence from context
            base_influence = context.get(trait, 0.0)
            
            # Emotional modulation
            if emotional_state:
                emotional_factor = sum(
                    emotional_state.values()
                ) / len(emotional_state)
                base_influence *= (1 + emotional_factor)
            
            influences[trait] = base_influence
        
        return influences
    
    def _calculate_trait_interactions(self, trait: str) -> float:
        """Calculate influence of other traits."""
        
        trait_idx = self.config.base_traits.index(trait)
        interaction_sum = 0.0
        
        for other_trait, other_value in self.traits.items():
            if other_trait != trait:
                other_idx = self.config.base_traits.index(other_trait)
                interaction_sum += (
                    other_value *
                    self.interaction_matrix[trait_idx, other_idx]
                )
        
        return interaction_sum * self.config.evolution_rate
    
    def get_trait_correlation(self, trait1: str, trait2: str) -> float:
        """Calculate correlation between two traits."""
        
        if len(self.trait_history) < 2:
            return 0.0
            
        history = np.array([
            [state[trait1], state[trait2]]
            for state in self.trait_history
        ])
        
        return np.corrcoef(history.T)[0, 1]
    
    def get_trait_stability(self, trait: str) -> float:
        """Calculate stability of a trait over time."""
        
        if len(self.trait_history) < 2:
            return 1.0
            
        values = np.array([
            state[trait] for state in self.trait_history
        ])
        
        return 1.0 - np.std(values)
    
    def get_dominant_traits(self, threshold: float = 0.7) -> List[str]:
        """Get list of traits above threshold."""
        
        return [
            trait for trait, value in self.traits.items()
            if value >= threshold
        ]
