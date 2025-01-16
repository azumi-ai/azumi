from typing import List, Dict
import numpy as np
from .identity import Identity

class DynamicTraits:
    def __init__(self, identity: Identity):
        self.identity = identity
        self.evolution_rate = 0.1
        self.trait_interactions = self._initialize_trait_interactions()
    
    def _initialize_trait_interactions(self) -> Dict[str, Dict[str, float]]:
        traits = self.identity.core.traits
        interactions = {}
        for trait1 in traits:
            interactions[trait1] = {
                trait2: np.random.random() * 0.2 - 0.1
                for trait2 in traits if trait2 != trait1
            }
        return interactions
    
    def evolve_traits(self, context: Dict[str, float]) -> None:
        for trait, value in context.items():
            if trait in self.identity.core.traits:
                current_value = self.identity.core.values[trait]
                new_value = current_value + self.evolution_rate * (value - current_value)
                self.identity.update_trait(trait, new_value)
