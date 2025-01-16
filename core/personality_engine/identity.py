from dataclasses import dataclass
from typing import List, Dict
import numpy as np

@dataclass
class CoreIdentity:
    name: str
    traits: List[str]
    values: Dict[str, float]

class Identity:
    def __init__(self, name: str, base_traits: List[str]):
        self.core = CoreIdentity(
            name=name,
            traits=base_traits,
            values={trait: np.random.random() for trait in base_traits}
        )
        self.personality_vector = self._generate_personality_vector()
    
    def _generate_personality_vector(self) -> np.ndarray:
        return np.array([self.core.values[trait] for trait in self.core.traits])
    
    def update_trait(self, trait: str, value: float) -> None:
        if trait in self.core.traits:
            self.core.values[trait] = max(0.0, min(1.0, value))
            self.personality_vector = self._generate_personality_vector()
