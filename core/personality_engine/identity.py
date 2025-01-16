from dataclasses import dataclass, field
from typing import List, Dict, Optional
import numpy as np
from concurrent.futures import ThreadPoolExecutor

@dataclass
class CoreIdentity:
    name: str
    traits: List[str]
    values: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    creation_timestamp: float = field(default_factory=time.time)

class Identity:
    def __init__(self, 
                 name: str, 
                 base_traits: List[str],
                 trait_weights: Optional[Dict[str, float]] = None):
        self.core = CoreIdentity(
            name=name,
            traits=base_traits,
            values=self._initialize_trait_values(base_traits, trait_weights)
        )
        self.personality_vector = self._generate_personality_vector()
        self._trait_lock = threading.Lock()
        self._executor = ThreadPoolExecutor(max_workers=4)
        
    def _initialize_trait_values(self, 
                               traits: List[str], 
                               weights: Optional[Dict[str, float]] = None) -> Dict[str, float]:
        if weights:
            return {trait: weights.get(trait, np.random.random()) for trait in traits}
        return {trait: np.random.random() for trait in traits}
    
    def _generate_personality_vector(self) -> np.ndarray:
        with self._trait_lock:
            return np.array([self.core.values[trait] for trait in self.core.traits])
    
    async def update_trait(self, trait: str, value: float) -> None:
        if trait not in self.core.traits:
            raise ValueError(f"Trait '{trait}' not found in personality")
            
        with self._trait_lock:
            self.core.values[trait] = np.clip(value, 0.0, 1.0)
            self.personality_vector = self._generate_personality_vector()
            
    async def update_multiple_traits(self, 
                                   trait_updates: Dict[str, float]) -> None:
        updates = []
        for trait, value in trait_updates.items():
            updates.append(self.update_trait(trait, value))
        await asyncio.gather(*updates)
    
    def get_trait_correlation(self, trait1: str, trait2: str) -> float:
        if trait1 not in self.core.traits or trait2 not in self.core.traits:
            raise ValueError("Invalid trait names")
        return np.corrcoef([self.core.values[trait1]], [self.core.values[trait2]])[0, 1]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.core.name,
            'traits': self.core.traits,
            'values': self.core.values,
            'metadata': self.core.metadata,
            'creation_timestamp': self.core.creation_timestamp
        }
