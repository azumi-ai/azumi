from typing import Dict, List, Any
import numpy as np
from ..core.personality_engine import Identity

class Environment:
    def __init__(self):
        self.state = {}
        self.entities = []
        self.rules = self._initialize_rules()
    
    def _initialize_rules(self) -> Dict[str, Any]:
        return {
            'interaction_range': 1.0,
            'influence_threshold': 0.5,
            'update_frequency': 10
        }
    
    def add_entity(self, entity: Identity, position: List[float]) -> None:
        self.entities.append({
            'identity': entity,
            'position': np.array(position),
            'state': {}
        })
    
    def update(self) -> None:
        """Update environment state based on entities and rules"""
        for entity in self.entities:
            nearby = self._get_nearby_entities(entity)
            self._process_interactions(entity, nearby)
    
    def _get_nearby_entities(self, entity: Dict[str, Any]) -> List[Dict[str, Any]]:
        nearby = []
        for other in self.entities:
            if other != entity:
                distance = np.linalg.norm(
                    entity['position'] - other['position']
                )
                if distance <= self.rules['interaction_range']:
                    nearby.append(other)
        return nearby
    
    def _process_interactions(self, 
                            entity: Dict[str, Any],
                            nearby: List[Dict[str, Any]]) -> None:
        for other in nearby:
            influence = self._calculate_influence(entity, other)
            if influence > self.rules
          
         def _calculate_influence(self, 
                           entity: Dict[str, Any],
                           other: Dict[str, Any]) -> float:
        personality_similarity = np.dot(
            entity['identity'].personality_vector,
            other['identity'].personality_vector
        )
        distance = np.linalg.norm(entity['position'] - other['position'])
        return personality_similarity * (1 / (1 + distance))
