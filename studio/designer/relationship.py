from typing import Dict, List, Optional, Any
import numpy as np
from dataclasses import dataclass
import networkx as nx
import torch
import logging

logger = logging.getLogger(__name__)

@dataclass
class RelationshipConfig:
    min_strength: float = 0.0
    max_strength: float = 1.0
    decay_rate: float = 0.01
    influence_factor: float = 0.3

class RelationshipManager:
    """Manages character relationships and interactions."""
    
    def __init__(self, config: RelationshipConfig):
        self.config = config
        self.relationships = {}
        self.history = []
        self.graph = nx.DiGraph()
        
    def create_relationship(
        self,
        char1_id: str,
        char2_id: str,
        initial_strength: Optional[float] = None,
        attributes: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Create or update a relationship between characters."""
        if initial_strength is None:
            initial_strength = np.random.uniform(
                self.config.min_strength,
                self.config.max_strength
            )
            
        relationship = {
            'strength': initial_strength,
            'attributes': attributes or {},
            'history': [],
            'last_interaction': time.time()
        }
        
        # Store bidirectional relationship
        if char1_id not in self.relationships:
            self.relationships[char1_id] = {}
        if char2_id not in self.relationships:
            self.relationships[char2_id] = {}
            
        self.relationships[char1_id][char2_id] = relationship
        
        # Update graph
        self.graph.add_edge(
            char1_id,
            char2_id,
            **relationship
        )
        
        return relationship
    
    async def update_relationship(
        self,
        char1_id: str,
        char2_id: str,
        interaction: Dict[str, Any]
    ) -> None:
        """Update relationship based on interaction."""
        if not self._relationship_exists(char1_id, char2_id):
            self.create_relationship(char1_id, char2_id)
            
        rel = self.relationships[char1_id][char2_id]
        
        # Calculate interaction impact
        impact = self._calculate_interaction_impact(interaction)
        
        # Update relationship strength
        old_strength = rel['strength']
        new_strength = np.clip(
            old_strength + impact,
            self.config.min_strength,
            self.config.max_strength
        )
        
        rel['strength'] = new_strength
        rel['last_interaction'] = time.time()
        rel['history'].append({
            'timestamp': time.time(),
            'old_strength': old_strength,
            'new_strength': new_strength,
            'interaction': interaction
        })
        
        # Update graph
        self.graph[char1_id][char2_id]['strength'] = new_strength
