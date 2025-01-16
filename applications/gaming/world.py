from typing import Dict, List, Optional, Any
import numpy as np
from dataclasses import dataclass
import asyncio
import logging

logger = logging.getLogger(__name__)

@dataclass
class WorldConfig:
    size: List[int]  # [x, y, z]
    update_rate: float = 1/60  # 60 FPS
    interaction_radius: float = 10.0
    max_entities: int = 1000

class GameWorld:
    """Dynamic game world management system."""
    
    def __init__(self, config: WorldConfig):
        self.config = config
        self.entities = {}
        self.regions = {}
        self.physics = PhysicsEngine()
        self.event_queue = asyncio.Queue()
        self._running = False
        
    async def start(self) -> None:
        """Start world simulation."""
        self._running = True
        await self._simulation_loop()
        
    async def stop(self) -> None:
        """Stop world simulation."""
        self._running = False
        
    async def add_entity(
        self,
        entity_data: Dict[str, Any],
        position: Optional[List[float]] = None
    ) -> str:
        """Add entity to world."""
        if len(self.entities) >= self.config.max_entities:
            raise ValueError("Maximum entity limit reached")
            
        entity_id = str(uuid.uuid4())
        
        entity = {
            'id': entity_id,
            'data': entity_data,
            'position': position or [0, 0, 0],
            'velocity': [0, 0, 0],
            'state': 'active'
        }
        
        self.entities[entity_id] = entity
        await self._update_spatial_index(entity)
        
        return entity_id
