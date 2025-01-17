from typing import Dict, List, Optional, Any, Tuple, Set
import asyncio
from dataclasses import dataclass
import numpy as np
import uuid
from collections import defaultdict
import logging
from rtree import index
import heapq
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)

@dataclass
class WorldConfig:
    size: Tuple[float, float, float] = (1000.0, 1000.0, 1000.0)
    cell_size: float = 50.0
    update_rate: float = 1/60  # 60 FPS
    interaction_radius: float = 10.0
    max_entities: int = 10000
    max_events_per_frame: int = 1000
    physics_substeps: int = 3
    thread_pool_size: int = 4

class SpatialGrid:
    """Optimized spatial partitioning system."""
    
    def __init__(self, size: Tuple[float, float, float], cell_size: float):
        self.size = size
        self.cell_size = cell_size
        self.grid_dimensions = tuple(
            int(np.ceil(s / cell_size)) for s in size
        )
        self.cells = defaultdict(set)
        self.entity_positions = {}
        
        # R-tree spatial index
        self.spatial_index = index.Index()
        self.index_counter = 0
    
    def add_entity(self, entity_id: str, position: np.ndarray) -> None:
        """Add entity to spatial grid."""
        cell = self._get_cell(position)
        self.cells[cell].add(entity_id)
        self.entity_positions[entity_id] = position
        
        # Update R-tree
        bounds = self._get_entity_bounds(position)
        self.spatial_index.insert(self.index_counter, bounds)
        self.index_counter += 1
    
    def remove_entity(self, entity_id: str) -> None:
        """Remove entity from spatial grid."""
        if entity_id in self.entity_positions:
            position = self.entity_positions[entity_id]
            cell = self._get_cell(position)
            self.cells[cell].discard(entity_id)
            del self.entity_positions[entity_id]
    
    def update_entity(self, entity_id: str, position: np.ndarray) -> None:
        """Update entity position."""
        if entity_id in self.entity_positions:
            old_cell = self._get_cell(self.entity_positions[entity_id])
            new_cell = self._get_cell(position)
            
            if old_cell != new_cell:
                self.cells[old_cell].discard(entity_id)
                self.cells[new_cell].add(entity_id)
            
            self.entity_positions[entity_id] = position
    
    def get_nearby_entities(
        self,
        position: np.ndarray,
        radius: float
    ) -> Set[str]:
        """Get entities within radius using R-tree."""
        bounds = self._get_search_bounds(position, radius)
        return set(self.spatial_index.intersection(bounds))
    
    def _get_cell(self, position: np.ndarray) -> Tuple[int, int, int]:
        """Get grid cell for position."""
        return tuple(
            int(p / self.cell_size) for p in position
        )
    
    def _get_entity_bounds(
        self,
        position: np.ndarray
    ) -> Tuple[float, float, float, float, float, float]:
        """Get entity bounds for R-tree."""
        return (
            position[0] - 0.5,
            position[1] - 0.5,
            position[2] - 0.5,
            position[0] + 0.5,
            position[1] + 0.5,
            position[2] + 0.5
        )
    
    def _get_search_bounds(
        self,
        position: np.ndarray,
        radius: float
    ) -> Tuple[float, float, float, float, float, float]:
        """Get search bounds for R-tree."""
        return (
            position[0] - radius,
            position[1] - radius,
            position[2] - radius,
            position[0] + radius,
            position[1] + radius,
            position[2] + radius
        )

class PhysicsEngine:
    """Optimized physics simulation system."""
    
    def __init__(self, substeps: int = 3):
        self.substeps = substeps
        self.gravity = np.array([0, -9.81, 0])
        self.colliders = {}
        self.dynamic_entities = set()
    
    async def update(
        self,
        dt: float,
        entities: Dict[str, Dict[str, Any]]
    ) -> None:
        """Update physics with substeps."""
        substep_dt = dt / self.substeps
        
        for _ in range(self.substeps):
            await self._update_substep(substep_dt, entities)
    
    async def _update_substep(
        self,
        dt: float,
        entities: Dict[str, Dict[str, Any]]
    ) -> None:
        """Process single physics substep."""
        for entity_id in self.dynamic_entities:
            if entity_id not in entities:
                continue
                
            entity = entities[entity_id]
            if 'physics' not in entity:
                continue
            
            # Update velocity
            entity['physics']['velocity'] += self.gravity * dt
            
            # Update position
            new_position = (
                entity['position'] +
                entity['physics']['velocity'] * dt
            )
            
            # Check collisions
            collisions = self._check_collisions(
                entity_id,
                new_position
            )
            
            if not collisions:
                entity['position'] = new_position
            else:
                # Handle collisions
                self._resolve_collisions(entity, collisions)

class GameWorld:
    """Optimized game world management system."""
    
    def __init__(self, config: WorldConfig):
        self.config = config
        self.entities = {}
        self.spatial_grid = SpatialGrid(config.size, config.cell_size)
        self.physics = PhysicsEngine(config.physics_substeps)
        self.event_queue = asyncio.PriorityQueue(
            maxsize=config.max_events_per_frame
        )
        self.update_callbacks = defaultdict(list)
        self._running = False
        self._executor = ThreadPoolExecutor(
            max_workers=config.thread_pool_size
        )
        
        # Performance monitoring
        self.metrics = defaultdict(float)
        self.frame_times = []
    
    async def start(self) -> None:
        """Start world simulation."""
        self._running = True
        try:
            await self._simulation_loop()
        except Exception as e:
            logger.error(f"Simulation error: {e}")
            raise
        finally:
            self._running = False
    
    async def stop(self) -> None:
        """Stop world simulation."""
        self._running = False
        self._executor.shutdown(wait=True)
    
    async def add_entity(
        self,
        entity_data: Dict[str, Any],
        position: Optional[np.ndarray] = None
    ) -> str:
        """Add entity to world."""
        if len(self.entities) >= self.config.max_entities:
            raise ValueError("Maximum entity limit reached")
            
        entity_id = str(uuid.uuid4())
        position = position or np.zeros(3)
        
        entity = {
            'id': entity_id,
            'data': entity_data,
            'position': position,
            'active': True,
            'created_at': asyncio.get_event_loop().time()
        }
        
        # Add physics component if needed
        if entity_data.get('physics_enabled', False):
            entity['physics'] = {
                'velocity': np.zeros(3),
                'mass': entity_data.get('mass', 1.0)
            }
            self.physics.dynamic_entities.add(entity_id)
        
        self.entities[entity_id] = entity
        self.spatial_grid.add_entity(entity_id, position)
        
        return entity_id
    
    async def remove_entity(self, entity_id: str) -> None:
        """Remove entity from world."""
        if entity_id in self.entities:
            self.spatial_grid.remove_entity(entity_id)
            self.physics.dynamic_entities.discard(entity_id)
            del self.entities[entity_id]
    
    async def update_entity(
        self,
        entity_id: str,
        updates: Dict[str, Any]
    ) -> None:
        """Update entity state."""
        if entity_id not in self.entities:
            return
            
        entity = self.entities[entity_id]
        
        # Update position if provided
        if 'position' in updates:
            new_position = np.array(updates['position'])
            self.spatial_grid.update_entity(entity_id, new_position)
            entity['position'] = new_position
        
        # Update other attributes
        entity['data'].update(updates.get('data', {}))
        
        # Queue interaction events if needed
        if 'interaction' in updates:
            await self._queue_interaction(entity_id, updates['interaction'])
    
    async def _simulation_loop(self) -> None:
        """Main simulation loop."""
        while self._running:
            start_time = asyncio.get_event_loop().time()
            
            try:
                # Process physics
                await self.physics.update(
                    self.config.update_rate,
                    self.entities
                )
                
                # Process events
                await self._process_events()
                
                # Update entities
                await self._update_entities()
                
                # Calculate frame time
                frame_time = (
                    asyncio.get_event_loop().time() - start_time
                )
                self.frame_times.append(frame_time)
                
                # Maintain frame rate
                if frame_time < self.config.update_rate:
                    await asyncio.sleep(
                        self.config.update_rate - frame_time
                    )
                
            except Exception as e:
                logger.error(f"Simulation loop error: {e}")
    
    async def _process_events(self) -> None:
        """Process world events."""
        processed = 0
        while not self.event_queue.empty() and processed < self.config.max_events_per_frame:
            try:
                priority, event = await self.event_queue.get()
                await self._handle_event(event)
                processed += 1
            except Exception as e:
                logger.error(f"Event processing error: {e}")
    
    async def _update_entities(self) -> None:
        """Update all entities."""
        update_tasks = []
        
        for entity_id, entity in self.entities.items():
            if not entity['active']:
                continue
                
            for callback in self.update_callbacks[entity_id]:
                task = asyncio.create_task(callback(entity))
                update_tasks.append(task)
        
        if update_tasks:
            await asyncio.gather(*update_tasks, return_exceptions=True)
    
    async def get_nearby_entities(
        self,
        position: np.ndarray,
        radius: Optional[float] = None
    ) -> List[str]:
        """Get entities near position."""
        radius = radius or self.config.interaction_radius
        return list(
            self.spatial_grid.get_nearby_entities(position, radius)
        )
    
    async def get_metrics(self) -> Dict[str, float]:
        """Get world performance metrics."""
        return {
            'entity_count': len(self.entities),
            'active_entities': sum(
                1 for e in self.entities.values() if e['active']
            ),
            'average_frame_time': np.mean(self.frame_times[-100:]),
            'event_queue_size': self.event_queue.qsize(),
            'physics_entities': len(self.physics.dynamic_entities)
        }
    
    def __del__(self):
        """Cleanup resources."""
        if hasattr(self, '_executor'):
            self._executor.shutdown(wait=False)
