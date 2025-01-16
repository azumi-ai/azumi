from typing import Dict, List, Optional, Any
import asyncio
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class GameEngineConfig:
    engine_type: str
    update_rate: float = 1/60  # 60 FPS
    buffer_size: int = 1000

class GameEngineConnector:
    """Game engine integration connector."""
    
    def __init__(self, config: GameEngineConfig):
        self.config = config
        self.characters = {}
        self.event_queue = asyncio.Queue(maxsize=config.buffer_size)
        self._running = False
    
    async def start(self) -> None:
        """Start game engine connector."""
        self._running = True
        await self._process_events()
    
    async def stop(self) -> None:
        """Stop game engine connector."""
        self._running = False
    
    async def spawn_character(
        self,
        character_data: Dict[str, Any],
        position: Optional[List[float]] = None
    ) -> str:
        """Spawn a character in the game world."""
        character_id = str(uuid.uuid4())
        
        character = {
            'id': character_id,
            'data': character_data,
            'position': position or [0, 0, 0],
            'state': 'spawning'
        }
        
        self.characters[character_id] = character
        
        # Queue spawn event
        await self.event_queue.put({
            'type': 'spawn',
            'character': character
        })
        
        return character_id
    
    async def update_character(
        self,
        character_id: str,
        updates: Dict[str, Any]
    ) -> None:
        """Update character in game world."""
        if character_id not in self.characters:
            raise ValueError(f"Character not found: {character_id}")
            
        character = self.characters[character_id]
        
        # Apply updates
        character['data'].update(updates)
        
        # Queue update event
        await self.event_queue.put({
            'type': 'update',
            'character_id': character_id,
            'updates': updates
        })
    
    async def _process_events(self) -> None:
        """Process game engine events."""
        while self._running:
            try:
                # Process all events in queue
                while not self.event_queue.empty():
                    event = await self.event_queue.get()
                    await self._handle_event(event)
                
                # Wait for next update
                await asyncio.sleep(self.config.update_rate)
                
            except Exception as e:
                logger.error(f"Event processing error: {e}")
    
    async def _handle_event(self, event: Dict[str, Any]) -> None:
        """Handle a single game engine event."""
        event_type = event.get('type')
        
        if event_type == 'spawn':
            await self._handle_spawn(event['character'])
        elif event_type == 'update':
            await self._handle_update(
                event['character_id'],
                event['updates']
            )
        else:
            logger.warning(f"Unknown event type: {event_type}")

# Initialize logging for all modules
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
