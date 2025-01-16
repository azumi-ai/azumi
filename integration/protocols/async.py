from typing import Dict, List, Optional, Any, Callable
import asyncio
from dataclasses import dataclass
import logging
import time
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)

@dataclass
class AsyncProtocolConfig:
    buffer_size: int = 1000
    flush_interval: float = 1.0
    max_concurrent: int = 100
    timeout: float = 30.0

class AsyncProtocol:
    """Asynchronous communication protocol implementation."""
    
    def __init__(self, config: AsyncProtocolConfig):
        self.config = config
        self.buffer = asyncio.Queue(maxsize=config.buffer_size)
        self.handlers: Dict[str, Callable] = {}
        self._semaphore = asyncio.Semaphore(config.max_concurrent)
        self._active = False
        
    async def register_handler(
        self,
        event_type: str,
        handler: Callable
    ) -> None:
        """Register event handler."""
        self.handlers[event_type] = handler
        
    async def emit(
        self,
        event_type: str,
        data: Any,
        priority: int = 0
    ) -> None:
        """Emit asynchronous event."""
        if not self._active:
            raise RuntimeError("Protocol not started")
            
        event = {
            'type': event_type,
            'data': data,
            'priority': priority,
            'timestamp': time.time()
        }
        
        await self.buffer.put(event)
