from typing import Dict, List, Optional, Any, Union
import asyncio
from dataclasses import dataclass
import logging
from contextlib import asynccontextmanager
import time

logger = logging.getLogger(__name__)

@dataclass
class SyncProtocolConfig:
    retry_attempts: int = 3
    timeout: float = 30.0
    batch_size: int = 100
    max_queue_size: int = 1000

class SyncProtocol:
    """Synchronous communication protocol implementation."""
    
    def __init__(self, config: SyncProtocolConfig):
        self.config = config
        self.queue = asyncio.Queue(maxsize=config.max_queue_size)
        self._active = False
        self._lock = asyncio.Lock()
        self._retry_delays = [1, 3, 5]  # Exponential backoff
        
    @asynccontextmanager
    async def session(self):
        """Create a managed protocol session."""
        try:
            await self.start()
            yield self
        finally:
            await self.stop()
    
    async def start(self) -> None:
        """Start protocol operations."""
        async with self._lock:
            if not self._active:
                self._active = True
                self._processor = asyncio.create_task(self._process_queue())
                
    async def stop(self) -> None:
        """Stop protocol operations."""
        async with self._lock:
            if self._active:
                self._active = False
                if hasattr(self, '_processor'):
                    await self._processor
                await self._clear_queue()
