from typing import Dict, List, Optional, Any, AsyncGenerator
import asyncio
from dataclasses import dataclass
import logging
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)

@dataclass
class StreamingConfig:
    chunk_size: int = 1024
    buffer_size: int = 8192
    compression_level: int = 6
    heartbeat_interval: float = 5.0

class StreamingProtocol:
    """Streaming data protocol implementation."""
    
    def __init__(self, config: StreamingConfig):
        self.config = config
        self.streams = {}
        self._active = False
        self._heartbeat_task = None
        
    @asynccontextmanager
    async def stream(self, stream_id: str):
        """Create and manage data stream."""
        try:
            stream = await self._create_stream(stream_id)
            yield stream
        finally:
            await self._close_stream(stream_id)
            
    async def write(
        self,
        stream_id: str,
        data: Any
    ) -> None:
        """Write data to stream."""
        if stream_id not in self.streams:
            raise ValueError(f"Stream not found: {stream_id}")
            
        stream = self.streams[stream_id]
        chunk = await self._prepare_chunk(data)
        await stream.write(chunk)
