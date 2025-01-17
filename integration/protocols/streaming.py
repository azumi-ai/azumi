from typing import Dict, List, Optional, Any, Union, Callable, AsyncGenerator
import asyncio
from dataclasses import dataclass
import logging
import msgpack
import zlib
from collections import deque
import time
import aiobuffer
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)

@dataclass
class StreamConfig:
    chunk_size: int = 8192  # 8KB
    buffer_size: int = 32768  # 32KB
    compression_level: int = 6
    heartbeat_interval: float = 5.0
    flow_control_window: int = 65536  # 64KB
    max_retry_attempts: int = 3
    backoff_factor: float = 1.5
    compression_threshold: int = 1024  # 1KB

class CircularBuffer:
    """Efficient circular buffer implementation."""
    
    def __init__(self, max_size: int):
        self.buffer = aiobuffer.AsyncCircularBuffer(max_size)
        self.write_lock = asyncio.Lock()
        self.read_lock = asyncio.Lock()
        self._closed = False
        
    async def write(self, data: bytes) -> int:
        """Write data to buffer."""
        if self._closed:
            raise ValueError("Buffer is closed")
            
        async with self.write_lock:
            return await self.buffer.write(data)
            
    async def read(self, size: int = -1) -> bytes:
        """Read data from buffer."""
        if self._closed and self.buffer.empty():
            return b""
            
        async with self.read_lock:
            return await self.buffer.read(size)
            
    def close(self) -> None:
        """Close buffer."""
        self._closed = True
        
    @property
    def available(self) -> int:
        """Get available space."""
        return self.buffer.maxsize - len(self.buffer)

class FlowController:
    """Stream flow control system."""
    
    def __init__(self, window_size: int):
        self.window_size = window_size
        self.current_window = window_size
        self.window_lock = asyncio.Lock()
        self.window_update = asyncio.Event()
        self.window_update.set()
        
    async def acquire(self, size: int) -> None:
        """Acquire flow control window."""
        while size > 0:
            async with self.window_lock:
                if self.current_window >= size:
                    self.current_window -= size
                    return
                    
                available = self.current_window
                size -= available
                self.current_window = 0
                
            self.window_update.clear()
            await self.window_update.wait()
            
    async def release(self, size: int) -> None:
        """Release flow control window."""
        async with self.window_lock:
            self.current_window = min(
                self.current_window + size,
                self.window_size
            )
            self.window_update.set()

class DataSerializer:
    """Efficient data serialization system."""
    
    def __init__(self, compression_level: int = 6, threshold: int = 1024):
        self.compression_level = compression_level
        self.threshold = threshold
        
    def serialize(self, data: Any) -> bytes:
        """Serialize data with optional compression."""
        try:
            # Serialize with MessagePack
            serialized = msgpack.packb(
                data,
                use_bin_type=True
            )
            
            # Compress if over threshold
            if len(serialized) > self.threshold:
                compressed = zlib.compress(
                    serialized,
                    level=self.compression_level
                )
                if len(compressed) < len(serialized):
                    return b'c' + compressed
                    
            return b'r' + serialized
            
        except Exception as e:
            logger.error(f"Serialization error: {e}")
            raise
            
    def deserialize(self, data: bytes) -> Any:
        """Deserialize data."""
        try:
            # Check compression flag
            flag = data[0:1]
            payload = data[1:]
            
            if flag == b'c':
                # Decompress
                decompressed = zlib.decompress(payload)
                return msgpack.unpackb(
                    decompressed,
                    raw=False
                )
            else:
                # Raw data
                return msgpack.unpackb(
                    payload,
                    raw=False
                )
                
        except Exception as e:
            logger.error(f"Deserialization error: {e}")
            raise

class StreamingProtocol:
    """Enhanced streaming protocol implementation."""
    
    def __init__(self, config: Optional[StreamConfig] = None):
        self.config = config or StreamConfig()
        self.streams = {}
        self.buffers = {}
        self.flow_controllers = {}
        self.serializer = DataSerializer(
            compression_level=self.config.compression_level,
            threshold=self.config.compression_threshold
        )
        self._active = False
        self._heartbeat_task = None
        
        # Error handling
        self.error_handlers = {}
        self.retry_counters = {}
        
        # Metrics
        self.metrics = {
            'bytes_sent': 0,
            'bytes_received': 0,
            'messages_sent': 0,
            'messages_received': 0,
            'errors': 0,
            'retries': 0
        }
        
    @asynccontextmanager
    async def stream(self, stream_id: str):
        """Create and manage data stream."""
        try:
            stream = await self._create_stream(stream_id)
            yield stream
        finally:
            await self._close_stream(stream_id)
            
    async def _create_stream(self, stream_id: str) -> Dict[str, Any]:
        """Create new stream."""
        if stream_id in self.streams:
            raise ValueError(f"Stream already exists: {stream_id}")
            
        stream = {
            'id': stream_id,
            'created_at': time.time(),
            'status': 'active',
            'sequence': 0
        }
        
        # Initialize buffer
        self.buffers[stream_id] = CircularBuffer(
            self.config.buffer_size
        )
        
        # Initialize flow control
        self.flow_controllers[stream_id] = FlowController(
            self.config.flow_control_window
        )
        
        self.streams[stream_id] = stream
        return stream
        
    async def _close_stream(self, stream_id: str) -> None:
        """Close stream and cleanup resources."""
        if stream_id in self.streams:
            self.streams[stream_id]['status'] = 'closed'
            self.buffers[stream_id].close()
            del self.streams[stream_id]
            del self.buffers[stream_id]
            del self.flow_controllers[stream_id]
            
    async def write(
        self,
        stream_id: str,
        data: Any,
        timeout: Optional[float] = None
    ) -> int:
        """Write data to stream."""
        if stream_id not in self.streams:
            raise ValueError(f"Stream not found: {stream_id}")
            
        try:
            # Serialize data
            serialized = self.serializer.serialize(data)
            
            # Wait for flow control
            await self.flow_controllers[stream_id].acquire(
                len(serialized)
            )
            
            # Write to buffer
            bytes_written = await self.buffers[stream_id].write(
                serialized
            )
            
            # Update metrics
            self.metrics['bytes_sent'] += bytes_written
            self.metrics['messages_sent'] += 1
            
            return bytes_written
            
        except Exception as e:
            logger.error(f"Write error: {e}")
            self.metrics['errors'] += 1
            
            # Handle error
            await self._handle_error(stream_id, 'write', e)
            raise
            
    async def read(
        self,
        stream_id: str,
        size: int = -1,
        timeout: Optional[float] = None
    ) -> Any:
        """Read data from stream."""
        if stream_id not in self.streams:
            raise ValueError(f"Stream not found: {stream_id}")
            
        try:
            # Read from buffer
            data = await self.buffers[stream_id].read(size)
            
            if not data:
                return None
                
            # Deserialize data
            deserialized = self.serializer.deserialize(data)
            
            # Release flow control
            await self.flow_controllers[stream_id].release(
                len(data)
            )
            
            # Update metrics
            self.metrics['bytes_received'] += len(data)
            self.metrics['messages_received'] += 1
            
            return deserialized
            
        except Exception as e:
            logger.error(f"Read error: {e}")
            self.metrics['errors'] += 1
            
            # Handle error
            await self._handle_error(stream_id, 'read', e)
            raise
            
    async def start(self) -> None:
        """Start protocol."""
        self._active = True
        self._heartbeat_task = asyncio.create_task(
            self._heartbeat_loop()
        )
        
    async def stop(self) -> None:
        """Stop protocol."""
        self._active = False
        if self._heartbeat_task:
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass
                
        # Close all streams
        stream_ids = list(self.streams.keys())
        for stream_id in stream_ids:
            await self._close_stream(stream_id)
            
    async def register_error_handler(
        self,
        stream_id: str,
        handler: Callable
    ) -> None:
        """Register stream error handler."""
        self.error_handlers[stream_id] = handler
        
    async def _handle_error(
        self,
        stream_id: str,
        operation: str,
        error: Exception
    ) -> None:
        """Handle stream error."""
        if stream_id not in self.retry_counters:
            self.retry_counters[stream_id] = 0
            
        self.retry_counters[stream_id] += 1
        
        # Check retry limit
        if self.retry_counters[stream_id] > self.config.max_retry_attempts:
            logger.error(f"Max retries exceeded for stream {stream_id}")
            await self._close_stream(stream_id)
            return
            
        # Exponential backoff
        await asyncio.sleep(
            self.config.backoff_factor ** (self.retry_counters[stream_id] - 1)
        )
        
        # Call error handler
        if stream_id in self.error_handlers:
            try:
                await self.error_handlers[stream_id](
                    stream_id,
                    operation,
                    error
                )
            except Exception as e:
                logger.error(f"Error handler failed: {e}")
                
    async def _heartbeat_loop(self) -> None:
        """Heartbeat monitoring loop."""
        while self._active:
            try:
                await self._send_heartbeats()
                await asyncio.sleep(self.config.heartbeat_interval)
            except Exception as e:
                logger.error(f"Heartbeat error: {e}")
                await asyncio.sleep(1)
                
    async def _send_heartbeats(self) -> None:
        """Send heartbeats to active streams."""
        for stream_id, stream in self.streams.items():
            if stream['status'] != 'active':
                continue
                
            try:
                await self.write(
                    stream_id,
                    {'type': 'heartbeat', 'sequence': stream['sequence']},
                    timeout=1.0
                )
                stream['sequence'] += 1
            except Exception as e:
                logger.error(f"Heartbeat failed for stream {stream_id}: {e}")
                
    async def get_metrics(self) -> Dict[str, Any]:
        """Get protocol metrics."""
        return {
            **self.metrics,
            'active_streams': len(self.streams),
            'buffer_usage': {
                stream_id: len(buffer.buffer) / buffer.buffer.maxsize
                for stream_id, buffer in self.buffers.items()
            }
        }
        
    def __del__(self):
        """Cleanup resources."""
        if self._active:
            asyncio.create_task(self.stop())
