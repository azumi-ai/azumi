from typing import Dict, List, Optional, Any, Callable
import asyncio
import json
import time
from dataclasses import dataclass
import logging
import numpy as np
from collections import defaultdict, deque
import plotly.graph_objects as go
from dash import dcc, html
from dash.dependencies import Input, Output
import pandas as pd
from functools import lru_cache

logger = logging.getLogger(__name__)

@dataclass
class InteractiveConfig:
    update_interval: int = 1000  # ms
    max_points: int = 1000
    batch_size: int = 100
    height: str = "600px"
    cache_ttl: int = 60  # seconds
    throttle_rate: float = 0.1  # seconds
    websocket_buffer: int = 1000

class DataBuffer:
    """Efficient data buffering system."""
    
    def __init__(self, max_size: int = 1000):
        self.buffer = deque(maxlen=max_size)
        self.timestamps = deque(maxlen=max_size)
        self._last_update = time.time()
        
    def add(self, data: Any) -> None:
        """Add data to buffer."""
        self.buffer.append(data)
        self.timestamps.append(time.time())
        
    def get_since(self, timestamp: float) -> List[Any]:
        """Get data since timestamp."""
        if not self.timestamps:
            return []
            
        idx = self._binary_search(timestamp)
        return list(self.buffer)[idx:]
    
    def _binary_search(self, timestamp: float) -> int:
        """Binary search for timestamp index."""
        timestamps = list(self.timestamps)
        left, right = 0, len(timestamps)
        
        while left < right:
            mid = (left + right) // 2
            if timestamps[mid] < timestamp:
                left = mid + 1
            else:
                right = mid
                
        return left

class DataStreamManager:
    """Real-time data stream management."""
    
    def __init__(self, buffer_size: int = 1000):
        self.streams = defaultdict(
            lambda: DataBuffer(buffer_size)
        )
        self.subscribers = defaultdict(set)
        self._lock = asyncio.Lock()
        
    async def publish(
        self,
        stream_id: str,
        data: Any
    ) -> None:
        """Publish data to stream."""
        async with self._lock:
            self.streams[stream_id].add(data)
            
            # Notify subscribers
            subscribers = self.subscribers[stream_id].copy()
            for callback in subscribers:
                try:
                    await callback(data)
                except Exception as e:
                    logger.error(f"Subscriber error: {e}")
                    
    async def subscribe(
        self,
        stream_id: str,
        callback: Callable
    ) -> None:
        """Subscribe to data stream."""
        self.subscribers[stream_id].add(callback)
        
    async def unsubscribe(
        self,
        stream_id: str,
        callback: Callable
    ) -> None:
        """Unsubscribe from data stream."""
        self.subscribers[stream_id].discard(callback)

class WebSocketManager:
    """WebSocket connection manager."""
    
    def __init__(self, buffer_size: int = 1000):
        self.connections = {}
        self.buffer = asyncio.Queue(maxsize=buffer_size)
        self._task = None
        
    async def start(self) -> None:
        """Start WebSocket manager."""
        self._task = asyncio.create_task(self._process_queue())
        
    async def stop(self) -> None:
        """Stop WebSocket manager."""
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            
    async def connect(self, websocket: Any, client_id: str) -> None:
        """Handle new WebSocket connection."""
        self.connections[client_id] = websocket
        
    async def disconnect(self, client_id: str) -> None:
        """Handle WebSocket disconnection."""
        if client_id in self.connections:
            del self.connections[client_id]
            
    async def broadcast(self, message: Dict[str, Any]) -> None:
        """Broadcast message to all connections."""
        await self.buffer.put(message)
        
    async def _process_queue(self) -> None:
        """Process message queue."""
        while True:
            try:
                message = await self.buffer.get()
                await self._send_to_all(message)
            except Exception as e:
                logger.error(f"Queue processing error: {e}")
                
    async def _send_to_all(self, message: Dict[str, Any]) -> None:
        """Send message to all connected clients."""
        disconnected = []
        
        for client_id, websocket in self.connections.items():
            try:
                await websocket.send_json(message)
            except Exception as e:
                logger.error(f"WebSocket error: {e}")
                disconnected.append(client_id)
                
        # Clean up disconnected clients
        for client_id in disconnected:
            await self.disconnect(client_id)

class InteractiveVisualizer:
    """Real-time interactive visualization system."""
    
    def __init__(self, config: Optional[InteractiveConfig] = None):
        self.config = config or InteractiveConfig()
        self.stream_manager = DataStreamManager(
            self.config.websocket_buffer
        )
        self.websocket_manager = WebSocketManager(
            self.config.websocket_buffer
        )
        self.figures = {}
        self.layouts = {}
        self.update_callbacks = {}
        
        # Initialize Dash app
        self.app = self._setup_dash_app()
        
    def _setup_dash_app(self):
        """Setup Dash application."""
        app = dash.Dash(__name__)
        
        app.layout = html.Div([
            html.H1("Azumi Analytics Dashboard"),
            html.Div(id='real-time-content'),
            dcc.Interval(
                id='interval-component',
                interval=self.config.update_interval,
                n_intervals=0
            )
        ])
        
        @app.callback(
            Output('real-time-content', 'children'),
            [Input('interval-component', 'n_intervals')]
        )
        def update_content(n):
            return self._generate_layout()
            
        return app
    
    @lru_cache(maxsize=100)
    def _generate_layout(self) -> List[html.Div]:
        """Generate dashboard layout."""
        layout = []
        
        for figure_id, figure_config in self.layouts.items():
            layout.append(
                html.Div([
                    html.H3(figure_config['title']),
                    dcc.Graph(
                        id=f'graph-{figure_id}',
                        figure=self.figures.get(figure_id),
                        config={'displayModeBar': False}
                    )
                ])
            )
            
        return layout
    
    async def add_visualization(
        self,
        viz_id: str,
        viz_type: str,
        config: Dict[str, Any]
    ) -> None:
        """Add new visualization."""
        self.layouts[viz_id] = {
            'type': viz_type,
            'title': config.get('title', ''),
            'config': config
        }
        
        # Create initial figure
        self.figures[viz_id] = self._create_figure(
            viz_type,
            config
        )
        
        # Setup update callback
        self.update_callbacks[viz_id] = self._create_update_callback(
            viz_type,
            config
        )
        
    async def update_data(
        self,
        viz_id: str,
        data: Any
    ) -> None:
        """Update visualization data."""
        if viz_id not in self.update_callbacks:
            return
            
        # Update figure
        self.figures[viz_id] = await self.update_callbacks[viz_id](
            self.figures[viz_id],
            data
        )
        
        # Broadcast update
        await self.websocket_manager.broadcast({
            'type': 'update',
            'viz_id': viz_id,
            'data': data
        })
        
    def _create_figure(
        self,
        viz_type: str,
        config: Dict[str, Any]
    ) -> go.Figure:
        """Create initial figure."""
        if viz_type == 'line':
            return self._create_line_chart(config)
        elif viz_type == 'scatter':
            return self._create_scatter_plot(config)
        elif viz_type == 'bar':
            return self._create_bar_chart(config)
        else:
            raise ValueError(f"Unknown visualization type: {viz_type}")
            
    def _create_update_callback(
        self,
        viz_type: str,
        config: Dict[str, Any]
    ) -> Callable:
        """Create visualization update callback."""
        if viz_type == 'line':
            return self._update_line_chart
        elif viz_type == 'scatter':
            return self._update_scatter_plot
        elif viz_type == 'bar':
            return self._update_bar_chart
        else:
            raise ValueError(f"Unknown visualization type: {viz_type}")
            
    async def run(self, host: str = "0.0.0.0", port: int = 8050) -> None:
        """Run visualization server."""
        await self.websocket_manager.start()
        self.app.run_server(host=host, port=port)
        
    async def shutdown(self) -> None:
        """Shutdown visualization server."""
        await self.websocket_manager.stop()
        
    def __del__(self):
        """Cleanup resources."""
        asyncio.create_task(self.shutdown())
