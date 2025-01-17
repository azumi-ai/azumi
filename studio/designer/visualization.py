from typing import Dict, List, Optional, Any, Union
import asyncio
from dataclasses import dataclass
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd
from dash import Dash, dcc, html, Input, Output, State
from dash.exceptions import PreventUpdate
import networkx as nx
from cachetools import TTLCache, LRUCache
import time

@dataclass
class VisualizationConfig:
    update_interval: int = 100  # milliseconds
    max_points: int = 10000
    cache_ttl: int = 60  # seconds
    batch_size: int = 1000
    data_retention: int = 3600  # seconds

class DataStreamer:
    """Efficient data streaming for real-time visualization."""
    
    def __init__(self, config: VisualizationConfig):
        self.config = config
        self.streams = {}
        self.buffers = {}
        self._callbacks = {}
        
    async def register_stream(
        self,
        stream_id: str,
        callback: callable
    ) -> None:
        """Register data stream with callback."""
        self.streams[stream_id] = {
            'last_update': time.time(),
            'data': []
        }
        self._callbacks[stream_id] = callback
        
    async def update_stream(
        self,
        stream_id: str,
        data: Any
    ) -> None:
        """Update stream data."""
        if stream_id not in self.streams:
            return
            
        stream = self.streams[stream_id]
        stream['data'].append(data)
        stream['last_update'] = time.time()
        
        # Trim old data
        if len(stream['data']) > self.config.max_points:
            stream['data'] = stream['data'][-self.config.max_points:]
        
        # Trigger callback
        if stream_id in self._callbacks:
            await self._callbacks[stream_id](data)

class VisualizationManager:
    """Enhanced visualization management system."""
    
    def __init__(self, config: Optional[VisualizationConfig] = None):
        self.config = config or VisualizationConfig()
        self.app = Dash(__name__)
        self.figures = {}
        self.data_streamer = DataStreamer(self.config)
        self.cache = TTLCache(maxsize=1000, ttl=self.config.cache_ttl)
        
        self._setup_layout()
        self._setup_callbacks()
    
    def _setup_layout(self) -> None:
        """Setup dashboard layout."""
        self.app.layout = html.Div([
            dcc.Store(id='visualization-store'),
            html.Div(id='visualization-container'),
            dcc.Interval(
                id='interval-component',
                interval=self.config.update_interval,
                n_intervals=0
            )
        ])
    
    def _setup_callbacks(self) -> None:
        """Setup interactive callbacks."""
        @self.app.callback(
            Output('visualization-store', 'data'),
            Input('interval-component', 'n_intervals')
        )
        def update_store(n):
            if not n:
                raise PreventUpdate
            return self._get_current_data()
        
        @self.app.callback(
            Output('visualization-container', 'children'),
            Input('visualization-store', 'data')
        )
        def update_visualizations(data):
            if not data:
                raise PreventUpdate
            return self._create_visualization_layout(data)
    
    async def create_visualization(
        self,
        viz_id: str,
        viz_type: str,
        data: Any,
        options: Optional[Dict[str, Any]] = None
    ) -> None:
        """Create new visualization."""
        if viz_type == 'real_time_line':
            fig = await self._create_real_time_line(data, options)
        elif viz_type == 'network_graph':
            fig = await self._create_network_graph(data, options)
        elif viz_type == 'heatmap':
            fig = await self._create_heatmap(data, options)
        elif viz_type == '3d_scatter':
            fig = await self._create_3d_scatter(data, options)
        else:
            raise ValueError(f"Unknown visualization type: {viz_type}")
        
        self.figures[viz_id] = {
            'type': viz_type,
            'figure': fig,
            'options': options or {}
        }
    
    async def _create_real_time_line(
        self,
        data: pd.DataFrame,
        options: Optional[Dict[str, Any]] = None
    ) -> go.Figure:
        """Create real-time line chart."""
        fig = go.Figure()
        
        for column in data.select_dtypes(include=[np.number]).columns:
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data[column],
                    name=column,
                    mode='lines',
                    line=dict(width=2),
                )
            )
        
        fig.update_layout(
            uirevision=True,  # Preserve zoom on updates
            showlegend=True,
            margin=dict(l=20, r=20, t=40, b=20),
            hovermode='closest',
            **options.get('layout', {}) if options else {}
        )
        
        return fig
    
    async def _create_network_graph(
        self,
        data: nx.Graph,
        options: Optional[Dict[str, Any]] = None
    ) -> go.Figure:
        """Create interactive network graph."""
        pos = nx.spring_layout(data)
        
        edge_trace = go.Scatter(
            x=[],
            y=[],
            line=dict(width=0.5, color='#888'),
            hoverinfo='none',
            mode='lines'
        )
        
        for edge in data.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_trace['x'] += (x0, x1, None)
            edge_trace['y'] += (y0, y1, None)
        
        node_trace = go.Scatter(
            x=[pos[node][0] for node in data.nodes()],
            y=[pos[node][1] for node in data.nodes()],
            mode='markers+text',
            hoverinfo='text',
            marker=dict(
                size=10,
                line=dict(width=2)
            )
        )
        
        fig = go.Figure(
            data=[edge_trace, node_trace],
            layout=go.Layout(
                showlegend=False,
                hovermode='closest',
                margin=dict(b=20, l=5, r=5, t=40),
                **options.get('layout', {}) if options else {}
            )
        )
        
        return fig
    
    async def update_visualization(
        self,
        viz_id: str,
        data: Any
    ) -> None:
        """Update existing visualization."""
        if viz_id not in self.figures:
            return
            
        viz = self.figures[viz_id]
        
        if viz['type'] == 'real_time_line':
            await self._update_real_time_line(viz, data)
        elif viz['type'] == 'network_graph':
            await self._update_network_graph(viz, data)
        elif viz['type'] == 'heatmap':
            await self._update_heatmap(viz, data)
        elif viz['type'] == '3d_scatter':
            await self._update_3d_scatter(viz, data)
    
    async def _update_real_time_line(
        self,
        viz: Dict[str, Any],
        data: pd.DataFrame
    ) -> None:
        """Update real-time line chart."""
        with viz['figure'].batch_update():
            for i, column in enumerate(
                data.select_dtypes(include=[np.number]).columns
            ):
                viz['figure'].data[i].x = data.index
                viz['figure'].data[i].y = data[column]
    
    def run(
        self,
        host: str = '0.0.0.0',
        port: int = 8050,
        debug: bool = False
    ) -> None:
        """Run visualization server."""
        self.app.run_server(
            host=host,
            port=port,
            debug=debug
        )
