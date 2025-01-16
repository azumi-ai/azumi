from typing import Dict, List, Optional, Any
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from dataclasses import dataclass

@dataclass
class DashboardConfig:
    update_interval: int = 5000  # ms
    max_points: int = 1000
    color_scheme: str = 'viridis'

class AnalyticsDashboard:
    """Interactive analytics dashboard."""
    
    def __init__(self, config: DashboardConfig):
        self.config = config
        self.figures = {}
        self.data_sources = {}
        
    def create_dashboard(
        self,
        metrics: Dict[str, pd.DataFrame],
        layout: Optional[Dict[str, Any]] = None
    ) -> go.Figure:
        """Create interactive dashboard."""
        
        # Create subplots
        fig = self._create_layout(layout)
        
        # Add metric plots
        for metric_name, data in metrics.items():
            self._add_metric_plot(fig, metric_name, data)
            
        # Update layout
        fig.update_layout(
            showlegend=True,
            height=800,
            title_text="Azumi Analytics Dashboard",
            title_x=0.5
        )
        
        return fig
        
    def _create_layout(
        self,
        layout: Optional[Dict[str, Any]] = None
    ) -> go.Figure:
        """Create dashboard layout."""
        default_layout = {
            'rows': 3,
            'cols': 2,
            'specs': [[{'type': 'scatter'}] * 2] * 3
        }
        
        layout = layout or default_layout
        return make_subplots(**layout)
    
    def _add_metric_plot(
        self,
        fig: go.Figure,
        metric_name: str,
        data: pd.DataFrame,
        row: int = 1,
        col: int = 1
    ) -> None:
        """Add metric visualization to dashboard."""
        plot_type = self._determine_plot_type(data)
        
        if plot_type == 'time_series':
            self._add_time_series(fig, metric_name, data, row, col)
        elif plot_type == 'distribution':
            self._add_distribution(fig, metric_name, data, row, col)
        elif plot_type == 'correlation':
            self._add_correlation(fig, metric_name, data, row, col)
    
    def update_dashboard(
        self,
        metrics: Dict[str, pd.DataFrame]
    ) -> None:
        """Update dashboard with new data."""
        for metric_name, data in metrics.items():
            if metric_name in self.figures:
                self._update_metric_plot(metric_name, data)
