from typing import Dict, List, Optional, Any
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd
from dataclasses import dataclass

@dataclass
class PlotConfig:
    width: int = 800
    height: int = 600
    template: str = "plotly_white"
    color_scheme: str = "viridis"

class PlotGenerator:
    """Data visualization plot generator."""
    
    def __init__(self, config: PlotConfig):
        self.config = config
        
    def create_plot(
        self,
        data: Union[pd.DataFrame, Dict[str, Any]],
        plot_type: str,
        parameters: Optional[Dict[str, Any]] = None
    ) -> go.Figure:
        """Create visualization plot."""
        if isinstance(data, dict):
            data = pd.DataFrame(data)
            
        if plot_type == 'line':
            return self._create_line_plot(data, parameters)
        elif plot_type == 'scatter':
            return self._create_scatter_plot(data, parameters)
        elif plot_type == 'bar':
            return self._create_bar_plot(data, parameters)
        elif plot_type == 'heatmap':
            return self._create_heatmap(data, parameters)
        else:
            raise ValueError(f"Unknown plot type: {plot_type}")
