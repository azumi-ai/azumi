import torch
import torch.nn as nn
from typing import Dict, List, Optional, Any
import plotly.graph_objects as go
import networkx as nx
import numpy as np
from dataclasses import dataclass

@dataclass
class VisualizationConfig:
    width: int = 800
    height: int = 600
    theme: str = "light"
    animation_frame_duration: int = 500

class PersonalityVisualizer:
    """Visualizes personality traits and relationships."""
    
    def __init__(self, config: VisualizationConfig):
        self.config = config
        self.figures = {}
        
    def create_trait_visualization(
        self,
        traits: Dict[str, float],
        history: Optional[List[Dict[str, float]]] = None
    ) -> go.Figure:
        """Create interactive trait visualization."""
        # Create radar chart for traits
        categories = list(traits.keys())
        values = list(traits.values())
        
        fig = go.Figure()
        
        # Add current traits
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories,
            fill='toself',
            name='Current Traits'
        ))
        
        # Add historical traits if available
        if history:
            for i, past_traits in enumerate(history[-5:]):  # Show last 5 states
                fig.add_trace(go.Scatterpolar(
                    r=list(past_traits.values()),
                    theta=categories,
                    name=f'Past State {i}',
                    opacity=0.3
                ))
        
        # Update layout
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )
            ),
            showlegend=True,
            width=self.config.width,
            height=self.config.height,
            template=self.config.theme
        )
        
        return fig
    
    def create_relationship_graph(
        self,
        relationships: Dict[str, Dict[str, float]],
        character_data: Optional[Dict[str, Any]] = None
    ) -> go.Figure:
        """Create interactive relationship graph visualization."""
        G = nx.Graph()
        
        # Add nodes (characters)
        for character in relationships.keys():
            G.add_node(
                character,
                data=character_data.get(character, {}) if character_data else {}
            )
        
        # Add edges (relationships)
        for char1, relations in relationships.items():
            for char2, strength in relations.items():
                G.add_edge(char1, char2, weight=strength)
        
        # Calculate layout
        pos = nx.spring_layout(G)
        
        # Create visualization
        fig = go.Figure()
        
        # Add edges
        edge_x = []
        edge_y = []
        edge_weights = []
        
        for edge in G.edges(data=True):
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            edge_weights.append(edge[2]['weight'])
        
        fig.add_trace(go.Scatter(
            x=edge_x,
            y=edge_y,
            line=dict(
                width=1,
                color='#888'
            ),
            hoverinfo='none',
            mode='lines'
        ))
        
        # Add nodes
        node_x = []
        node_y = []
        node_text = []
        
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            node_text.append(str(node))
        
        fig.add_trace(go.Scatter(
            x=node_x,
            y=node_y,
            mode='markers+text',
            hoverinfo='text',
            text=node_text,
            textposition='top center',
            marker=dict(
                size=20,
                color='#1f77b4',
                line=dict(
                    width=2,
                    color='#fff'
                )
            )
        ))
        
        # Update layout
        fig.update_layout(
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20, l=5, r=5, t=40),
            width=self.config.width,
            height=self.config.height,
            template=self.config.theme
        )
        
        return fig
