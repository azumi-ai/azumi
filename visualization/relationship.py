from typing import Dict, List, Optional, Any
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from dataclasses import dataclass

@dataclass
class RelationshipVisualizerConfig:
    node_size_factor: float = 1000
    edge_width_factor: float = 2
    color_scheme: str = 'plasma'

class RelationshipVisualizer:
    """Character relationship visualization system."""
    
    def __init__(self, config: RelationshipVisualizerConfig):
        self.config = config
        self.graph = nx.Graph()
        self.pos = None
        
    def create_visualization(
        self,
        relationships: Dict[str, Dict[str, float]],
        character_data: Optional[Dict[str, Any]] = None
    ) -> plt.Figure:
        """Generate relationship visualization."""
        
        # Create graph
        self.graph.clear()
        for char1, relations in relationships.items():
            for char2, strength in relations.items():
                self.graph.add_edge(
                    char1,
                    char2,
                    weight=strength
                )
        
        # Calculate layout
        self.pos = nx.spring_layout(self.graph)
        
        # Create visualization
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Draw nodes
        nx.draw_networkx_nodes(
            self.graph,
            self.pos,
            node_size=[
                self.config.node_size_factor * d
                for n, d in self.graph.degree()
            ],
            node_color=list(range(len(self.graph))),
            cmap=plt.get_cmap(self.config.color_scheme)
        )
        
        # Draw edges
        nx.draw_networkx_edges(
            self.graph,
            self.pos,
            width=[
                self.config.edge_width_factor * d['weight']
                for (u, v, d) in self.graph.edges(data=True)
            ]
        )
        
        # Add labels
        nx.draw_networkx_labels(self.graph, self.pos)
        
        return fig
