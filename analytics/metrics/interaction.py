from typing import Dict, List, Optional, Any
import numpy as np
from dataclasses import dataclass
import pandas as pd
import logging
from scipy import stats

logger = logging.getLogger(__name__)

@dataclass
class InteractionMetricsConfig:
    sequence_length: int = 10
    min_confidence: float = 0.8
    smoothing_factor: float = 0.1

class InteractionMetrics:
    """Interaction analysis metrics system."""
    
    def __init__(self, config: InteractionMetricsConfig):
        self.config = config
        self.sequences = []
        self.patterns = {}
        
    async def analyze_interaction(
        self,
        interaction: Dict[str, Any],
        history: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """Analyze interaction patterns."""
        # Update sequence history
        sequence = await self._extract_sequence(interaction)
        self.sequences.append(sequence)
        
        # Trim old sequences
        if len(self.sequences) > self.config.sequence_length:
            self.sequences.pop(0)
            
        # Analyze patterns
        patterns = await self._analyze_patterns(
            sequence,
            history
        )
        
        # Calculate metrics
        metrics = await self._calculate_metrics(patterns)
        
        return {
            'sequence': sequence,
            'patterns': patterns,
            'metrics': metrics,
            'confidence': await self._calculate_confidence(patterns)
        }
