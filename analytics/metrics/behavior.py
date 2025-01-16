from typing import Dict, List, Optional, Any
import numpy as np
from dataclasses import dataclass
import pandas as pd
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)

@dataclass
class BehaviorMetricsConfig:
    window_size: int = 1000
    min_samples: int = 10
    update_interval: float = 1.0

class BehaviorMetrics:
    """Behavior pattern analysis metrics."""
    
    def __init__(self, config: BehaviorMetricsConfig):
        self.config = config
        self.metrics = defaultdict(list)
        self.patterns = {}
        self._initialize_analyzers()
        
    def _initialize_analyzers(self) -> None:
        """Initialize behavior analyzers."""
        self.analyzers = {
            'consistency': ConsistencyAnalyzer(),
            'variability': VariabilityAnalyzer(),
            'trend': TrendAnalyzer(),
            'anomaly': AnomalyDetector()
        }
        
    async def analyze_behavior(
        self,
        behavior_data: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Analyze behavior patterns."""
        results = {}
        
        for name, analyzer in self.analyzers.items():
            try:
                results[name] = await analyzer.analyze(
                    behavior_data,
                    context
                )
            except Exception as e:
                logger.error(f"Analyzer error ({name}): {e}")
                results[name] = None
                
        return results
