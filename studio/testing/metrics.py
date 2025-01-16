from typing import Dict, List, Optional, Any
import numpy as np
from dataclasses import dataclass
import time
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)

@dataclass
class MetricsConfig:
    window_size: int = 100
    save_interval: int = 10
    metrics_types: List[str] = None

class TestingMetrics:
    """Collects and analyzes testing metrics."""
    
    def __init__(self, config: MetricsConfig):
        self.config = config
        self.metrics = defaultdict(list)
        self.aggregated = defaultdict(dict)
        
        # Initialize default metric types if none provided
        if not self.config.metrics_types:
            self.config.metrics_types = [
                'response_time',
                'memory_usage',
                'decision_accuracy',
                'personality_stability'
            ]
    
    async def record_metric(
        self,
        metric_type: str,
        value: float,
        context: Optional[Dict[str, Any]] = None
    ) -> None:
        """Record a new metric value."""
        if metric_type not in self.config.metrics_types:
            logger.warning(f"Unknown metric type: {metric_type}")
            return
            
        metric_data = {
            'value': value,
            'timestamp': time.time(),
            'context': context or {}
        }
        
        self.metrics[metric_type].append(metric_data)
        
        # Aggregate if interval reached
        if len(self.metrics[metric_type]) % self.config.save_interval == 0:
            await self._aggregate_metrics(metric_type)
    
    async def get_metrics_summary(
        self,
        metric_type: Optional[str] = None,
        window: Optional[int] = None
    ) -> Dict[str, Any]:
        """Get summary of recorded metrics."""
        window = window or self.config.window_size
        
        if metric_type:
            return await self._get_single_metric_summary(metric_type, window)
        
        return {
            mtype: await self._get_single_metric_summary(mtype, window)
            for mtype in self.config.metrics_types
        }
