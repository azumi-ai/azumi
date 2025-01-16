from typing import Dict, List, Any
import numpy as np
from dataclasses import dataclass
import time
import logging

logger = logging.getLogger(__name__)

@dataclass
class MetricsConfig:
    window_size: int = 1000
    update_interval: float = 1.0
    retention_period: int = 30  # days

class PerformanceMetrics:
    """Performance monitoring and metrics system."""
    
    def __init__(self, config: MetricsConfig):
        self.config = config
        self.metrics = {
            'response_time': [],
            'memory_usage': [],
            'interaction_success': [],
            'personality_stability': []
        }
        self.start_time = time.time()
        
    async def record_metric(
        self,
        metric_type: str,
        value: float,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Record a new metric value."""
        if metric_type not in self.metrics:
            raise ValueError(f"Unknown metric type: {metric_type}")
            
        timestamp = time.time()
        self.metrics[metric_type].append({
            'value': value,
            'timestamp': timestamp,
            'metadata': metadata or {}
        })
        
        # Cleanup old metrics
        await self._cleanup_metrics()
        
    async def get_metrics_summary(
        self,
        metric_type: str,
        window: Optional[int] = None
    ) -> Dict[str, float]:
        """Get summary statistics for a metric."""
        if metric_type not in self.metrics:
            raise ValueError(f"Unknown metric type: {metric_type}")
            
        window = window or self.config.window_size
        recent_metrics = [
            m['value'] for m in self.metrics[metric_type][-window:]
        ]
        
        if not recent_metrics:
            return {}
            
        return {
            'mean': np.mean(recent_metrics),
            'std': np.std(recent_metrics),
            'min': np.min(recent_metrics),
            'max': np.max(recent_metrics),
            'median': np.median(recent_metrics)
        }
