from typing import Dict, List, Optional, Any
import numpy as np
from dataclasses import dataclass
import time
import torch
import logging

logger = logging.getLogger(__name__)

@dataclass
class PerformanceConfig:
    metrics_window: int = 1000
    update_interval: float = 1.0
    alert_threshold: float = 0.8

class PerformanceAnalyzer:
    """Analyzes system performance metrics."""
    
    def __init__(self, config: PerformanceConfig):
        self.config = config
        self.metrics = {}
        self.alerts = []
        self.start_time = time.time()
    
    async def record_performance(
        self,
        metric_type: str,
        value: float,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Record a performance metric."""
        if metric_type not in self.metrics:
            self.metrics[metric_type] = []
            
        metric_data = {
            'value': value,
            'timestamp': time.time(),
            'metadata': metadata or {}
        }
        
        self.metrics[metric_type].append(metric_data)
        
        # Check for alerts
        if value > self.config.alert_threshold:
            await self._create_alert(metric_type, value, metadata)
    
    async def get_performance_report(
        self,
        metric_types: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Generate performance report."""
        report = {
            'timestamp': time.time(),
            'duration': time.time() - self.start_time,
            'metrics': {},
            'alerts': len(self.alerts)
        }
        
        for metric_type in (metric_types or self.metrics.keys()):
            if metric_type in self.metrics:
                report['metrics'][metric_type] = await self._analyze_metric(
                    metric_type
                )
                
        return report
