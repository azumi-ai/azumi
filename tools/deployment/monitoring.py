from typing import Dict, List, Optional, Any
import asyncio
import psutil
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class MonitoringConfig:
    check_interval: float = 1.0  # seconds
    memory_threshold: float = 0.9  # 90%
    cpu_threshold: float = 0.8    # 80%
    disk_threshold: float = 0.9   # 90%

class DeploymentMonitor:
    """System deployment monitoring."""
    
    def __init__(self, config: MonitoringConfig):
        self.config = config
        self.metrics = {}
        self.alerts = []
        self._monitoring = False
    
    async def start_monitoring(self) -> None:
        """Start system monitoring."""
        self._monitoring = True
        
        while self._monitoring:
            try:
                # Collect system metrics
                metrics = await self._collect_metrics()
                
                # Check thresholds
                alerts = await self._check_thresholds(metrics)
                
                # Store metrics and alerts
                self.metrics[time.time()] = metrics
                if alerts:
                    self.alerts.extend(alerts)
                    
                # Clean old metrics
                await self._clean_old_metrics()
                
                await asyncio.sleep(self.config.check_interval)
                
            except Exception as e:
                logger.error(f"Monitoring error: {e}")
    
    async def _collect_metrics(self) -> Dict[str, float]:
        """Collect system performance metrics."""
        return {
            'cpu_percent': psutil.cpu_percent(),
            'memory_percent': psutil.virtual_memory().percent,
            'disk_percent': psutil.disk_usage('/').percent,
            'network_io': psutil.net_io_counters()._asdict()
        }
    
    async def _check_thresholds(
        self,
        metrics: Dict[str, float]
    ) -> List[Dict[str, Any]]:
        """Check metrics against thresholds."""
        alerts = []
        
        if metrics['cpu_percent'] > self.config.cpu_threshold * 100:
            alerts.append({
                'type': 'cpu_high',
                'value': metrics['cpu_percent'],
                'threshold': self.config.cpu_threshold * 100,
                'timestamp': time.time()
            })
            
        if metrics['memory_percent'] > self.config.memory_threshold * 100:
            alerts.append({
                'type': 'memory_high',
                'value': metrics['memory_percent'],
                'threshold': self.config.memory_threshold * 100,
                'timestamp': time.time()
            })
            
        return alerts
