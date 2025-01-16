from typing import Dict, List, Optional, Any
import time
import logging
from dataclasses import dataclass
import asyncio
from prometheus_client import Counter, Gauge, Histogram
import json

logger = logging.getLogger(__name__)

@dataclass
class MonitoringConfig:
    check_interval: int = 60  # seconds
    alert_threshold: float = 0.8
    retention_days: int = 30
    max_events: int = 10000

class SecurityMonitor:
    def __init__(self, config: Optional[MonitoringConfig] = None):
        self.config = config or MonitoringConfig()
        self.initialize_metrics()
        self.events = []
        self.alerts = []
        
    def initialize_metrics(self):
        # Prometheus metrics
        self.requests_total = Counter(
            'azumi_requests_total',
            'Total number of requests processed'
        )
        self.active_sessions = Gauge(
            'azumi_active_sessions',
            'Number of active sessions'
        )
        self.response_time = Histogram(
            'azumi_response_time_seconds',
            'Response time in seconds'
        )
        self.memory_usage = Gauge(
            'azumi_memory_usage_bytes',
            'Current memory usage'
        )
        
    async def monitor_session(self, session_id: str) -> None:
        try:
            start_time = time.time()
            self.active_sessions.inc()
            
            while True:
                await self._check_session_health(session_id)
                await asyncio.sleep(self.config.check_interval)
                
        except Exception as e:
            logger.error(f"Session monitoring error: {e}")
        finally:
            self.active_sessions.dec()
            
    async def _check_session_health(self, session_id: str) -> None:
        metrics = await self._collect_session_metrics(session_id)
        self._update_metrics(metrics)
        
        if self._should_alert(metrics):
            await self._generate_alert(session_id, metrics)
            
    def _should_alert(self, metrics: Dict[str, float]) -> bool:
        return any(
            value > self.config.alert_threshold
            for value in metrics.values()
        )

    async def log_security_event(self, event: Dict[str, Any]) -> None:
        timestamp = time.time()
        event_data = {
            "timestamp": timestamp,
            "event_type": event.get("type", "unknown"),
            "severity": event.get("severity", "info"),
            "details": event.get("details", {}),
            "session_id": event.get("session_id")
        }
        
        self.events.append(event_data)
        
        # Trim old events
        if len(self.events) > self.config.max_events:
            self.events = self.events[-self.config.max_events:]
            
        # Log high severity events
        if event_data["severity"] in ["high", "critical"]:
            logger.warning(f"High severity security event: {event_data}")
            await self._generate_alert(event_data["session_id"], event_data)
