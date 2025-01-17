from typing import Dict, List, Optional, Any, Set, Union
import asyncio
from dataclasses import dataclass
import logging
import time
from collections import defaultdict, deque
import psutil
import numpy as np
from datetime import datetime, timedelta
import json
import aiohttp
import ssl
from cryptography.fernet import Fernet

logger = logging.getLogger(__name__)

@dataclass
class MonitorConfig:
    check_interval: float = 1.0  # seconds
    memory_threshold: float = 0.9  # 90%
    cpu_threshold: float = 0.8    # 80%
    disk_threshold: float = 0.9   # 90%
    alert_cooldown: int = 300     # 5 minutes
    max_alerts: int = 1000
    retention_days: int = 30
    encryption_key: Optional[str] = None
    threat_sensitivity: float = 0.7

class ThreatDetector:
    """Advanced threat detection system."""
    
    def __init__(self, config: MonitorConfig):
        self.config = config
        self.threat_patterns = self._load_threat_patterns()
        self.anomaly_detector = AnomalyDetector()
        self.incident_history = deque(maxlen=10000)
        self.active_threats = set()
        self.threat_scores = defaultdict(float)
        
        # Initialize encryption
        if config.encryption_key:
            self.cipher = Fernet(config.encryption_key.encode())
        else:
            self.cipher = None
    
    def _load_threat_patterns(self) -> Dict[str, Any]:
        """Load threat detection patterns."""
        # Implementation would load from configuration
        return {
            'brute_force': {
                'pattern': 'repeated_auth_failure',
                'threshold': 5,
                'window': 300  # 5 minutes
            },
            'data_exfiltration': {
                'pattern': 'large_data_transfer',
                'threshold': 100_000_000,  # 100MB
                'window': 60  # 1 minute
            },
            'injection_attempt': {
                'pattern': 'malicious_input',
                'threshold': 3,
                'window': 600  # 10 minutes
            }
        }
    
    async def analyze_event(
        self,
        event: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze security event for threats."""
        threat_level = 0.0
        detected_threats = set()
        
        # Encrypt sensitive data
        if self.cipher and 'sensitive_data' in event:
            event['sensitive_data'] = self.cipher.encrypt(
                json.dumps(event['sensitive_data']).encode()
            )
        
        # Pattern matching
        for threat_type, pattern in self.threat_patterns.items():
            if await self._match_pattern(event, pattern):
                threat_level = max(threat_level, 0.7)
                detected_threats.add(threat_type)
        
        # Anomaly detection
        anomaly_score = await self.anomaly_detector.analyze(event)
        if anomaly_score > self.config.threat_sensitivity:
            threat_level = max(threat_level, anomaly_score)
            detected_threats.add('anomaly')
        
        # Update threat scores
        for threat in detected_threats:
            self.threat_scores[threat] += 1
        
        # Store incident
        self.incident_history.append({
            'timestamp': time.time(),
            'event': event,
            'threat_level': threat_level,
            'threats': list(detected_threats)
        })
        
        return {
            'threat_level': threat_level,
            'threats': list(detected_threats),
            'anomaly_score': anomaly_score,
            'timestamp': time.time()
        }

class ResourceMonitor:
    """Enhanced resource monitoring system."""
    
    def __init__(self, config: MonitorConfig):
        self.config = config
        self.metrics = defaultdict(deque)
        self.thresholds = {
            'cpu_percent': config.cpu_threshold,
            'memory_percent': config.memory_threshold,
            'disk_percent': config.disk_threshold
        }
        self.alerts = deque(maxlen=config.max_alerts)
        self._last_alert = defaultdict(float)
    
    async def check_resources(self) -> Dict[str, Any]:
        """Check system resources."""
        metrics = {
            'cpu_percent': psutil.cpu_percent(),
            'memory_percent': psutil.virtual_memory().percent,
            'disk_percent': psutil.disk_usage('/').percent,
            'network': psutil.net_io_counters()._asdict(),
            'timestamp': time.time()
        }
        
        # Store metrics
        for key, value in metrics.items():
            if key != 'timestamp':
                self.metrics[key].append((time.time(), value))
        
        # Check thresholds
        alerts = []
        for metric, threshold in self.thresholds.items():
            if metrics[metric] > threshold * 100:
                alert = await self._create_alert(
                    metric,
                    metrics[metric],
                    threshold * 100
                )
                if alert:
                    alerts.append(alert)
        
        return {
            'metrics': metrics,
            'alerts': alerts
        }
    
    async def _create_alert(
        self,
        metric: str,
        value: float,
        threshold: float
    ) -> Optional[Dict[str, Any]]:
        """Create resource alert."""
        now = time.time()
        
        # Check cooldown
        if now - self._last_alert[metric] < self.config.alert_cooldown:
            return None
            
        alert = {
            'type': 'resource_alert',
            'metric': metric,
            'value': value,
            'threshold': threshold,
            'timestamp': now
        }
        
        self.alerts.append(alert)
        self._last_alert[metric] = now
        
        return alert

class AuditLogger:
    """Enhanced audit logging system."""
    
    def __init__(self, config: MonitorConfig):
        self.config = config
        self.logs = deque(maxlen=100000)
        self.indexed_logs = defaultdict(list)
        self.sensitive_fields = {'password', 'token', 'key'}
        
        # Initialize encryption
        if config.encryption_key:
            self.cipher = Fernet(config.encryption_key.encode())
        else:
            self.cipher = None
    
    async def log_event(
        self,
        event: Dict[str, Any],
        event_type: str,
        severity: str = 'info'
    ) -> str:
        """Log security event."""
        # Sanitize sensitive data
        event = await self._sanitize_event(event)
        
        # Create log entry
        entry = {
            'id': str(time.time()),
            'timestamp': time.time(),
            'type': event_type,
            'severity': severity,
            'event': event
        }
        
        # Encrypt if needed
        if self.cipher:
            entry['event'] = self.cipher.encrypt(
                json.dumps(event).encode()
            ).decode()
        
        # Store log
        self.logs.append(entry)
        self.indexed_logs[event_type].append(entry)
        
        return entry['id']
    
    async def _sanitize_event(
        self,
        event: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Sanitize sensitive data in event."""
        sanitized = {}
        for key, value in event.items():
            if key in self.sensitive_fields:
                sanitized[key] = '[REDACTED]'
            elif isinstance(value, dict):
                sanitized[key] = await self._sanitize_event(value)
            else:
                sanitized[key] = value
        return sanitized
    
    async def query_logs(
        self,
        filters: Dict[str, Any],
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Query audit logs with filters."""
        results = []
        
        for log in reversed(self.logs):
            if len(results) >= limit:
                break
                
            if start_time and log['timestamp'] < start_time:
                continue
                
            if end_time and log['timestamp'] > end_time:
                continue
                
            if all(
                log.get(key) == value
                for key, value in filters.items()
            ):
                # Decrypt if needed
                if self.cipher and isinstance(log['event'], str):
                    log['event'] = json.loads(
                        self.cipher.decrypt(
                            log['event'].encode()
                        ).decode()
                    )
                    
                results.append(log)
        
        return results

class AlertManager:
    """Enhanced alert management system."""
    
    def __init__(self, config: MonitorConfig):
        self.config = config
        self.alerts = deque(maxlen=config.max_alerts)
        self.active_alerts = set()
        self.alert_handlers = defaultdict(list)
        self.alert_history = defaultdict(list)
        self._last_notification = defaultdict(float)
    
    async def create_alert(
        self,
        alert_type: str,
        details: Dict[str, Any],
        severity: str = 'info'
    ) -> str:
        """Create and process security alert."""
        alert_id = str(time.time())
        
        alert = {
            'id': alert_id,
            'type': alert_type,
            'details': details,
            'severity': severity,
            'status': 'new',
            'timestamp': time.time()
        }
        
        # Store alert
        self.alerts.append(alert)
        self.active_alerts.add(alert_id)
        self.alert_history[alert_type].append(alert)
        
        # Process alert
        await self._process_alert(alert)
        
        return alert_id
    
    async def _process_alert(
        self,
        alert: Dict[str, Any]
    ) -> None:
        """Process security alert."""
        handlers = self.alert_handlers.get(alert['type'], [])
        
        for handler in handlers:
            try:
                await handler(alert)
            except Exception as e:
                logger.error(f"Alert handler error: {e}")
        
        # Check if notification needed
        await self._check_notification(alert)
    
    async def _check_notification(
        self,
        alert: Dict[str, Any]
    ) -> None:
        """Check if alert requires notification."""
        alert_type = alert['type']
        now = time.time()
        
        # Check cooldown
        if now - self._last_notification[alert_type] < self.config.alert_cooldown:
            return
            
        # Send notification based on severity
        if alert['severity'] in {'high', 'critical'}:
            await self._send_notification(alert)
            self._last_notification[alert_type] = now

class SecurityMonitor:
    """Enhanced security monitoring system."""
    
    def __init__(self, config: Optional[MonitorConfig] = None):
        self.config = config or MonitorConfig()
        self.threat_detector = ThreatDetector(self.config)
        self.resource_monitor = ResourceMonitor(self.config)
        self.audit_logger = AuditLogger(self.config)
        self.alert_manager = AlertManager(self.config)
        
        self._monitoring = False
        self._monitor_task = None
    
    async def start(self) -> None:
        """Start security monitoring."""
        self._monitoring = True
        self._monitor_task = asyncio.create_task(
            self._monitoring_loop()
        )
    
    async def stop(self) -> None:
        """Stop security monitoring."""
        self._monitoring = False
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
    
    async def process_event(
        self,
        event: Dict[str, Any],
        event_type: str
    ) -> Dict[str, Any]:
        """Process security event."""
        try:
            # Analyze for threats
            threat_analysis = await self.threat_detector.analyze_event(event)
            
            # Log event
            log_id = await self.audit_logger.log_event(
                event,
                event_type,
                'warning' if threat_analysis['threat_level'] > 0.5 else 'info'
            )
            
            # Create alert if needed
            if threat_analysis['threat_level'] > self.config.threat_sensitivity:
                alert_id = await self.alert_manager.create_alert(
                    'security_threat',
                    {
                        'event': event,
                        'analysis': threat_analysis
                    },
                    'high' if threat_analysis['threat_level'] > 0.8 else 'medium'
                )
            
            return {
                'status': 'processed',
                'log_id': log_id,
                'threat_analysis': threat_analysis
            }
            
        except Exception as e:
            logger.error(f"Event processing error: {e}")
            raise
    
    async def _monitoring_loop(self) -> None:
        """Main monitoring loop."""
        while self._monitoring:
            try:
                # Check resources
                resources = await self.resource_monitor.check_resources()
                
                # Log resource status
                await self.audit_logger.log_event(
                    resources['metrics'],
                    'resource_check'
                )
                
                # Process resource alerts
                for alert in resources['alerts']:
                    await self.alert_manager.create_alert(
                        'resource_warning',
                        alert,
                        'medium'
                    )
                
                await asyncio.sleep(self.config.check_interval)
                
            except Exception as e:
                logger.error(f"Monitoring error: {e}")
                await asyncio.sleep(1)
    
    async def get_security_status(self) -> Dict[str, Any]:
        """Get current security status."""
        return {
            'active_threats': len(self.threat_detector.active_threats),
            'threat_scores': dict(self.threat_detector.threat_scores),
            'resource_status': await self.resource_monitor.check_resources(),
            'active_alerts': len(self.alert_manager.active_alerts)
        }
    
    def __del__(self):
        """Cleanup resources."""
        if self._monitoring:
            asyncio.create_task(self.stop())
