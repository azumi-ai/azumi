from typing import Dict, List, Optional, Any, Union, Callable
import numpy as np
import pandas as pd
from dataclasses import dataclass
import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict
from scipy import stats
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mutual_info_score
import torch
from cachetools import TTLCache
import time

logger = logging.getLogger(__name__)

@dataclass
class AnalysisConfig:
    window_size: int = 1000
    update_interval: float = 0.1
    cache_ttl: int = 3600
    min_samples: int = 30
    confidence_level: float = 0.95
    batch_size: int = 100
    max_dimensions: int = 50

class StreamingStats:
    """Real-time statistical calculations."""
    
    def __init__(self):
        self.count = 0
        self.mean = 0.0
        self.M2 = 0.0
        self.min_value = float('inf')
        self.max_value = float('-inf')
        self.values = []  # For quantile calculations
    
    def update(self, value: float) -> None:
        """Update streaming statistics."""
        self.count += 1
        delta = value - self.mean
        self.mean += delta / self.count
        delta2 = value - self.mean
        self.M2 += delta * delta2
        
        self.min_value = min(self.min_value, value)
        self.max_value = max(self.max_value, value)
        self.values.append(value)
    
    @property
    def variance(self) -> float:
        """Calculate variance."""
        return self.M2 / (self.count - 1) if self.count > 1 else 0.0
    
    @property
    def std(self) -> float:
        """Calculate standard deviation."""
        return np.sqrt(self.variance)

class MetricAggregator:
    """Efficient metric aggregation system."""
    
    def __init__(self, config: AnalysisConfig):
        self.config = config
        self.metrics = defaultdict(StreamingStats)
        self.temporal_data = defaultdict(list)
        self.correlation_cache = TTLCache(
            maxsize=1000,
            ttl=config.cache_ttl
        )
    
    async def add_metric(
        self,
        metric_name: str,
        value: float,
        timestamp: Optional[float] = None
    ) -> None:
        """Add metric value."""
        self.metrics[metric_name].update(value)
        
        if timestamp is None:
            timestamp = time.time()
            
        self.temporal_data[metric_name].append((timestamp, value))
        
        # Trim old data
        if len(self.temporal_data[metric_name]) > self.config.window_size:
            self.temporal_data[metric_name].pop(0)
    
    async def get_statistics(
        self,
        metric_name: str
    ) -> Dict[str, float]:
        """Get metric statistics."""
        stats = self.metrics[metric_name]
        return {
            'mean': stats.mean,
            'std': stats.std,
            'min': stats.min_value,
            'max': stats.max_value,
            'count': stats.count
        }
    
    async def calculate_correlation(
        self,
        metric1: str,
        metric2: str
    ) -> Optional[float]:
        """Calculate metric correlation."""
        cache_key = f"{metric1}_{metric2}"
        if cache_key in self.correlation_cache:
            return self.correlation_cache[cache_key]
            
        if (metric1 not in self.metrics or 
            metric2 not in self.metrics):
            return None
            
        values1 = [v for _, v in self.temporal_data[metric1]]
        values2 = [v for _, v in self.temporal_data[metric2]]
        
        if len(values1) != len(values2):
            return None
            
        correlation = stats.pearsonr(values1, values2)[0]
        self.correlation_cache[cache_key] = correlation
        return correlation

class AnalysisMetrics:
    """Enhanced research analysis metrics system."""
    
    def __init__(self, config: Optional[AnalysisConfig] = None):
        self.config = config or AnalysisConfig()
        self.aggregator = MetricAggregator(self.config)
        self._executor = ThreadPoolExecutor(max_workers=4)
        self._update_task = None
        self._metric_callbacks = defaultdict(list)
        
        # Advanced analysis components
        self.dimension_reducer = torch.nn.Linear(
            self.config.max_dimensions,
            3
        )  # For visualization
        
    async def start(self) -> None:
        """Start metrics system."""
        self._update_task = asyncio.create_task(
            self._update_loop()
        )
        
    async def stop(self) -> None:
        """Stop metrics system."""
        if self._update_task:
            self._update_task.cancel()
            try:
                await self._update_task
            except asyncio.CancelledError:
                pass
    
    async def record_metric(
        self,
        name: str,
        value: float,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Record metric value."""
        await self.aggregator.add_metric(name, value)
        
        # Notify callbacks
        for callback in self._metric_callbacks[name]:
            try:
                await callback(name, value, metadata)
            except Exception as e:
                logger.error(f"Callback error: {e}")
    
    async def analyze_metrics(
        self,
        metrics: List[str],
        analysis_types: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Perform comprehensive metric analysis."""
        if not analysis_types:
            analysis_types = ['basic', 'temporal', 'correlation']
            
        results = {}
        
        try:
            for analysis_type in analysis_types:
                analyzer = getattr(
                    self,
                    f'_analyze_{analysis_type}',
                    None
                )
                if analyzer:
                    results[analysis_type] = await analyzer(metrics)
            
            return results
            
        except Exception as e:
            logger.error(f"Analysis error: {e}")
            raise
    
    async def _analyze_basic(
        self,
        metrics: List[str]
    ) -> Dict[str, Dict[str, float]]:
        """Basic statistical analysis."""
        results = {}
        
        for metric in metrics:
            stats = await self.aggregator.get_statistics(metric)
            
            # Calculate confidence intervals
            if stats['count'] >= self.config.min_samples:
                ci = stats.t.interval(
                    self.config.confidence_level,
                    stats['count'] - 1,
                    stats['mean'],
                    stats['std'] / np.sqrt(stats['count'])
                )
                stats['ci_lower'] = ci[0]
                stats['ci_upper'] = ci[1]
            
            results[metric] = stats
            
        return results
    
    async def _analyze_temporal(
        self,
        metrics: List[str]
    ) -> Dict[str, Dict[str, Any]]:
        """Temporal analysis of metrics."""
        results = {}
        
        for metric in metrics:
            temporal_data = self.aggregator.temporal_data[metric]
            if len(temporal_data) < self.config.min_samples:
                continue
                
            values = [v for _, v in temporal_data]
            timestamps = [t for t, _ in temporal_data]
            
            # Trend analysis
            trend = await self._analyze_trend(values)
            
            # Seasonality detection
            seasonality = await self._detect_seasonality(values)
            
            # Stationarity test
            stationarity = await self._test_stationarity(values)
            
            results[metric] = {
                'trend': trend,
                'seasonality': seasonality,
                'stationarity': stationarity
            }
            
        return results
    
    async def _analyze_trend(
        self,
        values: List[float]
    ) -> Dict[str, float]:
        """Analyze metric trend."""
        x = np.arange(len(values))
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, values)
        
        return {
            'slope': slope,
            'intercept': intercept,
            'r_squared': r_value ** 2,
            'p_value': p_value
        }
    
    async def _detect_seasonality(
        self,
        values: List[float]
    ) -> Dict[str, Any]:
        """Detect metric seasonality."""
        if len(values) < self.config.min_samples:
            return None
            
        # Calculate autocorrelation
        acf = np.correlate(values, values, mode='full')
        acf = acf[len(acf)//2:]
        
        # Find peaks in autocorrelation
        peaks = []
        for i in range(1, len(acf)-1):
            if acf[i] > acf[i-1] and acf[i] > acf[i+1]:
                peaks.append((i, acf[i]))
        
        if not peaks:
            return {'seasonal': False}
            
        # Sort peaks by correlation strength
        peaks.sort(key=lambda x: x[1], reverse=True)
        
        return {
            'seasonal': True,
            'period': peaks[0][0],
            'strength': peaks[0][1] / acf[0]
        }
    
    async def _test_stationarity(
        self,
        values: List[float]
    ) -> Dict[str, Any]:
        """Test for stationarity."""
        result = adfuller(values)
        
        return {
            'stationary': result[1] < 0.05,
            'adf_statistic': result[0],
            'p_value': result[1],
            'critical_values': result[4]
        }
    
    async def calculate_mutual_information(
        self,
        metric1: str,
        metric2: str
    ) -> float:
        """Calculate mutual information between metrics."""
        values1 = self.aggregator.temporal_data[metric1]
        values2 = self.aggregator.temporal_data[metric2]
        
        if len(values1) != len(values2):
            return 0.0
            
        # Convert to numpy arrays
        x = np.array([v for _, v in values1])
        y = np.array([v for _, v in values2])
        
        # Calculate mutual information
        mi = mutual_info_score(
            np.digitize(x, bins=20),
            np.digitize(y, bins=20)
        )
        
        return mi
    
    async def get_dimensionality_reduction(
        self,
        metrics: List[str]
    ) -> torch.Tensor:
        """Reduce metrics dimensionality for visualization."""
        if len(metrics) > self.config.max_dimensions:
            raise ValueError("Too many metrics for dimensionality reduction")
            
        values = []
        for metric in metrics:
            metric_values = [v for _, v in self.aggregator.temporal_data[metric]]
            if metric_values:
                values.append(np.mean(metric_values))
            else:
                values.append(0.0)
                
        # Convert to tensor and reduce
        with torch.no_grad():
            input_tensor = torch.tensor(values, dtype=torch.float32)
            reduced = self.dimension_reducer(input_tensor)
            
        return reduced
    
    async def _update_loop(self) -> None:
        """Main update loop."""
        while True:
            try:
                await self._process_updates()
                await asyncio.sleep(self.config.update_interval)
            except Exception as e:
                logger.error(f"Update loop error: {e}")
                await asyncio.sleep(1)
    
    async def _process_updates(self) -> None:
        """Process metric updates."""
        # Implementation depends on specific update requirements
        pass
    
    async def get_health_check(self) -> Dict[str, Any]:
        """Get system health metrics."""
        return {
            'total_metrics': len(self.aggregator.metrics),
            'active_callbacks': sum(
                len(callbacks)
                for callbacks in self._metric_callbacks.values()
            ),
            'memory_usage': self._get_memory_usage()
        }
    
    def _get_memory_usage(self) -> Dict[str, int]:
        """Get memory usage statistics."""
        return {
            'metrics': sum(
                sys.getsizeof(stats.values)
                for stats in self.aggregator.metrics.values()
            ),
            'temporal_data': sum(
                sys.getsizeof(data)
                for data in self.aggregator.temporal_data.values()
            ),
            'cache': sys.getsizeof(self.aggregator.correlation_cache)
        }
    
    def __del__(self):
        """Cleanup resources."""
        if hasattr(self, '_executor'):
            self._executor.shutdown(wait=False)
