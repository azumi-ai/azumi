from typing import Dict, List, Optional, Any
import psutil
import gc
import logging
from dataclasses import dataclass
import time

logger = logging.getLogger(__name__)

@dataclass
class MemoryDebugConfig:
    check_interval: float = 1.0
    threshold_mb: int = 1000
    log_level: str = "DEBUG"

class MemoryDebugger:
    """Memory usage debugging tool."""
    
    def __init__(self, config: MemoryDebugConfig):
        self.config = config
        self.snapshots = []
        self._setup_logging()
        
    def _setup_logging(self) -> None:
        """Setup debugging logger."""
        logging.basicConfig(
            level=self.config.log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
    async def take_snapshot(self) -> Dict[str, Any]:
        """Take memory usage snapshot."""
        snapshot = {
            'timestamp': time.time(),
            'memory': psutil.Process().memory_info(),
            'gc_stats': gc.get_stats(),
            'objects': self._count_objects()
        }
        
        self.snapshots.append(snapshot)
        return snapshot
