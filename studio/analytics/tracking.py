from typing import Dict, List, Optional, Any
import numpy as np
from dataclasses import dataclass
import time
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)

@dataclass
class TrackingConfig:
    tracking_interval: float = 0.1
    save_interval: int = 100
    max_history: int = 1000

class BehaviorTracker:
    """Tracks and analyzes behavior patterns."""
    
    def __init__(self, config: TrackingConfig):
        self.config = config
        self.behaviors = defaultdict(list)
        self.patterns = {}
        self._tracking = False
        
    async def start_tracking(self, entity_id: str) -> None:
        """Start tracking an entity's behavior."""
        self._tracking = True
        
        while self._tracking:
            try:
                await self._record_behavior(entity_id)
                await asyncio.sleep(self.config.tracking_interval)
            except Exception as e:
                logger.error(f"Tracking error: {e}")
                
    async def stop_tracking(self, entity_id: str) -> None:
        """Stop tracking an entity's behavior."""
        self._tracking = False
        await self._analyze_patterns(entity_id)
