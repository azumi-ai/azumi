from typing import Dict, List, Optional, Any
import logging
from dataclasses import dataclass
import time
import asyncio

logger = logging.getLogger(__name__)

@dataclass
class InteractionDebugConfig:
    trace_depth: int = 10
    log_level: str = "DEBUG"
    save_traces: bool = True

class InteractionDebugger:
    """Interaction debugging tool."""
    
    def __init__(self, config: InteractionDebugConfig):
        self.config = config
        self.traces = []
        self._setup_logging()
        
    async def trace_interaction(
        self,
        interaction_id: str,
        data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Trace interaction flow."""
        trace = {
            'id': interaction_id,
            'timestamp': time.time(),
            'data': data,
            'stack': await self._capture_stack(),
            'context': await self._capture_context()
        }
        
        if self.config.save_traces:
            self.traces.append(trace)
            
        return trace
