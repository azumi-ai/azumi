from typing import Dict, List, Optional, Any
import torch
import numpy as np
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class DebuggerConfig:
    trace_depth: int = 10
    log_level: str = "DEBUG"
    save_state: bool = True

class PersonalityDebugger:
    """Debugging tools for personality system."""
    
    def __init__(self, config: DebuggerConfig):
        self.config = config
        self.traces = []
        self.states = {}
        self._setup_logging()
        
    def _setup_logging(self):
        logging.basicConfig(
            level=self.config.log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
    async def trace_personality(
        self,
        personality_id: str,
        action: str,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Trace personality system execution."""
        
        trace_data = {
            'timestamp': time.time(),
            'personality_id': personality_id,
            'action': action,
            'context': context,
            'stack': []
        }
        
        try:
            # Collect execution trace
            trace_data['stack'] = await self._collect_trace()
            
            # Save state if enabled
            if self.config.save_state:
                await self._save_state(personality_id)
            
            self.traces.append(trace_data)
            return trace_data
            
        except Exception as e:
            logger.error(f"Trace error: {e}")
            trace_data['error'] = str(e)
            return trace_data
            
    async def analyze_trace(
        self,
        trace_id: str
    ) -> Dict[str, Any]:
        """Analyze a specific trace."""
        trace = self._get_trace(trace_id)
        if not trace:
            raise ValueError(f"Trace not found: {trace_id}")
            
        analysis = {
            'execution_time': self._analyze_execution_time(trace),
            'memory_usage': self._analyze_memory_usage(trace),
            'decision_points': self._analyze_decisions(trace),
            'state_changes': self._analyze_state_changes(trace)
        }
        
        return analysis
