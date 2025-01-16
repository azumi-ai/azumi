from typing import Dict, List, Any
from .short_term import ShortTermMemory
from .long_term import LongTermMemory

class MemoryIntegration:
    def __init__(self, 
                 short_term: ShortTermMemory, 
                 long_term: LongTermMemory):
        self.short_term = short_term
        self.long_term = long_term
        self.importance_threshold = 0.7
    
    def consolidate_memories(self) -> None:
        """Transfer important short-term memories to long-term storage"""
        for memory_item in self.short_term.memory:
            importance = self._calculate_importance(memory_item)
            if importance >= self.importance_threshold:
                self.long_term.store(memory_item['content'], importance)
    
    def _calculate_importance(self, memory_item: Dict[str, Any]) -> float:
        # Calculate importance based on access count, emotional intensity, etc.
        base_importance = min(1.0, memory_item['access_count'] / 10)
        emotional_intensity = memory_item['content'].get('emotional_intensity', 0.5)
        return (base_importance + emotional_intensity) / 2
