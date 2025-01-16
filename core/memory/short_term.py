from typing import Dict, List, Any
from collections import deque
import time

class ShortTermMemory:
    def __init__(self, capacity: int = 1000):
        self.capacity = capacity
        self.memory = deque(maxlen=capacity)
        self.index = {}
    
    def add(self, item: Dict[str, Any]) -> None:
        timestamp = time.time()
        memory_item = {
            'content': item,
            'timestamp': timestamp,
            'access_count': 0
        }
        self.memory.append(memory_item)
        
        # Update index
        key = item.get('id') or str(timestamp)
        self.index[key] = memory_item
    
    def get(self, key: str) -> Dict[str, Any]:
        if key in self.index:
            memory_item = self.index[key]
            memory_item['access_count'] += 1
            return memory_item['content']
        return None
