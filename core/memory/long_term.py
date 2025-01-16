from typing import Dict, List, Optional, Any
import sqlite3
import json
import time
import asyncio
import aiosqlite
from contextlib import asynccontextmanager
import numpy as np
from dataclasses import dataclass
import logging
from cachetools import TTLCache, LRUCache

logger = logging.getLogger(__name__)

@dataclass
class MemoryConfig:
    importance_threshold: float = 0.5
    retention_period: int = 30 * 24 * 3600  # 30 days
    cleanup_interval: int = 24 * 3600  # 24 hours
    max_memories: int = 10000
    cache_ttl: int = 3600  # 1 hour
    cache_size: int = 1000
    batch_size: int = 100

class LongTermMemory:
    def __init__(self, 
                 db_path: str = 'memory.db',
                 config: Optional[MemoryConfig] = None):
        self.db_path = db_path
        self.config = config or MemoryConfig()
        self.cleanup_task = None
        self.cache = TTLCache(
            maxsize=self.config.cache_size,
            ttl=self.config.cache_ttl
        )
        self.embedding_cache = LRUCache(maxsize=1000)
        self._init_db()
        self._start_cleanup_task()
    
    async def _init_db(self) -> None:
        """Initialize database with optimized indexes."""
        async with self._get_db() as db:
            await db.execute('''
                CREATE TABLE IF NOT EXISTS memories (
                    id TEXT PRIMARY KEY,
                    content TEXT,
                    timestamp REAL,
                    importance REAL,
                    last_access REAL,
                    access_count INTEGER,
                    metadata TEXT,
                    embedding BLOB
                )
            ''')
            
            # Optimized indexes
            await db.execute('''
                CREATE INDEX IF NOT EXISTS idx_importance_timestamp 
                ON memories(importance DESC, timestamp DESC)
            ''')
            await db.execute('''
                CREATE INDEX IF NOT EXISTS idx_last_access 
                ON memories(last_access)
            ''')
            await db.execute('''
                CREATE INDEX IF NOT EXISTS idx_importance_access 
                ON memories(importance DESC, last_access DESC)
            ''')
    
    @asynccontextmanager
    async def _get_db(self):
        """Get database connection with connection pooling."""
        if not hasattr(self, '_pool'):
            self._pool = await aiosqlite.connect(
                self.db_path,
                isolation_level=None  # Enable autocommit mode
            )
        try:
            yield self._pool
        except Exception as e:
            logger.error(f"Database error: {e}")
            raise
    
    async def store(self, 
                    memory: Dict[str, Any],
                    importance: float = 0.5,
                    metadata: Optional[Dict[str, Any]] = None,
                    embedding: Optional[np.ndarray] = None) -> str:
        """Store memory with batch processing for efficiency."""
        if importance < self.config.importance_threshold:
            return None
            
        memory_id = str(time.time())
        
        try:
            async with self._get_db() as db:
                await db.execute('''
                    INSERT INTO memories 
                    (id, content, timestamp, importance, last_access, 
                     access_count, metadata, embedding)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    memory_id,
                    json.dumps(memory),
                    time.time(),
                    importance,
                    time.time(),
                    0,
                    json.dumps(metadata) if metadata else None,
                    embedding.tobytes() if embedding is not None else None
                ))
                
            # Update cache
            cache_key = self._get_cache_key(memory_id)
            self.cache[cache_key] = {
                'memory': memory,
                'metadata': metadata,
                'importance': importance
            }
            
            if embedding is not None:
                self.embedding_cache[memory_id] = embedding
                
            await self._enforce_memory_limit()
            return memory_id
            
        except Exception as e:
            logger.error(f"Store error: {e}")
            raise
    
    async def retrieve(self, 
                      query: Dict[str, Any],
                      limit: int = 10,
                      min_importance: float = 0.0) -> List[Dict[str, Any]]:
        """Retrieve memories with caching and optimized query."""
        cache_key = f"query_{hash(json.dumps(query, sort_keys=True))}"
        
        # Check cache
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        try:
            async with self._get_db() as db:
                # Optimized query with index usage
                cursor = await db.execute('''
                    SELECT content, importance, metadata
                    FROM memories
                    WHERE importance >= ?
                    ORDER BY importance DESC, last_access DESC
                    LIMIT ?
                ''', (min_importance, limit))
                
                results = await cursor.fetchall()
                
                memories = []
                for content, importance, metadata in results:
                    memory_data = json.loads(content)
                    if metadata:
                        memory_data['metadata'] = json.loads(metadata)
                    memory_data['importance'] = importance
                    memories.append(memory_data)
                
                # Update cache
                self.cache[cache_key] = memories
                return memories
                
        except Exception as e:
            logger.error(f"Retrieval error: {e}")
            raise
    
    async def batch_store(self, 
                         memories: List[Dict[str, Any]]) -> List[str]:
        """Batch store memories for better performance."""
        if not memories:
            return []
            
        memory_ids = []
        async with self._get_db() as db:
            async with db.cursor() as cursor:
                for i in range(0, len(memories), self.config.batch_size):
                    batch = memories[i:i + self.config.batch_size]
                    values = [
                        (
                            str(time.time() + idx),
                            json.dumps(mem['content']),
                            time.time(),
                            mem.get('importance', 0.5),
                            time.time(),
                            0,
                            json.dumps(mem.get('metadata')),
                            mem.get('embedding').tobytes() 
                            if 'embedding' in mem else None
                        )
                        for idx, mem in enumerate(batch)
                    ]
                    
                    await cursor.executemany('''
                        INSERT INTO memories 
                        (id, content, timestamp, importance, last_access, 
                         access_count, metadata, embedding)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ''', values)
                    
                    memory_ids.extend([v[0] for v in values])
                    
        return memory_ids

    def _get_cache_key(self, memory_id: str) -> str:
        """Generate consistent cache key."""
        return f"memory_{memory_id}"
