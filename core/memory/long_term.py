from typing import Dict, List, Any, Optional
import sqlite3
import json
import time
import asyncio
import aiosqlite
from contextlib import asynccontextmanager
import numpy as np
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class MemoryConfig:
    importance_threshold: float = 0.5
    retention_period: int = 30 * 24 * 3600  # 30 days in seconds
    cleanup_interval: int = 24 * 3600  # 24 hours in seconds
    max_memories: int = 10000

class LongTermMemory:
    def __init__(self, 
                 db_path: str = 'memory.db',
                 config: Optional[MemoryConfig] = None):
        self.db_path = db_path
        self.config = config or MemoryConfig()
        self.cleanup_task = None
        self._init_db()
        self._start_cleanup_task()
    
    async def _init_db(self) -> None:
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
            await db.execute('CREATE INDEX IF NOT EXISTS idx_importance ON memories(importance)')
            await db.execute('CREATE INDEX IF NOT EXISTS idx_timestamp ON memories(timestamp)')
    
    @asynccontextmanager
    async def _get_db(self):
        db = await aiosqlite.connect(self.db_path)
        try:
            yield db
        finally:
            await db.close()
    
    async def store(self, 
                    memory: Dict[str, Any],
                    importance: float = 0.5,
                    metadata: Optional[Dict[str, Any]] = None,
                    embedding: Optional[np.ndarray] = None) -> str:
        if importance < self.config.importance_threshold:
            return None
            
        memory_id = str(time.time())
        
        async with self._get_db() as db:
            await db.execute('''
                INSERT INTO memories 
                (id, content, timestamp, importance, last_access, access_count, metadata, embedding)
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
            await db.commit()
            
        await self._enforce_memory_limit()
        return memory_id
    
    async def retrieve(self, 
                      query: Dict[str, Any],
                      limit: int = 10,
                      min_importance: float = 0.0) -> List[Dict[str, Any]]:
        try:
            async with self._get_db() as db:
                async with db.execute('''
                    SELECT content, importance, metadata
                    FROM memories
                    WHERE importance >= ?
                    ORDER BY importance DESC, last_access DESC
                    LIMIT ?
                ''', (min_importance, limit)) as cursor:
                    results = await cursor.fetchall()
                    
                    memories = []
                    for content, importance, metadata in results:
                        memory_data = json.loads(content)
                        if metadata:
                            memory_data['metadata'] = json.loads(metadata)
                        memory_data['importance'] = importance
                        memories.append(memory_data)
                        
                    return memories
                    
        except Exception as e:
            logger.error(f"Error retrieving memories: {e}")
            return []
    
    async def _enforce_memory_limit(self) -> None:
        async with self._get_db() as db:
            async with db.execute('SELECT COUNT(*) FROM memories') as cursor:
                count = (await cursor.fetchone())[0]
                
            if count > self.config.max_memories:
                excess = count - self.config.max_memories
                await db.execute('''
                    DELETE FROM memories 
                    WHERE id IN (
                        SELECT id FROM memories 
                        ORDER BY importance ASC, last_access ASC 
                        LIMIT ?
                    )
                ''', (excess,))
                await db.commit()
    
    async def _cleanup_old_memories(self) -> None:
        current_time = time.time()
        cutoff_time = current_time - self.config.retention_period
        
        async with self._get_db() as db:
            await db.execute('''
                DELETE FROM memories 
                WHERE timestamp < ? AND importance < ?
            ''', (cutoff_time, self.config.importance_threshold))
            await db.commit()
    
    def _start_cleanup_task(self) -> None:
        async def cleanup_loop():
            while True:
                try:
                    await self._cleanup_old_memories()
                except Exception as e:
                    logger.error(f"Error in cleanup task: {e}")
                await asyncio.sleep(self.config.cleanup_interval)
        
        self.cleanup_task = asyncio.create_task(cleanup_loop())
    
    async def close(self) -> None:
        if self.cleanup_task:
            self.cleanup_task.cancel()
            try:
                await self.cleanup_task
            except asyncio.CancelledError:
                pass
