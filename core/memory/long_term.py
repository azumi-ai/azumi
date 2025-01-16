from typing import Dict, List, Any
import sqlite3
import json
import time

class LongTermMemory:
    def __init__(self, db_path: str = 'memory.db'):
        self.conn = sqlite3.connect(db_path)
        self.create_tables()
    
    def create_tables(self) -> None:
        with self.conn:
            self.conn.execute('''
                CREATE TABLE IF NOT EXISTS memories (
                    id TEXT PRIMARY KEY,
                    content TEXT,
                    timestamp REAL,
                    importance REAL,
                    last_access REAL,
                    access_count INTEGER
                )
            ''')
    
    def store(self, memory: Dict[str, Any], importance: float = 0.5) -> None:
        memory_id = str(time.time())
        with self.conn:
            self.conn.execute('''
                INSERT INTO memories 
                (id, content, timestamp, importance, last_access, access_count)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                memory_id,
                json.dumps(memory),
                time.time(),
                importance,
                time.time(),
                0
            ))
    
    def retrieve(self, query: Dict[str, Any]) -> List[Dict[str, Any]]:
        # Implement more sophisticated retrieval logic based on query parameters
        with self.conn:
            cursor = self.conn.execute('''
                SELECT content, importance FROM memories
                ORDER BY importance DESC, last_access DESC
                LIMIT 10
            ''')
            return [json.loads(row[0]) for row in cursor.fetchall()]
