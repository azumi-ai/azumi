from typing import Dict, List, Optional, Any, Callable
import jsonschema
from dataclasses import dataclass
import logging
import time

logger = logging.getLogger(__name__)

@dataclass
class ValidationConfig:
    schema_dir: str = "schemas"
    strict_mode: bool = True
    cache_ttl: int = 3600  # seconds

class TestValidator:
    """Test validation system."""
    
    def __init__(self, config: ValidationConfig):
        self.config = config
        self.schemas = self._load_schemas()
        self.validators = {}
        self.cache = {}
        
    async def validate(
        self,
        data: Dict[str, Any],
        schema_name: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Validate test data against schema."""
        # Get validator
        validator = await self._get_validator(schema_name)
        
        # Check cache
        cache_key = self._generate_cache_key(data, schema_name)
        if cache_key in self.cache:
            cache_entry = self.cache[cache_key]
            if time.time() - cache_entry['timestamp'] < self.config.cache_ttl:
                return cache_entry['result']
        
        # Perform validation
        try:
            validator.validate(data)
            result = {
                'valid': True,
                'schema': schema_name,
                'timestamp': time.time()
            }
        except jsonschema.exceptions.ValidationError as e:
            result = {
                'valid': False,
                'schema': schema_name,
                'error': str(e),
                'timestamp': time.time()
            }
            
        # Update cache
        self.cache[cache_key] = {
            'result': result,
            'timestamp': time.time()
        }
        
        return result
