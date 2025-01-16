from typing import Dict, List, Any
import re

class Validator:
    @staticmethod
    def validate_personality_config(config: Dict[str, Any]) -> bool:
        required_fields = ['name', 'base_traits', 'values']
        return all(field in config for field in required_fields)
    
    @staticmethod
    def validate_interaction(text: str) -> bool:
        if not text or len(text.strip()) == 0:
            return False
        return True
    
    @staticmethod
    def sanitize_input(text: str) -> str:
        # Remove potential harmful characters
        return re.sub(r'[^\w\s\-.,!?]', '', text)
