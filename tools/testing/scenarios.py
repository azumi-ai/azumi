from typing import Dict, List, Optional, Any
import yaml
import json
import asyncio
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class ScenarioConfig:
    template_dir: str = "scenarios"
    max_complexity: int = 5
    max_duration: int = 3600  # seconds
    parallel_execution: bool = True

class TestScenarioManager:
    """Test scenario management system."""
    
    def __init__(self, config: ScenarioConfig):
        self.config = config
        self.scenarios = {}
        self.templates = self._load_templates()
        self._validation_rules = self._init_validation_rules()
    
    async def create_scenario(
        self,
        scenario_type: str,
        parameters: Dict[str, Any],
        complexity: Optional[int] = None
    ) -> Dict[str, Any]:
        """Create test scenario from template."""
        template = self.templates.get(scenario_type)
        if not template:
            raise ValueError(f"Unknown scenario type: {scenario_type}")
            
        # Validate complexity
        if complexity and complexity > self.config.max_complexity:
            raise ValueError(f"Complexity exceeds maximum: {complexity}")
            
        # Generate scenario
        scenario = await self._generate_scenario(
            template,
            parameters,
            complexity or 1
        )
        
        # Validate scenario
        if not await self._validate_scenario(scenario):
            raise ValueError("Invalid scenario configuration")
            
        scenario_id = str(uuid.uuid4())
        self.scenarios[scenario_id] = scenario
        
        return {
            'id': scenario_id,
            'type': scenario_type,
            'config': scenario
        }
    
    async def _generate_scenario(
        self,
        template: Dict[str, Any],
        parameters: Dict[str, Any],
        complexity: int
    ) -> Dict[str, Any]:
        """Generate scenario from template."""
        scenario = template.copy()
        
        # Apply parameters
        scenario.update(parameters)
        
        # Scale complexity
        await self._scale_complexity(scenario, complexity)
        
        # Add metadata
        scenario['metadata'] = {
            'complexity': complexity,
            'created_at': time.time(),
            'parameters': parameters
        }
        
        return scenario
