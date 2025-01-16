from typing import Dict, List, Optional, Any
import numpy as np
from dataclasses import dataclass
import uuid
import json
import logging

logger = logging.getLogger(__name__)

@dataclass
class ScenarioConfig:
    complexity: int = 3  # 1-5 scale
    duration: int = 1000  # in steps
    interaction_frequency: float = 0.1
    branching_factor: int = 3

class ScenarioBuilder:
    """Builds interactive scenarios for character testing."""
    
    def __init__(self, config: ScenarioConfig):
        self.config = config
        self.scenarios = {}
        self.event_templates = self._load_event_templates()
        
    def create_scenario(
        self,
        scenario_type: str,
        characters: List[Dict[str, Any]],
        environment: Dict[str, Any]
    ) -> str:
        """Create a new interaction scenario."""
        scenario_id = str(uuid.uuid4())
        
        # Generate scenario structure
        events = self._generate_event_sequence(
            scenario_type,
            characters,
            environment
        )
        
        # Create scenario object
        scenario = {
            'id': scenario_id,
            'type': scenario_type,
            'characters': characters,
            'environment': environment,
            'events': events,
            'metadata': {
                'complexity': self.config.complexity,
                'duration': self.config.duration,
                'creation_time': time.time()
            }
        }
        
        self.scenarios[scenario_id] = scenario
        return scenario_id
        
    def _generate_event_sequence(
        self,
        scenario_type: str,
        characters: List[Dict[str, Any]],
        environment: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Generate sequence of events for scenario."""
        events = []
        current_time = 0
        
        while current_time < self.config.duration:
            # Determine next event
            event_template = self._select_event_template(
                scenario_type,
                characters,
                environment
            )
            
            # Generate event variations
            variations = self._generate_event_variations(
                event_template,
                self.config.branching_factor
            )
            
            # Add event to sequence
            events.append({
                'time': current_time,
                'template': event_template,
                'variations': variations,
                'participants': self._select_participants(characters),
                'conditions': self._generate_conditions(environment)
            })
            
            # Update time
            current_time += int(1 / self.config.interaction_frequency)
            
        return events
    
    def export_scenario(
        self,
        scenario_id: str,
        format: str = 'json'
    ) -> str:
        """Export scenario in specified format."""
        if scenario_id not in self.scenarios:
            raise ValueError(f"Scenario not found: {scenario_id}")
            
        scenario = self.scenarios[scenario_id]
        
        if format == 'json':
            return json.dumps(scenario, indent=2)
        else:
            raise ValueError(f"Unsupported format: {format}")
