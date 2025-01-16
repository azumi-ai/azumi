from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import torch
import numpy as np

@dataclass
class NPCConfig:
    personality_complexity: int = 16
    memory_span: int = 1000
    interaction_radius: float = 10.0
    evolution_rate: float = 0.1

class DynamicNPC:
    """Dynamic NPC system with evolving personality."""
    
    def __init__(self, config: NPCConfig):
        self.config = config
        self.personality = NPCPersonality(config.personality_complexity)
        self.memory = NPCMemory(config.memory_span)
        self.behavior = NPCBehavior(config.evolution_rate)
        
    async def interact(
        self,
        player_action: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Process player interaction and generate response."""
        
        # Update memory with interaction
        await self.memory.store_interaction(player_action, context)
        
        # Process through personality system
        personality_response = await self.personality.process_interaction(
            player_action,
            self.memory.get_relevant_memories(context)
        )
        
        # Generate behavior response
        behavior_response = await self.behavior.generate_response(
            personality_response,
            context
        )
        
        # Evolve personality based on interaction
        await self.personality.evolve(player_action, behavior_response)
        
        return {
            'response': behavior_response,
            'personality_state': self.personality.get_state(),
            'memory_summary': self.memory.summarize()
        }
