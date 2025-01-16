from typing import Dict, Any
import numpy as np
from .identity import Identity

class CognitiveSystem:
    def __init__(self, identity: Identity):
        self.identity = identity
        self.decision_threshold = 0.6
        self.learning_rate = 0.05
    
    def make_decision(self, options: Dict[str, Dict[str, float]]) -> str:
        scores = {}
        personality_weights = self.identity.personality_vector
        
        for option, attributes in options.items():
            attribute_values = np.array(list(attributes.values()))
            scores[option] = np.dot(personality_weights, attribute_values)
        
        return max(scores.items(), key=lambda x: x[1])[0]
    
    def learn_from_outcome(self, decision: str, outcome: float) -> None:
        if outcome > self.decision_threshold:
            self.learning_rate *= 1.1
        else:
            self.learning_rate *= 0.9
