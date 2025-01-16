from typing import Dict, List
import numpy as np
from .detection import EmotionDetector
from ..memory import short_term, long_term

class EmotionalLearning:
    def __init__(self, emotion_detector: EmotionDetector):
        self.emotion_detector = emotion_detector
        self.learning_rate = 0.1
        self.emotional_memory = {}
    
    def learn_from_interaction(self, 
                             text: str, 
                             response: str, 
                             feedback: float) -> None:
        emotions = self.emotion_detector.detect_emotions(text)
        response_emotions = self.emotion_detector.detect_emotions(response)
        
        for emotion, intensity in emotions.items():
            if emotion not in self.emotional_memory:
                self.emotional_memory[emotion] = []
            
            self.emotional_memory[emotion].append({
                'intensity': intensity,
                'response_emotions': response_emotions,
                'feedback': feedback
            })
    
    def optimize_emotional_responses(self) -> None:
        for emotion, memories in self.emotional_memory.items():
            if len(memories) > 10:  # Minimum sample size
                successful_responses = [m for m in memories if m['feedback'] > 0.7]
                if successful_responses:
                    avg_intensity = np.mean([m['intensity'] for m in successful_responses])
                    self.learning_rate = max(0.05, min(0.2, avg_intensity))
