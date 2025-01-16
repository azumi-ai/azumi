from typing import Dict, List
import numpy as np
from .detection import EmotionDetector

class ResponseGenerator:
    def __init__(self, emotion_detector: EmotionDetector):
        self.emotion_detector = emotion_detector
        self.response_templates = self._load_response_templates()
    
    def _load_response_templates(self) -> Dict[str, List[str]]:
        # In practice, load from a database or file
        return {
            "joy": ["That's wonderful!", "I'm so happy to hear that!"],
            "sadness": ["I understand how you feel.", "I'm here for you."],
            "anger": ["I sense that you're frustrated.", "Let's work through this."],
            "fear": ["It's okay to feel scared.", "We can face this together."],
        }
    
    def generate_response(self, text: str, context: Dict[str, Any]) -> str:
        emotions = self.emotion_detector.detect_emotions(text)
        dominant_emotion = max(emotions.items(), key=lambda x: x[1])[0]
        
        templates = self.response_templates.get(dominant_emotion, ["I understand."])
        return np.random.choice(templates)
