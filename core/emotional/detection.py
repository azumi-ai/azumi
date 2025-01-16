from typing import Dict, List
import numpy as np
from transformers import pipeline

class EmotionDetector:
    def __init__(self):
        self.emotion_classifier = pipeline("text-classification", 
                                        model="j-hartmann/emotion-english-distilroberta-base", 
                                        return_all_scores=True)
        self.emotion_threshold = 0.3
    
    def detect_emotions(self, text: str) -> Dict[str, float]:
        predictions = self.emotion_classifier(text)[0]
        return {pred['label']: pred['score'] 
                for pred in predictions 
                if pred['score'] > self.emotion_threshold}
