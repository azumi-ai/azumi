from typing import Dict, List, Optional
import numpy as np
from transformers import pipeline
from concurrent.futures import ThreadPoolExecutor
import torch
from torch.nn.functional import softmax
import logging

logger = logging.getLogger(__name__)

class EmotionDetector:
    def __init__(self, 
                 model_name: str = "j-hartmann/emotion-english-distilroberta-base",
                 batch_size: int = 8,
                 device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.batch_size = batch_size
        self.emotion_classifier = pipeline(
            "text-classification",
            model=model_name,
            return_all_scores=True,
            device=self.device
        )
        self.emotion_threshold = 0.3
        self.cache = LRUCache(maxsize=1000)
        self._executor = ThreadPoolExecutor(max_workers=4)
    
    async def detect_emotions(self, 
                            text: str,
                            context: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
        cache_key = f"{text}_{hash(str(context))}"
        if cached_result := self.cache.get(cache_key):
            return cached_result
        
        try:
            predictions = await self._get_predictions(text)
            emotions = {
                pred['label']: pred['score']
                for pred in predictions[0]
                if pred['score'] > self.emotion_threshold
            }
            
            if context:
                emotions = self._adjust_emotions_by_context(emotions, context)
            
            self.cache[cache_key] = emotions
            return emotions
            
        except Exception as e:
            logger.error(f"Error detecting emotions: {e}")
            return {}
    
    async def detect_emotions_batch(self, 
                                  texts: List[str]) -> List[Dict[str, float]]:
        batches = [texts[i:i + self.batch_size] 
                  for i in range(0, len(texts), self.batch_size)]
        
        async def process_batch(batch):
            predictions = await self._get_predictions(batch)
            return [
                {pred['label']: pred['score']
                 for pred in prediction
                 if pred['score'] > self.emotion_threshold}
                for prediction in predictions
            ]
        
        results = []
        for batch in batches:
            batch_results = await process_batch(batch)
            results.extend(batch_results)
        
        return results
    
    async def _get_predictions(self, text: str | List[str]) -> List[Dict[str, float]]:
        return await asyncio.to_thread(self.emotion_classifier, text)
    
    def _adjust_emotions_by_context(self,
                                  emotions: Dict[str, float],
                                  context: Dict[str, Any]) -> Dict[str, float]:
        if 'mood' in context:
            mood_factor = context['mood']
            emotions = {k: v * mood_factor for k, v in emotions.items()}
        
        if 'personality' in context:
            personality_factor = context['personality'].get('emotional_intensity', 1.0)
            emotions = {k: v * personality_factor for k, v in emotions.items()}
        
        return emotions
    
    def update_threshold(self, new_threshold: float) -> None:
        if not 0 <= new_threshold <= 1:
            raise ValueError("Threshold must be between 0 and 1")
        self.emotion_threshold = new_threshold
        self.cache.clear()
