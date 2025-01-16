from typing import Dict, List, Optional, Any, Union, Tuple
import torch
import torch.nn as nn
from transformers import pipeline, AutoTokenizer, AutoModel
from concurrent.futures import ThreadPoolExecutor
import asyncio
from dataclasses import dataclass
import numpy as np
from cachetools import TTLCache, LRUCache
import logging
from contextlib import asynccontextmanager
from collections import defaultdict

logger = logging.getLogger(__name__)

@dataclass
class EmotionDetectionConfig:
    model_name: str = "j-hartmann/emotion-english-distilroberta-base"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size: int = 32
    max_length: int = 512
    cache_ttl: int = 3600  # 1 hour
    cache_size: int = 10000
    embedding_cache_size: int = 1000
    emotion_threshold: float = 0.3
    num_workers: int = 4
    use_quantization: bool = True
    enable_fp16: bool = True

class EmotionDetector:
    """Optimized emotion detection system."""
    
    def __init__(self, config: Optional[EmotionDetectionConfig] = None):
        self.config = config or EmotionDetectionConfig()
        
        # Initialize caches
        self.result_cache = TTLCache(
            maxsize=self.config.cache_size,
            ttl=self.config.cache_ttl
        )
        self.embedding_cache = LRUCache(
            maxsize=self.config.embedding_cache_size
        )
        
        # Initialize processing pools
        self._executor = ThreadPoolExecutor(
            max_workers=self.config.num_workers
        )
        self._setup_model()
        
        # Batch processing queues
        self.batch_queue = asyncio.Queue()
        self.batch_results = defaultdict(asyncio.Future)
        self._batch_processor_task = None
        
        # Performance monitoring
        self.metrics = {
            'processed': 0,
            'cache_hits': 0,
            'batch_size_avg': 0,
            'processing_time_avg': 0
        }
    
    def _setup_model(self) -> None:
        """Setup emotion detection model with optimizations."""
        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_name
            )
            
            # Load model with optimizations
            self.model = AutoModel.from_pretrained(
                self.config.model_name
            ).to(self.config.device)
            
            # Apply optimizations
            if self.config.use_quantization and self.config.device == "cpu":
                self.model = torch.quantization.quantize_dynamic(
                    self.model,
                    {torch.nn.Linear},
                    dtype=torch.qint8
                )
            
            if self.config.enable_fp16 and self.config.device == "cuda":
                self.model = self.model.half()
            
            # Create pipeline with optimized settings
            self.emotion_classifier = pipeline(
                "text-classification",
                model=self.model,
                tokenizer=self.tokenizer,
                device=self.config.device,
                return_all_scores=True,
                batch_size=self.config.batch_size
            )
            
            # JIT compile for faster inference
            self.model = torch.jit.script(self.model)
            
        except Exception as e:
            logger.error(f"Model initialization error: {e}")
            raise
    
    async def start(self) -> None:
        """Start the batch processor."""
        if self._batch_processor_task is None:
            self._batch_processor_task = asyncio.create_task(
                self._process_batches()
            )
    
    async def stop(self) -> None:
        """Stop the batch processor."""
        if self._batch_processor_task:
            self._batch_processor_task.cancel()
            try:
                await self._batch_processor_task
            except asyncio.CancelledError:
                pass
            self._batch_processor_task = None
    
    async def detect_emotions(
        self,
        text: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, float]:
        """Detect emotions in text with caching and optimizations."""
        cache_key = self._generate_cache_key(text, context)
        
        # Check cache
        if cache_key in self.result_cache:
            self.metrics['cache_hits'] += 1
            return self.result_cache[cache_key]
        
        # Add to batch queue
        future = asyncio.Future()
        self.batch_results[cache_key] = future
        await self.batch_queue.put((cache_key, text, context))
        
        try:
            result = await future
            return result
        except Exception as e:
            logger.error(f"Emotion detection error: {e}")
            raise
    
    async def detect_emotions_batch(
        self,
        texts: List[str],
        contexts: Optional[List[Dict[str, Any]]] = None
    ) -> List[Dict[str, float]]:
        """Batch emotion detection for multiple texts."""
        if not texts:
            return []
            
        contexts = contexts or [None] * len(texts)
        futures = []
        
        # Process in optimal batch sizes
        for i in range(0, len(texts), self.config.batch_size):
            batch_texts = texts[i:i + self.config.batch_size]
            batch_contexts = contexts[i:i + self.config.batch_size]
            
            batch_futures = await self._process_text_batch(
                batch_texts,
                batch_contexts
            )
            futures.extend(batch_futures)
        
        # Gather results
        try:
            results = await asyncio.gather(*futures)
            return results
        except Exception as e:
            logger.error(f"Batch detection error: {e}")
            raise
    
    @torch.no_grad()
    async def _process_text_batch(
        self,
        texts: List[str],
        contexts: List[Optional[Dict[str, Any]]]
    ) -> List[asyncio.Future]:
        """Process a batch of texts efficiently."""
        futures = []
        cache_keys = []
        uncached_indices = []
        uncached_texts = []
        
        # Check cache and collect uncached texts
        for idx, (text, context) in enumerate(zip(texts, contexts)):
            cache_key = self._generate_cache_key(text, context)
            cache_keys.append(cache_key)
            
            if cache_key in self.result_cache:
                future = asyncio.Future()
                future.set_result(self.result_cache[cache_key])
                futures.append(future)
            else:
                uncached_indices.append(idx)
                uncached_texts.append(text)
                future = asyncio.Future()
                self.batch_results[cache_key] = future
                futures.append(future)
        
        if uncached_texts:
            # Process uncached texts
            try:
                inputs = self.tokenizer(
                    uncached_texts,
                    padding=True,
                    truncation=True,
                    max_length=self.config.max_length,
                    return_tensors="pt"
                ).to(self.config.device)
                
                outputs = await asyncio.to_thread(
                    self._run_inference,
                    inputs
                )
                
                # Process results
                for idx, text_idx in enumerate(uncached_indices):
                    cache_key = cache_keys[text_idx]
                    emotions = self._process_output(
                        outputs[idx],
                        contexts[text_idx]
                    )
                    
                    # Cache result
                    self.result_cache[cache_key] = emotions
                    
                    # Set future result
                    self.batch_results[cache_key].set_result(emotions)
                    
            except Exception as e:
                logger.error(f"Batch processing error: {e}")
                # Set error for all pending futures
                for idx in uncached_indices:
                    cache_key = cache_keys[idx]
                    self.batch_results[cache_key].set_exception(e)
        
        return futures
    
    def _run_inference(
        self,
        inputs: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Run model inference with performance optimizations."""
        with torch.cuda.amp.autocast() if self.config.enable_fp16 else nullcontext():
            outputs = self.model(**inputs)
            predictions = self.emotion_classifier(outputs)
            return predictions
    
    def _process_output(
        self,
        output: torch.Tensor,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, float]:
        """Process model output into emotion predictions."""
        emotions = {
            pred['label']: pred['score']
            for pred in output
            if pred['score'] > self.config.emotion_threshold
        }
        
        if context:
            emotions = self._adjust_emotions_by_context(emotions, context)
            
        return emotions
    
    def _generate_cache_key(
        self,
        text: str,
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Generate consistent cache key."""
        key = f"emotion_{hash(text)}"
        if context:
            key += f"_{hash(str(sorted(context.items())))}"
        return key
    
    async def get_emotion_embeddings(
        self,
        text: str
    ) -> torch.Tensor:
        """Get emotion embeddings with caching."""
        cache_key = f"embed_{hash(text)}"
        
        if cache_key in self.embedding_cache:
            return self.embedding_cache[cache_key]
            
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.config.max_length
        ).to(self.config.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            embeddings = outputs.last_hidden_state.mean(dim=1)
            
        self.embedding_cache[cache_key] = embeddings
        return embeddings
    
    def _adjust_emotions_by_context(
        self,
        emotions: Dict[str, float],
        context: Dict[str, Any]
    ) -> Dict[str, float]:
        """Adjust emotion predictions based on context."""
        if 'mood' in context:
            mood_factor = context['mood']
            emotions = {k: v * mood_factor for k, v in emotions.items()}
        
        if 'personality' in context:
            personality_factor = context['personality'].get(
                'emotional_intensity',
                1.0
            )
            emotions = {k: v * personality_factor for k, v in emotions.items()}
        
        return emotions

    async def get_metrics(self) -> Dict[str, float]:
        """Get performance metrics."""
        return {
            'processed_count': self.metrics['processed'],
            'cache_hit_rate': (
                self.metrics['cache_hits'] / max(1, self.metrics['processed'])
            ),
            'average_batch_size': self.metrics['batch_size_avg'],
            'average_processing_time': self.metrics['processing_time_avg']
        }
    
    async def clear_caches(self) -> None:
        """Clear all caches."""
        self.result_cache.clear()
        self.embedding_cache.clear()

    def __del__(self):
        """Cleanup resources."""
        if hasattr(self, '_executor'):
            self._executor.shutdown(wait=False)
