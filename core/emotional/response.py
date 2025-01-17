from typing import Dict, List, Optional, Any, Tuple
import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel, GPT2Config
import numpy as np
from dataclasses import dataclass
import logging
from cachetools import TTLCache, LRUCache
import asyncio
from collections import defaultdict
import time
import json

logger = logging.getLogger(__name__)

@dataclass
class ResponseConfig:
    model_name: str = "gpt2-medium"
    max_length: int = 100
    temperature: float = 0.7
    top_p: float = 0.9
    cache_ttl: int = 3600  # 1 hour
    cache_size: int = 10000
    context_window: int = 5
    personality_weight: float = 0.7
    emotion_threshold: float = 0.3

class ResponseCache:
    """Advanced response caching system."""
    
    def __init__(self, config: ResponseConfig):
        self.config = config
        self.response_cache = TTLCache(
            maxsize=config.cache_size,
            ttl=config.cache_ttl
        )
        self.context_cache = LRUCache(maxsize=1000)
        self.pattern_cache = {}
        
    def get_cached_response(
        self,
        key: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Optional[Dict[str, Any]]:
        """Get cached response with context matching."""
        if key not in self.response_cache:
            return None
            
        cached = self.response_cache[key]
        
        # Check context similarity if provided
        if context and 'context' in cached:
            similarity = self._calculate_context_similarity(
                context,
                cached['context']
            )
            if similarity < 0.8:  # Threshold for context similarity
                return None
                
        return cached
    
    def cache_response(
        self,
        key: str,
        response: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> None:
        """Cache response with context."""
        self.response_cache[key] = {
            'response': response,
            'context': context,
            'timestamp': time.time()
        }
        
        # Cache context patterns
        if context:
            pattern = self._extract_context_pattern(context)
            self.pattern_cache[key] = pattern
    
    def _calculate_context_similarity(
        self,
        context1: Dict[str, Any],
        context2: Dict[str, Any]
    ) -> float:
        """Calculate similarity between contexts."""
        pattern1 = self._extract_context_pattern(context1)
        pattern2 = self._extract_context_pattern(context2)
        
        # Compare patterns
        common_keys = set(pattern1.keys()) & set(pattern2.keys())
        if not common_keys:
            return 0.0
            
        similarities = [
            abs(pattern1[k] - pattern2[k])
            for k in common_keys
        ]
        
        return 1.0 - (sum(similarities) / len(similarities))
    
    def _extract_context_pattern(
        self,
        context: Dict[str, Any]
    ) -> Dict[str, float]:
        """Extract pattern from context."""
        pattern = {}
        
        # Extract emotional patterns
        if 'emotions' in context:
            pattern.update(context['emotions'])
            
        # Extract situational patterns
        if 'situation' in context:
            for key, value in context['situation'].items():
                if isinstance(value, (int, float)):
                    pattern[f'situation_{key}'] = float(value)
        
        return pattern

class PersonalityModule:
    """Enhanced personality-aware response generation."""
    
    def __init__(self, config: ResponseConfig):
        self.config = config
        self.personality_embeddings = {}
        self.interaction_history = defaultdict(list)
        self.style_preferences = {}
        
    async def adapt_response(
        self,
        response: str,
        personality: Dict[str, Any],
        interaction_history: List[Dict[str, Any]]
    ) -> str:
        """Adapt response to personality."""
        # Get personality embedding
        embedding = await self._get_personality_embedding(personality)
        
        # Apply personality-specific adaptations
        response = await self._apply_style_preferences(
            response,
            personality
        )
        
        # Maintain consistency with interaction history
        response = await self._ensure_consistency(
            response,
            interaction_history
        )
        
        return response
    
    async def _get_personality_embedding(
        self,
        personality: Dict[str, Any]
    ) -> np.ndarray:
        """Generate personality embedding."""
        personality_id = personality.get('id', 'default')
        
        if personality_id in self.personality_embeddings:
            return self.personality_embeddings[personality_id]
            
        # Generate new embedding
        embedding = np.array([
            personality.get('traits', {}).get(trait, 0.5)
            for trait in ['openness', 'conscientiousness', 'extraversion', 'agreeableness', 'neuroticism']
        ])
        
        self.personality_embeddings[personality_id] = embedding
        return embedding
    
    async def _apply_style_preferences(
        self,
        response: str,
        personality: Dict[str, Any]
    ) -> str:
        """Apply personality-specific style preferences."""
        style = personality.get('communication_style', {})
        
        # Apply formality
        if 'formality' in style:
            response = await self._adjust_formality(
                response,
                style['formality']
            )
        
        # Apply expressiveness
        if 'expressiveness' in style:
            response = await self._adjust_expressiveness(
                response,
                style['expressiveness']
            )
        
        return response

class EmotionalResponseGenerator:
    """Enhanced emotional response generation system."""
    
    def __init__(self, config: Optional[ResponseConfig] = None):
        self.config = config or ResponseConfig()
        
        # Initialize components
        self.model = self._init_model()
        self.cache = ResponseCache(self.config)
        self.personality = PersonalityModule(self.config)
        
        # Response generation
        self.templates = self._load_templates()
        self.emotion_embeddings = self._init_emotion_embeddings()
        
    def _init_model(self) -> nn.Module:
        """Initialize response generation model."""
        model_config = GPT2Config.from_pretrained(self.config.model_name)
        model_config.emotion_embedding_size = 64
        
        model = GPT2LMHeadModel.from_pretrained(
            self.config.model_name,
            config=model_config
        )
        
        return model
    
    def _init_emotion_embeddings(self) -> nn.Embedding:
        """Initialize emotion embeddings."""
        return nn.Embedding(
            8,  # Number of basic emotions
            self.config.emotion_embedding_size
        )
    
    async def generate_response(
        self,
        input_text: str,
        emotions: Dict[str, float],
        context: Optional[Dict[str, Any]] = None,
        personality: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Generate emotional response."""
        # Check cache
        cache_key = self._generate_cache_key(
            input_text,
            emotions,
            context
        )
        
        cached_response = self.cache.get_cached_response(
            cache_key,
            context
        )
        if cached_response:
            return cached_response['response']
        
        try:
            # Process input
            input_embedding = await self._process_input(input_text)
            emotion_embedding = await self._process_emotions(emotions)
            context_embedding = await self._process_context(context)
            
            # Generate base response
            response = await self._generate_base_response(
                input_embedding,
                emotion_embedding,
                context_embedding
            )
            
            # Apply personality adaptations
            if personality:
                response = await self.personality.adapt_response(
                    response,
                    personality,
                    context.get('interaction_history', []) if context else []
                )
            
            # Process response
            processed_response = await self._process_response(
                response,
                emotions,
                context
            )
            
            # Cache response
            self.cache.cache_response(
                cache_key,
                processed_response,
                context
            )
            
            return processed_response
            
        except Exception as e:
            logger.error(f"Response generation error: {e}")
            raise
    
    async def _process_input(
        self,
        text: str
    ) -> torch.Tensor:
        """Process input text."""
        # Tokenize and encode
        tokens = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.max_length
        )
        
        return self.model.transformer(
            tokens.input_ids
        ).last_hidden_state
    
    async def _process_emotions(
        self,
        emotions: Dict[str, float]
    ) -> torch.Tensor:
        """Process emotion values."""
        emotion_vector = torch.zeros(8)
        for i, (emotion, value) in enumerate(emotions.items()):
            if value >= self.config.emotion_threshold:
                emotion_vector[i] = value
        
        return self.emotion_embeddings(emotion_vector)
    
    async def _process_context(
        self,
        context: Optional[Dict[str, Any]]
    ) -> Optional[torch.Tensor]:
        """Process context information."""
        if not context:
            return None
            
        # Extract relevant context features
        features = []
        
        # Add emotional context
        if 'emotional_state' in context:
            features.append(
                await self._process_emotions(context['emotional_state'])
            )
        
        # Add situational context
        if 'situation' in context:
            features.append(
                await self._process_situation(context['situation'])
            )
        
        if not features:
            return None
            
        return torch.cat(features, dim=-1)
    
    async def _generate_base_response(
        self,
        input_embedding: torch.Tensor,
        emotion_embedding: torch.Tensor,
        context_embedding: Optional[torch.Tensor]
    ) -> str:
        """Generate base response."""
        # Combine embeddings
        if context_embedding is not None:
            combined_embedding = torch.cat([
                input_embedding,
                emotion_embedding,
                context_embedding
            ], dim=-1)
        else:
            combined_embedding = torch.cat([
                input_embedding,
                emotion_embedding
            ], dim=-1)
        
        # Generate response
        outputs = self.model.generate(
            inputs_embeds=combined_embedding,
            max_length=self.config.max_length,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            do_sample=True,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id
        )
        
        return self.tokenizer.decode(outputs[0])
    
    def _generate_cache_key(
        self,
        text: str,
        emotions: Dict[str, float],
        context: Optional[Dict[str, Any]]
    ) -> str:
        """Generate cache key."""
        key_parts = [
            text,
            json.dumps(emotions, sort_keys=True)
        ]
        
        if context:
            key_parts.append(
                json.dumps(
                    self._extract_cache_context(context),
                    sort_keys=True
                )
            )
        
        return "|".join(key_parts)
    
    def _extract_cache_context(
        self,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Extract relevant context for caching."""
        cache_context = {}
        
        # Include emotional state
        if 'emotional_state' in context:
            cache_context['emotions'] = {
                k: v for k, v in context['emotional_state'].items()
                if v >= self.config.emotion_threshold
            }
        
        # Include situation type
        if 'situation' in context:
            cache_context['situation_type'] = context['situation'].get('type')
        
        return cache_context
    
    def __del__(self):
        """Cleanup resources."""
        # Cleanup caches
        self.cache.response_cache.clear()
        self.cache.context_cache.clear()
