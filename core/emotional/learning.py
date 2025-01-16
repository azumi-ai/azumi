from typing import Dict, List, Optional, Any
import torch
import torch.nn as nn
import numpy as np
from dataclasses import dataclass
from collections import deque
import logging

logger = logging.getLogger(__name__)

@dataclass
class EmotionalLearningConfig:
    learning_rate: float = 0.01
    memory_size: int = 1000
    batch_size: int = 32
    update_frequency: int = 100
    min_samples: int = 50
    emotion_dims: int = 8

class EmotionalLearning:
    def __init__(
        self,
        config: Optional[EmotionalLearningConfig] = None,
        emotion_detector: Optional[Any] = None
    ):
        self.config = config or EmotionalLearningConfig()
        self.emotion_detector = emotion_detector
        self.emotional_memory = deque(maxlen=self.config.memory_size)
        self.learning_step = 0
        
        # Initialize emotion embeddings
        self.emotion_embeddings = nn.Parameter(
            torch.randn(self.config.emotion_dims, 128)
        )
        
        self.optimizer = torch.optim.Adam(
            [self.emotion_embeddings],
            lr=self.config.learning_rate
        )
        
        self._setup_metrics()
    
    def _setup_metrics(self):
        self.metrics = {
            'total_interactions': 0,
            'successful_responses': 0,
            'learning_iterations': 0,
            'average_feedback': 0.0,
        }
    
    async def learn_from_interaction(
        self,
        text: str,
        response: str,
        feedback: float,
        context: Optional[Dict[str, Any]] = None
    ) -> None:
        try:
            # Detect emotions
            emotions = await self.emotion_detector.detect_emotions(text)
            response_emotions = await self.emotion_detector.detect_emotions(response)
            
            # Store interaction
            interaction = {
                'text_emotions': emotions,
                'response_emotions': response_emotions,
                'feedback': feedback,
                'context': context
            }
            
            self.emotional_memory.append(interaction)
            self._update_metrics(feedback)
            
            # Perform learning if enough samples
            if (len(self.emotional_memory) >= self.config.min_samples and
                self.learning_step % self.config.update_frequency == 0):
                await self._update_emotion_model()
            
            self.learning_step += 1
            
        except Exception as e:
            logger.error(f"Error in emotional learning: {e}")
    
    async def _update_emotion_model(self) -> None:
        if len(self.emotional_memory) < self.config.batch_size:
            return
        
        # Sample batch
        batch = random.sample(
            self.emotional_memory,
            self.config.batch_size
        )
        
        # Prepare tensors
        emotion_tensors = []
        feedback_tensors = []
        
        for interaction in batch:
            emotions = torch.tensor(
                list(interaction['text_emotions'].values()),
                dtype=torch.float32
            )
            emotion_tensors.append(emotions)
            feedback_tensors.append(interaction['feedback'])
        
        emotions = torch.stack(emotion_tensors)
        feedback = torch.tensor(feedback_tensors)
        
        # Update emotion embeddings
        self.optimizer.zero_grad()
        loss = self._compute_learning_loss(emotions, feedback)
        loss.backward()
        self.optimizer.step()
        
        self.metrics['learning_iterations'] += 1
    
    def _compute_learning_loss(
        self,
        emotions: torch.Tensor,
        feedback: torch.Tensor
    ) -> torch.Tensor:
        # Compute similarity between emotions and embeddings
        similarity = torch.matmul(emotions, self.emotion_embeddings)
        
        # Weight similarity by feedback
        weighted_similarity = similarity * feedback.unsqueeze(1)
        
        # Compute loss
        loss = -torch.mean(weighted_similarity)
        return loss
    
    def _update_metrics(self, feedback: float) -> None:
        self.metrics['total_interactions'] += 1
        if feedback > 0.7:
            self.metrics['successful_responses'] += 1
        
        # Update running average
        self.metrics['average_feedback'] = (
            (self.metrics['average_feedback'] * (self.metrics['total_interactions'] - 1) +
             feedback) / self.metrics['total_interactions']
        )
    
    def get_metrics(self) -> Dict[str, float]:
        return {
            **self.metrics,
            'success_rate': (
                self.metrics['successful_responses'] /
                max(1, self.metrics['total_interactions'])
            )
        }
    
    def save_state(self, path: str) -> None:
        state = {
            'emotion_embeddings': self.emotion_embeddings.data,
            'metrics': self.metrics,
            'config': self.config.__dict__,
            'optimizer': self.optimizer.state_dict()
        }
        torch.save(state, path)
    
    def load_state(self, path: str) -> None:
        state = torch.load(path)
        self.emotion_embeddings.data = state['emotion_embeddings']
        self.metrics = state['metrics']
        self.config = EmotionalLearningConfig(**state['config'])
        self.optimizer.load_state_dict(state['optimizer'])
