import torch
import torch.nn as nn
from transformers import BertModel, AutoTokenizer
from typing import Dict, List, Optional, Union, Tuple
import numpy as np
from dataclasses import dataclass
from torch.cuda.amp import autocast
import logging

logger = logging.getLogger(__name__)

@dataclass
class EmotionClassifierConfig:
    model_name: str = 'bert-base-uncased'
    num_emotions: int = 8
    hidden_dropout: float = 0.1
    attention_dropout: float = 0.1
    classifier_dropout: float = 0.2
    hidden_size: int = 768
    intermediate_size: int = 1024

class EmotionClassifier(nn.Module):
    def __init__(self, config: EmotionClassifierConfig):
        super().__init__()
        self.config = config
        
        # Load base model with custom configuration
        self.bert = BertModel.from_pretrained(
            config.model_name,
            hidden_dropout_prob=config.hidden_dropout,
            attention_probs_dropout_prob=config.attention_dropout
        )
        
        # Enhanced classifier architecture
        self.classifier = nn.Sequential(
            nn.Linear(config.hidden_size, config.intermediate_size),
            nn.LayerNorm(config.intermediate_size),
            nn.GELU(),
            nn.Dropout(config.classifier_dropout),
            nn.Linear(config.intermediate_size, config.num_emotions)
        )
        
        self.emotion_embeddings = nn.Parameter(
            torch.randn(config.num_emotions, config.hidden_size)
        )
        
        self.register_buffer(
            'emotion_attention_mask',
            torch.ones(config.num_emotions)
        )
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_attentions=return_attention
        )
        
        pooled_output = outputs.pooler_output
        
        # Enhanced emotion detection with attention mechanism
        emotion_attention = torch.matmul(
            pooled_output,
            self.emotion_embeddings.t()
        )
        emotion_attention = emotion_attention.softmax(dim=-1)
        
        # Apply classifier
        logits = self.classifier(pooled_output)
        probs = torch.softmax(logits, dim=-1)
        
        # Weight probabilities with attention
        weighted_probs = probs * emotion_attention
        
        if return_attention:
            return weighted_probs, {
                'emotion_attention': emotion_attention,
                'bert_attention': outputs.attentions
            }
        
        return weighted_probs
    
    def predict_emotions(
        self,
        texts: List[str],
        tokenizer: Optional[AutoTokenizer] = None,
        batch_size: int = 32,
        threshold: float = 0.3
    ) -> List[Dict[str, float]]:
        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        
        device = next(self.parameters()).device
        self.eval()
        
        all_predictions = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            encoded = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors='pt'
            ).to(device)
            
            with torch.no_grad(), autocast():
                predictions = self(
                    input_ids=encoded['input_ids'],
                    attention_mask=encoded['attention_mask']
                )
            
            # Convert to numpy for processing
            batch_predictions = predictions.cpu().numpy()
            
            for pred in batch_predictions:
                emotions = {
                    f'emotion_{i}': float(score)
                    for i, score in enumerate(pred)
                    if score >= threshold
                }
                all_predictions.append(emotions)
        
        return all_predictions

    def get_attention_patterns(
        self,
        text: str,
        tokenizer: Optional[AutoTokenizer] = None
    ) -> Dict[str, np.ndarray]:
        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        
        encoded = tokenizer(
            text,
            return_tensors='pt',
            padding=True,
            truncation=True
        ).to(next(self.parameters()).device)
        
        with torch.no_grad():
            _, attention_data = self(
                input_ids=encoded['input_ids'],
                attention_mask=encoded['attention_mask'],
                return_attention=True
            )
        
        return {
            'emotion_attention': attention_data['emotion_attention'].cpu().numpy(),
            'token_attention': [
                layer.cpu().numpy()
                for layer in attention_data['bert_attention']
            ]
        }
