import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel, GPT2Config
from typing import Dict, List, Optional, Union, Any
import numpy as np
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class NarrativeConfig:
    base_model: str = 'gpt2'
    max_length: int = 1024
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    num_return_sequences: int = 1
    narrative_embedding_dim: int = 256

class NarrativeGenerator(nn.Module):
    def __init__(self, config: Optional[NarrativeConfig] = None):
        super().__init__()
        self.config = config or NarrativeConfig()
        
        # Load base model
        self.gpt2_config = GPT2Config.from_pretrained(self.config.base_model)
        self.gpt2_config.narrative_embedding_dim = self.config.narrative_embedding_dim
        
        self.gpt2 = GPT2LMHeadModel.from_pretrained(
            self.config.base_model,
            config=self.gpt2_config
        )
        
        # Add narrative-specific components
        self.narrative_embeddings = nn.Embedding(
            10,  # Number of narrative elements
            self.config.narrative_embedding_dim
        )
        
        self.narrative_projection = nn.Linear(
            self.config.narrative_embedding_dim,
            self.gpt2_config.hidden_size
        )
        
        self.context_encoder = nn.LSTM(
            self.gpt2_config.hidden_size,
            self.config.narrative_embedding_dim,
            batch_first=True
        )
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        narrative_ids: Optional[torch.Tensor] = None,
        context_embeds: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Process narrative elements
        if narrative_ids is not None:
            narrative_embeds = self.narrative_embeddings(narrative_ids)
            narrative_hidden = self.narrative_projection(narrative_embeds)
        else:
            narrative_hidden = None
        
        # Process context
        if context_embeds is not None:
            context_out, _ = self.context_encoder(context_embeds)
            context_hidden = context_out[:, -1, :]  # Take last hidden state
            context_hidden = self.narrative_projection(context_hidden)
        else:
            context_hidden = None
        
        # Generate base outputs
        outputs = self.gpt2(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=None
        )
        
        hidden_states = outputs.hidden_states
        
        # Combine with narrative and context
        if narrative_hidden is not None:
            hidden_states = hidden_states + narrative_hidden.unsqueeze(1)
        
        if context_hidden is not None:
            hidden_states = hidden_states + context_hidden.unsqueeze(1)
        
        return self.gpt2.lm_head(hidden_states)
    
    def generate_narrative(
        self,
        prompt: str,
        context: Optional[Dict[str, Any]] = None,
        narrative_elements: Optional[List[str]] = None,
        max_length: Optional[int] = None
    ) -> List[str]:
        try:
            device = next(self.parameters()).device
            
            # Encode prompt
            inputs = self.tokenizer(
                prompt,
                return_tensors='pt',
                padding=True,
                truncation=True
            ).to(device)
            
            # Prepare narrative elements
            if narrative_elements:
                narrative_ids = self._encode_narrative_elements(
                    narrative_elements
                ).to(device)
            else:
                narrative_ids = None
            
            # Prepare context
            if context:
                context_embeds = self._encode_context(context).to(device)
            else:
                context_embeds = None
            
            # Generate
            outputs = self.generate(
                **inputs,
                narrative_ids=narrative_ids,
                context_embeds=context_embeds,
                max_length=max_length or self.config.max_length,
                num_return_sequences=self.config.num_return_sequences,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                top_k=self.config.top_k,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
            
            # Decode outputs
            generated_texts = self.tokenizer.batch_decode(
                outputs,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )
            
            return generated_texts
            
        except Exception as e:
            logger.error(f"Error generating narrative: {e}")
            return [prompt]  # Fallback to original prompt
    
    def _encode_narrative_elements(
        self,
        elements: List[str]
    ) -> torch.Tensor:
        """Encode narrative elements into tensor representation."""
        # Map elements to indices
        element_to_idx = {
            'character': 0,
            'setting': 1,
            'conflict': 2,
            'resolution': 3,
            'emotion': 4,
            'action': 5,
            'dialogue': 6,
            'description': 7,
            'transition': 8,
            'theme': 9
        }
        
        indices = [element_to_idx.get(elem, 0) for elem in elements]
        return torch.tensor(indices, dtype=torch.long)
    
    def _encode_context(
        self,
        context: Dict[str, Any]
    ) -> torch.Tensor:
        """Encode context information into embeddings."""
        context_keys = ['setting', 'mood', 'style', 'tone']
        context_values = []
        
        for key in context_keys:
            if key in context:
                # Encode context value using GPT2 and get hidden states
                encoded = self.tokenizer(
                    str(context[key]),
                    return_tensors='pt',
                    padding=True,
                    truncation=True
                )
                
                with torch.no_grad():
                    outputs = self.gpt2(
                        input_ids=encoded['input_ids'],
                        attention_mask=encoded['attention_mask']
                    )
                    context_values.append(outputs.last_hidden_state.mean(dim=1))
            else:
                # Use zero embedding if context key not present
                context_values.append(
                    torch.zeros(1, self.gpt2_config.hidden_size)
                )
        
        return torch.cat(context_values, dim=-1)
    
    def adjust_narrative_style(
        self,
        style: str,
        intensity: float = 1.0
    ) -> None:
        """Adjust narrative generation style."""
        if not 0 <= intensity <= 1:
            raise ValueError("Intensity must be between 0 and 1")
            
        # Adjust temperature based on style
        style_temps = {
            'creative': 0.9,
            'formal': 0.5,
            'casual': 0.7,
            'poetic': 0.8,
            'technical': 0.4
        }
        
        self.config.temperature = style_temps.get(style, 0.7) * intensity
        
        # Adjust other generation parameters
        if style == 'creative':
            self.config.top_p = 0.9
            self.config.top_k = 50
        elif style == 'formal':
            self.config.top_p = 0.7
            self.config.top_k = 30
        elif style == 'technical':
            self.config.top_p = 0.5
            self.config.top_k = 20
    
    def add_special_tokens(
        self,
        special_tokens: List[str]
    ) -> None:
        """Add special tokens for narrative control."""
        special_tokens_dict = {'additional_special_tokens': special_tokens}
        self.tokenizer.add_special_tokens(special_tokens_dict)
        self.gpt2.resize_token_embeddings(len(self.tokenizer))
    
    def save_pretrained(
        self,
        path: str
    ) -> None:
        """Save model, tokenizer, and configuration."""
        os.makedirs(path, exist_ok=True)
        
        # Save model components
        self.gpt2.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        
        # Save narrative-specific components
        torch.save({
            'narrative_embeddings': self.narrative_embeddings.state_dict(),
            'narrative_projection': self.narrative_projection.state_dict(),
            'context_encoder': self.context_encoder.state_dict(),
            'config': self.config.__dict__
        }, os.path.join(path, 'narrative_components.pt'))
    
    @classmethod
    def from_pretrained(
        cls,
        path: str
    ) -> 'NarrativeGenerator':
        """Load pretrained model and components."""
        # Load configuration and create model
        config_dict = torch.load(
            os.path.join(path, 'narrative_components.pt')
        )['config']
        model = cls(NarrativeConfig(**config_dict))
        
        # Load GPT2 components
        model.gpt2 = GPT2LMHeadModel.from_pretrained(path)
        model.tokenizer = AutoTokenizer.from_pretrained(path)
        
        # Load narrative components
        narrative_components = torch.load(
            os.path.join(path, 'narrative_components.pt')
        )
        model.narrative_embeddings.load_state_dict(
            narrative_components['narrative_embeddings']
        )
        model.narrative_projection.load_state_dict(
            narrative_components['narrative_projection']
        )
        model.context_encoder.load_state_dict(
            narrative_components['context_encoder']
        )
        
        return model
    
    def get_narrative_attention(
        self,
        text: str,
        narrative_elements: List[str]
    ) -> Dict[str, torch.Tensor]:
        """Get attention patterns for narrative elements."""
        encoded = self.tokenizer(
            text,
            return_tensors='pt',
            padding=True,
            truncation=True
        )
        
        narrative_ids = self._encode_narrative_elements(narrative_elements)
        
        with torch.no_grad():
            outputs = self.gpt2(
                input_ids=encoded['input_ids'],
                attention_mask=encoded['attention_mask'],
                output_attentions=True
            )
            
            narrative_embeds = self.narrative_embeddings(narrative_ids)
            attention_weights = torch.matmul(
                outputs.last_hidden_state,
                narrative_embeds.transpose(0, 1)
            )
            
        return {
            'token_attention': outputs.attentions,
            'narrative_attention': attention_weights
        }
    
    def interpolate_narratives(
        self,
        narrative1: str,
        narrative2: str,
        alpha: float = 0.5
    ) -> str:
        """Interpolate between two narratives."""
        if not 0 <= alpha <= 1:
            raise ValueError("Alpha must be between 0 and 1")
            
        # Encode both narratives
        encoded1 = self.tokenizer(narrative1, return_tensors='pt')
        encoded2 = self.tokenizer(narrative2, return_tensors='pt')
        
        with torch.no_grad():
            # Get hidden states for both narratives
            hidden1 = self.gpt2(
                input_ids=encoded1['input_ids']
            ).last_hidden_state
            hidden2 = self.gpt2(
                input_ids=encoded2['input_ids']
            ).last_hidden_state
            
            # Interpolate hidden states
            interpolated = alpha * hidden1 + (1 - alpha) * hidden2
            
            # Generate from interpolated hidden states
            outputs = self.gpt2.generate(
                inputs_embeds=interpolated,
                max_length=self.config.max_length,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                top_k=self.config.top_k,
                do_sample=True
            )
        
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

class NarrativeOptimizer:
    """Optimizer for narrative generation parameters."""
    
    def __init__(
        self,
        generator: NarrativeGenerator,
        learning_rate: float = 0.01
    ):
        self.generator = generator
        self.learning_rate = learning_rate
        self.history = []
    
    def optimize(
        self,
        target_metrics: Dict[str, float],
        num_iterations: int = 100
    ) -> None:
        """Optimize generation parameters based on target metrics."""
        for _ in range(num_iterations):
            current_metrics = self._evaluate_metrics()
            updates = self._compute_updates(current_metrics, target_metrics)
            self._apply_updates(updates)
            self.history.append(current_metrics)
    
    def _evaluate_metrics(self) -> Dict[str, float]:
        """Evaluate current generation metrics."""
        # Implement metric evaluation logic
        pass
    
    def _compute_updates(
        self,
        current: Dict[str, float],
        target: Dict[str, float]
    ) -> Dict[str, float]:
        """Compute parameter updates based on metrics."""
        # Implement update computation logic
        pass
    
    def _apply_updates(
        self,
        updates: Dict[str, float]
    ) -> None:
        """Apply computed updates to generator parameters."""
        # Implement parameter update logic
        pass

# Initialize logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
