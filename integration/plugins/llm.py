from typing import Dict, List, Optional, Any
import torch
from transformers import AutoModel, AutoTokenizer
import asyncio
from dataclasses import dataclass

@dataclass
class LLMConfig:
    model_name: str
    max_length: int = 512
    batch_size: int = 16
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

class LLMIntegration:
    """Integration with external LLM models."""
    
    def __init__(self, config: LLMConfig):
        self.config = config
        self.model = AutoModel.from_pretrained(config.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        self.model.to(self.config.device)
        
    async def process_text(
        self,
        text: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Process text through LLM."""
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            max_length=self.config.max_length,
            truncation=True,
            padding=True
        ).to(self.config.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            
        return {
            'embeddings': outputs.last_hidden_state.cpu(),
            'attention': outputs.attentions[-1].cpu() if outputs.attentions else None
        }
