import torch
import torch.nn as nn
from transformers import BertModel

class EmotionClassifier(nn.Module):
    def __init__(self, num_emotions: int = 8):
        super().__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(768, num_emotions)
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return self.softmax(logits)
