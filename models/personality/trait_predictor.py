import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional
from transformers import AutoModel

class TraitPredictor(nn.Module):
    def __init__(self,
                 num_traits: int = 5,
                 hidden_dims: List[int] = [512, 256],
                 dropout_rate: float = 0.2,
                 pretrained_model: str = "bert-base-uncased"):
        super().__init__()
        
        # Load pretrained transformer
        self.encoder = AutoModel.from_pretrained(pretrained_model)
        
        # Freeze encoder parameters for efficiency
        for param in self.encoder.parameters():
            param.requires_grad = False
            
        # Build MLP layers
        input_dim = self.encoder.config.hidden_size
        self.mlp = self._build_mlp(input_dim, hidden_dims, num_traits, dropout_rate)
        
        self.layer_norm = nn.LayerNorm(num_traits)
        self.dropout = nn.Dropout(dropout_rate)
        
    def _build_mlp(self, 
                   input_dim: int,
                   hidden_dims: List[int],
                   output_dim: int,
                   dropout_rate: float) -> nn.Sequential:
        layers = []
        dims = [input_dim] + hidden_dims + [output_dim]
        
        for i in range(len(dims) - 1):
            layers.extend([
                nn.Linear(dims[i], dims[i + 1]),
                nn.LayerNorm(dims[i + 1]),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            
        return nn.Sequential(*layers[:-2])  # Remove last ReLU and Dropout
        
    def forward(self,
                input_ids: torch.Tensor,
                attention_mask: torch.Tensor,
                token_type_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Get encoder outputs
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        
        # Use pooled output for classification
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        
        # Pass through MLP
        traits = self.mlp(pooled_output)
        traits = self.layer_norm(traits)
        
        # Apply sigmoid for trait probabilities
        return torch.sigmoid(traits)
    
    def predict_traits(self,
                      texts: List[str],
                      tokenizer,
                      device: str = "cuda" if torch.cuda.is_available() else "cpu",
                      batch_size: int = 32) -> np.ndarray:
        self.eval()
        self.to(device)
        
        all_predictions = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            # Tokenize batch
            encoded = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            ).to(device)
            
            with torch.no_grad():
                predictions = self(
                    input_ids=encoded.input_ids,
                    attention_mask=encoded.attention_mask,
                    token_type_ids=encoded.get('token_type_ids', None)
                )
                
            all_predictions.append(predictions.cpu().numpy())
            
        return np.vstack(all_predictions)
    
    def interpret_prediction(self,
                           prediction: torch.Tensor,
                           trait_names: List[str],
                           threshold: float = 0.5) -> Dict[str, float]:
        """
        Interpret model predictions and return trait probabilities.
        
        Args:
            prediction: Model output tensor
            trait_names: List of trait names corresponding to output dimensions
            threshold: Minimum probability threshold for trait attribution
            
        Returns:
            Dictionary mapping trait names to their probabilities
        """
        if len(trait_names) != prediction.size(-1):
            raise ValueError("Number of trait names must match model output dimension")
            
        probs = prediction.squeeze().cpu().numpy()
        return {
            trait: float(prob)
            for trait, prob in zip(trait_names, probs)
            if prob >= threshold
        }
    
    def save_pretrained(self, path: str) -> None:
        """Save model weights and configuration."""
        os.makedirs(path, exist_ok=True)
        
        # Save model state
        torch.save(self.state_dict(), os.path.join(path, "model.pt"))
        
        # Save config
        config = {
            "num_traits": self.mlp[-2].out_features,  # Get output dim from final linear layer
            "hidden_dims": [layer.out_features for layer in self.mlp if isinstance(layer, nn.Linear)][:-1],
            "dropout_rate": self.dropout.p,
            "pretrained_model": self.encoder.config._name_or_path
        }
        
        with open(os.path.join(path, "config.json"), "w") as f:
            json.dump(config, f)
    
    @classmethod
    def from_pretrained(cls, path: str) -> "TraitPredictor":
        """Load model from pretrained weights."""
        with open(os.path.join(path, "config.json"), "r") as f:
            config = json.load(f)
            
        model = cls(**config)
        model.load_state_dict(torch.load(os.path.join(path, "model.pt")))
        return model
    
    def get_attention_weights(self,
                            input_ids: torch.Tensor,
                            attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Get attention weights for interpretability.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask tensor
            
        Returns:
            Attention weights tensor
        """
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=True
        )
        
        # Average attention weights across all layers and heads
        attention_weights = torch.stack(outputs.attentions).mean(dim=[0, 1])
        return attention_weights
    
    def get_trait_embeddings(self) -> torch.Tensor:
        """Get learned trait embeddings from the final layer."""
        return self.mlp[-2].weight.data
    
    def compute_trait_similarities(self) -> torch.Tensor:
        """Compute cosine similarities between trait embeddings."""
        trait_embeddings = self.get_trait_embeddings()
        return F.cosine_similarity(
            trait_embeddings.unsqueeze(1),
            trait_embeddings.unsqueeze(0),
            dim=2
        )

class TraitPredictorConfig:
    """Configuration class for TraitPredictor model."""
    
    def __init__(self,
                num_traits: int = 5,
                hidden_dims: List[int] = [512, 256],
                dropout_rate: float = 0.2,
                pretrained_model: str = "bert-base-uncased",
                learning_rate: float = 1e-4,
                warmup_steps: int = 1000,
                weight_decay: float = 0.01,
                max_grad_norm: float = 1.0):
        self.num_traits = num_traits
        self.hidden_dims = hidden_dims
        self.dropout_rate = dropout_rate
        self.pretrained_model = pretrained_model
        self.learning_rate = learning_rate
        self.warmup_steps = warmup_steps
        self.weight_decay = weight_decay
        self.max_grad_norm = max_grad_norm
        
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "TraitPredictorConfig":
        """Create config from dictionary."""
        return cls(**config_dict)
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return self.__dict__

class TraitPredictorTrainer:
    """Trainer class for TraitPredictor model."""
    
    def __init__(self,
                model: TraitPredictor,
                config: TraitPredictorConfig,
                train_dataset: Dataset,
                eval_dataset: Optional[Dataset] = None,
                device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.model = model.to(device)
        self.config = config
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.device = device
        
        # Setup optimizer and scheduler
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=config.warmup_steps,
            num_training_steps=len(train_dataset)
        )
        
        self.scaler = GradScaler()  # For mixed precision training
        
    def train(self,
             num_epochs: int,
             batch_size: int = 32,
             eval_steps: int = 100,
             save_steps: int = 1000,
             save_dir: Optional[str] = None):
        """Train the model."""
        train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        
        best_eval_loss = float('inf')
        global_step = 0
        
        for epoch in range(num_epochs):
            self.model.train()
            epoch_loss = 0
            
            for batch in tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                with autocast():  # Mixed precision
                    outputs = self.model(
                        input_ids=batch['input_ids'],
                        attention_mask=batch['attention_mask']
                    )
                    loss = F.binary_cross_entropy(outputs, batch['traits'])
                
                self.scaler.scale(loss).backward()
                
                # Gradient clipping
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.max_grad_norm
                )
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.scheduler.step()
                self.optimizer.zero_grad()
                
                epoch_loss += loss.item()
                global_step += 1
                
                if global_step % eval_steps == 0 and self.eval_dataset is not None:
                    eval_loss = self.evaluate()
                    self.model.train()
                    
                    if eval_loss < best_eval_loss and save_dir is not None:
                        best_eval_loss = eval_loss
                        self.save_checkpoint(
                            os.path.join(save_dir, f"checkpoint-{global_step}")
                        )
            
            epoch_loss /= len(train_dataloader)
            print(f"Epoch {epoch + 1} average loss: {epoch_loss:.4f}")
    
    def evaluate(self, batch_size: int = 32) -> float:
        """Evaluate the model."""
        if self.eval_dataset is None:
            raise ValueError("No evaluation dataset provided")
            
        eval_dataloader = DataLoader(
            self.eval_dataset,
            batch_size=batch_size,
            num_workers=4,
            pin_memory=True
        )
        
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch in tqdm(eval_dataloader, desc="Evaluating"):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask']
                )
                loss = F.binary_cross_entropy(outputs, batch['traits'])
                total_loss += loss.item()
        
        return total_loss / len(eval_dataloader)
    
    def save_checkpoint(self, path: str) -> None:
        """Save training checkpoint."""
        os.makedirs(path, exist_ok=True)
        
        # Save model
        self.model.save_pretrained(path)
        
        # Save optimizer and scheduler states
        torch.save({
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'scaler': self.scaler.state_dict(),
            'global_step': self.global_step,
        }, os.path.join(path, "training_state.pt"))
    
    def load_checkpoint(self, path: str) -> None:
        """Load training checkpoint."""
        # Load model
        self.model = TraitPredictor.from_pretrained(path)
        self.model.to(self.device)
        
        # Load training state
        training_state = torch.load(os.path.join(path, "training_state.pt"))
        self.optimizer.load_state_dict(training_state['optimizer'])
        self.scheduler.load_state_dict(training_state['scheduler'])
        self.scaler.load_state_dict(training_state['scaler'])
        self.global_step = training_state['global_step']
