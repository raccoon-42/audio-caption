"""GPT-2 language model implementation."""

from typing import (
    Any,
    Dict,
    Optional,
)

import torch
from transformers import (
    GPT2LMHeadModel,
    GPT2Tokenizer,
)

from audiocaption.core.base import LanguageModel
from audiocaption.core.registry import ModelRegistry


@ModelRegistry.register_language_model("gpt2")
class GPT2LanguageModel(LanguageModel):
    """GPT-2 language model for text generation."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        model_name = config.get("model_name", "gpt2")
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = GPT2LMHeadModel.from_pretrained(model_name).to(self.device)
        
        if self.freeze:
            self.freeze_weights()
        else:
            self.unfreeze_weights()
    
    def get_embedding_dim(self) -> int:
        """Return GPT-2 embedding dimension."""
        return self.model.config.n_embd
    
    def get_tokenizer(self):
        """Return the GPT-2 tokenizer."""
        return self.tokenizer
    
    def get_text_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Get GPT-2 text embeddings from input token IDs."""
        return self.model.transformer.wte(input_ids)
    
    def forward_with_prefix(
        self,
        prefix_embeddings: torch.Tensor,
        text_embeddings: torch.Tensor,
        labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with prefix embeddings.
        
        Args:
            prefix_embeddings: Tensor of shape (batch_size, prefix_len, embedding_dim)
            text_embeddings: Tensor of shape (batch_size, seq_len, embedding_dim)
            labels: Optional labels tensor of shape (batch_size, prefix_len + seq_len)
            
        Returns:
            Dictionary with 'loss' key if labels provided
        """
        # Concatenate prefix + text embeddings
        inputs_embeds = torch.cat([prefix_embeddings, text_embeddings], dim=1)
        
        outputs = self.model(
            inputs_embeds=inputs_embeds,
            labels=labels
        )
        
        return {"loss": outputs.loss} if labels is not None else {}
