"""Abstract base classes for all model components."""

from abc import (
    ABC,
    abstractmethod,
)
from typing import (
    Any,
    Dict,
    Optional,
)

import torch
import torch.nn as nn


class AudioEncoder(ABC, nn.Module):
    """Abstract base class for audio encoders."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        self.freeze = config.get("freeze", True)
        self.device = config.get("device", "cpu")
        
    @abstractmethod
    def get_embedding_dim(self) -> int:
        """Return the output embedding dimension."""
        pass
    
    @abstractmethod
    def encode_audio(self, audio: Dict[str, Any]) -> torch.Tensor:
        """
        Encode audio to embeddings.
        
        Args:
            audio: Dictionary with 'array' and 'sampling_rate' keys
            
        Returns:
            Tensor of shape (batch_size, embedding_dim)
        """
        pass
    
    def freeze_weights(self):
        """Freeze all parameters."""
        for param in self.parameters():
            param.requires_grad = False
        self.eval()
    
    def unfreeze_weights(self):
        """Unfreeze all parameters."""
        for param in self.parameters():
            param.requires_grad = True
        self.train()


class LanguageModel(ABC, nn.Module):
    """Abstract base class for language models."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        self.freeze = config.get("freeze", True)
        self.device = config.get("device", "cpu")
        
    @abstractmethod
    def get_embedding_dim(self) -> int:
        """Return the input embedding dimension."""
        pass
    
    @abstractmethod
    def get_tokenizer(self):
        """Return the tokenizer for this language model."""
        pass
    
    @abstractmethod
    def get_text_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Get text embeddings from input token IDs.
        
        Args:
            input_ids: Tensor of shape (batch_size, seq_len)
            
        Returns:
            Tensor of shape (batch_size, seq_len, embedding_dim)
        """
        pass
    
    @abstractmethod
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
            labels: Optional labels tensor
            
        Returns:
            Dictionary with at least 'loss' key if labels provided
        """
        pass
    
    def freeze_weights(self):
        """Freeze all parameters."""
        for param in self.parameters():
            param.requires_grad = False
        self.eval()
    
    def unfreeze_weights(self):
        """Unfreeze all parameters."""
        for param in self.parameters():
            param.requires_grad = True
        self.train()


class Projection(ABC, nn.Module):
    """Abstract base class for projection layers."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        self.device = config.get("device", "cpu")
        
    @abstractmethod
    def forward(self, audio_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Project audio embeddings to language model embedding space.
        
        Args:
            audio_embeddings: Tensor of shape (batch_size, audio_embedding_dim)
            
        Returns:
            Tensor of shape (batch_size, prefix_len * lm_embedding_dim)
        """
        pass
    
    @abstractmethod
    def get_output_shape(self, prefix_len: int, lm_embedding_dim: int) -> tuple:
        """Return the expected output shape (prefix_len, lm_embedding_dim)."""
        pass
