"""Sequential projection layer implementation."""

from typing import (
    Any,
    Dict,
)

import torch
import torch.nn as nn

from audiocaption.core.base import Projection
from audiocaption.core.registry import ModelRegistry


@ModelRegistry.register_projection("sequential")
class SequentialProjection(Projection):
    """Sequential projection layer."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        audio_dim = config["audio_embedding_dim"]
        lm_dim = config["language_model_embedding_dim"]
        prefix_len = config["prefix_len"]
        
        self.network = nn.Sequential(
            nn.Linear(audio_dim, lm_dim * 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(lm_dim * 2, lm_dim * 4),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(lm_dim * 4, prefix_len * lm_dim),
        ).to(self.device)
    
    def forward(self, audio_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Project audio embeddings to language model space.
        
        Args:
            audio_embeddings: Tensor of shape (batch_size, audio_embedding_dim)
            
        Returns:
            Tensor of shape (batch_size, prefix_len * lm_embedding_dim)
        """
        return self.network(audio_embeddings)
    
    def get_output_shape(self, prefix_len: int, lm_embedding_dim: int) -> tuple:
        """Return the expected output shape."""
        return (prefix_len, lm_embedding_dim)
