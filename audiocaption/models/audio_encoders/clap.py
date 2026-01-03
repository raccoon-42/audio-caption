"""CLAP audio encoder implementation."""

from typing import (
    Any,
    Dict,
)

import librosa
import torch
from transformers import (
    ClapModel,
    ClapProcessor,
)

from audiocaption.core.base import AudioEncoder
from audiocaption.core.registry import ModelRegistry


@ModelRegistry.register_audio_encoder("clap")
class CLAPAudioEncoder(AudioEncoder):
    """CLAP (Contrastive Language-Audio Pretraining) audio encoder."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        model_name = config.get("model_name", "laion/clap-htsat-unfused")
        self.processor = ClapProcessor.from_pretrained(model_name)
        self.model = ClapModel.from_pretrained(model_name).to(self.device)
        
        if self.freeze:
            self.freeze_weights()
        else:
            self.unfreeze_weights()
    
    def get_embedding_dim(self) -> int:
        """Return CLAP embedding dimension."""
        return self.model.config.projection_dim
    
    def encode_audio(self, audio: Dict[str, Any]) -> torch.Tensor:
        """
        Encode audio to CLAP embeddings.
        
        Args:
            audio: Dictionary with 'array' and 'sampling_rate' keys
            
        Returns:
            Tensor of shape (1, embedding_dim)
        """
        audio_array = audio["array"]
        sr = audio["sampling_rate"]
        
        # Handle stereo audio
        if audio_array.ndim == 2:
            audio_array = audio_array.mean(axis=1)
        
        # Resample to 48000 Hz if needed (CLAP requirement)
        if sr != 48000:
            audio_array = librosa.resample(
                audio_array, 
                orig_sr=sr, 
                target_sr=48000
            )
        
        # Process with CLAP
        inputs = self.processor(
            audio=audio_array,
            sampling_rate=48000,
            return_tensors="pt"
        ).to(self.device)
        
        with torch.no_grad():
            embeddings = self.model.get_audio_features(**inputs)
        
        return embeddings  # (1, embedding_dim)
