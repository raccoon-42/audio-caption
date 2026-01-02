"""Model registry for easy model registration and retrieval."""

import inspect
from typing import (
    Any,
    Callable,
    Dict,
    Type,
)


class ModelRegistry:
    """Registry for model components."""
    
    _audio_encoders: Dict[str, Type] = {}
    _language_models: Dict[str, Type] = {}
    _projections: Dict[str, Type] = {}
    
    @classmethod
    def register_audio_encoder(cls, name: str):
        """Decorator to register an audio encoder."""
        def decorator(model_class):
            cls._audio_encoders[name] = model_class
            return model_class
        return decorator
    
    @classmethod
    def register_language_model(cls, name: str):
        """Decorator to register a language model."""
        def decorator(model_class):
            cls._language_models[name] = model_class
            return model_class
        return decorator
    
    @classmethod
    def register_projection(cls, name: str):
        """Decorator to register a projection layer."""
        def decorator(projection_class):
            cls._projections[name] = projection_class
            return projection_class
        return decorator
    
    @classmethod
    def get_audio_encoder(cls, name: str, config: Dict[str, Any]):
        """Get an audio encoder instance by name."""
        if name not in cls._audio_encoders:
            raise ValueError(f"Audio encoder '{name}' not found. Available: {list(cls._audio_encoders.keys())}")
        return cls._audio_encoders[name](config)
    
    @classmethod
    def get_language_model(cls, name: str, config: Dict[str, Any]):
        """Get a language model instance by name."""
        if name not in cls._language_models:
            raise ValueError(f"Language model '{name}' not found. Available: {list(cls._language_models.keys())}")
        return cls._language_models[name](config)
    
    @classmethod
    def get_projection(cls, name: str, config: Dict[str, Any]):
        """Get a projection layer instance by name."""
        if name not in cls._projections:
            raise ValueError(f"Projection '{name}' not found. Available: {list(cls._projections.keys())}")
        return cls._projections[name](config)
    
    @classmethod
    def list_audio_encoders(cls) -> list:
        """List all registered audio encoders."""
        return list(cls._audio_encoders.keys())
    
    @classmethod
    def list_language_models(cls) -> list:
        """List all registered language models."""
        return list(cls._language_models.keys())
    
    @classmethod
    def list_projections(cls) -> list:
        """List all registered projections."""
        return list(cls._projections.keys())
