"""Model implementations."""

# Import all models to register them
from .audio_encoders.clap import CLAPAudioEncoder
from .language_models.gpt2 import GPT2LanguageModel
from .language_models.t5 import T5LanguageModel
from .projections.sequential import SequentialProjection

__all__ = [
    "CLAPAudioEncoder",
    "GPT2LanguageModel",
    "T5LanguageModel",
    "SequentialProjection",
]
