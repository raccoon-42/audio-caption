"""Audio encoder implementations."""

from .base import BaseAudioEncoder
from .clap import CLAPAudioEncoder

__all__ = ["BaseAudioEncoder", "CLAPAudioEncoder"]
