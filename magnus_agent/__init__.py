"""Magnus — Gemini 3.8 Live Speech-to-Speech Voice Agent Package."""

from .magnus_config import MagnusSettings, GEMINI_VOICE_MODELS, GEMINI_VOICES, load_gemini_api_key_from_env
from .gemini_live_engine import GeminiSpeechToSpeechEngine, SpeechToSpeechResult
from .active_listener import MagnusActiveController

__all__ = [
    "MagnusSettings",
    "GEMINI_VOICE_MODELS",
    "GEMINI_VOICES",
    "load_gemini_api_key_from_env",
    "GeminiSpeechToSpeechEngine",
    "SpeechToSpeechResult",
    "MagnusActiveController",
]
