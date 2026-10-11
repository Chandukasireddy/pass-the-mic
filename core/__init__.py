"""Core utilities and turn-taking protocol for pass-the-mic."""

from .protocol import MicPassTurn, MessageType, TurnGrant, TurnSubmission, AgentRegistration, parse_agent_turn
from .tts import VOICE_REGISTRY, synthesize_speech, get_voice_for_team, list_curated_voices
from .audio import play_audio_file, speak, async_speak
from .llm import LLMClient, get_llm_client

__all__ = [
    "MicPassTurn",
    "MessageType",
    "TurnGrant",
    "TurnSubmission",
    "AgentRegistration",
    "parse_agent_turn",
    "VOICE_REGISTRY",
    "synthesize_speech",
    "get_voice_for_team",
    "list_curated_voices",
    "play_audio_file",
    "speak",
    "async_speak",
    "LLMClient",
    "get_llm_client",
]
