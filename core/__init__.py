"""Core utilities for acoustic Speaker + Microphone turn-taking in pass-the-mic."""

from .protocol import (
    MicPassTurn,
    MessageType,
    TurnGrant,
    TurnSubmission,
    AgentRegistration,
    parse_agent_turn,
    detect_mic_pass_in_transcript,
    is_mic_passed_to_me,
    TEAM_HOMOPHONES,
)
from .tts import VOICE_REGISTRY, synthesize_speech, synthesize_speech_sync, get_voice_for_team, list_curated_voices
from .audio import (
    play_audio_file,
    play_audio_file_async,
    speak,
    async_speak,
    record_from_mic,
    async_record_from_mic,
    calibrate_microphone,
    list_audio_devices,
    is_currently_speaking,
)
from .stt import STTClient, get_stt_client
from .llm import LLMClient, get_llm_client

__all__ = [
    "MicPassTurn",
    "MessageType",
    "TurnGrant",
    "TurnSubmission",
    "AgentRegistration",
    "parse_agent_turn",
    "detect_mic_pass_in_transcript",
    "is_mic_passed_to_me",
    "TEAM_HOMOPHONES",
    "VOICE_REGISTRY",
    "synthesize_speech",
    "synthesize_speech_sync",
    "get_voice_for_team",
    "list_curated_voices",
    "play_audio_file",
    "play_audio_file_async",
    "speak",
    "async_speak",
    "record_from_mic",
    "async_record_from_mic",
    "calibrate_microphone",
    "list_audio_devices",
    "is_currently_speaking",
    "STTClient",
    "get_stt_client",
    "LLMClient",
    "get_llm_client",
]
