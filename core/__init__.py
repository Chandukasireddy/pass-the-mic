"""Core utilities and turn-taking protocol for pass-the-mic."""

from importlib import import_module

# Keep legacy exports available without importing optional TTS/LLM dependencies
# when the standalone live-audio client imports a core submodule.
_MODULE_EXPORTS = {
    "protocol": {"MicPassTurn", "MessageType", "TurnGrant", "TurnSubmission", "AgentRegistration", "parse_agent_turn"},
    "tts": {"VOICE_REGISTRY", "synthesize_speech", "get_voice_for_team", "list_curated_voices"},
    "audio": {"play_audio_file", "speak", "async_speak"},
    "llm": {"LLMClient", "get_llm_client"},
}


def __getattr__(name):
    for module, names in _MODULE_EXPORTS.items():
        if name in names:
            value = getattr(import_module(f".{module}", __name__), name)
            globals()[name] = value
            return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

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
