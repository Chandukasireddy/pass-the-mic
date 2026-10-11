"""Configuration, Model Registry, and .env Key Loader for Magnus (Gemini Speech-to-Speech Agent)."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Dict, List, Optional
from pydantic import BaseModel, Field


def load_gemini_api_key_from_env() -> str:
    """Load the Gemini API key from environment or .env files (supports `Gemini-API_KEY` and `GEMINI_API_KEY`)."""
    # Check existing environment variables first
    for env_key in ("GEMINI_API_KEY", "Gemini-API_KEY", "GOOGLE_API_KEY"):
        val = os.environ.get(env_key)
        if val and val.strip() and not val.startswith("your_"):
            return val.strip()

    # Check both local folder .env and parent project .env
    candidates = [
        Path(__file__).resolve().parent / ".env",
        Path(__file__).resolve().parent.parent / ".env",
    ]
    for env_path in candidates:
        if env_path.exists():
            try:
                for line in env_path.read_text(encoding="utf-8").splitlines():
                    clean = line.strip()
                    if not clean or clean.startswith("#") or "=" not in clean:
                        continue
                    k, v = clean.split("=", 1)
                    k_clean = k.strip()
                    v_clean = v.strip().strip('"').strip("'")
                    os.environ[k_clean] = v_clean
                    if k_clean.lower() in ("gemini-api_key", "gemini_api_key", "google_api_key"):
                        if v_clean and not v_clean.startswith("your_"):
                            os.environ["GEMINI_API_KEY"] = v_clean
                            return v_clean
            except Exception:
                pass
    return ""


# Latest Gemini Speech-to-Speech & Native Audio Models
GEMINI_VOICE_MODELS: List[Dict[str, str]] = [
    {
        "id": "gemini-3.8-live",
        "name": "Gemini 3.8 Live (Speech-to-Speech Bidi)",
        "mode": "live_bidi",
        "description": "Latest flagship real-time bidirectional Speech-to-Speech model",
    },
    {
        "id": "gemini-3.8-live-extended-thinking",
        "name": "Gemini 3.8 Live Extended Thinking (Speech-to-Speech)",
        "mode": "live_bidi",
        "description": "Real-time Speech-to-Speech with deep strategic reasoning",
    },
    {
        "id": "gemini-2.5-flash-native-audio-latest",
        "name": "Gemini 2.5 Flash Native Audio Live",
        "mode": "live_bidi",
        "description": "Low-latency native audio stream over Gemini Live API",
    },
    {
        "id": "gemini-3.1-flash-live-preview",
        "name": "Gemini 3.1 Flash Live Preview",
        "mode": "live_bidi",
        "description": "Fast preview multimodal live voice agent model",
    },
    {
        "id": "gemini-3.8-flash",
        "name": "Gemini 3.8 Flash + Gemini 3.8 Flash TTS",
        "mode": "multimodal_s2s",
        "description": "Multimodal audio input understanding paired with Gemini 3.8 Flash TTS",
    },
]

# Gemini Native Prebuilt Voices
GEMINI_VOICES: List[Dict[str, str]] = [
    {
        "id": "Charon",
        "name": "Charon (Deep & Commanding — Magnus Default)",
        "edge_fallback": "en-GB-RyanNeural",
        "description": "Deep, authoritative, grandmaster broadcast presence",
    },
    {
        "id": "Fenrir",
        "name": "Fenrir (Bold & Intense)",
        "edge_fallback": "en-US-ChristopherNeural",
        "description": "Strong, assertive, high-conviction voice",
    },
    {
        "id": "Puck",
        "name": "Puck (Dynamic & Fast-Witted)",
        "edge_fallback": "en-US-GuyNeural",
        "description": "Energetic, quick, conversational tone",
    },
    {
        "id": "Kore",
        "name": "Kore (Calm & Analytical)",
        "edge_fallback": "en-AU-NatashaNeural",
        "description": "Clear, composed, precision-focused voice",
    },
    {
        "id": "Aoede",
        "name": "Aoede (Warm & Expressive)",
        "edge_fallback": "en-US-AriaNeural",
        "description": "Articulate, engaging storytelling voice",
    },
]

PRESET_TOPICS: List[str] = [
    "Will decentralized open-source AI models outpace closed frontier labs by 2027?",
    "Will autonomous AI agents completely replace manual software coding within 5 years?",
    "Should frontier AI training runs above 10^26 FLOPs require international licensing?",
    "Will personal on-device voice agents replace smartphones and apps by 2030?",
    "Which will generate greater economic impact first: humanoid robotics or digital agents?",
]


class MagnusSettings(BaseModel):
    """Live runtime settings for Magnus, controllable from the Web UI."""
    agent_name: str = Field(default="Magnus")
    persona: str = Field(
        default=(
            "You are Magnus, a Grandmaster AI Strategist and Systems Architect participating in a live "
            "voice roundtable. You combine first-principles physics, compute economics, and strategic "
            "foresight. You speak with calm, commanding authority in 2 to 4 punchy sentences."
        )
    )
    style: str = Field(default="Commanding, strategic, razor-sharp, concise (2-4 sentences)")
    gemini_model: str = Field(default="gemini-3.8-live")
    tts_model: str = Field(default="gemini-3.8-flash-tts")
    gemini_voice: str = Field(default="Charon")
    fallback_edge_voice: str = Field(default="en-GB-RyanNeural")

    # Active Listening Modes:
    # - "wake_word": Listens to all room speech, speaks when "Magnus" or a pass cue is heard
    # - "always_on": Responds to every spoken utterance heard on the microphone (1-on-1 live voice chat)
    # - "manual": Only speaks when triggered from the Web UI
    listening_mode: str = Field(default="wake_word")
    wake_phrases: List[str] = Field(
        default_factory=lambda: [
            "magnus",
            "team magnus",
            "pass the mic to magnus",
            "passing the mic to magnus",
            "over to magnus",
            "hey magnus",
        ]
    )
    peer_teams: List[str] = Field(
        default_factory=lambda: ["Team Orange", "Team Red", "Team Blue", "Moderator"]
    )
    pass_mic_at_end: bool = Field(default=True)
    topic: str = Field(
        default="Will decentralized open-source AI models outpace closed frontier labs by 2027?"
    )

    # Microphone & Speaker Hardware Settings
    mic_device_index: Optional[int] = Field(default=None)
    energy_threshold: float = Field(default=0.015)
    silence_duration: float = Field(default=1.6)
    max_record_seconds: float = Field(default=25.0)
    speaker_enabled: bool = Field(default=True)
    speech_rate: str = Field(default="+0%")


# Phonetic / STT homophones for "Magnus" so microphone wake detection is ultra-reliable
MAGNUS_HOMOPHONES = [
    "magnus",
    "magness",
    "magnis",
    "magnes",
    "magnet",
    "magna",
    "team magnus",
]


def check_magnus_wake_phrase(transcript: str, settings: MagnusSettings) -> bool:
    """Return True if the microphone transcript triggers Magnus to speak."""
    if not transcript or not transcript.strip():
        return False

    if settings.listening_mode == "always_on":
        return True

    clean = re.sub(r"[^\w\s]", " ", transcript.lower())
    clean = re.sub(r"\s+", " ", clean).strip()

    # Check user-configured wake phrases
    for phrase in settings.wake_phrases:
        p_clean = phrase.lower().strip()
        if p_clean and p_clean in clean:
            return True

    # Check phonetic homophones for Magnus
    for alias in MAGNUS_HOMOPHONES:
        if re.search(r"\b" + re.escape(alias) + r"\b", clean):
            return True

    return False
