"""Zero-cost, reliable neural TTS pipeline using Microsoft Edge Neural Voices.

Requires zero API keys, zero subscriptions, has no rate limits, and delivers
high-fidelity, human-like voice synthesis.
"""

from __future__ import annotations

import asyncio
import os
import re
import tempfile
from pathlib import Path
from typing import Dict, Optional
import edge_tts

# Curated voice registry specified in the architecture, plus fun attendee additions
VOICE_REGISTRY: Dict[str, dict] = {
    # Core Meetup Team Voices
    "Moderator": {
        "voice_id": "en-US-GuyNeural",
        "description": "Authoritative, clear broadcast tone",
        "accent": "US English (Male)",
    },
    "Team Orange": {
        "voice_id": "en-US-AriaNeural",
        "description": "Energetic, upbeat, and dynamic",
        "accent": "US English (Female)",
    },
    "Team Red": {
        "voice_id": "en-GB-RyanNeural",
        "description": "British, analytical, crisp, sharp",
        "accent": "British English (Male)",
    },
    "Team Blue": {
        "voice_id": "en-AU-NatashaNeural",
        "description": "Australian, relaxed, contemplative, articulate",
        "accent": "Australian English (Female)",
    },
    # Additional Curated Attendee Voices
    "Team Emerald": {
        "voice_id": "en-US-ChristopherNeural",
        "description": "Confident, direct, technical",
        "accent": "US English (Male)",
    },
    "Team Purple": {
        "voice_id": "en-GB-SoniaNeural",
        "description": "Expressive, thoughtful, philosophical",
        "accent": "British English (Female)",
    },
    "Team Amber": {
        "voice_id": "en-IN-NeerjaNeural",
        "description": "Warm, engaging, articulate",
        "accent": "Indian English (Female)",
    },
    "Team Silver": {
        "voice_id": "en-US-JennyNeural",
        "description": "Natural, conversational, versatile",
        "accent": "US English (Female)",
    },
    "Team Onyx": {
        "voice_id": "en-CA-LiamNeural",
        "description": "Deep, calm, measured",
        "accent": "Canadian English (Male)",
    },
}

# Quick lookup aliases
DEFAULT_VOICES = {
    "moderator": "en-US-GuyNeural",
    "host": "en-US-GuyNeural",
    "orange": "en-US-AriaNeural",
    "red": "en-GB-RyanNeural",
    "blue": "en-AU-NatashaNeural",
    "emerald": "en-US-ChristopherNeural",
    "purple": "en-GB-SoniaNeural",
    "amber": "en-IN-NeerjaNeural",
    "silver": "en-US-JennyNeural",
}


def sanitize_text_for_speech(text: str) -> str:
    """Clean text before feeding to TTS to ensure crisp, natural voice delivery."""
    # Remove markdown formatting like bold, italics, backticks
    clean = re.sub(r"[*_`#]", "", text)
    # Remove markdown links [text](url) -> text
    clean = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", clean)
    # Collapse multiple whitespaces and linebreaks
    clean = re.sub(r"\s+", " ", clean).strip()
    return clean


def get_voice_for_team(team_name: str, fallback_voice: Optional[str] = None) -> str:
    """Intelligently map a team name or color to a curated edge-tts voice ID.
    
    If team_name is already a valid edge-tts voice identifier (contains 'Neural'),
    it will be returned directly.
    """
    if "Neural" in team_name:
        return team_name

    # Check exact match in registry
    for name, data in VOICE_REGISTRY.items():
        if name.lower() == team_name.lower():
            return data["voice_id"]

    # Check partial color keywords
    lower_name = team_name.lower()
    for key, voice_id in DEFAULT_VOICES.items():
        if key in lower_name:
            return voice_id

    # Fallback default
    return fallback_voice or "en-US-AriaNeural"


def list_curated_voices() -> Dict[str, dict]:
    """Return dictionary of curated voices and descriptions for workshop attendees."""
    return VOICE_REGISTRY


async def synthesize_speech(
    text: str,
    voice: str = "en-US-GuyNeural",
    output_path: Optional[str | Path] = None,
    rate: str = "+0%",
    volume: str = "+0%",
    pitch: str = "+0Hz",
) -> Path:
    """Synthesize text to an MP3 file using Microsoft Edge Neural TTS.
    
    Args:
        text: Text to speak out loud.
        voice: Edge-TTS voice identifier (e.g. 'en-US-GuyNeural').
        output_path: Destination path for the mp3 file. If None, a temporary file is created.
        rate: Speed adjustment (e.g. '+0%', '+10%').
        volume: Volume adjustment (e.g. '+0%').
        pitch: Pitch adjustment (e.g. '+0Hz').
        
    Returns:
        Path to the synthesized audio file.
    """
    clean_text = sanitize_text_for_speech(text)
    if not clean_text:
        clean_text = "Mic check."

    # If voice is a team name/color rather than raw voice ID, resolve it
    resolved_voice = get_voice_for_team(voice, fallback_voice=voice)

    if output_path is None:
        fd, tmp_file = tempfile.mkstemp(suffix=".mp3", prefix="ptm_speech_")
        os.close(fd)
        dest = Path(tmp_file)
    else:
        dest = Path(output_path)
        dest.parent.mkdir(parents=True, exist_ok=True)

    communicate = edge_tts.Communicate(
        text=clean_text,
        voice=resolved_voice,
        rate=rate,
        volume=volume,
        pitch=pitch,
    )
    
    await communicate.save(str(dest))
    return dest


def synthesize_speech_sync(
    text: str,
    voice: str = "en-US-GuyNeural",
    output_path: Optional[str | Path] = None,
    rate: str = "+0%",
) -> Path:
    """Synchronous wrapper for synthesize_speech."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        # In an existing loop, execute in thread pool or create task
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as pool:
            return pool.submit(
                asyncio.run,
                synthesize_speech(text, voice, output_path, rate=rate)
            ).result()
    else:
        return asyncio.run(synthesize_speech(text, voice, output_path, rate=rate))
