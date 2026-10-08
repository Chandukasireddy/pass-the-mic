"""Character Agent Builder for Meetup Attendees.

Provides a clean, modular class to configure a personalized voice agent,
generate context-aware debate turns, and speak out loud on the attendee's laptop.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

from core.audio import async_speak, speak
from core.llm import LLMClient, get_llm_client
from core.protocol import MicPassTurn
from core.tts import get_voice_for_team


class CharacterAgent:
    """Modular attendee voice agent class."""

    def __init__(
        self,
        name: str = "Team Orange",
        voice: str = "en-US-AriaNeural",
        persona: str = "Tech Optimist",
        style: str = "Energetic, visionary, concise",
        test_phrase: Optional[str] = None,
        llm_client: Optional[LLMClient] = None,
        speech_rate: str = "+0%",
    ):
        self.name = name
        self.voice = get_voice_for_team(name, fallback_voice=voice)
        self.persona = persona
        self.style = style
        self.test_phrase = test_phrase or f"Hello everyone! This is {self.name}, ready for the debate."
        self.speech_rate = speech_rate
        self.llm_client = llm_client or get_llm_client()

    @classmethod
    def from_preset(cls, preset_path: str | Path, **overrides) -> CharacterAgent:
        """Instantiate an agent directly from a preset JSON file."""
        path = Path(preset_path)
        if not path.exists():
            # Check inside character_agents/presets/
            alt_path = Path(__file__).parent / "presets" / path.name
            if alt_path.exists():
                path = alt_path
            else:
                raise FileNotFoundError(f"Preset file not found: {preset_path}")

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        data.update(overrides)
        return cls(
            name=data.get("name", "Team Orange"),
            voice=data.get("voice", "en-US-AriaNeural"),
            persona=data.get("persona", ""),
            style=data.get("style", ""),
            test_phrase=data.get("test_phrase"),
            speech_rate=data.get("speech_rate", "+0%"),
        )

    def generate_turn(
        self,
        topic: str,
        history: List[dict],
        available_agents: List[str],
    ) -> MicPassTurn:
        """Synchronously generate response turn using LLM."""
        full_persona = f"{self.persona}. Style: {self.style}"
        return self.llm_client.generate_mic_turn(
            speaker_name=self.name,
            persona=full_persona,
            topic=topic,
            history=history,
            available_agents=available_agents,
        )

    async def async_generate_turn(
        self,
        topic: str,
        history: List[dict],
        available_agents: List[str],
    ) -> MicPassTurn:
        """Asynchronously generate response turn using LLM without stalling loop."""
        full_persona = f"{self.persona}. Style: {self.style}"
        return await self.llm_client.async_generate_mic_turn(
            speaker_name=self.name,
            persona=full_persona,
            topic=topic,
            history=history,
            available_agents=available_agents,
        )

    def speak_turn(self, turn: MicPassTurn, block: bool = True) -> None:
        """Speak the turn out loud through the laptop's speakers (blocking)."""
        speak(turn.spoken_text, voice=self.voice, rate=self.speech_rate, block=block)

    async def async_speak_turn(self, turn: MicPassTurn) -> None:
        """Speak the turn out loud asynchronously while keeping event loop responsive."""
        await async_speak(turn.spoken_text, voice=self.voice, rate=self.speech_rate)

    async def test_voice(self, custom_text: Optional[str] = None) -> None:
        """Play test speech to check audio levels and pronunciation."""
        text = custom_text or self.test_phrase
        await async_speak(text, voice=self.voice, rate=self.speech_rate)
