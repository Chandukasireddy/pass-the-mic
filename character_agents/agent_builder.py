"""Character Agent Builder for Meetup Attendees (Speaker + Microphone Edition).

Provides a clean, modular class to configure a personalized voice agent that:
1. LISTENS through the laptop's physical microphone (via VAD + Speech-to-Text)
2. DETECTS when its team name is called over the air ("Passing the mic to Team Orange")
3. THINKS of a contextual response using its persona and LLM
4. SPEAKS out loud through the laptop's physical speakers (via Edge-TTS + pygame)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, List, Optional, Tuple

from core.audio import async_speak, calibrate_microphone, speak
from core.llm import LLMClient, get_llm_client
from core.protocol import MicPassTurn, detect_mic_pass_in_transcript, is_mic_passed_to_me
from core.stt import STTClient, get_stt_client
from core.tts import get_voice_for_team


class CharacterAgent:
    """Modular attendee voice agent with physical Speaker output and Microphone input."""

    def __init__(
        self,
        name: str = "Team Orange",
        voice: str = "en-US-AriaNeural",
        persona: str = "Tech Optimist",
        style: str = "Energetic, visionary, concise",
        test_phrase: Optional[str] = None,
        llm_client: Optional[LLMClient] = None,
        stt_client: Optional[STTClient] = None,
        speech_rate: str = "+0%",
        energy_threshold: float = 0.015,
    ):
        self.name = name
        self.voice = get_voice_for_team(name, fallback_voice=voice)
        self.persona = persona
        self.style = style
        self.test_phrase = (
            test_phrase
            or f"Hello everyone! This is {self.name}. My speaker and microphone are live for the roundtable."
        )
        self.speech_rate = speech_rate
        self.energy_threshold = energy_threshold
        self.llm_client = llm_client or get_llm_client()
        self.stt_client = stt_client or get_stt_client()

    @classmethod
    def from_preset(cls, preset_path: str | Path, **overrides) -> CharacterAgent:
        """Instantiate an agent directly from a preset JSON file."""
        path = Path(preset_path)
        if not path.exists():
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
            llm_client=data.get("llm_client"),
            stt_client=data.get("stt_client"),
        )

    def calibrate_mic(self, duration_sec: float = 1.0) -> float:
        """Calibrate microphone energy threshold to ambient room noise."""
        self.energy_threshold = calibrate_microphone(duration_sec=duration_sec)
        return self.energy_threshold

    def listen_on_mic(
        self,
        silence_duration: float = 1.8,
        max_duration: float = 25.0,
        wait_timeout: float = 30.0,
        fixed_duration: Optional[float] = None,
        on_speech_start: Optional[Callable[[], None]] = None,
    ) -> Tuple[str, float]:
        """Listen on the laptop's microphone and return (transcribed_text, duration_sec)."""
        return self.stt_client.listen_and_transcribe(
            energy_threshold=self.energy_threshold,
            silence_duration=silence_duration,
            max_duration=max_duration,
            wait_timeout=wait_timeout,
            fixed_duration=fixed_duration,
            on_speech_start=on_speech_start,
        )

    async def async_listen_on_mic(
        self,
        silence_duration: float = 1.8,
        max_duration: float = 25.0,
        wait_timeout: float = 30.0,
        fixed_duration: Optional[float] = None,
        on_speech_start: Optional[Callable[[], None]] = None,
    ) -> Tuple[str, float]:
        """Asynchronously listen on the laptop's microphone and transcribe speech."""
        return await self.stt_client.async_listen_and_transcribe(
            energy_threshold=self.energy_threshold,
            silence_duration=silence_duration,
            max_duration=max_duration,
            wait_timeout=wait_timeout,
            fixed_duration=fixed_duration,
            on_speech_start=on_speech_start,
        )

    def check_wake_trigger(self, transcript: str, known_teams: Optional[List[str]] = None) -> bool:
        """Return True if the transcribed room speech passes the mic to this agent."""
        return is_mic_passed_to_me(transcript, self.name, known_teams=known_teams)

    def detect_recipient(self, transcript: str, known_teams: Optional[List[str]] = None) -> Optional[str]:
        """Return the team name that the mic was passed to in the transcript, if any."""
        return detect_mic_pass_in_transcript(transcript, known_teams=known_teams)

    def generate_turn(
        self,
        topic: str,
        history: List[dict],
        available_agents: List[str],
        heard_from_mic: str = "",
    ) -> MicPassTurn:
        """Synchronously generate response turn using LLM."""
        full_persona = f"{self.persona}. Style: {self.style}"
        return self.llm_client.generate_mic_turn(
            speaker_name=self.name,
            persona=full_persona,
            topic=topic,
            history=history,
            available_agents=available_agents,
            heard_from_mic=heard_from_mic,
        )

    async def async_generate_turn(
        self,
        topic: str,
        history: List[dict],
        available_agents: List[str],
        heard_from_mic: str = "",
    ) -> MicPassTurn:
        """Asynchronously generate response turn using LLM without stalling loop."""
        full_persona = f"{self.persona}. Style: {self.style}"
        return await self.llm_client.async_generate_mic_turn(
            speaker_name=self.name,
            persona=full_persona,
            topic=topic,
            history=history,
            available_agents=available_agents,
            heard_from_mic=heard_from_mic,
        )

    def speak_turn(self, turn: MicPassTurn, block: bool = True) -> None:
        """Speak the turn out loud through the laptop's speakers (blocking)."""
        speak(turn.spoken_text, voice=self.voice, rate=self.speech_rate, block=block)

    async def async_speak_turn(self, turn: MicPassTurn) -> None:
        """Speak the turn out loud asynchronously while keeping event loop responsive."""
        await async_speak(turn.spoken_text, voice=self.voice, rate=self.speech_rate)

    async def test_voice(self, custom_text: Optional[str] = None) -> None:
        """Play test speech out of laptop speakers to check volume and pronunciation."""
        text = custom_text or self.test_phrase
        await async_speak(text, voice=self.voice, rate=self.speech_rate)
