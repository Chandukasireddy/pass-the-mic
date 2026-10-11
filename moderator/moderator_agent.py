"""The Host's Moderator Voice Agent ('Team Black') — Speaker + Microphone Edition.

1. Introduces the debate topic out loud via the Host laptop's SPEAKER
2. Listens to the room debate via the Host laptop's MICROPHONE (STT)
3. Intervenes out loud if the room goes silent or a mic pass is missed
4. Summarizes and concludes the roundtable out loud via the Host's SPEAKER
"""

from __future__ import annotations

import logging
from typing import Callable, List, Optional, Tuple

from core.audio import async_speak, calibrate_microphone
from core.llm import LLMClient, get_llm_client
from core.protocol import detect_mic_pass_in_transcript
from core.stt import STTClient, get_stt_client

logger = logging.getLogger("pass_the_mic.moderator")


class ModeratorAgent:
    """The host-side Moderator agent with Speaker output and Microphone input."""

    def __init__(
        self,
        name: str = "Moderator",
        voice: str = "en-US-GuyNeural",
        llm_client: Optional[LLMClient] = None,
        stt_client: Optional[STTClient] = None,
        speak_aloud: bool = True,
        energy_threshold: float = 0.015,
    ):
        self.name = name
        self.voice = voice
        self.llm_client = llm_client or get_llm_client()
        self.stt_client = stt_client or get_stt_client()
        self.speak_aloud = speak_aloud
        self.energy_threshold = energy_threshold

    def calibrate_mic(self, duration_sec: float = 1.0) -> float:
        """Calibrate host microphone threshold to ambient room noise."""
        self.energy_threshold = calibrate_microphone(duration_sec=duration_sec)
        return self.energy_threshold

    async def introduce_topic(
        self,
        topic: str,
        first_speaker: str,
        custom_intro: Optional[str] = None,
    ) -> str:
        """Introduce the debate topic out loud and acoustically pass the mic to the first speaker."""
        if custom_intro:
            text = custom_intro
            if first_speaker.lower() not in text.lower():
                text = f"{text.rstrip('.!?')}. Passing the mic to {first_speaker}!"
        else:
            text = (
                f"Welcome everyone to Pass-The-Mic! Tonight's debate question is: {topic} "
                f"Every laptop is listening through its microphone and speaking through its speaker. "
                f"Keep your points sharp and always pass the mic out loud. "
                f"To kick us off, I am passing the mic to {first_speaker}!"
            )

        logger.info(f"[{self.name}] Speaking intro: {text}")
        if self.speak_aloud:
            await async_speak(text, voice=self.voice)
        return text

    async def async_listen_to_room(
        self,
        silence_duration: float = 1.8,
        max_duration: float = 25.0,
        wait_timeout: float = 25.0,
        on_speech_start: Optional[Callable[[], None]] = None,
    ) -> Tuple[str, float]:
        """Listen to the room via the Host laptop's microphone and transcribe what was spoken."""
        return await self.stt_client.async_listen_and_transcribe(
            energy_threshold=self.energy_threshold,
            silence_duration=silence_duration,
            max_duration=max_duration,
            wait_timeout=wait_timeout,
            on_speech_start=on_speech_start,
        )

    def detect_next_speaker(
        self,
        transcript: str,
        participating_teams: List[str],
        current_speaker: Optional[str] = None,
    ) -> Optional[str]:
        """Detect which team received the mic in the transcribed room audio."""
        return detect_mic_pass_in_transcript(
            transcript,
            known_teams=participating_teams,
            current_speaker=current_speaker,
        )

    async def intervene(
        self,
        reason: str,
        faulty_speaker: str,
        redirected_speaker: str,
    ) -> str:
        """Speak out loud to re-trigger the next laptop's microphone if a handoff was missed."""
        text = (
            f"Moderator intervention! {faulty_speaker} didn't complete a clear acoustic handoff. "
            f"Stepping in to keep the roundtable moving — I am passing the mic to {redirected_speaker}!"
        )

        logger.info(f"[{self.name}] Intervening: {text}")
        if self.speak_aloud:
            await async_speak(text, voice=self.voice)
        return text

    async def summarize_and_conclude(
        self,
        topic: str,
        history: List[dict],
    ) -> str:
        """Deliver a crisp summary verdict out loud wrapping up the roundtable debate."""
        speakers = list({turn.get("speaker") for turn in history if turn.get("speaker")})
        speakers_str = ", ".join(speakers) if speakers else "all participating teams"

        summary_script = (
            f"And that brings our Pass-The-Mic acoustic roundtable to a close! "
            f"We explored {topic} with live voice arguments from {speakers_str}. "
            f"Both optimistic acceleration and pragmatic engineering realism were passionately defended over the air. "
            f"Fantastic job to everyone in the room tonight!"
        )

        logger.info(f"[{self.name}] Concluding: {summary_script}")
        if self.speak_aloud:
            await async_speak(summary_script, voice=self.voice)
        return summary_script
