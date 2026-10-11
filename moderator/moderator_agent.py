"""The Host's Moderator Voice Agent ('Team Black').

Introduces topics, enforces rules, intervenes on protocol anomalies,
and synthesizes the debate at the conclusion.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Dict
from core.audio import async_speak, speak
from core.llm import LLMClient, get_llm_client

logger = logging.getLogger("pass_the_mic.moderator")


class ModeratorAgent:
    """The host-side Moderator agent who coordinates the roundtable voice banter."""

    def __init__(
        self,
        name: str = "Moderator",
        voice: str = "en-US-GuyNeural",
        llm_client: Optional[LLMClient] = None,
        speak_aloud: bool = True,
    ):
        self.name = name
        self.voice = voice
        self.llm_client = llm_client or get_llm_client()
        self.speak_aloud = speak_aloud

    async def introduce_topic(
        self,
        topic: str,
        first_speaker: str,
        custom_intro: Optional[str] = None,
    ) -> str:
        """Introduce the debate topic, lay out the rules, and call the first speaker."""
        if custom_intro:
            text = custom_intro
        else:
            text = (
                f"Welcome everyone to Pass-The-Mic! Tonight's debate question is: {topic}. "
                f"The rules are simple: keep responses punchy, speak with conviction, and always pass the mic to an opposing perspective. "
                f"To kick us off, {first_speaker}, what is your stance on this?"
            )

        logger.info(f"[{self.name}] Introducing: {text}")
        if self.speak_aloud:
            await async_speak(text, voice=self.voice)
        return text

    async def intervene(
        self,
        reason: str,
        faulty_speaker: str,
        redirected_speaker: str,
    ) -> str:
        """Intervene when an agent hallucinates, passes to itself, or times out."""
        text = (
            f"Moderator intervention! {faulty_speaker} passed to an invalid or offline participant. "
            f"Stepping in to keep the debate alive — redirecting the microphone to {redirected_speaker}. "
            f"{redirected_speaker}, the floor is yours!"
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
        """Deliver a crisp summary verdict wrapping up the roundtable debate."""
        # Synthesize debate points
        speakers = list({turn.get("speaker") for turn in history if turn.get("speaker")})
        speakers_str = ", ".join(speakers) if speakers else "all teams"

        summary_script = (
            f"And that brings our Pass-The-Mic debate to a close! "
            f"We explored {topic} from multiple angles, with sharp insights from {speakers_str}. "
            f"Both the optimistic breakthroughs and pragmatic risks were passionately defended. "
            f"Tremendous job to all participating agents tonight. That's a wrap for this round!"
        )

        logger.info(f"[{self.name}] Concluding: {summary_script}")
        if self.speak_aloud:
            await async_speak(summary_script, voice=self.voice)
        return summary_script
