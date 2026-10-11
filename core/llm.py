"""Fast LLM client with Groq Llama-3 free tier, OpenAI fallback, and Mock simulation mode.

Designed for acoustic Speaker-to-Microphone roundtables: takes what the laptop's
microphone just heard (`heard_from_mic`), generates a sharp counter-argument,
and ends with a clear spoken wake cue ("Passing the mic to Team Red") so the
next laptop's microphone detects the handoff over the air.
"""

from __future__ import annotations

import logging
import os
import random
from typing import List, Optional
from dotenv import load_dotenv

from .protocol import MicPassTurn, parse_agent_turn

load_dotenv()

logger = logging.getLogger("pass_the_mic.llm")


def _build_system_prompt(speaker_name: str, persona: str, available_agents: List[str]) -> str:
    """Construct a system prompt enforcing persona, acoustic wake phrases, and strict JSON."""
    agents_list_str = ", ".join(f'"{a}"' for a in available_agents) if available_agents else '"Moderator"'

    return f"""You are {speaker_name}, an autonomous voice agent running on a laptop in a live, in-person roundtable debate.
Every laptop in this room speaks out loud through its physical speaker and listens through its physical microphone (Speech-to-Text).
Your persona/perspective: {persona or "Knowledgeable, articulate tech specialist"}.

CRITICAL ACOUSTIC ROUNDTABLE RULES:
1. Spoken length: 2 to 4 punchy, conversational sentences. Speak directly to the room. Avoid markdown or bullet points.
2. Direct engagement: Directly address or rebut what your microphone just heard from the previous speaker.
3. The Acoustic Wake Phrase: Because the next laptop is listening via its physical microphone, you MUST end your `spoken_text` with the exact phrase: "Passing the mic to [Next Agent]!" where [Next Agent] is chosen from [{agents_list_str}].
4. Output Format: Reply with ONLY a single valid JSON object matching this exact schema:

{{
  "speaker": "{speaker_name}",
  "spoken_text": "<2-4 conversational sentences ending with: Passing the mic to [Next Agent]!>",
  "next_agent": "<Exactly one name chosen from: [{agents_list_str}]>"
}}

DO NOT return any text outside of the JSON block. Do not pass the mic to yourself."""


def _mock_generate_turn(
    speaker_name: str,
    persona: str,
    topic: str,
    history: List[dict],
    available_agents: List[str],
    heard_from_mic: str = "",
) -> MicPassTurn:
    """Intelligent simulation engine when running offline without API keys."""
    next_agent = random.choice(available_agents) if available_agents else "Moderator"

    context_source = f"{topic} {heard_from_mic}".lower()

    if "open source" in context_source or "weights" in context_source or "closed" in context_source:
        if "optimist" in persona.lower() or "orange" in speaker_name.lower():
            speech = (
                f"I heard the question loud and clear on my microphone! Decentralized open weights are democratizing intelligence at breakneck speed. "
                f"When millions of developers fine-tune models locally, closed walled gardens simply cannot keep pace. "
                f"Passing the mic to {next_agent}!"
            )
        elif "skeptic" in persona.lower() or "red" in speaker_name.lower():
            speech = (
                f"Listening to that optimism on my mic, I have to push back on the compute reality. "
                f"Frontier training runs require massive hyperscaler clusters and energy grids that open-source groups simply don't have. "
                f"Passing the mic to {next_agent}!"
            )
        else:
            speech = (
                f"Hearing both sides in the room, the deeper question is cognitive sovereignty. "
                f"Whether intelligence is controlled by three corporations or shared as a public commons will shape human freedom. "
                f"Passing the mic to {next_agent}!"
            )
    elif "coding" in context_source or "engineer" in context_source or "code" in context_source:
        if "optimist" in persona.lower() or "orange" in speaker_name.lower():
            speech = (
                f"We are witnessing software engineering evolve into high-level system orchestration. "
                f"Autonomous agents amplify human creativity, turning every developer into an architect directing a fleet of builders. "
                f"Passing the mic to {next_agent}!"
            )
        elif "skeptic" in persona.lower() or "red" in speaker_name.lower():
            speech = (
                f"Generating boilerplate is easy, but debugging distributed race conditions in production is where autonomous agents fail. "
                f"Without deterministic verification, human engineers will spend twice as long auditing AI-generated bugs. "
                f"Passing the mic to {next_agent}!"
            )
        else:
            speech = (
                f"As we delegate code synthesis to machines, we must preserve human algorithmic reasoning. "
                f"Software architecture is fundamentally an expression of human intent and ethics. "
                f"Passing the mic to {next_agent}!"
            )
    else:
        if "optimist" in persona.lower() or "orange" in speaker_name.lower():
            speech = (
                f"Hearing the discussion in the room, the exponential trajectory of these breakthroughs is undeniable. "
                f"Embracing open innovation will unlock extraordinary capabilities across science and engineering. "
                f"Passing the mic to {next_agent}!"
            )
        elif "skeptic" in persona.lower() or "red" in speaker_name.lower():
            speech = (
                f"We need to separate exciting demos from production engineering reality. "
                f"Until we solve reliability, latency bottlenecks, and unit economics, skepticism is mandatory. "
                f"Passing the mic to {next_agent}!"
            )
        else:
            speech = (
                f"This debate goes right to the heart of human agency and long-term governance. "
                f"We must balance rapid technological capability with thoughtful societal alignment. "
                f"Passing the mic to {next_agent}!"
            )

    return MicPassTurn(
        speaker=speaker_name,
        spoken_text=speech,
        next_agent=next_agent,
        heard_from_mic=heard_from_mic,
    )


class LLMClient:
    """Unified LLM client supporting Groq (primary), OpenAI (fallback), and Mock simulation."""

    def __init__(
        self,
        provider: Optional[str] = None,
        groq_api_key: Optional[str] = None,
        openai_api_key: Optional[str] = None,
        groq_model: Optional[str] = None,
        openai_model: Optional[str] = None,
    ):
        self.groq_api_key = groq_api_key or os.getenv("GROQ_API_KEY")
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        self.groq_model = groq_model or os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
        self.openai_model = openai_model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")

        self._groq_client = None
        self._openai_client = None

        if provider:
            self.provider = provider.lower()
        elif self.groq_api_key and self.groq_api_key.strip() and not self.groq_api_key.startswith("your_"):
            self.provider = "groq"
        elif self.openai_api_key and self.openai_api_key.strip() and not self.openai_api_key.startswith("your_"):
            self.provider = "openai"
        else:
            self.provider = "mock"

        self._init_clients()

    def _init_clients(self) -> None:
        if self.provider == "groq" and self.groq_api_key:
            try:
                from groq import Groq
                self._groq_client = Groq(api_key=self.groq_api_key)
            except Exception as e:
                logger.warning(f"Failed to initialize Groq client: {e}. Falling back to mock mode.")
                self.provider = "mock"

        elif self.provider == "openai" and self.openai_api_key:
            try:
                from openai import OpenAI
                self._openai_client = OpenAI(api_key=self.openai_api_key)
            except Exception as e:
                logger.warning(f"Failed to initialize OpenAI client: {e}. Falling back to mock mode.")
                self.provider = "mock"

    def generate_mic_turn(
        self,
        speaker_name: str,
        persona: str,
        topic: str,
        history: List[dict],
        available_agents: List[str],
        heard_from_mic: str = "",
    ) -> MicPassTurn:
        """Synchronously generate an agent's speech based on what its microphone heard."""
        if not heard_from_mic and history:
            heard_from_mic = history[-1].get("spoken_text", "")

        if self.provider == "mock":
            return _mock_generate_turn(
                speaker_name, persona, topic, history, available_agents, heard_from_mic=heard_from_mic
            )

        system_prompt = _build_system_prompt(speaker_name, persona, available_agents)

        history_summary = []
        for turn in history[-5:]:
            spk = turn.get("speaker", "Speaker")
            txt = turn.get("spoken_text", "")
            history_summary.append(f"{spk}: \"{txt}\"")

        history_block = "\n".join(history_summary) if history_summary else "Debate has just begun."
        mic_block = f"WHAT YOUR LAPTOP MICROPHONE JUST HEARD IN THE ROOM:\n\"{heard_from_mic}\"\n\n" if heard_from_mic else ""

        user_prompt = (
            f"DEBATE TOPIC: {topic}\n\n"
            f"{mic_block}"
            f"RECENT ROOM TRANSCRIPT:\n{history_block}\n\n"
            f"AVAILABLE PEERS TO PASS THE MIC TO: {available_agents}\n\n"
            f"Deliver your spoken response as {speaker_name} and end by passing the mic out loud."
        )

        try:
            if self.provider == "groq" and self._groq_client:
                response = self._groq_client.chat.completions.create(
                    model=self.groq_model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.7,
                    max_tokens=300,
                )
                raw_text = response.choices[0].message.content
                return parse_agent_turn(raw_text, speaker_name, available_agents, heard_from_mic=heard_from_mic)

            elif self.provider == "openai" and self._openai_client:
                response = self._openai_client.chat.completions.create(
                    model=self.openai_model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.7,
                    max_tokens=300,
                )
                raw_text = response.choices[0].message.content
                return parse_agent_turn(raw_text, speaker_name, available_agents, heard_from_mic=heard_from_mic)

        except Exception as e:
            logger.warning(f"Error during {self.provider} completion: {e}. Using simulated fallback turn.")
            return _mock_generate_turn(
                speaker_name, persona, topic, history, available_agents, heard_from_mic=heard_from_mic
            )

        return _mock_generate_turn(
            speaker_name, persona, topic, history, available_agents, heard_from_mic=heard_from_mic
        )

    async def async_generate_mic_turn(
        self,
        speaker_name: str,
        persona: str,
        topic: str,
        history: List[dict],
        available_agents: List[str],
        heard_from_mic: str = "",
    ) -> MicPassTurn:
        """Asynchronously generate turn without blocking the event loop."""
        import asyncio
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.generate_mic_turn(
                speaker_name=speaker_name,
                persona=persona,
                topic=topic,
                history=history,
                available_agents=available_agents,
                heard_from_mic=heard_from_mic,
            ),
        )


def get_llm_client() -> LLMClient:
    """Convenience factory to get configured LLM client."""
    return LLMClient()
