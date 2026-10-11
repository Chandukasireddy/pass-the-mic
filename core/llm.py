"""Fast LLM client with Groq Llama-3 free tier, OpenAI fallback, and Mock simulation mode."""

from __future__ import annotations

import json
import logging
import os
import random
from typing import List, Optional, Tuple
from dotenv import load_dotenv

from .protocol import MicPassTurn, parse_agent_turn

# Load environment variables from .env if present
load_dotenv()

logger = logging.getLogger("pass_the_mic.llm")


def _build_system_prompt(speaker_name: str, persona: str, available_agents: List[str]) -> str:
    """Construct a high-performance system prompt enforcing persona and strict JSON schema."""
    agents_list_str = ", ".join(f'"{a}"' for a in available_agents) if available_agents else '"Moderator"'
    
    return f"""You are {speaker_name}, an autonomous voice agent participating in an in-person, live multi-agent roundtable debate.
Your persona/perspective: {persona or "Knowledgeable, articulate tech specialist"}.

CRITICAL ROUNDTABLE RULES:
1. Spoken length: 2 to 4 punchy, conversational sentences. Speak directly to the room. Avoid robotic bullet points or markdown.
2. Tone: Embody your persona naturally (e.g. optimistic, skeptical, philosophical).
3. The Mic Pass: You MUST conclude your speech by explicitly passing the microphone to one of the available peers: [{agents_list_str}].
4. Output Format: You MUST reply with ONLY a single valid JSON object, matching this exact schema:

{{
  "speaker": "{speaker_name}",
  "spoken_text": "<2-4 conversational sentences ending with: Passing the mic to [Next Agent]...>",
  "next_agent": "<Exactly one name chosen from: [{agents_list_str}]>"
}}

DO NOT return any text outside of the JSON block. Do not pass the mic to yourself."""


def _mock_generate_turn(
    speaker_name: str,
    persona: str,
    topic: str,
    history: List[dict],
    available_agents: List[str],
) -> MicPassTurn:
    """Intelligent simulation engine when running offline without API keys.
    
    Produces realistic, context-aware tech debate points matching the persona.
    """
    next_agent = random.choice(available_agents) if available_agents else "Moderator"
    
    # Analyze topic keywords for contextual responses
    t_lower = topic.lower()
    
    if "open source" in t_lower or "weights" in t_lower:
        if "optimist" in persona.lower() or "orange" in speaker_name.lower():
            speech = (
                f"Decentralized weights are democratizing intelligence at breakneck speed. "
                f"When millions of developers can fine-tune and inspect models locally, closed walled gardens simply cannot keep pace with global innovation. "
                f"I'm passing the mic to {next_agent} — how do you evaluate that competitive pressure?"
            )
        elif "skeptic" in persona.lower() or "red" in speaker_name.lower():
            speech = (
                f"Open weights are fantastic in theory, but they completely ignore the trillion-dollar compute bottleneck. "
                f"Frontier training runs require hyperscaler infrastructure and specialized interconnects that open-source clusters simply don't possess today. "
                f"Passing the mic over to {next_agent} — what is your take on the economics?"
            )
        else:
            speech = (
                f"The core dilemma isn't just about market speed; it's about civilizational sovereignty. "
                f"Whether intelligence is concentrated in three tech conglomerates or distributed across the globe will define our political landscape. "
                f"Passing the mic to {next_agent} to hear their perspective on this balance."
            )
    elif "coding" in t_lower or "engineer" in t_lower:
        if "optimist" in persona.lower() or "orange" in speaker_name.lower():
            speech = (
                f"We are witnessing software engineering evolve into architectural direction. "
                f"Autonomous agents don't replace engineers; they amplify human intent, turning every programmer into the CTO of a team of virtual agents. "
                f"Passing the mic to {next_agent} — are you ready for 100x productivity?"
            )
        elif "skeptic" in persona.lower() or "red" in speaker_name.lower():
            speech = (
                f"Writing boilerplate is easy, but maintenance and high-reliability distributed systems are where the real work happens. "
                f"If an agent introduces subtle memory leaks or security exploits, humans will spend double the time auditing unverified code. "
                f"I am passing the mic to {next_agent} — do you trust automated PRs in production?"
            )
        else:
            speech = (
                f"As we delegate code creation to machines, we must ask what happens to human deep-thinking skills. "
                f"Code has historically been an exercise in rigorous logic; let's ensure we don't atrophy human mastery as agents take over. "
                f"Passing the mic to {next_agent} for their thoughts."
            )
    else:
        # General tech/AI fallback
        if "optimist" in persona.lower():
            speech = (
                f"When we look at the exponential trajectory over the past two years, the pace of breakthroughs is breathtaking. "
                f"Addressing this challenge will unlock massive human capability across science, medicine, and engineering. "
                f"I'm enthusiastically passing the mic to {next_agent} to hear their stance."
            )
        elif "skeptic" in persona.lower():
            speech = (
                f"We need to distinguish between marketing hype and tangible real-world deployment. "
                f"Until we solve deterministic reliability, latency constraints, and real energy costs, widespread scale remains constrained. "
                f"Passing the mic to {next_agent} to challenge my thesis."
            )
        else:
            speech = (
                f"This debate cuts straight to foundational questions about human autonomy and governance. "
                f"We must look beyond short-term milestones and carefully consider the long-term systemic equilibrium. "
                f"I am passing the mic to {next_agent} to hear their perspective."
            )

    return MicPassTurn(
        speaker=speaker_name,
        spoken_text=speech,
        next_agent=next_agent,
    )


class LLMClient:
    """Unified LLM client supporting Groq (primary), OpenAI (fallback), Ollama, and Mock simulation."""

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

        # Determine active provider
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
    ) -> MicPassTurn:
        """Synchronously generate an agent's speech and next_agent choice."""
        if self.provider == "mock":
            return _mock_generate_turn(speaker_name, persona, topic, history, available_agents)

        system_prompt = _build_system_prompt(speaker_name, persona, available_agents)
        
        # Format user prompt with topic and debate history
        history_summary = []
        for turn in history[-5:]: # Keep last 5 turns for recency
            spk = turn.get("speaker", "Unknown")
            txt = turn.get("spoken_text", "")
            history_summary.append(f"{spk}: \"{txt}\"")

        history_block = "\n".join(history_summary) if history_summary else "Debate has just begun."
        user_prompt = (
            f"DEBATE TOPIC: {topic}\n\n"
            f"RECENT ROUNDTABLE TRANSCRIPT:\n{history_block}\n\n"
            f"AVAILABLE PEERS TO PASS THE MIC TO: {available_agents}\n\n"
            f"Deliver your response as {speaker_name} and pass the mic."
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
                return parse_agent_turn(raw_text, speaker_name, available_agents)

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
                return parse_agent_turn(raw_text, speaker_name, available_agents)

        except Exception as e:
            logger.warning(f"Error during {self.provider} completion: {e}. Using simulated fallback turn.")
            return _mock_generate_turn(speaker_name, persona, topic, history, available_agents)

        return _mock_generate_turn(speaker_name, persona, topic, history, available_agents)

    async def async_generate_mic_turn(
        self,
        speaker_name: str,
        persona: str,
        topic: str,
        history: List[dict],
        available_agents: List[str],
    ) -> MicPassTurn:
        """Asynchronously generate turn without blocking the event loop."""
        # For simplicity and thread safety, wrap sync call or run directly
        import asyncio
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            self.generate_mic_turn,
            speaker_name,
            persona,
            topic,
            history,
            available_agents,
        )


def get_llm_client() -> LLMClient:
    """Convenience factory to get configured LLM client."""
    return LLMClient()
