"""Turn-taking protocol and message schemas for pass-the-mic."""

from __future__ import annotations

import json
import re
import time
from enum import Enum
from typing import Any, List, Optional
from pydantic import BaseModel, Field


class MessageType(str, Enum):
    """Protocol message types exchanged between Host and Participants."""
    REGISTER = "register"
    REGISTER_ACK = "register_ack"
    TOPIC_ANNOUNCED = "topic_announced"
    YOUR_TURN = "your_turn"
    TURN_COMPLETE = "turn_complete"
    TURN_BROADCAST = "turn_broadcast"
    MODERATOR_INTERVENTION = "moderator_intervention"
    ROUNDTABLE_FINISHED = "roundtable_finished"
    PING = "ping"
    PONG = "pong"
    STATUS_UPDATE = "status_update"


class MicPassTurn(BaseModel):
    """Core turn data structure returned by each voice agent.
    
    Guarantees clean turn-taking by specifying who spoke, the spoken script,
    and who should receive the mic next.
    """
    speaker: str = Field(..., description="Name of the team speaking (e.g. 'Team Orange')")
    spoken_text: str = Field(..., description="The script spoken out loud by the agent")
    next_agent: str = Field(..., description="Name of the next team to receive the mic")
    timestamp: float = Field(default_factory=time.time, description="Unix timestamp of when the turn occurred")

    def to_json(self) -> str:
        return self.model_dump_json()

    @classmethod
    def from_json(cls, data: str | dict) -> MicPassTurn:
        if isinstance(data, str):
            return cls.model_validate_json(data)
        return cls.model_validate(data)


class AgentRegistration(BaseModel):
    """Registration packet sent by an attendee when connecting to the Host."""
    type: str = Field(default=MessageType.REGISTER.value)
    name: str = Field(..., description="Team name (e.g. 'Team Orange')")
    voice: str = Field(default="en-US-AriaNeural", description="Selected Edge-TTS voice identifier")
    persona: str = Field(default="", description="Brief persona summary or tagline")


class TurnGrant(BaseModel):
    """Payload sent by the Host server to instruct an agent to speak."""
    type: str = Field(default=MessageType.YOUR_TURN.value)
    topic: str = Field(..., description="The current roundtable debate topic")
    history: List[dict] = Field(default_factory=list, description="Transcript of previous turns in the debate")
    available_agents: List[str] = Field(default_factory=list, description="List of eligible peer team names excluding self")
    round_number: int = Field(default=1, description="Current round index")
    time_limit_sec: int = Field(default=30, description="Recommended speaking time ceiling")


class TurnSubmission(BaseModel):
    """Payload sent by an attendee after speaking, passing the mic."""
    type: str = Field(default=MessageType.TURN_COMPLETE.value)
    turn: MicPassTurn


def extract_json_block(text: str) -> Optional[str]:
    """Extract a JSON object from text that may contain markdown or extra commentary."""
    text = text.strip()
    
    # 1. Direct parse attempt
    if text.startswith("{") and text.endswith("}"):
        return text

    # 2. Extract from markdown code blocks ```json ... ``` or ``` ... ```
    code_block_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if code_block_match:
        return code_block_match.group(1).strip()

    # 3. Find outermost curly braces in free text
    brace_match = re.search(r"(\{[\s\S]*\})", text)
    if brace_match:
        return brace_match.group(1).strip()

    return None


def parse_agent_turn(
    raw_response: str,
    current_speaker: str,
    available_agents: List[str],
) -> MicPassTurn:
    """Robust parser that guarantees a valid MicPassTurn even if the LLM hallucinates.
    
    Args:
        raw_response: Raw string output from the LLM.
        current_speaker: The name of the agent currently speaking.
        available_agents: List of connected peer names that can receive the mic.
        
    Returns:
        A validated, well-formed MicPassTurn object.
    """
    json_str = extract_json_block(raw_response)
    data = None
    
    if json_str:
        try:
            data = json.loads(json_str)
        except json.JSONDecodeError:
            # Try minor repairs like fixing trailing commas
            repaired = re.sub(r",\s*([\]}])", r"\1", json_str)
            try:
                data = json.loads(repaired)
            except Exception:
                data = None

    if not isinstance(data, dict):
        # Fallback if LLM output was completely unstructured text
        clean_text = raw_response.strip().replace('"', "'")
        next_candidate = available_agents[0] if available_agents else "Moderator"
        return MicPassTurn(
            speaker=current_speaker,
            spoken_text=clean_text or f"I yield my perspective on this topic. Passing the mic to {next_candidate}.",
            next_agent=next_candidate,
        )

    # Extract fields with safe defaults
    speaker = str(data.get("speaker") or current_speaker).strip()
    spoken_text = str(data.get("spoken_text") or "").strip()
    next_agent = str(data.get("next_agent") or "").strip()

    # Ensure speaker matches current speaker
    if not speaker:
        speaker = current_speaker

    # If spoken_text is empty, construct a sensible default
    if not spoken_text:
        spoken_text = f"That is an essential question. Passing the mic onward to hear more perspectives."

    # Validate next_agent: Cannot pass to oneself!
    if not available_agents:
        next_agent = "Moderator"
    else:
        # Match case-insensitively with available agents
        match = None
        for candidate in available_agents:
            if candidate.lower() == next_agent.lower():
                match = candidate
                break
        
        # If no direct match or passed to self, check if candidate is mentioned in spoken_text
        if not match or next_agent.lower() == current_speaker.lower():
            for candidate in available_agents:
                if candidate.lower() in spoken_text.lower():
                    match = candidate
                    break
                    
        # Fallback to the first available peer
        if not match or match.lower() == current_speaker.lower():
            # Pick first available agent that isn't current speaker
            eligible = [a for a in available_agents if a.lower() != current_speaker.lower()]
            match = eligible[0] if eligible else "Moderator"

        next_agent = match

    return MicPassTurn(
        speaker=speaker,
        spoken_text=spoken_text,
        next_agent=next_agent,
    )
