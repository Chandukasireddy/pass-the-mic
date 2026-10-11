"""Turn-taking protocol, acoustic wake-phrase detection, and schemas for pass-the-mic.

Supports both:
1. Over-the-Air Acoustic Detection (detecting who received the mic from Microphone STT transcripts)
2. Structured JSON LLM output validation (`MicPassTurn`)
"""

from __future__ import annotations

import json
import re
import time
from enum import Enum
from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class MessageType(str, Enum):
    """Protocol message types for optional hybrid/telemetry coordination."""
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
    spoken_text: str = Field(..., description="The script spoken out loud by the agent's speaker")
    next_agent: str = Field(..., description="Name of the next team to receive the mic")
    heard_from_mic: str = Field(default="", description="The microphone transcript this agent heard before speaking")
    timestamp: float = Field(default_factory=time.time, description="Unix timestamp of when the turn occurred")

    def to_json(self) -> str:
        return self.model_dump_json()

    @classmethod
    def from_json(cls, data: str | dict) -> MicPassTurn:
        if isinstance(data, str):
            return cls.model_validate_json(data)
        return cls.model_validate(data)


class AgentRegistration(BaseModel):
    """Registration packet for hybrid monitoring."""
    type: str = Field(default=MessageType.REGISTER.value)
    name: str = Field(..., description="Team name (e.g. 'Team Orange')")
    voice: str = Field(default="en-US-AriaNeural", description="Selected Edge-TTS voice identifier")
    persona: str = Field(default="", description="Brief persona summary or tagline")


class TurnGrant(BaseModel):
    """Payload for turn coordination."""
    type: str = Field(default=MessageType.YOUR_TURN.value)
    topic: str = Field(..., description="The current roundtable debate topic")
    history: List[dict] = Field(default_factory=list, description="Transcript of previous turns in the debate")
    available_agents: List[str] = Field(default_factory=list, description="List of eligible peer team names excluding self")
    round_number: int = Field(default=1, description="Current round index")
    time_limit_sec: int = Field(default=30, description="Recommended speaking time ceiling")


class TurnSubmission(BaseModel):
    """Payload sent after an agent finishes speaking."""
    type: str = Field(default=MessageType.TURN_COMPLETE.value)
    turn: MicPassTurn


# ==============================================================================
# ACOUSTIC MICROPHONE WAKE-WORD & MIC-PASS DETECTION
# ==============================================================================

# Phonetic and STT homophone aliases so real-world microphone transcription is rock-solid
TEAM_HOMOPHONES: Dict[str, List[str]] = {
    "Team Orange": ["team orange", "orange team"],
    "Team Red": ["team red", "team read", "team redd", "red team"],
    "Team Blue": ["team blue", "team blew", "blue team"],
    "Team Emerald": ["team emerald", "team emrald", "emerald team"],
    "Team Purple": ["team purple", "purple team"],
    "Team Amber": ["team amber", "amber team"],
    "Team Silver": ["team silver", "silver team"],
    "Team Onyx": ["team onyx", "team onix", "onyx team"],
    "Moderator": ["moderator", "team black", "the host"],
}

MIC_PASS_CUE_PATTERNS: List[str] = [
    r"pass(?:ing)?\s+(?:the\s+)?(?:mic|mike|microphone|floor|turn|baton)\s+(?:over\s+)?to\s+",
    r"hand(?:ing)?\s+(?:the\s+)?(?:mic|mike|microphone|floor)\s+(?:over\s+)?to\s+",
    r"over\s+to\s+(?:you\s+)?",
    r"start(?:ing)?\s+(?:the\s+roundtable\s+)?with\s+",
    r"kick(?:ing)?\s+(?:us\s+)?off\s+with\s+",
    r"hear\s+(?:first\s+)?from\s+",
    r"redirect(?:ing)?\s+(?:the\s+)?(?:mic|mike|microphone)\s+to\s+",
    r"call(?:ing)?\s+on\s+",
]


def _get_aliases_for_team(team_name: str) -> List[str]:
    """Return lowercase phonetic aliases for a given team name."""
    for canonical, aliases in TEAM_HOMOPHONES.items():
        if canonical.lower() == team_name.lower():
            return aliases
    lower = team_name.lower().strip()
    aliases = [lower]
    if lower.startswith("team "):
        color = lower.replace("team ", "").strip()
        aliases.append(f"{color} team")
    return aliases


def detect_mic_pass_in_transcript(
    transcript: str,
    known_teams: Optional[List[str]] = None,
    current_speaker: Optional[str] = None,
) -> Optional[str]:
    """Analyze a microphone STT transcript and detect which team was passed the mic.
    
    Handles natural speech variations and STT homophones (e.g., 'team read' -> 'Team Red',
    'passing the mike to team orange' -> 'Team Orange').
    
     If multiple teams are mentioned (e.g., 'I agree with Team Blue, but I am passing
    the mic to Team Red'), prioritizes explicit mic-pass cue phrases and the final team
    mentioned at the end of the speech.
    """
    if not transcript or not transcript.strip():
        return None

    clean = re.sub(r"[^\w\s]", " ", transcript.lower())
    clean = re.sub(r"\s+", " ", clean).strip()

    teams_to_check = known_teams or list(TEAM_HOMOPHONES.keys())
    if current_speaker:
        teams_to_check = [t for t in teams_to_check if t.lower() != current_speaker.lower()]

    # 1. First check explicit cue phrases (e.g., "passing the mic to <Team>")
    best_cue_team: Optional[str] = None
    best_cue_pos: int = -1

    for team in teams_to_check:
        for alias in _get_aliases_for_team(team):
            for cue in MIC_PASS_CUE_PATTERNS:
                pattern = cue + re.escape(alias)
                for match in re.finditer(pattern, clean):
                    if match.start() > best_cue_pos:
                        best_cue_pos = match.start()
                        best_cue_team = team

    if best_cue_team:
        return best_cue_team

    # 2. Check the tail end of the transcript (last 65% of spoken text) for any team mention
    tail_start = int(len(clean) * 0.35) if len(clean) > 40 else 0
    tail_text = clean[tail_start:]

    last_mentioned_team: Optional[str] = None
    last_pos: int = -1

    for team in teams_to_check:
        for alias in _get_aliases_for_team(team):
            for match in re.finditer(r"\b" + re.escape(alias) + r"\b", tail_text):
                if match.end() > last_pos:
                    last_pos = match.end()
                    last_mentioned_team = team

    return last_mentioned_team


def is_mic_passed_to_me(
    transcript: str,
    my_team_name: str,
    known_teams: Optional[List[str]] = None,
) -> bool:
    """Return True if the microphone transcript passes the mic to `my_team_name`."""
    detected = detect_mic_pass_in_transcript(transcript, known_teams=known_teams)
    if not detected:
        return False
    return detected.lower() == my_team_name.lower()


# ==============================================================================
# STRUCTURED LLM OUTPUT PARSING
# ==============================================================================

def extract_json_block(text: str) -> Optional[str]:
    """Extract a JSON object from text that may contain markdown or extra commentary."""
    text = text.strip()

    if text.startswith("{") and text.endswith("}"):
        return text

    code_block_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if code_block_match:
        return code_block_match.group(1).strip()

    brace_match = re.search(r"(\{[\s\S]*\})", text)
    if brace_match:
        return brace_match.group(1).strip()

    return None


def parse_agent_turn(
    raw_response: str,
    current_speaker: str,
    available_agents: List[str],
    heard_from_mic: str = "",
) -> MicPassTurn:
    """Robust parser that guarantees a valid MicPassTurn even if the LLM hallucinates."""
    json_str = extract_json_block(raw_response)
    data = None

    if json_str:
        try:
            data = json.loads(json_str)
        except json.JSONDecodeError:
            repaired = re.sub(r",\s*([\]}])", r"\1", json_str)
            try:
                data = json.loads(repaired)
            except Exception:
                data = None

    if not isinstance(data, dict):
        clean_text = raw_response.strip().replace('"', "'")
        next_candidate = available_agents[0] if available_agents else "Moderator"
        if next_candidate.lower() not in clean_text.lower():
            clean_text = f"{clean_text} Passing the mic to {next_candidate}.".strip()
        return MicPassTurn(
            speaker=current_speaker,
            spoken_text=clean_text,
            next_agent=next_candidate,
            heard_from_mic=heard_from_mic,
        )

    speaker = str(data.get("speaker") or current_speaker).strip() or current_speaker
    spoken_text = str(data.get("spoken_text") or "").strip()
    next_agent = str(data.get("next_agent") or "").strip()

    if not available_agents:
        next_agent = "Moderator"
    else:
        match = None
        for candidate in available_agents:
            if candidate.lower() == next_agent.lower():
                match = candidate
                break

        if not match or next_agent.lower() == current_speaker.lower():
            detected_in_speech = detect_mic_pass_in_transcript(
                spoken_text,
                known_teams=available_agents,
                current_speaker=current_speaker,
            )
            if detected_in_speech:
                match = detected_in_speech

        if not match or match.lower() == current_speaker.lower():
            eligible = [a for a in available_agents if a.lower() != current_speaker.lower()]
            match = eligible[0] if eligible else "Moderator"

        next_agent = match

    if not spoken_text:
        spoken_text = f"That is a vital point for our roundtable. Passing the mic to {next_agent}."
    elif next_agent.lower() not in spoken_text.lower():
        # Ensure spoken_text explicitly says the next_agent name out loud so their microphone wakes up!
        spoken_text = f"{spoken_text.rstrip('.!?')}. Passing the mic to {next_agent}!"

    return MicPassTurn(
        speaker=speaker,
        spoken_text=spoken_text,
        next_agent=next_agent,
        heard_from_mic=heard_from_mic,
    )
