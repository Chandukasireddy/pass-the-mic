"""Unit and Integration Tests for pass-the-mic."""

import json
import pytest
from pathlib import Path
from fastapi.testclient import TestClient

from core.protocol import (
    MicPassTurn,
    MessageType,
    parse_agent_turn,
    TurnGrant,
    TurnSubmission,
)
from core.tts import (
    get_voice_for_team,
    sanitize_text_for_speech,
    VOICE_REGISTRY,
)
from core.llm import LLMClient
from character_agents.agent_builder import CharacterAgent
from moderator.server import app, load_topics
from moderator.moderator_agent import ModeratorAgent


def test_protocol_json_serialization():
    turn = MicPassTurn(
        speaker="Team Orange",
        spoken_text="I think open-source models are advancing faster than closed labs. Passing the mic to Team Red.",
        next_agent="Team Red",
    )
    json_str = turn.to_json()
    assert "Team Orange" in json_str
    assert "Team Red" in json_str

    deserialized = MicPassTurn.from_json(json_str)
    assert deserialized.speaker == "Team Orange"
    assert deserialized.next_agent == "Team Red"


def test_protocol_robust_parser_clean_json():
    raw = '{"speaker": "Team Orange", "spoken_text": "We need open weights! Passing to Team Red.", "next_agent": "Team Red"}'
    turn = parse_agent_turn(raw, "Team Orange", ["Team Red", "Team Blue"])
    assert turn.speaker == "Team Orange"
    assert turn.next_agent == "Team Red"


def test_protocol_robust_parser_markdown_blocks():
    raw = '''```json
    {
        "speaker": "Team Orange",
        "spoken_text": "Testing markdown backticks! Passing to Team Blue.",
        "next_agent": "Team Blue"
    }
    ```'''
    turn = parse_agent_turn(raw, "Team Orange", ["Team Red", "Team Blue"])
    assert turn.speaker == "Team Orange"
    assert turn.next_agent == "Team Blue"


def test_protocol_self_pass_recovery():
    # Agent accidentally passes mic to itself
    raw = '{"speaker": "Team Orange", "spoken_text": "Passing to myself!", "next_agent": "Team Orange"}'
    turn = parse_agent_turn(raw, "Team Orange", ["Team Red", "Team Blue"])
    assert turn.next_agent != "Team Orange"
    assert turn.next_agent in ["Team Red", "Team Blue"]


def test_protocol_malformed_text_fallback():
    raw = "Hey guys, I totally disagree with closed models! What do you think Team Red?"
    turn = parse_agent_turn(raw, "Team Orange", ["Team Red", "Team Blue"])
    assert turn.speaker == "Team Orange"
    assert turn.next_agent == "Team Red"
    assert "disagree with closed models" in turn.spoken_text


def test_voice_registry_mapping():
    assert get_voice_for_team("Team Orange") == "en-US-AriaNeural"
    assert get_voice_for_team("Team Red") == "en-GB-RyanNeural"
    assert get_voice_for_team("Moderator") == "en-US-GuyNeural"
    assert get_voice_for_team("Team Blue") == "en-AU-NatashaNeural"
    # Custom direct voice
    assert get_voice_for_team("en-US-JennyNeural") == "en-US-JennyNeural"


def test_sanitize_text_for_speech():
    raw = "Hello **world**! Check [this link](http://example.com) and `code`."
    clean = sanitize_text_for_speech(raw)
    assert clean == "Hello world! Check this link and code."


def test_character_agent_presets():
    agent_orange = CharacterAgent.from_preset("presets/orange_optimist.json")
    assert agent_orange.name == "Team Orange"
    assert "AriaNeural" in agent_orange.voice
    assert "Optimist" in agent_orange.persona

    agent_red = CharacterAgent.from_preset("presets/red_skeptic.json")
    assert agent_red.name == "Team Red"
    assert "RyanNeural" in agent_red.voice
    assert "Skeptic" in agent_red.persona


def test_llm_mock_generation():
    llm = LLMClient(provider="mock")
    turn = llm.generate_mic_turn(
        speaker_name="Team Orange",
        persona="Tech Optimist",
        topic="Open Source AI",
        history=[],
        available_agents=["Team Red"],
    )
    assert turn.speaker == "Team Orange"
    assert turn.next_agent == "Team Red"
    assert len(turn.spoken_text) > 20


def test_server_rest_endpoints():
    client = TestClient(app)
    
    # Root HTML endpoint
    res_index = client.get("/")
    assert res_index.status_code == 200
    assert "Pass-The-Mic" in res_index.text

    # Status endpoint
    res_status = client.get("/status")
    assert res_status.status_code == 200
    data = res_status.json()
    assert "is_running" in data
    assert "connected_teams" in data

    # Topics endpoint
    res_topics = client.get("/topics")
    assert res_topics.status_code == 200
    topics = res_topics.json()
    assert len(topics) >= 1
    assert "topic" in topics[0]
