"""Unit and Integration Tests for pass-the-mic (Speaker + Microphone Edition)."""

import numpy as np
from fastapi.testclient import TestClient

from core.protocol import (
    MicPassTurn,
    parse_agent_turn,
    detect_mic_pass_in_transcript,
    is_mic_passed_to_me,
)
from core.tts import (
    get_voice_for_team,
    sanitize_text_for_speech,
)
from core.audio import (
    compute_rms_energy,
    list_audio_devices,
    is_currently_speaking,
)
from core.stt import STTClient, get_stt_client
from core.llm import LLMClient
from character_agents.agent_builder import CharacterAgent
from moderator.server import app


def test_protocol_json_serialization():
    turn = MicPassTurn(
        speaker="Team Orange",
        spoken_text="I think open-source models are advancing faster than closed labs. Passing the mic to Team Red!",
        next_agent="Team Red",
        heard_from_mic="Welcome to the roundtable! Passing the mic to Team Orange!",
    )
    json_str = turn.to_json()
    assert "Team Orange" in json_str
    assert "Team Red" in json_str

    deserialized = MicPassTurn.from_json(json_str)
    assert deserialized.speaker == "Team Orange"
    assert deserialized.next_agent == "Team Red"
    assert "Welcome to the roundtable" in deserialized.heard_from_mic


def test_protocol_robust_parser_clean_json():
    raw = '{"speaker": "Team Orange", "spoken_text": "We need open weights! Passing the mic to Team Red.", "next_agent": "Team Red"}'
    turn = parse_agent_turn(raw, "Team Orange", ["Team Red", "Team Blue"])
    assert turn.speaker == "Team Orange"
    assert turn.next_agent == "Team Red"


def test_protocol_enforces_spoken_wake_phrase():
    # If LLM forgot to say next_agent's name in spoken_text, parser appends it so next laptop's mic wakes up
    raw = '{"speaker": "Team Orange", "spoken_text": "Open weights are the future.", "next_agent": "Team Red"}'
    turn = parse_agent_turn(raw, "Team Orange", ["Team Red", "Team Blue"])
    assert turn.next_agent == "Team Red"
    assert "team red" in turn.spoken_text.lower()


def test_acoustic_wake_phrase_detection():
    # Standard spoken handoff
    t1 = "I believe open source will win. Passing the mic to Team Red!"
    assert detect_mic_pass_in_transcript(t1) == "Team Red"
    assert is_mic_passed_to_me(t1, "Team Red") is True
    assert is_mic_passed_to_me(t1, "Team Orange") is False

    # Homophone tolerance from STT ("mike", "team read", "team blew")
    t2 = "Great point, now passing the mike over to team read for their thoughts"
    assert detect_mic_pass_in_transcript(t2) == "Team Red"

    t3 = "I disagree completely and I am handing the floor to team blew"
    assert detect_mic_pass_in_transcript(t3) == "Team Blue"

    # Disambiguation when an earlier team is referenced first
    t4 = "I heard Team Orange's argument, but I am passing the mic to Team Blue!"
    assert detect_mic_pass_in_transcript(t4) == "Team Blue"


def test_protocol_self_pass_recovery():
    raw = '{"speaker": "Team Orange", "spoken_text": "Passing to myself!", "next_agent": "Team Orange"}'
    turn = parse_agent_turn(raw, "Team Orange", ["Team Red", "Team Blue"])
    assert turn.next_agent != "Team Orange"
    assert turn.next_agent in ["Team Red", "Team Blue"]


def test_voice_registry_mapping():
    assert get_voice_for_team("Team Orange") == "en-US-AriaNeural"
    assert get_voice_for_team("Team Red") == "en-GB-RyanNeural"
    assert get_voice_for_team("Moderator") == "en-US-GuyNeural"
    assert get_voice_for_team("Team Blue") == "en-AU-NatashaNeural"


def test_sanitize_text_for_speech():
    raw = "Hello **world**! Check [this link](http://example.com) and `code`."
    clean = sanitize_text_for_speech(raw)
    assert clean == "Hello world! Check this link and code."


def test_audio_rms_and_devices():
    silence = np.zeros((1600, 1), dtype=np.int16)
    assert compute_rms_energy(silence) == 0.0

    tone = np.full((1600, 1), 16384, dtype=np.int16)
    rms = compute_rms_energy(tone)
    assert 0.49 <= rms <= 0.51

    devices = list_audio_devices()
    assert "inputs" in devices
    assert "outputs" in devices
    assert is_currently_speaking() is False


def test_stt_client_initialization():
    stt = get_stt_client(provider="google")
    assert stt.provider == "google"


def test_character_agent_presets_and_wake_check():
    agent_orange = CharacterAgent.from_preset("presets/orange_optimist.json")
    assert agent_orange.name == "Team Orange"
    assert "AriaNeural" in agent_orange.voice
    assert agent_orange.check_wake_trigger("Welcome! Passing the mic to Team Orange!") is True
    assert agent_orange.check_wake_trigger("Passing the mic to Team Red!") is False


def test_llm_mock_generation_with_mic_context():
    llm = LLMClient(provider="mock")
    turn = llm.generate_mic_turn(
        speaker_name="Team Orange",
        persona="Tech Optimist",
        topic="Open Source AI",
        history=[],
        available_agents=["Team Red"],
        heard_from_mic="Welcome to the debate! Passing the mic to Team Orange!",
    )
    assert turn.speaker == "Team Orange"
    assert turn.next_agent == "Team Red"
    assert "team red" in turn.spoken_text.lower()
    assert "Welcome to the debate" in turn.heard_from_mic


def test_server_rest_endpoints():
    client = TestClient(app)
    res_status = client.get("/status")
    assert res_status.status_code == 200
    res_topics = client.get("/topics")
    assert res_topics.status_code == 200
    assert len(res_topics.json()) >= 1
