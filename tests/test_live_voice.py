"""Exercise the actual Live wire protocol without a key or audio devices."""

import asyncio
import base64
import json

import pytest

from core.live_voice import LiveCharacter, MentionGate, is_silent, start_session, stream_session


class Clock:
    now = 100.0

    def __call__(self):
        return self.now


@pytest.mark.parametrize("text", ["Hey REX, thoughts?", "I agree with Rex.", "Team Rex, go."])
def test_mentions_wake(text):
    gate = MentionGate(["Rex"])
    assert gate.feed(text)
    assert gate.is_open()


@pytest.mark.parametrize("text", ["Rexford disagrees.", "Tyrannosaurusrex!", "What do you think?", "rex_thing"])
def test_unrelated_words_do_not_wake(text):
    assert not MentionGate(["Rex"]).feed(text)


def test_fragmented_names_and_end_of_caption():
    clock = Clock()
    gate = MentionGate(["Rex"], clock)
    assert not gate.feed("Hey Re")
    assert not gate.feed("x")
    clock.now += 0.5
    assert gate.tick()
    assert not gate.tick()


def test_partial_longer_name_does_not_wake():
    gate = MentionGate(["Rex"])
    assert not gate.feed("Rex")
    assert not gate.feed("ford, hello.")


def test_idle_caption_fragments_are_not_joined():
    clock = Clock()
    gate = MentionGate(["Rex"], clock)
    gate.feed("Re")
    clock.now += 3
    assert not gate.feed("x, yes?")


def test_one_reply_then_rearm_without_replaying_old_mention():
    clock = Clock()
    gate = MentionGate(["Rex"], clock)
    assert not gate.allow_audio(48000)
    assert gate.feed("Rex, explain.")
    assert not gate.feed("Rex, explain again.")
    assert gate.allow_audio(48000)
    clock.now += 25  # Past the reply, its tail and the follow-up window.
    assert not gate.allow_audio(48000)
    assert not gate.tick()
    assert gate.feed("Rex, next question.")


def test_unused_wake_expires_and_echo_does_not_wake():
    clock = Clock()
    gate = MentionGate(["Rex"], clock)
    assert not gate.feed("Rex, hello.", suppressed=True)
    assert not gate.tick()
    gate.feed("Rex, hello.")
    clock.now += 26
    assert not gate.allow_audio(48000)


class FakeSocket:
    def __init__(self, events=()):
        self.events = asyncio.Queue()
        for event in events:
            self.events.put_nowait(json.dumps(event))
        self.sent = []

    async def send(self, message):
        event = json.loads(message)
        self.sent.append(event)
        if event["type"] == "session.close":
            self.events.put_nowait(json.dumps({"type": "session.closed", "usage": {"seconds": 3}}))

    async def recv(self):
        return await self.events.get()

    def __aiter__(self):
        return self

    async def __anext__(self):
        return await self.recv()


class FakeAudio:
    echo_suppressed = False

    def __init__(self, stop):
        self.played = []
        self.stop = stop
        self.cleared = False

    async def read(self):
        await asyncio.sleep(0.001)
        return bytes(960)

    def play(self, chunk):
        self.played.append(chunk)
        self.stop.set()

    def clear(self):
        self.cleared = True

    def meter(self):
        return "[MIC] test"


def test_microphone_meter_distinguishes_no_frames_silence_and_sound():
    from array import array
    from core.live_audio import LiveAudio
    audio = LiveAudio()
    assert "NO AUDIO FRAMES" in audio.meter()
    audio._capture(bytes(960), 480, None, None)
    assert "SILENT" in audio.meter()
    audio._capture(array("h", [4000, -4000] * 240).tobytes(), 480, None, None)
    assert "SOUND DETECTED" in audio.meter()
    assert "NO AUDIO FRAMES" in audio.meter()


def test_start_uses_live_model_and_waits_for_ack():
    async def scenario():
        ws = FakeSocket([{"type": "session.started"}])
        await start_session(ws, LiveCharacter())
        event = ws.sent[0]
        assert event["type"] == "session.start"
        assert event["session"]["model"] == "openai/gpt-live-1"
        assert event["session"]["audio"]["format"]["rate"] == 24000
    asyncio.run(scenario())


def test_start_error_and_timeout_are_visible():
    async def scenario():
        with pytest.raises(RuntimeError, match="startup refused"):
            await start_session(FakeSocket([{"type": "error", "error": "invalid key"}]), LiveCharacter())
        with pytest.raises(asyncio.TimeoutError):
            await start_session(FakeSocket(), LiveCharacter(), timeout=0.01)
    asyncio.run(scenario())


def test_real_event_names_gating_audio_and_graceful_close():
    async def scenario():
        stop = asyncio.Event()
        audio = FakeAudio(stop)
        blocked = {"type": "session.output_audio.delta", "delta": base64.b64encode(b"\0\0").decode()}
        allowed = {"type": "session.output_audio.delta", "delta": base64.b64encode(b"\0\x10").decode()}
        ws = FakeSocket([
            blocked,
            {"type": "session.output_transcript.delta", "delta": "Rex, hello."},
            blocked,  # The assistant's transcript must never open the gate.
            {"type": "session.input_transcript.delta", "delta": "Rex, what do you think?"},
            allowed,
        ])
        usage = await asyncio.wait_for(stream_session(ws, LiveCharacter(), audio, stop, lambda _: None), 1)
        assert audio.played == [b"\0\x10"]
        assert audio.cleared
        assert usage == {"seconds": 3}
        assert ws.sent[-1]["type"] == "session.close"
        assert any(e["type"] == "session.instructions.append" for e in ws.sent)
    asyncio.run(scenario())


def test_api_failure_closes_session_and_propagates():
    async def scenario():
        stop = asyncio.Event()
        audio = FakeAudio(stop)
        ws = FakeSocket([{"type": "error", "error": {"message": "budget exhausted"}}])
        with pytest.raises(RuntimeError, match="budget exhausted"):
            await stream_session(ws, LiveCharacter(), audio, stop, lambda _: None)
        assert ws.sent[-1]["type"] == "session.close"
        assert audio.cleared
    asyncio.run(scenario())


def test_server_close_stops_microphone_loop():
    async def scenario():
        stop = asyncio.Event()
        audio = FakeAudio(stop)
        ws = FakeSocket([{"type": "session.closed", "usage": {"seconds": 4}}])
        assert await stream_session(ws, LiveCharacter(), audio, stop, lambda _: None) == {"seconds": 4}
        assert audio.cleared
    asyncio.run(scenario())


def test_provider_filler_silence_does_not_hold_gate_open():
    assert is_silent(bytes(960))
    assert is_silent(b"\x05\x00" * 480)  # Faint noise floor.
    assert not is_silent(b"\x00\x10" * 480)

    async def scenario():
        stop = asyncio.Event()
        audio = FakeAudio(stop)
        speech = {"type": "session.output_audio.delta", "delta": base64.b64encode(b"\x00\x10" * 240).decode()}
        filler = {"type": "session.output_audio.delta", "delta": base64.b64encode(bytes(9600)).decode()}
        ws = FakeSocket([
            {"type": "session.input_transcript.delta", "delta": "Rex, can you hear me?"},
            speech, filler, filler, filler,
        ])
        await asyncio.wait_for(stream_session(ws, LiveCharacter(), audio, stop, lambda _: None), 1)
        # Short pauses after speech are kept; long filler is covered by the clocked tests.
        assert audio.played[0] == b"\x00\x10" * 240
    asyncio.run(scenario())


def test_follow_up_audio_passes_without_name():
    clock = Clock()
    gate = MentionGate(["Rex"], clock)
    assert gate.feed("Rex, can you hear me? ")
    assert gate.allow_audio(48000)  # One second of reply.
    clock.now += 4                  # Reply and gate tail finished.
    assert not gate.is_open()
    assert gate.allow_audio(24000)  # The model answers a nameless follow-up.


def test_no_follow_up_after_window_expires():
    clock = Clock()
    gate = MentionGate(["Rex"], clock)
    assert gate.feed("Rex, hello? ")
    assert gate.allow_audio(48000)
    clock.now += 1 + 20 + 1
    assert not gate.allow_audio(24000)


def test_provider_filler_silence_does_not_hold_gate_open():
    assert is_silent(bytes(960))
    assert is_silent(b"\x05\x00" * 480)  # Faint noise floor.
    assert not is_silent(b"\x00\x10" * 480)

    async def scenario():
        stop = asyncio.Event()
        audio = FakeAudio(stop)
        speech = {"type": "session.output_audio.delta", "delta": base64.b64encode(b"\x00\x10" * 240).decode()}
        filler = {"type": "session.output_audio.delta", "delta": base64.b64encode(bytes(9600)).decode()}
        ws = FakeSocket([
            {"type": "session.input_transcript.delta", "delta": "Rex, can you hear me?"},
            speech, filler, filler, filler,
        ])
        await asyncio.wait_for(stream_session(ws, LiveCharacter(), audio, stop, lambda _: None), 1)
        # Short pauses after speech are kept; long filler is covered by the clocked tests.
        assert audio.played[0] == b"\x00\x10" * 240
    asyncio.run(scenario())


def test_follow_up_audio_passes_without_name():
    clock = Clock()
    gate = MentionGate(["Rex"], clock)
    assert gate.feed("Rex, can you hear me? ")
    assert gate.allow_audio(48000)  # One second of reply.
    clock.now += 4                  # Reply and gate tail finished.
    assert not gate.is_open()
    assert gate.allow_audio(24000)  # The model answers a nameless follow-up.


def test_no_follow_up_after_window_expires():
    clock = Clock()
    gate = MentionGate(["Rex"], clock)
    assert gate.feed("Rex, hello? ")
    assert gate.allow_audio(48000)
    clock.now += 3 + 20 + 1
    assert not gate.feed("Unrelated side chatter")
    clock.now += 0.5
    assert not gate.tick()


def test_pause_inside_reply_does_not_cut_it_off():
    clock = Clock()
    gate = MentionGate(["Rex"], clock)
    assert gate.feed("Rex, explain offsets. ")
    assert gate.allow_audio(24000)             # Half a second of speech.
    clock.now += 0.5
    assert gate.allow_audio(9600, loud=False)  # A 0.2 s pause is kept.
    clock.now += 1.0
    assert gate.allow_audio(24000)             # Speech resumes after the gap.


def test_long_filler_is_dropped_and_wake_ignored_while_speaking():
    clock = Clock()
    gate = MentionGate(["Rex"], clock)
    assert gate.feed("Rex, hi. ")
    assert gate.allow_audio(24000)
    clock.now += 0.5
    assert gate.allow_audio(9600, loud=False)
    clock.now += 3
    assert not gate.allow_audio(9600, loud=False)
    gate.text, gate.last_caption = "stray words", clock.now - 1
    gate.follow_until = clock.now + 10
    assert not gate.tick(suppressed=True)


def test_reply_audio_that_beats_the_name_caption_is_not_lost():
    async def scenario():
        stop = asyncio.Event()
        audio = FakeAudio(stop)
        speech = {"type": "session.output_audio.delta", "delta": base64.b64encode(b"\x00\x10" * 240).decode()}
        ws = FakeSocket([
            speech, speech,  # Model starts answering before the caption arrives.
            {"type": "session.input_transcript.delta", "delta": "Rex, what is AI?"},
        ])
        await asyncio.wait_for(stream_session(ws, LiveCharacter(), audio, stop, lambda _: None), 1)
        assert audio.played == [b"\x00\x10" * 240] * 2
    asyncio.run(scenario())


def test_unaddressed_speech_is_never_played():
    async def scenario():
        stop = asyncio.Event()
        audio = FakeAudio(stop)
        speech = {"type": "session.output_audio.delta", "delta": base64.b64encode(b"\x00\x10" * 240).decode()}
        ws = FakeSocket([speech, {"type": "session.input_transcript.delta", "delta": "Nice weather today."}])
        asyncio.get_running_loop().call_later(0.1, stop.set)
        await asyncio.wait_for(stream_session(ws, LiveCharacter(), audio, stop, lambda _: None), 1)
        assert audio.played == []
    asyncio.run(scenario())
