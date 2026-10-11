"""BrowserAudio bridge against a real local WebSocket client (no browser, no mic)."""

import asyncio
import json

import pytest
from websockets.asyncio.client import connect
from websockets.exceptions import InvalidStatus

from core.browser_audio import BrowserAudio


def run(coro):
    return asyncio.run(coro)


def test_serves_page_and_rejects_foreign_origin():
    async def go():
        async with BrowserAudio(port=0) as audio:
            reader, writer = await asyncio.open_connection("127.0.0.1", audio.port)
            writer.write(b"GET / HTTP/1.1\r\nHost: x\r\nConnection: close\r\n\r\n")
            await writer.drain()
            body = (await reader.read()).decode()
            writer.close()
            assert "200 OK" in body and "getUserMedia" in body
            with pytest.raises(InvalidStatus) as err:
                await connect(f"ws://127.0.0.1:{audio.port}/ws", origin="https://evil.example")
            assert err.value.response.status_code == 403
    run(go())


def test_mic_frames_in_and_speaker_audio_out():
    async def go():
        async with BrowserAudio(port=0) as audio:
            origin = f"http://127.0.0.1:{audio.port}"
            async with connect(f"ws://127.0.0.1:{audio.port}/ws", origin=origin) as ws:
                await audio.wait_connected(2)
                assert json.loads(await asyncio.wait_for(ws.recv(), 2)) == audio.status  # Sent on connect.
                frame = (1000).to_bytes(2, "little", signed=True) * 480
                await ws.send(frame)
                assert await asyncio.wait_for(audio.read(), 2) == frame
                assert "SOUND DETECTED" in audio.meter()

                audio.play(b"\x01\x00" * 240)
                assert await asyncio.wait_for(ws.recv(), 2) == b"\x01\x00" * 240
                # Half duplex: the mic is zeroed while the reply plays.
                await ws.send(frame)
                assert await asyncio.wait_for(audio.read(), 2) == bytes(len(frame))

                audio.clear()
                assert json.loads(await asyncio.wait_for(ws.recv(), 2)) == {"type": "clear"}
    run(go())


def test_disconnect_raises_instead_of_hanging():
    async def go():
        async with BrowserAudio(port=0) as audio:
            origin = f"http://127.0.0.1:{audio.port}"
            async with connect(f"ws://127.0.0.1:{audio.port}/ws", origin=origin):
                await audio.wait_connected(2)
            with pytest.raises(RuntimeError, match="disconnected"):
                await asyncio.wait_for(audio.read(), 2)
    run(go())


def test_playback_backlog_is_bounded():
    audio = BrowserAudio(port=0)
    with pytest.raises(RuntimeError, match="fell behind"):
        audio.play(bytes(48000 * 6))


def test_page_commands_and_status_updates():
    async def go():
        async with BrowserAudio(port=0) as audio:
            origin = f"http://127.0.0.1:{audio.port}"
            async with connect(f"ws://127.0.0.1:{audio.port}/ws", origin=origin) as ws:
                await audio.wait_connected(2)
                assert json.loads(await ws.recv()) == {"type": "status", "state": "idle"}

                await ws.send(json.dumps({"type": "agent", "action": "start"}))
                await ws.send(json.dumps({"type": "agent", "action": "rm -rf"}))  # Ignored.
                await ws.send("not json")
                await ws.send(json.dumps({"type": "agent", "action": "stop"}))
                assert [await asyncio.wait_for(audio.commands.get(), 2) for _ in range(2)] == ["start", "stop"]
                assert audio.next_command() is None

                audio.set_status("live")
                audio.play(b"\x01\x00" * 10)
                audio.clear()  # A flush must not swallow the status update.
                seen = [json.loads(await asyncio.wait_for(ws.recv(), 2))]
                seen.append(json.loads(await asyncio.wait_for(ws.recv(), 2)))
                assert {"type": "status", "state": "live"} in seen
                assert {"type": "clear"} in seen
    run(go())
