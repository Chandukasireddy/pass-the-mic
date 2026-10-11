"""Run a name-triggered voice character using Merge Gateway's GPT-Live."""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import os
from pathlib import Path
import signal
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.live_voice import LiveCharacter, MERGE_LIVE_URL, start_session, stream_session


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", type=Path, default=Path(__file__).parent / "presets/rex_live.json")
    parser.add_argument("--name", help="Override the name that wakes the character")
    parser.add_argument("--voice", help="Override the GPT-Live voice")
    parser.add_argument("--audio", choices=("browser", "local"), default="browser",
                        help="browser: mic/speaker through a local web page (default); local: sounddevice")
    parser.add_argument("--port", type=int, default=8765, help="Port for the local browser audio page")
    parser.add_argument("--no-open", action="store_true", help="Print the browser URL instead of opening it")
    parser.add_argument("--input-device", type=int, help="Microphone device ID")
    parser.add_argument("--output-device", type=int, help="Speaker device ID")
    parser.add_argument("--list-devices", action="store_true")
    parser.add_argument("--check-mic", action="store_true", help="Show microphone levels for 10 seconds without connecting to Merge")
    parser.add_argument("--check-speaker", action="store_true", help="Play a short test tone through the speaker without connecting to Merge")
    parser.add_argument("--dry-run", action="store_true", help="Print configuration; no network or microphone")
    return parser.parse_args(argv)


@contextlib.asynccontextmanager
async def open_audio(args):
    """Yield an audio object; for the browser, wait until the page is streaming."""
    if args.audio == "local":
        from core.live_audio import LiveAudio
        with LiveAudio(args.input_device, args.output_device) as audio:
            import sounddevice as sd
            print(f"[DEVICE] Microphone: {sd.query_devices(audio.input_stream.device)['name']}", flush=True)
            yield audio
        return
    import webbrowser
    from core.browser_audio import BrowserAudio
    async with BrowserAudio(args.port) as audio:
        print(f"Open {audio.url} in Chrome, click 'Start microphone', then 'Start agent'.", flush=True)
        if not args.no_open:
            webbrowser.open(audio.url)
        await audio.wait_connected()
        yield audio


async def check_microphone(args):
    async with open_audio(args) as audio:
        print("Microphone check: speak for 10 seconds. No audio is sent to Merge.", flush=True)
        for _ in range(10):
            await asyncio.sleep(1)
            print(audio.meter(), flush=True)


async def check_speaker(args):
    import math
    from array import array
    from core.live_voice import SAMPLE_RATE
    async with open_audio(args) as audio:
        print("Speaker check: you should hear a two-tone beep.", flush=True)
        for freq in (440, 660):
            samples = array("h", (int(9000 * math.sin(2 * math.pi * freq * i / SAMPLE_RATE)) for i in range(SAMPLE_RATE // 2)))
            audio.play(samples.tobytes())
        await asyncio.sleep(1.5)


async def run_session(character, key, audio, stop, on_live=lambda: None):
    """One billed Merge connection: open, stream until `stop`, close. Returns usage."""
    from websockets.asyncio.client import connect

    async with connect(
        MERGE_LIVE_URL,
        additional_headers={"Authorization": f"Bearer {key}"},
        open_timeout=15, close_timeout=5, max_size=16 * 1024 * 1024,
    ) as ws:
        await start_session(ws, character)
        audio.clear()
        # Discard audio captured during the handshake.
        while not audio.incoming.empty():
            audio.incoming.get_nowait()
        on_live()
        print(f"[LISTENING] Say '{character.name}' in the conversation. Ctrl+C to stop.", flush=True)
        print("Room audio streams to Merge/OpenAI; connected time includes silence. Microphone is silenced during playback.", flush=True)
        return await stream_session(ws, character, audio, stop)


async def wait_for_command(audio, quit_event, wanted):
    """Poll the page's button clicks; None means quit or the tab went away."""
    while not quit_event.is_set():
        if audio.lost:
            return None
        command = audio.next_command()
        if command in wanted:
            return command
        await asyncio.sleep(0.05)
    return None


async def control_loop(character, key, audio, quit_event):
    """Browser UI mode: the mic stays open for free; billing runs only Start to Stop."""
    while True:
        audio.set_status("idle")
        print("[IDLE] Agent stopped, not billing. Click 'Start agent' in the browser.", flush=True)
        if await wait_for_command(audio, quit_event, ("start",)) is None:
            return
        stop = asyncio.Event()

        async def watch_stop():
            await wait_for_command(audio, quit_event, ("stop",))  # Also returns on quit or tab loss.
            audio.set_status("stopping")
            stop.set()

        watcher = asyncio.create_task(watch_stop())
        audio.set_status("connecting")
        message = ""
        try:
            usage = await run_session(character, key, audio, stop, lambda: audio.set_status("live"))
            if usage is not None:
                message = f"Last session billed {usage.get('seconds', 'unknown')} seconds."
        except Exception as exc:
            print(f"Session ended: {exc}", file=sys.stderr, flush=True)
            message = f"Session ended: {exc}"
        finally:
            watcher.cancel()
            await asyncio.gather(watcher, return_exceptions=True)
        if quit_event.is_set() or audio.lost:
            return
        audio.set_status("idle", message=message)
        # Fall through to the idle loop; stale clicks from the session are dropped.
        while audio.next_command() is not None:
            pass


async def run(character, key, args):
    # Open local audio before any billed session. Until signal handlers are
    # installed, Ctrl+C raises KeyboardInterrupt so waiting for the tab is cancellable.
    async with open_audio(args) as audio:
        quit_event = asyncio.Event()
        loop = asyncio.get_running_loop()
        previous = {}
        for sig in (signal.SIGINT, signal.SIGTERM):
            previous[sig] = signal.getsignal(sig)
            signal.signal(sig, lambda *_: loop.call_soon_threadsafe(quit_event.set))
        try:
            if args.audio == "browser":
                await control_loop(character, key, audio, quit_event)
            else:
                await run_session(character, key, audio, quit_event)
        finally:
            for sig, handler in previous.items():
                signal.signal(sig, handler)


def main(argv=None):
    args = parse_args(argv)
    try:
        from dotenv import load_dotenv
        load_dotenv(ROOT / ".env")
        if args.list_devices:
            import sounddevice as sd
            print(sd.query_devices())
            return 0
        if args.check_mic:
            asyncio.run(check_microphone(args))
            return 0
        if args.check_speaker:
            asyncio.run(check_speaker(args))
            return 0
        data = json.loads(args.preset.read_text(encoding="utf-8"))
        character = LiveCharacter(
            name=args.name or data["name"],
            aliases=tuple(data.get("aliases", [])) if not args.name else (),
            voice=args.voice or data.get("live_voice", "vesper"),
            model=os.getenv("MERGE_LIVE_MODEL", "openai/gpt-live-1"),
            persona=data["persona"], style=data["style"],
        )
        if args.dry_run:
            print(json.dumps(character.start_event(), indent=2))
            return 0
        key = os.getenv("MERGE_GATEWAY_API_KEY", "").strip()
        if not key or key.startswith("your_"):
            raise ValueError("Set MERGE_GATEWAY_API_KEY in the project .env before starting live audio.")
        asyncio.run(run(character, key, args))
        return 0
    except KeyboardInterrupt:
        return 0
    except ImportError as exc:
        print(f"Missing dependency: {exc}. Run: pip install -r requirements-live.txt", file=sys.stderr)
    except Exception as exc:
        print(f"Live voice stopped: {exc}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
