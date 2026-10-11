"""FastAPI + WebSocket Backend Server for Magnus (Gemini Speech-to-Speech Voice Agent).

Run with:
    python magnus_agent/app.py
or from inside `magnus_agent/`:
    python app.py
Then open http://localhost:8050 in your browser!
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import webbrowser
from pathlib import Path
from typing import Any, Dict, List, Optional

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel
import uvicorn

from magnus_agent.active_listener import MagnusActiveController, get_audio_devices
from magnus_agent.gemini_live_engine import GeminiSpeechToSpeechEngine
from magnus_agent.magnus_config import (
    GEMINI_VOICES,
    GEMINI_VOICE_MODELS,
    PRESET_TOPICS,
    MagnusSettings,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("magnus.server")

app = FastAPI(title="Magnus — Gemini 3.8 Live Speech-to-Speech Control Center")

# Global singletons
settings = MagnusSettings()
engine = GeminiSpeechToSpeechEngine()
controller = MagnusActiveController(settings=settings, engine=engine)

STATIC_DIR = Path(__file__).resolve().parent / "static"


@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    index_file = STATIC_DIR / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    return HTMLResponse("<h1>Magnus UI not found</h1>", status_code=404)


@app.get("/api/state")
async def get_full_state():
    """Return all current settings, models, voices, audio devices, and live status."""
    return {
        "settings": controller.settings.model_dump(),
        "is_listening": controller.is_listening,
        "state": controller.state,
        "turns_completed": controller.turns_completed,
        "api_status": engine.last_api_status,
        "has_api_key": bool(engine.api_key),
        "models": GEMINI_VOICE_MODELS,
        "voices": GEMINI_VOICES,
        "preset_topics": PRESET_TOPICS,
        "audio_devices": get_audio_devices(),
        "transcript_history": controller.transcript_history,
    }


class SettingsUpdatePayload(BaseModel):
    agent_name: Optional[str] = None
    persona: Optional[str] = None
    style: Optional[str] = None
    gemini_model: Optional[str] = None
    tts_model: Optional[str] = None
    gemini_voice: Optional[str] = None
    fallback_edge_voice: Optional[str] = None
    listening_mode: Optional[str] = None
    wake_phrases: Optional[List[str]] = None
    peer_teams: Optional[List[str]] = None
    pass_mic_at_end: Optional[bool] = None
    topic: Optional[str] = None
    mic_device_index: Optional[int] = None
    energy_threshold: Optional[float] = None
    silence_duration: Optional[float] = None
    max_record_seconds: Optional[float] = None
    speaker_enabled: Optional[bool] = None
    speech_rate: Optional[str] = None
    api_key: Optional[str] = None


@app.post("/api/settings")
async def update_settings(payload: SettingsUpdatePayload):
    """Update Magnus's configuration in real time from the Web UI."""
    data = payload.model_dump(exclude_unset=True)

    if "api_key" in data and data["api_key"]:
        engine.update_api_key(data.pop("api_key"))

    current_dict = controller.settings.model_dump()
    for k, v in data.items():
        if v is not None or k == "mic_device_index":
            current_dict[k] = v

    # Sync fallback edge voice when gemini_voice changes
    if "gemini_voice" in data and "fallback_edge_voice" not in data:
        for v_info in GEMINI_VOICES:
            if v_info["id"].lower() == str(data["gemini_voice"]).lower():
                current_dict["fallback_edge_voice"] = v_info["edge_fallback"]
                break

    controller.settings = MagnusSettings(**current_dict)
    await controller.broadcast_event({
        "type": "settings_updated",
        "settings": controller.settings.model_dump(),
        "api_status": engine.last_api_status,
    })
    return {
        "status": "updated",
        "settings": controller.settings.model_dump(),
        "api_status": engine.last_api_status,
    }


@app.post("/api/listening/start")
async def start_listening():
    """Start continuous microphone active listening."""
    await controller.start_active_listening()
    return {"status": "started", "is_listening": controller.is_listening, "state": controller.state}


@app.post("/api/listening/stop")
async def stop_listening():
    """Stop continuous microphone active listening."""
    await controller.stop_active_listening()
    return {"status": "stopped", "is_listening": controller.is_listening, "state": controller.state}


@app.post("/api/calibrate")
async def calibrate_mic():
    """Sample 1s of ambient room noise to auto-adjust the microphone VAD threshold."""
    threshold = await controller.calibrate_microphone()
    return {"status": "calibrated", "energy_threshold": threshold}


class TriggerPayload(BaseModel):
    prompt: Optional[str] = None


@app.post("/api/trigger")
async def trigger_magnus(payload: TriggerPayload):
    """Immediately trigger Magnus to speak out loud."""
    asyncio.create_task(controller.trigger_manual_turn(custom_prompt=payload.prompt))
    return {"status": "triggered"}


@app.post("/api/test-voice")
async def test_magnus_voice():
    """Play a quick speaker test line as Magnus."""
    test_line = (
        f"Greetings. I am {controller.settings.agent_name}, running on {controller.settings.gemini_model} "
        f"with the {controller.settings.gemini_voice} voice profile. My active microphone listener is ready."
    )
    asyncio.create_task(controller.trigger_manual_turn(custom_prompt=f"Say this exact greeting: {test_line}"))
    return {"status": "testing_voice"}


@app.post("/api/clear-history")
async def clear_history():
    controller.transcript_history.clear()
    controller.turns_completed = 0
    await controller.broadcast_event({"type": "history_cleared"})
    return {"status": "cleared"}


@app.websocket("/ws")
async def websocket_stream(ws: WebSocket):
    """Real-time telemetry WebSocket pushing mic RMS levels, state transitions, and transcripts."""
    await ws.accept()

    async def _send_event(event: Dict[str, Any]) -> None:
        await ws.send_text(json.dumps(event))

    controller.subscribe(_send_event)
    try:
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        controller.unsubscribe(_send_event)


def _ensure_port_available(host: str, port: int) -> int:
    """Ensure the requested port is free; if occupied by a previous instance, free it or pick next available."""
    import socket
    import psutil

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        if s.connect_ex((host, port)) != 0:
            return port

    # Try to terminate a previous stale python magnus_agent instance holding the port
    try:
        for conn in psutil.net_connections(kind="inet"):
            if conn.laddr and conn.laddr.port == port and conn.pid:
                proc = psutil.Process(conn.pid)
                cmdline = " ".join(proc.cmdline()).lower()
                if "magnus_agent" in cmdline or "app.py" in cmdline:
                    proc.terminate()
                    proc.wait(timeout=2)
                    return port
    except Exception:
        pass

    # Fallback: find the next available port
    for candidate in range(port + 1, port + 50):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex((host, candidate)) != 0:
                return candidate
    return port


def main():
    import threading

    parser = argparse.ArgumentParser(description="Run the Magnus Gemini Speech-to-Speech Web Control Center.")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host address (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8050, help="Port number (default: 8050)")
    parser.add_argument("--open-browser", action="store_true", help="Automatically open browser on launch")
    args = parser.parse_args()

    active_port = _ensure_port_available(args.host, args.port)
    url = f"http://{args.host}:{active_port}"
    print("=" * 72)
    print("  🎙️  MAGNUS — GEMINI 3.8 LIVE SPEECH-TO-SPEECH VOICE AGENT")
    print(f"  🌐 Web Control Center: {url}")
    print(f"  🔑 API Status:         {engine.last_api_status}")
    print("=" * 72)

    if args.open_browser:
        t = threading.Timer(1.0, lambda: webbrowser.open(url))
        t.daemon = True
        t.start()

    uvicorn.run(app, host=args.host, port=active_port, log_level="warning")


if __name__ == "__main__":
    main()

