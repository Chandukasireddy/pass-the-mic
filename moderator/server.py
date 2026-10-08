"""FastAPI and WebSocket coordination server for Pass-The-Mic.

Coordinates turn-taking across attendee laptops over local Wi-Fi, manages
the state machine, and orchestrates the Moderator Agent.
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
import uvicorn

from core.protocol import (
    MessageType,
    MicPassTurn,
    TurnGrant,
    TurnSubmission,
    parse_agent_turn,
)
from moderator.moderator_agent import ModeratorAgent
from moderator.dashboard import RoundtableDashboard, get_local_ip

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("pass_the_mic.server")

# Load topics from topics.json
TOPICS_FILE = Path(__file__).parent / "topics.json"


def load_topics() -> List[dict]:
    if TOPICS_FILE.exists():
        with open(TOPICS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return [
        {
            "id": "open-source-ai",
            "title": "Open Source vs Closed AI",
            "topic": "Will open weights outpace closed frontier labs by 2027?",
            "intro_script": "Welcome to Pass-The-Mic! Tonight's debate: Will open weights outpace closed labs by 2027?",
            "suggested_first_speaker": "Team Orange",
        }
    ]


app = FastAPI(title="Pass-The-Mic Coordinator Server")


class RoundtableCoordinator:
    """Central state machine coordinating turns between connected laptops."""

    def __init__(self, port: int = 8000, enable_dashboard: bool = True):
        self.port = port
        self.enable_dashboard = enable_dashboard
        self.dashboard = RoundtableDashboard(port=port)
        self.moderator = ModeratorAgent(speak_aloud=True)
        self.topics = load_topics()

        # Connection registry
        self.active_connections: Dict[str, WebSocket] = {}
        self.agent_metadata: Dict[str, dict] = {}

        # Roundtable runtime state
        self.current_topic_data: dict = self.topics[0] if self.topics else {}
        self.history: List[dict] = []
        self.active_speaker: Optional[str] = None
        self.is_running: bool = False
        self.max_turns: int = 6
        self.current_round: int = 0
        self._turn_event: Optional[asyncio.Event] = None
        self._latest_turn: Optional[MicPassTurn] = None

    async def register_agent(self, websocket: WebSocket, name: str, voice: str, persona: str) -> None:
        """Register a new participant agent."""
        self.active_connections[name] = websocket
        self.agent_metadata[name] = {
            "name": name,
            "voice": voice,
            "persona": persona,
        }
        self.dashboard.register_team(name, voice, persona)
        logger.info(f"Team registered: {name} ({voice})")

        # Send ACK
        await websocket.send_text(json.dumps({
            "type": MessageType.REGISTER_ACK.value,
            "team_name": name,
            "topic": self.current_topic_data.get("topic", ""),
            "connected_peers": [n for n in self.active_connections.keys() if n != name],
        }))

    def unregister_agent(self, name: str) -> None:
        """Remove a disconnected participant agent."""
        if name in self.active_connections:
            del self.active_connections[name]
        if name in self.agent_metadata:
            del self.agent_metadata[name]
        self.dashboard.unregister_team(name)
        logger.info(f"Team disconnected: {name}")

    async def broadcast(self, payload: dict) -> None:
        """Broadcast a message to all connected attendee laptops."""
        msg = json.dumps(payload)
        for name, ws in list(self.active_connections.items()):
            try:
                await ws.send_text(msg)
            except Exception:
                pass

    def get_eligible_peers(self, current_speaker: str) -> List[str]:
        """Get list of connected peers eligible to receive the mic."""
        return [name for name in self.active_connections.keys() if name != current_speaker]

    async def run_roundtable(
        self,
        topic_id: Optional[str] = None,
        first_speaker: Optional[str] = None,
        max_turns: int = 6,
    ) -> None:
        """Main orchestrator loop for the live roundtable debate."""
        if self.is_running:
            logger.warning("Roundtable is already running!")
            return

        # Select topic
        if topic_id:
            for t in self.topics:
                if t.get("id") == topic_id:
                    self.current_topic_data = t
                    break
        elif not self.current_topic_data:
            self.current_topic_data = self.topics[0]

        topic_str = self.current_topic_data.get("topic", "AI Evolution")
        intro_script = self.current_topic_data.get("intro_script")

        # Need at least one connected attendee
        connected_names = list(self.active_connections.keys())
        if not connected_names:
            logger.warning("No attendee teams connected. Cannot start debate.")
            return

        # Determine starting speaker
        if not first_speaker or first_speaker not in connected_names:
            suggested = self.current_topic_data.get("suggested_first_speaker")
            if suggested and suggested in connected_names:
                first_speaker = suggested
            else:
                first_speaker = connected_names[0]

        self.is_running = True
        self.max_turns = max_turns
        self.history = []
        self.current_round = 0
        self.dashboard.update_topic(topic_str)
        self.dashboard.set_status("ROUNDTABLE IN PROGRESS")

        logger.info(f"Starting roundtable on: '{topic_str}' with opener {first_speaker}")

        # 1. Announce topic to attendees via WebSocket
        await self.broadcast({
            "type": MessageType.TOPIC_ANNOUNCED.value,
            "topic": topic_str,
            "first_speaker": first_speaker,
        })

        # 2. Moderator speaks out loud on the Host laptop
        self.dashboard.set_active_speaker("Moderator")
        self.dashboard.refresh()
        await self.moderator.introduce_topic(
            topic=topic_str,
            first_speaker=first_speaker,
            custom_intro=intro_script,
        )

        current_speaker = first_speaker

        # 3. Turn-taking cycle
        while self.is_running and self.current_round < self.max_turns:
            self.current_round += 1
            self.active_speaker = current_speaker
            self.dashboard.set_active_speaker(current_speaker)
            self.dashboard.refresh()

            ws = self.active_connections.get(current_speaker)
            if not ws:
                # Speaker dropped or offline
                logger.warning(f"Speaker {current_speaker} disconnected. Moderator intervening.")
                available = [n for n in self.active_connections.keys() if n != current_speaker]
                if not available:
                    logger.warning("No more participants connected. Stopping roundtable.")
                    break
                next_target = available[0]
                await self.moderator.intervene("Participant disconnected", current_speaker, next_target)
                current_speaker = next_target
                continue

            # Grant turn to current speaker
            eligible_peers = self.get_eligible_peers(current_speaker)
            grant = TurnGrant(
                topic=topic_str,
                history=self.history,
                available_agents=eligible_peers,
                round_number=self.current_round,
                time_limit_sec=45,
            )

            self._turn_event = asyncio.Event()
            self._latest_turn = None

            try:
                await ws.send_text(grant.model_dump_json())
            except Exception as e:
                logger.error(f"Failed to send turn grant to {current_speaker}: {e}")
                self.unregister_agent(current_speaker)
                continue

            # Wait for attendee to speak out loud and submit their mic pass (timeout 45s)
            try:
                await asyncio.wait_for(self._turn_event.wait(), timeout=45.0)
            except asyncio.TimeoutError:
                logger.warning(f"Timeout waiting for {current_speaker} to pass mic.")
                available = self.get_eligible_peers(current_speaker)
                if not available:
                    break
                next_target = available[0]
                await self.moderator.intervene("Turn timeout", current_speaker, next_target)
                current_speaker = next_target
                continue

            # Turn successfully received
            turn = self._latest_turn
            if not turn:
                continue

            # Record in history and update dashboard
            turn_record = {
                "speaker": turn.speaker,
                "spoken_text": turn.spoken_text,
                "next_agent": turn.next_agent,
                "round": self.current_round,
            }
            self.history.append(turn_record)
            self.dashboard.record_turn(turn.speaker, turn.spoken_text, turn.next_agent)
            self.dashboard.refresh()

            # Broadcast turn to all attendees
            await self.broadcast({
                "type": MessageType.TURN_BROADCAST.value,
                "turn": turn.model_dump(),
            })

            # Determine next speaker
            candidate_next = turn.next_agent
            if candidate_next in self.active_connections and candidate_next != current_speaker:
                current_speaker = candidate_next
            else:
                # Next agent invalid or self-pass: Moderator intervenes
                available = self.get_eligible_peers(current_speaker)
                if available:
                    next_target = available[0]
                    await self.moderator.intervene("Invalid mic pass", current_speaker, next_target)
                    current_speaker = next_target
                else:
                    break

        # 4. Conclude the debate
        self.active_speaker = "Moderator"
        self.dashboard.set_active_speaker("Moderator")
        self.dashboard.set_status("DEBATE CONCLUDED")
        self.dashboard.refresh()

        await self.moderator.summarize_and_conclude(topic_str, self.history)
        await self.broadcast({
            "type": MessageType.ROUNDTABLE_FINISHED.value,
            "topic": topic_str,
            "total_turns": len(self.history),
        })

        self.is_running = False
        self.active_speaker = None
        self.dashboard.set_active_speaker(None)
        self.dashboard.refresh()
        logger.info("Roundtable finished successfully.")

    def handle_turn_complete(self, speaker: str, data: dict) -> None:
        """Called when an attendee WebSocket returns their turn submission."""
        raw_turn = data.get("turn") or data
        available = self.get_eligible_peers(speaker)

        if isinstance(raw_turn, dict):
            turn = parse_agent_turn(json.dumps(raw_turn), speaker, available)
        else:
            turn = parse_agent_turn(str(raw_turn), speaker, available)

        self._latest_turn = turn
        if self._turn_event:
            self._turn_event.set()


# Global coordinator instance
coordinator = RoundtableCoordinator()


@app.get("/", response_class=HTMLResponse)
async def index():
    """Simple status webpage for the room."""
    teams_html = "".join(
        f"<li><b>{name}</b> (Voice: {meta['voice']}) - {meta['persona']}</li>"
        for name, meta in coordinator.agent_metadata.items()
    ) or "<i>No teams connected yet</i>"

    local_ip = get_local_ip()
    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Pass-The-Mic: Coordinator Server</title>
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <style>
            body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; max-width: 800px; margin: 40px auto; padding: 20px; line-height: 1.6; background: #0f172a; color: #f8fafc; }}
            h1 {{ color: #38bdf8; }}
            .card {{ background: #1e293b; padding: 20px; border-radius: 8px; margin-bottom: 20px; border: 1px solid #334155; }}
            .highlight {{ color: #4ade80; font-family: monospace; font-size: 1.1em; }}
            ul {{ list-style-type: square; padding-left: 20px; }}
            button {{ background: #38bdf8; color: #0f172a; border: none; padding: 10px 20px; font-weight: bold; border-radius: 6px; cursor: pointer; }}
            button:hover {{ background: #0284c7; color: white; }}
        </style>
    </head>
    <body>
        <h1>🎙️ Pass-The-Mic Coordinator</h1>
        <div class="card">
            <h3>Host Network Information</h3>
            <p>WebSocket Address for Attendees: <span class="highlight">ws://{local_ip}:{coordinator.port}/ws</span></p>
            <p>Active Status: <b>{'DEBATE RUNNING' if coordinator.is_running else 'WAITING'}</b></p>
            <p>Current Speaker: <span class="highlight">{coordinator.active_speaker or 'None'}</span></p>
        </div>
        <div class="card">
            <h3>Connected Teams ({len(coordinator.active_connections)})</h3>
            <ul>{teams_html}</ul>
        </div>
        <div class="card">
            <h3>Current Topic</h3>
            <p><b>{coordinator.current_topic_data.get('title', 'Default')}</b>: {coordinator.current_topic_data.get('topic', '')}</p>
        </div>
    </body>
    </html>
    """


@app.get("/status")
async def get_status():
    """Return JSON status of the coordinator server."""
    return {
        "is_running": coordinator.is_running,
        "active_speaker": coordinator.active_speaker,
        "connected_teams": list(coordinator.agent_metadata.keys()),
        "round": coordinator.current_round,
        "topic": coordinator.current_topic_data,
        "turns_completed": len(coordinator.history),
    }


@app.get("/topics")
async def get_topics():
    """Return all available curated debate topics."""
    return coordinator.topics


class StartRequest(BaseModel):
    topic_id: Optional[str] = None
    first_speaker: Optional[str] = None
    max_turns: int = 6


@app.post("/start")
async def start_roundtable(req: StartRequest):
    """Trigger the start of the voice debate."""
    if coordinator.is_running:
        raise HTTPException(status_code=400, detail="Roundtable already running")

    asyncio.create_task(coordinator.run_roundtable(
        topic_id=req.topic_id,
        first_speaker=req.first_speaker,
        max_turns=req.max_turns,
    ))
    return {"status": "started", "topic_id": req.topic_id}


@app.post("/stop")
async def stop_roundtable():
    """Stop the ongoing debate."""
    coordinator.is_running = False
    if coordinator._turn_event:
        coordinator._turn_event.set()
    return {"status": "stopped"}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """Main WebSocket endpoint connecting attendee laptops to the host."""
    await websocket.accept()
    team_name = None

    try:
        # Await registration message
        raw_msg = await websocket.receive_text()
        data = json.loads(raw_msg)

        if data.get("type") == MessageType.REGISTER.value:
            team_name = data.get("name", f"Team_{len(coordinator.active_connections) + 1}")
            voice = data.get("voice", "en-US-AriaNeural")
            persona = data.get("persona", "")
            await coordinator.register_agent(websocket, team_name, voice, persona)
        else:
            await websocket.close(code=1003, reason="Registration required first")
            return

        # Continuous message listening loop
        while True:
            text = await websocket.receive_text()
            msg = json.loads(text)
            msg_type = msg.get("type")

            if msg_type == MessageType.TURN_COMPLETE.value:
                coordinator.handle_turn_complete(team_name, msg)
            elif msg_type == MessageType.PING.value:
                await websocket.send_text(json.dumps({"type": MessageType.PONG.value}))

    except WebSocketDisconnect:
        if team_name:
            coordinator.unregister_agent(team_name)
    except Exception as e:
        logger.error(f"WebSocket error for {team_name}: {e}")
        if team_name:
            coordinator.unregister_agent(team_name)


def main():
    """CLI Entrypoint to run the moderator coordinator server with live dashboard."""
    import argparse
    parser = argparse.ArgumentParser(description="Pass-The-Mic Moderator Host Server")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind server (default: 8000)")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host address to bind (default: 0.0.0.0)")
    parser.add_argument("--no-dashboard", action="store_true", help="Disable rich live dashboard")
    args = parser.parse_args()

    local_ip = get_local_ip()
    print("=" * 65)
    print("  PASS-THE-MIC: MULTI-AGENT VOICE ROUNDTABLE (HOST SERVER)")
    print(f"  Local LAN IP: {local_ip}:{args.port}")
    print(f"  WebSocket URI: ws://{local_ip}:{args.port}/ws")
    print("=" * 65)

    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
