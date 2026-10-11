"""Attendee Autonomous Voice Agent Runner (Acoustic Microphone + Speaker Loop).

Run this script on your laptop during the meetup!
Your laptop will:
1. LISTEN continuously through its physical MICROPHONE (using VAD + Speech-to-Text)
2. FOLLOW the live room conversation and detect when someone says:
   "Passing the mic to <Your Team Name>!"
3. THINK of a sharp response using your persona and LLM
4. SPEAK out loud through your laptop's physical SPEAKERS and pass the mic to the next team!
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import List

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from character_agents.agent_builder import CharacterAgent
from core.llm import LLMClient
from core.protocol import MessageType, MicPassTurn
from core.stt import STTClient

load_dotenv()
console = Console()

DEFAULT_ROOM_TEAMS = ["Team Orange", "Team Red", "Team Blue", "Moderator"]


async def run_acoustic_agent_loop(
    agent: CharacterAgent,
    peer_teams: List[str],
    topic: str,
    silence_duration: float = 1.8,
) -> None:
    """Main over-the-air acoustic loop: Listen on Mic -> Detect Wake Phrase -> Speak on Speaker."""
    # Calibrate microphone to room ambient noise
    console.print("[dim]Calibrating laptop microphone to ambient room noise (1s)...[/dim]")
    threshold = agent.calibrate_mic(duration_sec=1.0)

    status_text = Text()
    status_text.append("Mode: ", style="bold white")
    status_text.append("ACOUSTIC (Speaker 🔊 + Microphone 🎙️ — Zero Wi-Fi Needed)\n", style="bold green")
    status_text.append("Team Name: ", style="bold white")
    status_text.append(f"{agent.name}\n", style="bold cyan")
    status_text.append("Speaker Voice: ", style="bold white")
    status_text.append(f"{agent.voice}\n", style="yellow")
    status_text.append("Mic STT Engine: ", style="bold white")
    status_text.append(f"{agent.stt_client.provider.upper()} (VAD threshold={threshold})\n", style="magenta")
    status_text.append("LLM Engine: ", style="bold white")
    status_text.append(f"{agent.llm_client.provider.upper()}\n", style="bold cyan")
    status_text.append("Other Teams in Room: ", style="bold white")
    status_text.append(", ".join(peer_teams) + "\n", style="white")
    status_text.append(f"Persona: {agent.persona}", style="italic dim")

    console.print(Panel(status_text, title="🎙️ Pass-The-Mic: Acoustic Voice Agent Active", border_style="green"))
    console.print(
        f"\n[bold cyan]👂 Listening on microphone for wake phrase:[/] "
        f"[bold yellow]\"...passing the mic to {agent.name}\"[/bold yellow] (Ctrl+C to stop)\n"
    )

    history: List[dict] = []
    known_teams = [agent.name] + [t for t in peer_teams if t.lower() != agent.name.lower()]
    round_num = 0

    try:
        while True:
            console.print(f"[dim]🎙️  [{agent.name}] Microphone listening to room...[/dim]")

            def _on_speech_detected():
                console.print("[cyan]   🌊 Voice detected in room! Recording speech until silence...[/cyan]")

            transcript, duration = await agent.async_listen_on_mic(
                silence_duration=silence_duration,
                max_duration=25.0,
                wait_timeout=30.0,
                on_speech_start=_on_speech_detected,
            )

            if not transcript:
                continue

            # Detect who the speaker in the room passed the mic to
            recipient = agent.detect_recipient(transcript, known_teams=known_teams)

            console.print(
                Panel(
                    f"[bold white]\"{transcript}\"[/bold white]\n"
                    f"[dim]Recorded Duration: {duration:.1f}s | Detected Mic Recipient: {recipient or 'None'}[/dim]",
                    title="👂 Heard via Laptop Microphone (STT)",
                    border_style="blue",
                )
            )

            # Save what was heard in local debate memory
            history.append({
                "speaker": "Room Speaker",
                "spoken_text": transcript,
            })

            # Check if the mic was passed to THIS laptop!
            if agent.check_wake_trigger(transcript, known_teams=known_teams):
                round_num += 1
                console.print("\n" + "=" * 65)
                console.print(
                    f"[bold white on red] 🎤 WAKE TRIGGER DETECTED! THE MIC IS PASSED TO {agent.name.upper()}! [/bold white on red]"
                )
                console.print(f"[dim]Turn #{round_num} | Thinking of response to what we just heard...[/dim]")

                eligible_peers = [t for t in peer_teams if t.lower() != agent.name.lower() and t.lower() != "moderator"]
                if not eligible_peers:
                    eligible_peers = ["Team Red" if "orange" in agent.name.lower() else "Team Orange"]

                turn: MicPassTurn = await agent.async_generate_turn(
                    topic=topic,
                    history=history,
                    available_agents=eligible_peers,
                    heard_from_mic=transcript,
                )

                turn_box = (
                    f"[bold cyan]Speaker:[/] {turn.speaker}\n"
                    f"[bold yellow]Speaking Out Loud:[/] \"{turn.spoken_text}\"\n"
                    f"[bold magenta]Acoustic Handoff To:[/] [bold green]{turn.next_agent}[/]"
                )
                console.print(
                    Panel(turn_box, title="[bold green]🔊 Speaking Out Loud via Laptop Speaker[/bold green]", border_style="yellow")
                )

                # Speak out loud on physical speaker (microphone is automatically locked while speaking)
                await agent.async_speak_turn(turn)

                history.append({
                    "speaker": agent.name,
                    "spoken_text": turn.spoken_text,
                    "next_agent": turn.next_agent,
                })

                console.print(
                    f"[bold green]✔ Finished speaking! Passed the mic over the air to {turn.next_agent}.[/bold green]\n"
                    + "=" * 65
                    + "\n"
                )
            else:
                if recipient:
                    console.print(f"[dim]👉 Mic was passed to {recipient}. Continuing to listen...[/dim]\n")
                else:
                    console.print(f"[dim]   (No mic pass directed at {agent.name} yet. Continuing to listen...)[/dim]\n")

    except KeyboardInterrupt:
        console.print(f"\n[bold yellow]👋 {agent.name} microphone closed (Ctrl+C). Goodbye![/bold yellow]")


async def run_wifi_hybrid_loop(
    agent: CharacterAgent,
    host: str,
    port: int,
) -> None:
    """Optional hybrid mode: coordinates turn triggers over Wi-Fi while speaking out loud."""
    import websockets

    uri = f"ws://{host}:{port}/ws"
    console.print(Panel(f"Connecting to hybrid host at [bold cyan]{uri}[/bold cyan]", title="Wi-Fi Hybrid Mode"))

    while True:
        try:
            async with websockets.connect(uri) as ws:
                await ws.send(json.dumps({
                    "type": MessageType.REGISTER.value,
                    "name": agent.name,
                    "voice": agent.voice,
                    "persona": agent.persona,
                }))
                async for raw_message in ws:
                    msg = json.loads(raw_message)
                    if msg.get("type") == MessageType.YOUR_TURN.value:
                        topic = msg.get("topic", "")
                        history = msg.get("history", [])
                        available_peers = msg.get("available_agents", [])
                        last_heard = history[-1].get("spoken_text", "") if history else ""
                        turn = await agent.async_generate_turn(topic, history, available_peers, heard_from_mic=last_heard)
                        console.print(Panel(turn.spoken_text, title=f"🔊 {agent.name} Speaking"))
                        await agent.async_speak_turn(turn)
                        await ws.send(json.dumps({
                            "type": MessageType.TURN_COMPLETE.value,
                            "turn": turn.model_dump(),
                        }))
        except KeyboardInterrupt:
            break
        except Exception as e:
            console.print(f"[red]Connection error ({e}), retrying in 3s...[/red]")
            await asyncio.sleep(3)


def main():
    parser = argparse.ArgumentParser(
        description="Run your autonomous voice agent (Listens with Microphone, Speaks with Speaker)."
    )
    parser.add_argument(
        "--preset",
        type=str,
        default="presets/orange_optimist.json",
        help="Path or name of character preset JSON (default: presets/orange_optimist.json)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["acoustic", "wifi"],
        default="acoustic",
        help="Roundtable mode: 'acoustic' (pure Mic + Speaker over the air, default) or 'wifi' (hybrid server sync)",
    )
    parser.add_argument(
        "--peers",
        type=str,
        default="Team Orange,Team Red,Team Blue",
        help="Comma-separated list of team names participating in the room",
    )
    parser.add_argument(
        "--topic",
        type=str,
        default="Will decentralized open-source AI models outpace closed frontier labs by 2027?",
        help="Debate topic context",
    )
    parser.add_argument(
        "--silence-duration",
        type=float,
        default=float(os.getenv("MIC_SILENCE_DURATION", "1.8")),
        help="Seconds of silence after speech before microphone finishes recording (default: 1.8)",
    )
    parser.add_argument("--name", type=str, default=None, help="Override team name")
    parser.add_argument("--voice", type=str, default=None, help="Override Edge-TTS voice ID")
    parser.add_argument(
        "--provider",
        type=str,
        choices=["groq", "openai", "mock"],
        default=None,
        help="Force specific LLM provider",
    )
    parser.add_argument(
        "--stt-provider",
        type=str,
        choices=["auto", "groq", "google", "openai"],
        default=None,
        help="Force specific Speech-to-Text provider",
    )
    parser.add_argument("--host", type=str, default=os.getenv("ROUNDTABLE_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.getenv("ROUNDTABLE_PORT", "8000")))
    args = parser.parse_args()

    overrides = {}
    if args.name:
        overrides["name"] = args.name
    if args.voice:
        overrides["voice"] = args.voice

    agent = CharacterAgent.from_preset(args.preset, **overrides)
    if args.provider:
        agent.llm_client = LLMClient(provider=args.provider)
    if args.stt_provider:
        agent.stt_client = STTClient(provider=args.stt_provider)

    peer_list = [p.strip() for p in args.peers.split(",") if p.strip() and p.strip().lower() != agent.name.lower()]

    if args.mode == "wifi":
        asyncio.run(run_wifi_hybrid_loop(agent, host=args.host, port=args.port))
    else:
        asyncio.run(
            run_acoustic_agent_loop(
                agent=agent,
                peer_teams=peer_list,
                topic=args.topic,
                silence_duration=args.silence_duration,
            )
        )


if __name__ == "__main__":
    main()
