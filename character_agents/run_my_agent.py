"""Attendee Client Script to Connect Your Voice Agent to the Host.

Run this script during the meetup to connect your laptop to the Host's
Pass-The-Mic server over local Wi-Fi.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

# Add project root to sys.path so script can run from any working directory
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
import websockets
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from character_agents.agent_builder import CharacterAgent
from core.llm import LLMClient
from core.protocol import MessageType, MicPassTurn

load_dotenv()
console = Console()


async def run_agent_loop(
    preset_path: str,
    host: str,
    port: int,
    team_override: str | None = None,
    voice_override: str | None = None,
    provider_override: str | None = None,
) -> None:
    """Connect attendee voice agent to the host coordinator server."""
    # 1. Initialize Character Agent
    overrides = {}
    if team_override:
        overrides["name"] = team_override
    if voice_override:
        overrides["voice"] = voice_override

    try:
        agent = CharacterAgent.from_preset(preset_path, **overrides)
    except FileNotFoundError:
        console.print(f"[bold red]Error:[/] Could not locate preset at {preset_path}")
        return

    if provider_override:
        agent.llm_client = LLMClient(provider=provider_override)

    # 2. Display Status Card
    uri = f"ws://{host}:{port}/ws"
    status_text = Text()
    status_text.append(f"Team: ", style="bold white")
    status_text.append(f"{agent.name}\n", style="bold green")
    status_text.append(f"Voice: ", style="bold white")
    status_text.append(f"{agent.voice}\n", style="yellow")
    status_text.append(f"LLM Provider: ", style="bold white")
    status_text.append(f"{agent.llm_client.provider.upper()}\n", style="bold cyan")
    status_text.append(f"Target Server: ", style="bold white")
    status_text.append(f"{uri}\n", style="bold magenta")
    status_text.append(f"Persona: {agent.persona}", style="italic dim")

    console.print(Panel(status_text, title="🎙️ Pass-The-Mic Attendee Client", border_style="green"))

    # 3. Connection & Turn Listening Loop
    while True:
        try:
            console.print(f"[dim]Connecting to Host at {uri}...[/dim]")
            async with websockets.connect(uri) as ws:
                console.print(f"[bold green]✔ Connected to Host Server![/bold green]")

                # Send Registration Packet
                reg_payload = {
                    "type": MessageType.REGISTER.value,
                    "name": agent.name,
                    "voice": agent.voice,
                    "persona": agent.persona,
                }
                await ws.send(json.dumps(reg_payload))

                # Handle incoming events
                async for raw_message in ws:
                    msg = json.loads(raw_message)
                    msg_type = msg.get("type")

                    if msg_type == MessageType.REGISTER_ACK.value:
                        console.print(f"[bold cyan]Registration confirmed by Host![/bold cyan]")
                        current_topic = msg.get("topic")
                        if current_topic:
                            console.print(f"[dim]Current Room Topic: {current_topic}[/dim]")

                    elif msg_type == MessageType.TOPIC_ANNOUNCED.value:
                        topic = msg.get("topic", "")
                        first_speaker = msg.get("first_speaker", "")
                        console.print(f"\n[bold yellow]══════ NEW TOPIC ANNOUNCED ══════[/bold yellow]")
                        console.print(f"[bold white]{topic}[/bold white]")
                        console.print(f"[dim]First Speaker: {first_speaker}[/dim]\n")

                    elif msg_type == MessageType.YOUR_TURN.value:
                        # THE MIC IS PASSED TO THIS LAPTOP!
                        topic = msg.get("topic", "")
                        history = msg.get("history", [])
                        available_peers = msg.get("available_agents", [])
                        round_num = msg.get("round_number", 1)

                        console.print("\n" + "=" * 50)
                        console.print(f"[bold white on red] 🎤 THE MIC IS PASSED TO {agent.name.upper()}! [/bold white on red]")
                        console.print(f"[dim]Round {round_num} | Generating response...[/dim]")

                        # Generate turn via LLM
                        turn: MicPassTurn = await agent.async_generate_turn(
                            topic=topic,
                            history=history,
                            available_agents=available_peers,
                        )

                        turn_box = (
                            f"[bold cyan]Speaker:[/] {turn.speaker}\n"
                            f"[bold yellow]Spoken Script:[/] \"{turn.spoken_text}\"\n"
                            f"[bold magenta]Passing Mic To:[/] [bold]{turn.next_agent}[/]"
                        )
                        console.print(Panel(turn_box, title="[bold green]Speaking Out Loud Now[/bold green]", border_style="yellow"))

                        # Speak out loud through the laptop's speakers (blocking playback)
                        await agent.async_speak_turn(turn)

                        # Submit completed turn back to the host server
                        submission = {
                            "type": MessageType.TURN_COMPLETE.value,
                            "turn": turn.model_dump(),
                        }
                        await ws.send(json.dumps(submission))
                        console.print(f"[bold green]✔ Turn complete! Passed the mic to {turn.next_agent}.[/bold green]\n" + "=" * 50 + "\n")

                    elif msg_type == MessageType.TURN_BROADCAST.value:
                        turn_data = msg.get("turn", {})
                        spk = turn_data.get("speaker", "Unknown")
                        nxt = turn_data.get("next_agent", "Next")
                        txt = turn_data.get("spoken_text", "")
                        if spk != agent.name:
                            console.print(f"[bold cyan]{spk}[/] [dim]spoke and passed mic to[/] [bold magenta]{nxt}[/]: \"{txt[:80]}...\"")

                    elif msg_type == MessageType.ROUNDTABLE_FINISHED.value:
                        console.print("\n[bold magenta]🏁 ROUNDTABLE FINISHED — Great job everyone![/bold magenta]\n")

        except (websockets.ConnectionClosedError, websockets.ConnectionClosedOK):
            console.print("[yellow]Connection to host closed. Retrying in 3 seconds...[/yellow]")
            await asyncio.sleep(3)
        except ConnectionRefusedError:
            console.print(f"[red]Could not connect to {uri}. Is the host server running? Retrying in 3s...[/red]")
            await asyncio.sleep(3)
        except KeyboardInterrupt:
            console.print("\n[dim]Disconnected by attendee (Ctrl+C). Goodbye![/dim]")
            break
        except Exception as e:
            console.print(f"[red]Unexpected error: {e}. Retrying in 3s...[/red]")
            await asyncio.sleep(3)


def main():
    parser = argparse.ArgumentParser(description="Run attendee voice agent and connect to the host.")
    parser.add_argument(
        "--preset",
        type=str,
        default="presets/orange_optimist.json",
        help="Path or name of character preset JSON (default: presets/orange_optimist.json)",
    )
    parser.add_argument(
        "--host",
        type=str,
        default=os.getenv("ROUNDTABLE_HOST", "127.0.0.1"),
        help="Host server IP address (default: 127.0.0.1 or ROUNDTABLE_HOST)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.getenv("ROUNDTABLE_PORT", "8000")),
        help="Host server port (default: 8000)",
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Override team name",
    )
    parser.add_argument(
        "--voice",
        type=str,
        default=None,
        help="Override voice ID",
    )
    parser.add_argument(
        "--provider",
        type=str,
        choices=["groq", "openai", "mock"],
        default=None,
        help="Force specific LLM provider (groq, openai, mock)",
    )
    args = parser.parse_args()

    asyncio.run(
        run_agent_loop(
            preset_path=args.preset,
            host=args.host,
            port=args.port,
            team_override=args.name,
            voice_override=args.voice,
            provider_override=args.provider,
        )
    )


if __name__ == "__main__":
    main()
