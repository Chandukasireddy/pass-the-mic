"""[HOST TEST] Simulate a complete Multi-Agent Voice Roundtable on ONE laptop.

Runs 1 Moderator + 2 (or 3) Sample Character Agents in a single process,
demonstrating live speech synthesis, turn-taking, and mic-passing.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

# Add root directory to path
root_dir = Path(__file__).resolve().parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

# Ensure UTF-8 output encoding across Windows terminals
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from character_agents.agent_builder import CharacterAgent
from core.audio import async_speak
from core.llm import LLMClient
from core.protocol import MicPassTurn
from moderator.moderator_agent import ModeratorAgent
from moderator.server import load_topics

console = Console()


async def run_simulation(
    num_rounds: int = 3,
    topic_override: str | None = None,
    speak_audio: bool = True,
    include_blue: bool = False,
    provider: str | None = None,
) -> None:
    """Run an end-to-end multi-agent voice roundtable simulation."""
    console.print("\n[bold yellow]═══════════════════════════════════════════════════════════════════[/bold yellow]")
    console.print("   [bold white]🎙️  PASS-THE-MIC: MULTI-AGENT VOICE ROUNDTABLE SIMULATION[/bold white]")
    console.print("[bold yellow]═══════════════════════════════════════════════════════════════════[/bold yellow]\n")

    # 1. Initialize LLM Client
    llm = LLMClient(provider=provider)
    console.print(f"[bold cyan]Active LLM Provider:[/] [bold green]{llm.provider.upper()}[/bold green]")
    console.print(f"[bold cyan]Audio Playback:[/] {'[green]ENABLED (Laptop Speakers)[/]' if speak_audio else '[yellow]MUTED (Fast Text Mode)[/]'}\n")

    # 2. Initialize Agents
    moderator = ModeratorAgent(
        name="Moderator (Team Black)",
        voice="en-US-GuyNeural",
        llm_client=llm,
        speak_aloud=speak_audio,
    )

    team_orange = CharacterAgent.from_preset(
        root_dir / "character_agents" / "presets" / "orange_optimist.json",
        llm_client=llm,
    )
    team_red = CharacterAgent.from_preset(
        root_dir / "character_agents" / "presets" / "red_skeptic.json",
        llm_client=llm,
    )

    agents = {
        "Team Orange": team_orange,
        "Team Red": team_red,
    }

    if include_blue:
        team_blue = CharacterAgent.from_preset(
            root_dir / "character_agents" / "presets" / "blue_philosopher.json",
            llm_client=llm,
        )
        agents["Team Blue"] = team_blue

    # Display Roster Table
    roster_table = Table(title="Roundtable Participants", border_style="cyan")
    roster_table.add_column("Agent / Role", style="bold white")
    roster_table.add_column("Voice ID", style="green")
    roster_table.add_column("Persona / Style", style="yellow")

    roster_table.add_row("Moderator (Host)", moderator.voice, "Authoritative broadcaster & timekeeper")
    for name, ag in agents.items():
        roster_table.add_row(name, ag.voice, f"{ag.persona[:60]}...")

    console.print(roster_table)

    # 3. Select Debate Topic
    topics = load_topics()
    selected_topic_data = topics[0] if topics else {}
    topic_text = topic_override or selected_topic_data.get(
        "topic", "Will decentralized open-source AI models outpace closed frontier labs by 2027?"
    )
    first_speaker_name = "Team Orange"

    console.print(Panel(f"[bold yellow]DEBATE TOPIC:[/] {topic_text}", border_style="green"))

    # 4. Step 1: Moderator Introduces Topic
    console.print("\n[bold white on dark_blue] 🎙️ HOST MODERATOR SPEAKS [/bold white on dark_blue]")
    intro_speech = await moderator.introduce_topic(
        topic=topic_text,
        first_speaker=first_speaker_name,
        custom_intro=selected_topic_data.get("intro_script"),
    )
    console.print(f"[bold cyan]Moderator:[/] \"{intro_speech}\"\n")

    # 5. Debate Turns Loop
    history = []
    current_speaker_name = first_speaker_name

    for round_idx in range(1, num_rounds + 1):
        console.print(f"[bold magenta]─── ROUND {round_idx} OF {num_rounds} ───[/bold magenta]")
        agent = agents.get(current_speaker_name)
        if not agent:
            # Fallback
            current_speaker_name = list(agents.keys())[0]
            agent = agents[current_speaker_name]

        available_peers = [name for name in agents.keys() if name != current_speaker_name]

        # Terminal badge
        console.print(f"\n[bold white on red] 🎤 MIC ACTIVE: {current_speaker_name.upper()} [/bold white on red]")
        console.print(f"[dim]Generating response with persona: {agent.persona[:50]}...[/dim]")

        # Generate Turn
        turn: MicPassTurn = await agent.async_generate_turn(
            topic=topic_text,
            history=history,
            available_agents=available_peers,
        )

        # Print Spoken Box
        speech_panel = (
            f"[bold cyan]Speaker:[/] [bold]{turn.speaker}[/]\n"
            f"[bold yellow]Spoken Script:[/] \"{turn.spoken_text}\"\n"
            f"[bold magenta]Passing Mic To:[/] [bold green]{turn.next_agent}[/]"
        )
        console.print(Panel(speech_panel, border_style="yellow"))

        # Audio Playback
        if speak_audio:
            console.print("[dim]🔊 Speaking through laptop speakers...[/dim]")
            await agent.async_speak_turn(turn)

        # Update History
        history.append({
            "speaker": turn.speaker,
            "spoken_text": turn.spoken_text,
            "next_agent": turn.next_agent,
            "round": round_idx,
        })

        console.print(f"[bold green]✔ Mic handed over to {turn.next_agent}![/bold green]\n")
        await asyncio.sleep(0.5)

        # Advance Speaker
        if turn.next_agent in agents and turn.next_agent != current_speaker_name:
            current_speaker_name = turn.next_agent
        else:
            # Moderator Intervention Simulation
            fallback = available_peers[0]
            console.print(f"[bold red]⚠ Anomaly in mic pass! Moderator intervening...[/bold red]")
            await moderator.intervene("Invalid pass", current_speaker_name, fallback)
            current_speaker_name = fallback

    # 6. Step 3: Moderator Concludes Debate
    console.print("\n[bold white on dark_blue] 🎙️ HOST MODERATOR CLOSES ROUNDTABLE [/bold white on dark_blue]")
    closing_speech = await moderator.summarize_and_conclude(topic=topic_text, history=history)
    console.print(f"[bold cyan]Moderator Summary:[/] \"{closing_speech}\"\n")

    console.print(Panel("[bold green]✔ SIMULATION COMPLETED SUCCESSFULLY![/bold green]\nAll turns, voice syntheses, and mic passes executed without error.", border_style="green"))


def main():
    parser = argparse.ArgumentParser(description="Simulate multi-agent voice roundtable on one laptop.")
    parser.add_argument("--rounds", type=int, default=3, help="Number of turns to debate (default: 3)")
    parser.add_argument("--topic", type=str, default=None, help="Custom debate topic question")
    parser.add_argument("--mute", action="store_true", help="Mute audio playback for fast text-only test")
    parser.add_argument("--three-teams", action="store_true", help="Include Team Blue as a third participant")
    parser.add_argument("--provider", type=str, choices=["groq", "openai", "mock"], default=None, help="Force LLM provider")
    args = parser.parse_args()

    asyncio.run(
        run_simulation(
            num_rounds=args.rounds,
            topic_override=args.topic,
            speak_audio=not args.mute,
            include_blue=args.three_teams,
            provider=args.provider,
        )
    )


if __name__ == "__main__":
    main()
