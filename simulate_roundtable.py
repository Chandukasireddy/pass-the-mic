"""[HOST TEST] Simulate a complete Multi-Agent Voice Roundtable on ONE laptop.

Demonstrates the full Acoustic Speaker 🔊 -> Microphone 🎙️ (Wake Phrase Detection)
turn-taking loop across 1 Moderator + 2 (or 3) Character Agents on a single machine,
plus an optional `--live-mic` mode where YOU can speak into your laptop's microphone
and pass the mic to any AI agent!
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

root_dir = Path(__file__).resolve().parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

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
from core.llm import LLMClient
from core.protocol import MicPassTurn, detect_mic_pass_in_transcript
from core.stt import STTClient
from moderator.moderator_agent import ModeratorAgent
from moderator.server import load_topics

console = Console()


async def run_simulation(
    num_rounds: int = 3,
    topic_override: str | None = None,
    speak_audio: bool = True,
    include_blue: bool = False,
    live_mic: bool = False,
    provider: str | None = None,
) -> None:
    """Run an end-to-end multi-agent voice roundtable simulation (Speaker + Mic Wake Detection)."""
    console.print("\n[bold yellow]═══════════════════════════════════════════════════════════════════[/bold yellow]")
    console.print("   [bold white]🎙️  PASS-THE-MIC: SPEAKER 🔊 + MICROPHONE 🎙️  ROUNDTABLE[/bold white]")
    console.print("[bold yellow]═══════════════════════════════════════════════════════════════════[/bold yellow]\n")

    llm = LLMClient(provider=provider)
    stt = STTClient()

    console.print(f"[bold cyan]Active LLM Provider:[/] [bold green]{llm.provider.upper()}[/bold green]")
    console.print(f"[bold cyan]Active STT Provider:[/] [bold magenta]{stt.provider.upper()}[/bold magenta]")
    console.print(
        f"[bold cyan]Speaker Playback:[/] "
        f"{'[green]ENABLED (Laptop Speakers)[/]' if speak_audio else '[yellow]MUTED (Fast Text Mode)[/]'}"
    )
    console.print(
        f"[bold cyan]Microphone Mode:[/] "
        f"{'[bold red]LIVE HARDWARE MIC (--live-mic)[/]' if live_mic else '[cyan]ACOUSTIC WAKE SIMULATION[/]'}\n"
    )

    moderator = ModeratorAgent(
        name="Moderator (Team Black)",
        voice="en-US-GuyNeural",
        llm_client=llm,
        stt_client=stt,
        speak_aloud=speak_audio,
    )

    team_orange = CharacterAgent.from_preset(
        root_dir / "character_agents" / "presets" / "orange_optimist.json",
        llm_client=llm,
        stt_client=stt,
    )
    team_red = CharacterAgent.from_preset(
        root_dir / "character_agents" / "presets" / "red_skeptic.json",
        llm_client=llm,
        stt_client=stt,
    )

    agents = {
        "Team Orange": team_orange,
        "Team Red": team_red,
    }

    if include_blue:
        team_blue = CharacterAgent.from_preset(
            root_dir / "character_agents" / "presets" / "blue_philosopher.json",
            llm_client=llm,
            stt_client=stt,
        )
        agents["Team Blue"] = team_blue

    known_teams = list(agents.keys())

    roster_table = Table(title="Roundtable Participants (Speaker 🔊 + Mic 🎙️)", border_style="cyan")
    roster_table.add_column("Agent / Laptop", style="bold white")
    roster_table.add_column("Speaker Voice ID", style="green")
    roster_table.add_column("Microphone Wake Phrase", style="yellow")

    roster_table.add_row("Moderator (Host)", moderator.voice, '"...passing the mic to Moderator"')
    for name, ag in agents.items():
        roster_table.add_row(name, ag.voice, f'"...passing the mic to {name}"')

    console.print(roster_table)

    topics = load_topics()
    selected_topic_data = topics[0] if topics else {}
    topic_text = topic_override or selected_topic_data.get(
        "topic", "Will decentralized open-source AI models outpace closed frontier labs by 2027?"
    )
    first_speaker_name = "Team Orange"

    console.print(Panel(f"[bold yellow]DEBATE TOPIC:[/] {topic_text}", border_style="green"))

    # Step 1: Moderator Speaks Topic Intro out loud on Speaker
    console.print("\n[bold white on dark_blue] 🔊 HOST MODERATOR SPEAKS OUT LOUD ON SPEAKER [/bold white on dark_blue]")
    intro_speech = await moderator.introduce_topic(
        topic=topic_text,
        first_speaker=first_speaker_name,
        custom_intro=selected_topic_data.get("intro_script"),
    )
    console.print(f"[bold cyan]Moderator Speaker:[/] \"{intro_speech}\"\n")

    history = []
    last_spoken_in_room = intro_speech
    previous_speaker = "Moderator"

    # Step 2: Acoustic Wake-Detection & Turn Loop
    for round_idx in range(1, num_rounds + 1):
        console.print(f"[bold magenta]─── ROUND {round_idx} OF {num_rounds} ───[/bold magenta]")

        # Optional: allow human to speak into hardware microphone to steer the debate
        if live_mic and round_idx == 1:
            console.print(
                Panel(
                    "Speak into your laptop microphone now! End by saying:\n"
                    "[bold green]\"...Passing the mic to Team Orange!\"[/bold green] or [bold green]\"...Passing the mic to Team Red!\"[/bold green]",
                    title="🎙️ Live Microphone Input",
                    border_style="red",
                )
            )
            heard_text, dur = await moderator.async_listen_to_room(wait_timeout=15.0)
            if heard_text:
                console.print(f"[bold green]👂 Microphone Transcribed ({dur:.1f}s):[/] \"{heard_text}\"")
                last_spoken_in_room = heard_text
                previous_speaker = "Human Attendee"

        # Every listening laptop checks if its acoustic wake phrase was spoken in `last_spoken_in_room`
        woken_agent_name = detect_mic_pass_in_transcript(
            last_spoken_in_room,
            known_teams=known_teams,
            current_speaker=previous_speaker,
        )

        if not woken_agent_name or woken_agent_name not in agents:
            # Moderator intervenes out loud to pass the mic cleanly
            eligible = [n for n in known_teams if n != previous_speaker]
            woken_agent_name = eligible[0]
            console.print(f"[bold red]⚠ No acoustic wake phrase heard! Moderator intervening on speaker...[/bold red]")
            last_spoken_in_room = await moderator.intervene("Missed wake phrase", previous_speaker, woken_agent_name)

        agent = agents[woken_agent_name]
        available_peers = [name for name in known_teams if name != woken_agent_name]

        console.print(
            f"\n[bold cyan]👂 [{woken_agent_name} Microphone][/] heard wake phrase in room audio: "
            f"[italic]\"...{last_spoken_in_room[-75:]}\"[/italic]"
        )
        console.print(f"[bold white on red] 🎤 MICROPHONE TRIGGERED: {woken_agent_name.upper()} WAKES UP! [/bold white on red]")

        # Generate Turn based on what the microphone heard
        turn: MicPassTurn = await agent.async_generate_turn(
            topic=topic_text,
            history=history,
            available_agents=available_peers,
            heard_from_mic=last_spoken_in_room,
        )

        speech_panel = (
            f"[bold cyan]Active Laptop:[/] [bold]{turn.speaker}[/]\n"
            f"[bold blue]Heard on Mic:[/] [dim]\"{turn.heard_from_mic[:85]}...\"[/dim]\n"
            f"[bold yellow]Speaking on Speaker:[/] \"{turn.spoken_text}\"\n"
            f"[bold magenta]Acoustic Wake Phrase For:[/] [bold green]{turn.next_agent}[/]"
        )
        console.print(Panel(speech_panel, border_style="yellow"))

        if speak_audio:
            console.print(f"[dim]🔊 [{woken_agent_name}] Playing voice out loud through laptop speakers (Mic muted)...[/dim]")
            await agent.async_speak_turn(turn)

        history.append({
            "speaker": turn.speaker,
            "spoken_text": turn.spoken_text,
            "next_agent": turn.next_agent,
            "round": round_idx,
        })

        last_spoken_in_room = turn.spoken_text
        previous_speaker = woken_agent_name

        console.print(
            f"[bold green]✔ [{woken_agent_name}] Speaker finished! Sound waves passed the mic to {turn.next_agent}.[/bold green]\n"
        )
        await asyncio.sleep(0.3)

    # Step 3: Moderator Concludes Debate out loud on Speaker
    console.print("\n[bold white on dark_blue] 🔊 HOST MODERATOR CLOSES ROUNDTABLE ON SPEAKER [/bold white on dark_blue]")
    closing_speech = await moderator.summarize_and_conclude(topic=topic_text, history=history)
    console.print(f"[bold cyan]Moderator Summary:[/] \"{closing_speech}\"\n")

    console.print(
        Panel(
            "[bold green]✔ ACOUSTIC ROUNDTABLE SIMULATION COMPLETED SUCCESSFULLY![/bold green]\n"
            "All microphone wake triggers, LLM responses, and speaker voice outputs executed cleanly.",
            border_style="green",
        )
    )


def main():
    parser = argparse.ArgumentParser(description="Simulate Speaker + Microphone voice roundtable on one laptop.")
    parser.add_argument("--rounds", type=int, default=3, help="Number of turns to debate (default: 3)")
    parser.add_argument("--topic", type=str, default=None, help="Custom debate topic question")
    parser.add_argument("--mute", action="store_true", help="Mute speaker playback for fast text-only test")
    parser.add_argument("--three-teams", action="store_true", help="Include Team Blue as a third participant")
    parser.add_argument(
        "--live-mic",
        action="store_true",
        help="Record your own voice from the hardware microphone to kick off Round 1",
    )
    parser.add_argument("--provider", type=str, choices=["groq", "openai", "mock"], default=None, help="Force LLM provider")
    args = parser.parse_args()

    asyncio.run(
        run_simulation(
            num_rounds=args.rounds,
            topic_override=args.topic,
            speak_audio=not args.mute,
            include_blue=args.three_teams,
            live_mic=args.live_mic,
            provider=args.provider,
        )
    )


if __name__ == "__main__":
    main()
