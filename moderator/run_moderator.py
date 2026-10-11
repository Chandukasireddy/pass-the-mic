"""Acoustic Host Moderator Runner (Speaker 🔊 + Microphone 🎙️).

Run this on the Host's laptop during the meetup!
The Host laptop will:
1. SPEAK the debate topic and opening cue out loud via its SPEAKER ("Passing the mic to Team Orange!")
2. LISTEN through its physical MICROPHONE as attendee laptops speak out loud in the room
3. TRANSCRIBE each turn live, track who holds the mic, and intervene out loud if a handoff is missed
4. CONCLUDE and summarize the debate out loud once all rounds complete!
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path
from typing import List, Optional

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from core.llm import LLMClient
from core.stt import STTClient
from core.tts import get_voice_for_team
from moderator.dashboard import RoundtableDashboard
from moderator.moderator_agent import ModeratorAgent
from moderator.server import load_topics

console = Console()


async def run_acoustic_moderator(
    teams: List[str],
    num_rounds: int = 4,
    topic_id: Optional[str] = None,
    topic_override: Optional[str] = None,
    silence_duration: float = 1.8,
    turn_timeout: float = 25.0,
    provider: Optional[str] = None,
    stt_provider: Optional[str] = None,
) -> None:
    """Orchestrate a live in-person roundtable using the Host's Speaker and Microphone."""
    llm = LLMClient(provider=provider)
    stt = STTClient(provider=stt_provider)
    moderator = ModeratorAgent(
        name="Moderator (Team Black)",
        voice="en-US-GuyNeural",
        llm_client=llm,
        stt_client=stt,
        speak_aloud=True,
    )

    console.print("\n[bold yellow]═══════════════════════════════════════════════════════════════════[/bold yellow]")
    console.print("   [bold white]🎙️  PASS-THE-MIC: ACOUSTIC HOST MODERATOR (SPEAKER + MIC)[/bold white]")
    console.print("[bold yellow]═══════════════════════════════════════════════════════════════════[/bold yellow]\n")

    console.print("[dim]Calibrating Host microphone to ambient room noise (1s)...[/dim]")
    threshold = moderator.calibrate_mic(duration_sec=1.0)

    # Select topic
    topics = load_topics()
    selected_topic = topics[0] if topics else {}
    if topic_id:
        for t in topics:
            if t.get("id") == topic_id:
                selected_topic = t
                break

    topic_str = topic_override or selected_topic.get(
        "topic", "Will decentralized open-source AI models outpace closed frontier labs by 2027?"
    )
    first_speaker = selected_topic.get("suggested_first_speaker", teams[0])
    if first_speaker not in teams:
        first_speaker = teams[0]

    # Display room roster
    roster = Table(title="Participating Acoustic Teams in Room", border_style="cyan")
    roster.add_column("Team Name", style="bold white")
    roster.add_column("Expected Voice ID", style="green")
    roster.add_column("Channel", style="yellow")
    roster.add_row("Moderator (Host)", moderator.voice, "Speaker 🔊 + Mic 🎙️ (Coordinator)")
    for t_name in teams:
        roster.add_row(t_name, get_voice_for_team(t_name), "Speaker 🔊 + Mic 🎙️ (Attendee Laptop)")
    console.print(roster)

    console.print(
        Panel(
            f"[bold yellow]DEBATE TOPIC:[/] {topic_str}\n"
            f"[bold cyan]STT Engine:[/] {stt.provider.upper()} (VAD threshold={threshold}) | "
            f"[bold cyan]Planned Rounds:[/] {num_rounds}",
            border_style="green",
        )
    )

    # 1. Moderator speaks topic introduction out loud through Host Speaker
    console.print("\n[bold white on dark_blue] 🔊 HOST MODERATOR SPEAKING INTRO ON SPEAKER [/bold white on dark_blue]")
    intro_text = await moderator.introduce_topic(
        topic=topic_str,
        first_speaker=first_speaker,
        custom_intro=selected_topic.get("intro_script"),
    )
    console.print(f"[bold cyan]Moderator Spoke:[/] \"{intro_text}\"\n")

    history: List[dict] = []
    current_speaker = first_speaker

    # 2. Acoustic Listening & Turn-Tracking Loop
    try:
        for round_idx in range(1, num_rounds + 1):
            console.print(f"[bold magenta]─── ROUND {round_idx} OF {num_rounds} (Active Floor: {current_speaker}) ───[/bold magenta]")
            console.print(
                f"[dim]👂 Host Microphone listening for {current_speaker}'s laptop speaker (timeout={turn_timeout:.0f}s)...[/dim]"
            )

            def _on_speech_start():
                console.print(f"[cyan]   🌊 Hearing {current_speaker} speaking in the room! Recording...[/cyan]")

            transcript, duration = await moderator.async_listen_to_room(
                silence_duration=silence_duration,
                max_duration=25.0,
                wait_timeout=turn_timeout,
                on_speech_start=_on_speech_start,
            )

            eligible_next = [t for t in teams if t.lower() != current_speaker.lower()]
            if not eligible_next:
                eligible_next = teams

            if not transcript:
                # Room was silent — Moderator intervenes out loud on Speaker!
                next_speaker = eligible_next[0]
                console.print(
                    f"[bold yellow]⚠ No speech heard from {current_speaker} within {turn_timeout:.0f}s. "
                    f"Moderator intervening out loud![/bold yellow]"
                )
                await moderator.intervene("Silence timeout", current_speaker, next_speaker)
                current_speaker = next_speaker
                continue

            # Detect who the speaker passed the mic to over the air
            detected_next = moderator.detect_next_speaker(
                transcript,
                participating_teams=teams,
                current_speaker=current_speaker,
            )

            console.print(
                Panel(
                    f"[bold cyan]Speaker Heard:[/] {current_speaker} ({duration:.1f}s)\n"
                    f"[bold white]Transcribed via Host Mic:[/] \"{transcript}\"\n"
                    f"[bold magenta]Detected Acoustic Mic Pass To:[/] [bold green]{detected_next or 'None detected'}[/]",
                    title=f"👂 Round {round_idx} Captured by Host Microphone",
                    border_style="yellow",
                )
            )

            history.append({
                "speaker": current_speaker,
                "spoken_text": transcript,
                "next_agent": detected_next or eligible_next[0],
                "round": round_idx,
            })

            if detected_next and detected_next in teams and detected_next.lower() != current_speaker.lower():
                current_speaker = detected_next
            else:
                # Speaker forgot to say "Passing the mic to <Team>" clearly — Moderator announces it out loud!
                next_speaker = eligible_next[0]
                console.print(
                    f"[bold yellow]⚠ Clear mic pass not detected in transcript. "
                    f"Moderator announcing handoff to {next_speaker} out loud![/bold yellow]"
                )
                await moderator.intervene("Unclear acoustic handoff", current_speaker, next_speaker)
                current_speaker = next_speaker

        # 3. Moderator Concludes the Roundtable out loud on Speaker
        console.print("\n[bold white on dark_blue] 🔊 HOST MODERATOR SPEAKING CONCLUSION ON SPEAKER [/bold white on dark_blue]")
        closing_text = await moderator.summarize_and_conclude(topic=topic_str, history=history)
        console.print(f"[bold cyan]Moderator Summary:[/] \"{closing_text}\"\n")

    except KeyboardInterrupt:
        console.print("\n[bold yellow]👋 Roundtable stopped by Host (Ctrl+C).[/bold yellow]")


def main():
    parser = argparse.ArgumentParser(
        description="Run the Host Moderator Agent (Speaks on Speaker, Listens to room on Microphone)."
    )
    parser.add_argument(
        "--teams",
        type=str,
        default="Team Orange,Team Red,Team Blue",
        help="Comma-separated list of attendee team names in the room (default: Team Orange,Team Red,Team Blue)",
    )
    parser.add_argument("--rounds", type=int, default=4, help="Number of debate turns before wrap-up (default: 4)")
    parser.add_argument("--topic-id", type=str, default=None, help="Topic ID from moderator/topics.json")
    parser.add_argument("--topic", type=str, default=None, help="Custom debate topic question")
    parser.add_argument(
        "--silence-duration",
        type=float,
        default=1.8,
        help="Seconds of silence after speech before ending microphone recording (default: 1.8)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=25.0,
        help="Seconds to wait for a laptop to speak before Moderator intervenes (default: 25.0)",
    )
    parser.add_argument("--provider", type=str, choices=["groq", "openai", "mock"], default=None)
    parser.add_argument("--stt-provider", type=str, choices=["auto", "groq", "google", "openai"], default=None)
    args = parser.parse_args()

    team_list = [t.strip() for t in args.teams.split(",") if t.strip()]
    asyncio.run(
        run_acoustic_moderator(
            teams=team_list,
            num_rounds=args.rounds,
            topic_id=args.topic_id,
            topic_override=args.topic,
            silence_duration=args.silence_duration,
            turn_timeout=args.timeout,
            provider=args.provider,
            stt_provider=args.stt_provider,
        )
    )


if __name__ == "__main__":
    main()
