"""Attendee Speaker & Microphone Testing Utility.

Run this standalone script to verify:
1. Laptop SPEAKER output (Edge-TTS voice synthesis + pygame playback)
2. Laptop MICROPHONE input (sounddevice recording + Speech-to-Text transcription)
No external server required!
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

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

from character_agents.agent_builder import CharacterAgent
from core.audio import list_audio_devices
from core.tts import list_curated_voices

console = Console()


async def test_agent_speaker_and_mic(
    preset_path: str,
    custom_text: str | None = None,
    test_mic: bool = False,
    mic_seconds: float = 4.0,
) -> None:
    """Test the laptop's speaker output and optionally its microphone input."""
    console.print("\n[bold yellow]🎙️  PASS-THE-MIC: LOCAL SPEAKER & MICROPHONE TEST[/bold yellow]\n")

    try:
        agent = CharacterAgent.from_preset(preset_path)
    except FileNotFoundError:
        console.print(f"[bold red]Error:[/] Preset not found at {preset_path}")
        return

    text_to_speak = custom_text or agent.test_phrase

    info_panel = (
        f"[bold cyan]Team Name:[/] [bold white]{agent.name}[/]\n"
        f"[bold cyan]Speaker Voice ID:[/] [green]{agent.voice}[/]\n"
        f"[bold cyan]STT Provider:[/] [magenta]{agent.stt_client.provider.upper()}[/]\n"
        f"[bold cyan]Persona:[/] {agent.persona}\n"
        f"[bold cyan]Style:[/] {agent.style}\n\n"
        f"[bold yellow]Speaker Test Script:[/] \"{text_to_speak}\""
    )
    console.print(Panel(info_panel, title="[bold green]1. Testing Laptop Speaker (TTS)[/bold green]", border_style="cyan"))

    console.print("\n[dim]🔊 Playing synthesized voice through your laptop speakers...[/dim]")
    start_time = asyncio.get_event_loop().time()
    await agent.test_voice(custom_text=text_to_speak)
    elapsed = asyncio.get_event_loop().time() - start_time
    console.print(f"[bold green]✔ Speaker playback completed in {elapsed:.1f}s![/bold green]\n")

    if test_mic:
        console.print(
            Panel(
                f"Speak clearly into your laptop microphone for [bold yellow]{mic_seconds:.0f} seconds[/bold yellow]!\n"
                f"Try saying: [bold green]\"Passing the mic to {agent.name}!\"[/bold green]",
                title="[bold magenta]2. Testing Laptop Microphone (STT)[/bold magenta]",
                border_style="magenta",
            )
        )
        console.print("[bold red]🔴 RECORDING FROM MICROPHONE NOW... SPEAK OUT LOUD![/bold red]")
        transcript, dur = await agent.async_listen_on_mic(fixed_duration=mic_seconds)

        if transcript:
            woke = agent.check_wake_trigger(transcript)
            wake_status = (
                f"[bold green]YES — Wake trigger for {agent.name} detected![/bold green]"
                if woke
                else f"[yellow]No (Did not hear 'Passing the mic to {agent.name}')[/yellow]"
            )
            console.print(f"\n[bold green]✔ Microphone heard ({dur:.1f}s):[/] [bold white]\"{transcript}\"[/bold white]")
            console.print(f"[bold cyan]Wake Trigger Match:[/] {wake_status}\n")

            reply = f"I heard you say: {transcript}"
            console.print(f"[dim]🔊 Reading back what your microphone heard...[/dim]")
            await agent.test_voice(custom_text=reply)
        else:
            console.print(
                "\n[bold yellow]⚠ No clear speech transcribed from microphone.[/bold yellow] "
                "Check that your microphone is unmuted and selected as the default OS input device.\n"
            )


def display_voice_catalog() -> None:
    """Print the curated voice table for workshop attendees."""
    table = Table(title="Pass-The-Mic Curated Voice Catalog", border_style="blue")
    table.add_column("Team / Key", style="bold cyan")
    table.add_column("Voice ID", style="green")
    table.add_column("Accent / Gender", style="yellow")
    table.add_column("Tone Description", style="white")

    for team, meta in list_curated_voices().items():
        table.add_row(team, meta["voice_id"], meta["accent"], meta["description"])

    console.print(table)


def display_audio_devices() -> None:
    """Print detected microphone inputs and speaker outputs."""
    devices = list_audio_devices()

    in_table = Table(title="Detected Microphone Input Devices", border_style="magenta")
    in_table.add_column("ID", style="cyan", justify="right")
    in_table.add_column("Microphone Name", style="bold white")
    in_table.add_column("Channels", justify="right", style="green")

    for d in devices.get("inputs", []):
        in_table.add_row(str(d["index"]), d["name"], str(d["max_input_channels"]))

    out_table = Table(title="Detected Speaker Output Devices", border_style="green")
    out_table.add_column("ID", style="cyan", justify="right")
    out_table.add_column("Speaker Name", style="bold white")
    out_table.add_column("Channels", justify="right", style="green")

    for d in devices.get("outputs", []):
        out_table.add_row(str(d["index"]), d["name"], str(d["max_output_channels"]))

    console.print(in_table)
    console.print()
    console.print(out_table)


def main():
    parser = argparse.ArgumentParser(description="Test local laptop speaker (TTS) and microphone (STT).")
    parser.add_argument(
        "--preset",
        type=str,
        default="presets/orange_optimist.json",
        help="Path or name of the preset JSON file (default: presets/orange_optimist.json)",
    )
    parser.add_argument(
        "--text",
        type=str,
        default=None,
        help="Custom text string to speak out loud",
    )
    parser.add_argument(
        "--test-mic",
        action="store_true",
        help="Also record from your laptop microphone and test Speech-to-Text transcription",
    )
    parser.add_argument(
        "--mic-seconds",
        type=float,
        default=4.0,
        help="Duration in seconds for the microphone recording test (default: 4.0)",
    )
    parser.add_argument(
        "--list-voices",
        action="store_true",
        help="Display the catalog of curated Edge-TTS voice options",
    )
    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="Display detected microphone input and speaker output hardware devices",
    )
    args = parser.parse_args()

    if args.list_voices:
        display_voice_catalog()
        return

    if args.list_devices:
        display_audio_devices()
        return

    asyncio.run(
        test_agent_speaker_and_mic(
            args.preset,
            custom_text=args.text,
            test_mic=args.test_mic,
            mic_seconds=args.mic_seconds,
        )
    )


if __name__ == "__main__":
    main()
