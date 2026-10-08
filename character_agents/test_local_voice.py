"""Attendee Voice Testing Utility.

Run this standalone script to verify Edge-TTS synthesis and laptop speaker
playback before connecting to the host server. No API keys or server required!
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

# Add project root to sys.path so script can be run directly from character_agents/
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
from core.tts import VOICE_REGISTRY, list_curated_voices

console = Console()


async def test_agent_voice(preset_path: str, custom_text: str | None = None) -> None:
    """Load agent preset and test speaker playback."""
    console.print("\n[bold yellow]🎙️  PASS-THE-MIC: LOCAL VOICE TEST[/bold yellow]\n")

    try:
        agent = CharacterAgent.from_preset(preset_path)
    except FileNotFoundError:
        console.print(f"[bold red]Error:[/] Preset not found at {preset_path}")
        return

    text_to_speak = custom_text or agent.test_phrase

    # Display test info panel
    info_panel = (
        f"[bold cyan]Team Name:[/] [bold white]{agent.name}[/]\n"
        f"[bold cyan]Voice ID:[/] [green]{agent.voice}[/]\n"
        f"[bold cyan]Persona:[/] {agent.persona}\n"
        f"[bold cyan]Style:[/] {agent.style}\n\n"
        f"[bold yellow]Spoken Script:[/] \"{text_to_speak}\""
    )
    console.print(Panel(info_panel, title="[bold green]Testing Voice Agent[/bold green]", border_style="cyan"))

    console.print("\n[dim]Synthesizing audio with Edge Neural TTS and playing via speakers...[/dim]")
    
    start_time = asyncio.get_event_loop().time()
    await agent.test_voice(custom_text=text_to_speak)
    elapsed = asyncio.get_event_loop().time() - start_time

    console.print(f"[bold green]✔ Speech playback completed in {elapsed:.1f}s![/bold green]")
    console.print("[dim]If you heard clear, loud speech from your laptop speakers, your audio pipeline is 100% ready for the meetup![/dim]\n")


def display_voice_catalog() -> None:
    """Print the curated voice table for workshop attendees."""
    table = Table(title="Pass-The-Mic Curated Voice Catalog", border_style="blue")
    table.add_column("Team / Key", style="bold cyan")
    table.add_column("Voice ID", style="green")
    table.add_column("Accent / Gender", style="yellow")
    table.add_column("Tone Description", style="white")

    for team, meta in list_curated_voices().items():
        table.add_row(
            team,
            meta["voice_id"],
            meta["accent"],
            meta["description"],
        )

    console.print(table)


def main():
    parser = argparse.ArgumentParser(description="Test local voice agent audio without a server.")
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
        "--list-voices",
        action="store_true",
        help="Display the catalog of curated voice options",
    )
    args = parser.parse_args()

    if args.list_voices:
        display_voice_catalog()
        return

    asyncio.run(test_agent_voice(args.preset, custom_text=args.text))


if __name__ == "__main__":
    main()
