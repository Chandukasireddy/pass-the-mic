"""Rich Terminal Dashboard for the Host Moderator.

Displays live network IP, connected attendee teams, active speaker badge,
and real-time conversation transcripts.
"""

from __future__ import annotations

import socket
import sys
import time
from typing import Dict, List, Optional

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text


def get_local_ip() -> str:
    """Detect the local Wi-Fi / LAN IP address of this machine."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # Doesn't need to actually reach external internet; selects local interface
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
    except Exception:
        ip = "127.0.0.1"
    finally:
        s.close()
    return ip


class RoundtableDashboard:
    """Rich interactive dashboard for the meetup host."""

    def __init__(self, port: int = 8000):
        self.console = Console()
        self.port = port
        self.local_ip = get_local_ip()
        self.topic: str = "Awaiting topic selection..."
        self.active_speaker: Optional[str] = None
        self.status_message: str = "Awaiting participant teams..."
        self.connected_teams: Dict[str, dict] = {}
        self.transcript: List[dict] = []
        self.round_number: int = 0
        self._live: Optional[Live] = None

    def update_topic(self, topic: str) -> None:
        self.topic = topic

    def set_active_speaker(self, speaker: Optional[str]) -> None:
        self.active_speaker = speaker

    def set_status(self, status: str) -> None:
        self.status_message = status

    def register_team(self, name: str, voice: str, persona: str = "") -> None:
        self.connected_teams[name] = {
            "name": name,
            "voice": voice,
            "persona": persona or "Participant",
            "turns": 0,
            "connected_at": time.time(),
        }

    def unregister_team(self, name: str) -> None:
        if name in self.connected_teams:
            del self.connected_teams[name]

    def record_turn(self, speaker: str, text: str, next_agent: str) -> None:
        if speaker in self.connected_teams:
            self.connected_teams[speaker]["turns"] += 1
        self.transcript.append({
            "speaker": speaker,
            "spoken_text": text,
            "next_agent": next_agent,
            "time": time.strftime("%H:%M:%S"),
        })

    def generate_layout(self) -> Layout:
        layout = Layout()
        layout.split_column(
            Layout(name="header", size=4),
            Layout(name="body", ratio=1),
            Layout(name="footer", size=7),
        )

        layout["body"].split_row(
            Layout(name="teams", ratio=1),
            Layout(name="stage", ratio=1),
        )

        # Header
        ws_url = f"ws://{self.local_ip}:{self.port}/ws"
        header_text = Text()
        header_text.append("🎤 PASS-THE-MIC: MULTI-AGENT VOICE ROUNDTABLE\n", style="bold yellow")
        header_text.append(f"Host LAN IP: ", style="bold white")
        header_text.append(f"{self.local_ip}:{self.port}  ", style="bold cyan")
        header_text.append(f"WebSocket URL: ", style="bold white")
        header_text.append(f"{ws_url}  ", style="bold green")
        header_text.append(f"State: {self.status_message}", style="italic magenta")

        layout["header"].update(Panel(header_text, border_style="cyan"))

        # Teams Table
        table = Table(title="Connected Attendee Teams", expand=True, border_style="blue")
        table.add_column("Team", style="bold white")
        table.add_column("Voice ID", style="dim")
        table.add_column("Turns", justify="right", style="cyan")
        table.add_column("Status", justify="center")

        for team_name, info in self.connected_teams.items():
            is_active = (team_name == self.active_speaker)
            if is_active:
                status_badge = "[bold white on red] 🎙️ LIVE [/]"
                team_style = f"[bold green]{team_name}[/]"
            else:
                status_badge = "[green]Ready[/]"
                team_style = team_name

            table.add_row(
                team_style,
                info.get("voice", "default"),
                str(info.get("turns", 0)),
                status_badge,
            )

        if not self.connected_teams:
            table.add_row("[dim]No teams yet[/]", "[dim]waiting...[/]", "0", "[yellow]Connecting[/]")

        layout["teams"].update(Panel(table, border_style="blue"))

        # Stage / Active Debate panel
        stage_content = Text()
        stage_content.append("DEBATE TOPIC:\n", style="bold yellow")
        stage_content.append(f"{self.topic}\n\n", style="white")

        stage_content.append("CURRENT MICROPHONE HOLDER:\n", style="bold cyan")
        if self.active_speaker:
            stage_content.append(f"👉 {self.active_speaker}\n\n", style="bold green on black")
        else:
            stage_content.append("Standby (Moderator floor)\n\n", style="dim")

        if self.transcript:
            last = self.transcript[-1]
            stage_content.append("LAST MIC PASS:\n", style="bold magenta")
            stage_content.append(f"{last['speaker']} ➔ {last['next_agent']}\n", style="bold yellow")

        layout["stage"].update(Panel(stage_content, title="Roundtable Stage", border_style="green"))

        # Footer / Transcript log
        transcript_text = Text()
        recent = self.transcript[-3:]
        if recent:
            for item in recent:
                transcript_text.append(f"[{item['time']}] ", style="dim")
                transcript_text.append(f"{item['speaker']}: ", style="bold cyan")
                transcript_text.append(f"\"{item['spoken_text'][:120]}...\" ", style="white")
                transcript_text.append(f"➔ {item['next_agent']}\n", style="bold magenta")
        else:
            transcript_text.append("Transcript will stream here once debate commences.", style="dim")

        layout["footer"].update(Panel(transcript_text, title="Live Debate Transcript", border_style="yellow"))

        return layout

    def start_live(self) -> Live:
        """Initialize the rich live loop."""
        self._live = Live(self.generate_layout(), console=self.console, refresh_per_second=4, screen=False)
        self._live.start()
        return self._live

    def refresh(self) -> None:
        """Refresh current screen render."""
        if self._live:
            self._live.update(self.generate_layout())

    def stop_live(self) -> None:
        """Stop live rendering."""
        if self._live:
            self._live.stop()
            self._live = None

    def print_summary(self) -> None:
        """Print a summary table of the finished session."""
        self.console.print("\n[bold green]═══ ROUNDTABLE COMPLETE ═══[/bold green]")
        self.console.print(f"[bold yellow]Topic:[/] {self.topic}")
        self.console.print(f"[bold cyan]Total Turns:[/] {len(self.transcript)}")
        for i, turn in enumerate(self.transcript, 1):
            self.console.print(f" [dim]{i}.[/] [bold]{turn['speaker']}[/] (passed to [magenta]{turn['next_agent']}[/]): {turn['spoken_text']}")
