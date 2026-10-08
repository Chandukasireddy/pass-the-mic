"""Clean, blocking audio playback engine for pass-the-mic.

Ensures that an agent's speech finishes completely before the mic is passed,
preventing overlapping voices during in-person roundtables.
"""

from __future__ import annotations

import asyncio
import os
import sys
import time
from pathlib import Path
from typing import Optional

# Suppress pygame welcome message to keep the terminal output clean
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "hide"

import pygame
from .tts import synthesize_speech, synthesize_speech_sync


def _init_mixer_if_needed() -> None:
    """Initialize pygame mixer safely with standard audio parameters."""
    if not pygame.mixer.get_init():
        try:
            # 44100Hz, 16-bit, stereo, 2048 buffer
            pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=2048)
        except Exception as e:
            # Fallback initialization with default device settings
            pygame.mixer.init()


def play_audio_file(file_path: str | Path, block: bool = True, cleanup: bool = True) -> None:
    """Play an audio file synchronously.
    
    Args:
        file_path: Path to the mp3 or wav audio file.
        block: If True, blocks until the audio finishes playing.
        cleanup: If True, deletes the file from disk after playback finishes.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")

    _init_mixer_if_needed()

    try:
        pygame.mixer.music.load(str(path))
        pygame.mixer.music.play()

        if block:
            clock = pygame.time.Clock()
            while pygame.mixer.music.get_busy():
                clock.tick(30)
    finally:
        # Crucial for Windows: stop and unload to release OS file locks
        try:
            pygame.mixer.music.stop()
            pygame.mixer.music.unload()
        except Exception:
            pass

        if cleanup:
            try:
                if path.exists():
                    path.unlink()
            except Exception:
                # If file is still temporarily locked, retry after brief tick
                time.sleep(0.05)
                try:
                    if path.exists():
                        path.unlink()
                except Exception:
                    pass


async def play_audio_file_async(file_path: str | Path, cleanup: bool = True) -> None:
    """Play an audio file asynchronously while keeping the asyncio event loop responsive.
    
    This ensures WebSocket heartbeat pings and dashboard updates continue
    to process while the speaker is talking out loud.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")

    _init_mixer_if_needed()

    try:
        pygame.mixer.music.load(str(path))
        pygame.mixer.music.play()

        while pygame.mixer.music.get_busy():
            await asyncio.sleep(0.05)
    finally:
        try:
            pygame.mixer.music.stop()
            pygame.mixer.music.unload()
        except Exception:
            pass

        if cleanup:
            try:
                if path.exists():
                    path.unlink()
            except Exception:
                await asyncio.sleep(0.05)
                try:
                    if path.exists():
                        path.unlink()
                except Exception:
                    pass


def speak(
    text: str,
    voice: str = "en-US-GuyNeural",
    rate: str = "+0%",
    block: bool = True,
) -> None:
    """End-to-end sync helper: synthesizes speech via Edge-TTS and plays out loud."""
    audio_file = synthesize_speech_sync(text=text, voice=voice, rate=rate)
    play_audio_file(audio_file, block=block, cleanup=True)


async def async_speak(
    text: str,
    voice: str = "en-US-GuyNeural",
    rate: str = "+0%",
) -> None:
    """End-to-end async helper: synthesizes speech via Edge-TTS and plays out loud."""
    audio_file = await synthesize_speech(text=text, voice=voice, rate=rate)
    await play_audio_file_async(audio_file, cleanup=True)
