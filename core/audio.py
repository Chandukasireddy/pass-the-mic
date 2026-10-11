"""Half-Duplex Speaker Playback and Microphone Recording Engine for pass-the-mic.

Enables each laptop to:
1. SPEAK out loud through its physical speakers (via Edge-TTS + pygame)
2. LISTEN through its physical microphone with Voice Activity Detection (via sounddevice)
while enforcing half-duplex locking so a laptop never records its own speaker output.
"""

from __future__ import annotations

import asyncio
import os
import tempfile
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

# Suppress pygame welcome banner
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "hide"

import numpy as np
import pygame
import sounddevice as sd
import soundfile as sf

from .tts import synthesize_speech, synthesize_speech_sync

# Half-duplex state: True while this laptop is actively playing audio on its speakers
_IS_SPEAKING: bool = False
_LAST_SPOKE_TIMESTAMP: float = 0.0
REVERB_COOLDOWN_SEC: float = 0.35


def is_currently_speaking() -> bool:
    """Return True if this laptop's speaker is currently playing or in reverb cooldown."""
    if _IS_SPEAKING:
        return True
    if (time.time() - _LAST_SPOKE_TIMESTAMP) < REVERB_COOLDOWN_SEC:
        return True
    return False


def _init_mixer_if_needed() -> None:
    """Initialize pygame mixer safely with standard audio parameters."""
    if not pygame.mixer.get_init():
        try:
            pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=2048)
        except Exception:
            pygame.mixer.init()


def list_audio_devices() -> Dict[str, List[dict]]:
    """Return available microphone input and speaker output devices on this laptop."""
    inputs = []
    outputs = []
    try:
        devices = sd.query_devices()
        for idx, dev in enumerate(devices):
            info = {
                "index": idx,
                "name": dev.get("name", f"Device {idx}"),
                "max_input_channels": dev.get("max_input_channels", 0),
                "max_output_channels": dev.get("max_output_channels", 0),
                "default_samplerate": dev.get("default_samplerate", 16000),
            }
            if info["max_input_channels"] > 0:
                inputs.append(info)
            if info["max_output_channels"] > 0:
                outputs.append(info)
    except Exception:
        pass
    return {"inputs": inputs, "outputs": outputs}


# ==============================================================================
# SPEAKER OUTPUT (MOUTH)
# ==============================================================================

def play_audio_file(file_path: str | Path, block: bool = True, cleanup: bool = True) -> None:
    """Play an audio file out loud through the laptop's speakers synchronously."""
    global _IS_SPEAKING, _LAST_SPOKE_TIMESTAMP

    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")

    _init_mixer_if_needed()
    _IS_SPEAKING = True

    try:
        pygame.mixer.music.load(str(path))
        pygame.mixer.music.play()

        if block:
            clock = pygame.time.Clock()
            while pygame.mixer.music.get_busy():
                clock.tick(30)
    finally:
        try:
            pygame.mixer.music.stop()
            pygame.mixer.music.unload()
        except Exception:
            pass

        _IS_SPEAKING = False
        _LAST_SPOKE_TIMESTAMP = time.time()

        if cleanup:
            try:
                if path.exists():
                    path.unlink()
            except Exception:
                time.sleep(0.05)
                try:
                    if path.exists():
                        path.unlink()
                except Exception:
                    pass


async def play_audio_file_async(file_path: str | Path, cleanup: bool = True) -> None:
    """Play an audio file out loud asynchronously while keeping the event loop responsive."""
    global _IS_SPEAKING, _LAST_SPOKE_TIMESTAMP

    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")

    _init_mixer_if_needed()
    _IS_SPEAKING = True

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

        _IS_SPEAKING = False
        _LAST_SPOKE_TIMESTAMP = time.time()

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
    """Synthesize speech via Edge-TTS and play it out loud through laptop speakers."""
    audio_file = synthesize_speech_sync(text=text, voice=voice, rate=rate)
    play_audio_file(audio_file, block=block, cleanup=True)


async def async_speak(
    text: str,
    voice: str = "en-US-GuyNeural",
    rate: str = "+0%",
) -> None:
    """Asynchronously synthesize speech via Edge-TTS and play it out loud through speakers."""
    audio_file = await synthesize_speech(text=text, voice=voice, rate=rate)
    await play_audio_file_async(audio_file, cleanup=True)


# ==============================================================================
# MICROPHONE INPUT & VOICE ACTIVITY DETECTION (EARS)
# ==============================================================================

def compute_rms_energy(audio_chunk: np.ndarray) -> float:
    """Compute normalized Root-Mean-Square (RMS) energy of a float32/int16 audio chunk."""
    if audio_chunk.size == 0:
        return 0.0
    if audio_chunk.dtype == np.int16:
        floats = audio_chunk.astype(np.float32) / 32768.0
    else:
        floats = audio_chunk.astype(np.float32)
    return float(np.sqrt(np.mean(np.square(floats))))


def calibrate_microphone(
    duration_sec: float = 1.0,
    sample_rate: int = 16000,
    device: Optional[int] = None,
) -> float:
    """Sample ambient room noise and return an adaptive RMS speech threshold.
    
    Returns a sensible default (0.015) if no microphone hardware is available.
    """
    try:
        num_frames = int(duration_sec * sample_rate)
        recording = sd.rec(
            num_frames,
            samplerate=sample_rate,
            channels=1,
            dtype="float32",
            device=device,
        )
        sd.wait()
        ambient_rms = compute_rms_energy(recording)
        # Set threshold above ambient room floor, clamped to a practical range
        threshold = max(0.008, min(ambient_rms * 2.5 + 0.005, 0.08))
        return round(threshold, 4)
    except Exception:
        return float(os.getenv("MIC_ENERGY_THRESHOLD", "0.015"))


def record_from_mic(
    output_path: Optional[str | Path] = None,
    sample_rate: int = 16000,
    energy_threshold: float = 0.015,
    silence_duration: float = 1.8,
    max_duration: float = 25.0,
    wait_timeout: float = 30.0,
    fixed_duration: Optional[float] = None,
    device: Optional[int] = None,
    on_speech_start: Optional[Callable[[], None]] = None,
) -> Tuple[Optional[Path], float]:
    """Listen on the laptop's physical microphone and record speech to a WAV file.
    
    Uses Voice Activity Detection (VAD):
      1. Waits up to `wait_timeout` seconds for someone in the room to start speaking.
      2. Records continuously while speech energy is present.
      3. Stops automatically once `silence_duration` seconds of silence follow the speech.
      
    If `fixed_duration` is provided, records for that exact number of seconds.
    
    Returns:
        Tuple of (Path to recorded WAV file or None if no speech detected, duration in seconds).
    """
    # Wait if our own speaker is currently talking
    while is_currently_speaking():
        time.sleep(0.05)

    if output_path is None:
        fd, tmp_wav = tempfile.mkstemp(suffix=".wav", prefix="ptm_mic_")
        os.close(fd)
        dest = Path(tmp_wav)
    else:
        dest = Path(output_path)
        dest.parent.mkdir(parents=True, exist_ok=True)

    # Fixed duration mode (e.g., quick microphone check)
    if fixed_duration is not None and fixed_duration > 0:
        num_frames = int(fixed_duration * sample_rate)
        audio_data = sd.rec(
            num_frames,
            samplerate=sample_rate,
            channels=1,
            dtype="int16",
            device=device,
        )
        sd.wait()
        sf.write(str(dest), audio_data, sample_rate, subtype="PCM_16")
        return dest, fixed_duration

    # Dynamic Voice Activity Detection (VAD) mode
    chunk_duration = 0.1  # 100ms chunks
    chunk_samples = int(sample_rate * chunk_duration)
    recorded_chunks: List[np.ndarray] = []
    pre_buffer: List[np.ndarray] = []  # Keep last 0.4s before speech trigger so first syllable isn't clipped
    max_pre_chunks = 4

    speech_started = False
    silence_chunks = 0
    max_silence_chunks = max(1, int(silence_duration / chunk_duration))
    max_total_chunks = max(1, int(max_duration / chunk_duration))
    max_wait_chunks = max(1, int(wait_timeout / chunk_duration))
    wait_chunks = 0

    with sd.InputStream(
        samplerate=sample_rate,
        channels=1,
        dtype="int16",
        blocksize=chunk_samples,
        device=device,
    ) as stream:
        while True:
            # Abort if our own speaker starts playing
            if is_currently_speaking():
                break

            chunk, _overflow = stream.read(chunk_samples)
            chunk_copy = chunk.copy()
            rms = compute_rms_energy(chunk_copy)

            if not speech_started:
                pre_buffer.append(chunk_copy)
                if len(pre_buffer) > max_pre_chunks:
                    pre_buffer.pop(0)

                if rms >= energy_threshold:
                    speech_started = True
                    recorded_chunks.extend(pre_buffer)
                    silence_chunks = 0
                    if on_speech_start:
                        try:
                            on_speech_start()
                        except Exception:
                            pass
                else:
                    wait_chunks += 1
                    if wait_chunks >= max_wait_chunks:
                        # No one spoke within wait_timeout
                        if dest.exists():
                            dest.unlink(missing_ok=True)
                        return None, 0.0
            else:
                recorded_chunks.append(chunk_copy)
                if rms < energy_threshold:
                    silence_chunks += 1
                else:
                    silence_chunks = 0

                if silence_chunks >= max_silence_chunks or len(recorded_chunks) >= max_total_chunks:
                    break

    if not recorded_chunks:
        if dest.exists():
            dest.unlink(missing_ok=True)
        return None, 0.0

    full_audio = np.concatenate(recorded_chunks, axis=0)
    total_duration = len(full_audio) / float(sample_rate)
    sf.write(str(dest), full_audio, sample_rate, subtype="PCM_16")
    return dest, round(total_duration, 2)


async def async_record_from_mic(
    output_path: Optional[str | Path] = None,
    sample_rate: int = 16000,
    energy_threshold: float = 0.015,
    silence_duration: float = 1.8,
    max_duration: float = 25.0,
    wait_timeout: float = 30.0,
    fixed_duration: Optional[float] = None,
    device: Optional[int] = None,
    on_speech_start: Optional[Callable[[], None]] = None,
) -> Tuple[Optional[Path], float]:
    """Asynchronous wrapper around `record_from_mic` using a background executor thread."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        None,
        lambda: record_from_mic(
            output_path=output_path,
            sample_rate=sample_rate,
            energy_threshold=energy_threshold,
            silence_duration=silence_duration,
            max_duration=max_duration,
            wait_timeout=wait_timeout,
            fixed_duration=fixed_duration,
            device=device,
            on_speech_start=on_speech_start,
        ),
    )
