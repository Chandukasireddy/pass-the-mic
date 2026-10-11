"""Full-Duplex Active Microphone Listener, Voice Barge-In ('Stop' Interruption) & Speaker Engine for Magnus.

Key Capabilities:
1. Full-Duplex Active Listening: Continues listening on the microphone EVEN WHILE Magnus is speaking!
2. Instant Voice Interruption ("Stop", "Wait", "Pause", "Quiet", "Mic Off"):
   Saying "stop" while Magnus is talking immediately halts speaker playback mid-sentence.
3. Instant UI Controls: Stop Speaking button, Mic ON/OFF toggle, and Full Emergency Stop.
"""

from __future__ import annotations

import asyncio
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Coroutine, Dict, List, Optional

os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "hide"

import numpy as np
import pygame
import sounddevice as sd
import soundfile as sf

try:
    from .gemini_live_engine import GeminiSpeechToSpeechEngine, SpeechToSpeechResult
    from .magnus_config import MagnusSettings, check_magnus_wake_phrase, check_stop_command
except ImportError:
    from gemini_live_engine import GeminiSpeechToSpeechEngine, SpeechToSpeechResult
    from magnus_config import MagnusSettings, check_magnus_wake_phrase, check_stop_command


def _init_mixer() -> None:
    if not pygame.mixer.get_init():
        try:
            pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=2048)
        except Exception:
            pygame.mixer.init()


def compute_rms(chunk: np.ndarray) -> float:
    if chunk.size == 0:
        return 0.0
    if chunk.dtype == np.int16:
        floats = chunk.astype(np.float32) / 32768.0
    else:
        floats = chunk.astype(np.float32)
    return float(np.sqrt(np.mean(np.square(floats))))


def get_audio_devices() -> Dict[str, List[dict]]:
    inputs = []
    outputs = []
    try:
        for idx, dev in enumerate(sd.query_devices()):
            item = {
                "index": idx,
                "name": dev.get("name", f"Device {idx}"),
                "inputs": dev.get("max_input_channels", 0),
                "outputs": dev.get("max_output_channels", 0),
            }
            if item["inputs"] > 0:
                inputs.append(item)
            if item["outputs"] > 0:
                outputs.append(item)
    except Exception:
        pass
    return {"inputs": inputs, "outputs": outputs}


class MagnusActiveController:
    """Manages Magnus's full-duplex active listening, voice barge-in ('stop'), and speaker output."""

    def __init__(self, settings: MagnusSettings, engine: GeminiSpeechToSpeechEngine):
        self.settings = settings
        self.engine = engine

        self.is_listening: bool = False
        self.state: str = "IDLE"  # IDLE | LISTENING | RECORDING_SPEECH | PROCESSING_S2S | SPEAKING
        self.current_rms: float = 0.0
        self.is_currently_playing: bool = False
        self.speech_interrupted: bool = False
        self.last_spoke_time: float = 0.0
        self.turns_completed: int = 0
        self.transcript_history: List[dict] = []

        self._listen_task: Optional[asyncio.Task] = None
        self._active_turn_task: Optional[asyncio.Task] = None
        self._event_callbacks: List[Callable[[dict], Coroutine[Any, Any, None]]] = []

    def subscribe(self, callback: Callable[[dict], Coroutine[Any, Any, None]]) -> None:
        self._event_callbacks.append(callback)

    def unsubscribe(self, callback: Callable[[dict], Coroutine[Any, Any, None]]) -> None:
        if callback in self._event_callbacks:
            self._event_callbacks.remove(callback)

    async def broadcast_event(self, payload: dict) -> None:
        for cb in list(self._event_callbacks):
            try:
                await cb(payload)
            except Exception:
                pass

    async def set_state(self, new_state: str, detail: str = "") -> None:
        self.state = new_state
        await self.broadcast_event({
            "type": "state_change",
            "state": self.state,
            "detail": detail,
            "is_listening": self.is_listening,
            "is_speaking": self.is_currently_playing,
            "turns_completed": self.turns_completed,
            "api_status": self.engine.last_api_status,
        })

    async def interrupt_speech(self, reason: str = "Stopped by user") -> None:
        """Immediately stop Magnus from speaking out loud mid-sentence."""
        self.speech_interrupted = True
        try:
            if pygame.mixer.get_init() and pygame.mixer.music.get_busy():
                pygame.mixer.music.stop()
                pygame.mixer.music.unload()
        except Exception:
            pass

        self.is_currently_playing = False
        self.last_spoke_time = time.time()

        if self._active_turn_task and not self._active_turn_task.done():
            self._active_turn_task.cancel()

        next_state = "LISTENING" if self.is_listening else "IDLE"
        await self.set_state(next_state, f"🛑 Speech stopped ({reason})")
        await self.broadcast_event({
            "type": "speech_interrupted",
            "reason": reason,
        })

    async def stop_all_agent_activity(self) -> None:
        """Turn microphone OFF and immediately stop any active speech or generation."""
        self.speech_interrupted = True
        try:
            if pygame.mixer.get_init():
                pygame.mixer.music.stop()
                pygame.mixer.music.unload()
        except Exception:
            pass
        self.is_currently_playing = False
        await self.stop_active_listening()
        await self.set_state("IDLE", "🔇 Microphone OFF & Agent Stopped")

    async def handle_realtime_voice_command(self, transcript: str) -> bool:
        """Check any real-time transcript for 'stop' or 'mic off' commands and act immediately."""
        should_stop, should_mic_off = check_stop_command(transcript, self.settings)
        if should_mic_off:
            await self.interrupt_speech(reason=f"Voice command: '{transcript}'")
            await self.stop_active_listening()
            await self.set_state("IDLE", f"🔇 Mic turned OFF by voice ('{transcript}')")
            return True
        if should_stop:
            await self.interrupt_speech(reason=f"Voice command: '{transcript}'")
            return True
        return False

    async def play_audio_on_speaker(self, file_path: Path, cleanup: bool = True) -> None:
        """Play Magnus's voice on speakers while keeping the Barge-In Listener active for 'stop'."""
        if not file_path or not file_path.exists():
            return

        if not self.settings.speaker_enabled:
            if cleanup:
                file_path.unlink(missing_ok=True)
            return

        _init_mixer()
        self.is_currently_playing = True
        self.speech_interrupted = False

        barge_in_task: Optional[asyncio.Task] = None
        if self.settings.barge_in_enabled:
            barge_in_task = asyncio.create_task(self._barge_in_monitor_while_speaking())

        try:
            pygame.mixer.music.load(str(file_path))
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy() and not self.speech_interrupted:
                await asyncio.sleep(0.04)
        finally:
            if barge_in_task and not barge_in_task.done():
                barge_in_task.cancel()
                try:
                    await barge_in_task
                except asyncio.CancelledError:
                    pass

            try:
                pygame.mixer.music.stop()
                pygame.mixer.music.unload()
            except Exception:
                pass

            self.is_currently_playing = False
            self.last_spoke_time = time.time()

            if cleanup:
                await asyncio.sleep(0.04)
                try:
                    file_path.unlink(missing_ok=True)
                except Exception:
                    pass

    async def _barge_in_monitor_while_speaking(self) -> None:
        """Full-duplex listener that monitors the microphone WHILE Magnus is speaking.
        
        If the user says 'stop', 'wait', 'pause', 'quiet', or 'mic off', it immediately
        interrupts `pygame.mixer.music` mid-sentence!
        """
        sample_rate = 16000
        chunk_samples = 1600  # 100ms
        q: asyncio.Queue[np.ndarray] = asyncio.Queue()
        loop = asyncio.get_running_loop()

        def _cb(indata: np.ndarray, frames: int, time_info: Any, status: Any) -> None:
            loop.call_soon_threadsafe(q.put_nowait, indata.copy())

        # Barge-in threshold sits slightly above ambient/speaker bleed so saying "Stop!" triggers cleanly
        barge_threshold = max(0.025, self.settings.energy_threshold * 1.8)

        try:
            with sd.InputStream(
                samplerate=sample_rate,
                channels=1,
                dtype="int16",
                blocksize=chunk_samples,
                device=self.settings.mic_device_index,
                callback=_cb,
            ):
                voice_window: List[np.ndarray] = []
                while self.is_currently_playing and not self.speech_interrupted:
                    try:
                        chunk = await asyncio.wait_for(q.get(), timeout=0.3)
                    except asyncio.TimeoutError:
                        continue

                    rms = compute_rms(chunk)
                    self.current_rms = round(rms, 4)

                    if rms >= barge_threshold:
                        voice_window.append(chunk)
                        # After ~0.9s of voice activity while speaking, check if user said "stop"
                        if len(voice_window) >= 9:
                            fd, tmp_wav = tempfile.mkstemp(suffix=".wav", prefix="magnus_bargein_")
                            os.close(fd)
                            wav_path = Path(tmp_wav)
                            sf.write(str(wav_path), np.concatenate(voice_window, axis=0), sample_rate, subtype="PCM_16")
                            voice_window.clear()

                            transcript = await self.engine.async_transcribe_wav(wav_path)
                            wav_path.unlink(missing_ok=True)

                            if transcript:
                                stopped = await self.handle_realtime_voice_command(transcript)
                                if stopped:
                                    break
                    else:
                        if len(voice_window) >= 5:
                            # Short burst finished (e.g. a quick word like "Stop!") — transcribe immediately!
                            fd, tmp_wav = tempfile.mkstemp(suffix=".wav", prefix="magnus_bargein_")
                            os.close(fd)
                            wav_path = Path(tmp_wav)
                            sf.write(str(wav_path), np.concatenate(voice_window, axis=0), sample_rate, subtype="PCM_16")
                            voice_window.clear()

                            transcript = await self.engine.async_transcribe_wav(wav_path)
                            wav_path.unlink(missing_ok=True)

                            if transcript:
                                stopped = await self.handle_realtime_voice_command(transcript)
                                if stopped:
                                    break
                        else:
                            voice_window.clear()
        except asyncio.CancelledError:
            pass
        except Exception:
            pass

    async def calibrate_microphone(self) -> float:
        """Sample 1 second of ambient noise and update settings.energy_threshold."""
        loop = asyncio.get_running_loop()

        def _sample() -> float:
            try:
                rec = sd.rec(
                    16000,
                    samplerate=16000,
                    channels=1,
                    dtype="float32",
                    device=self.settings.mic_device_index,
                )
                sd.wait()
                rms = compute_rms(rec)
                return round(max(0.008, min(rms * 2.5 + 0.005, 0.08)), 4)
            except Exception:
                return 0.015

        threshold = await loop.run_in_executor(None, _sample)
        self.settings.energy_threshold = threshold
        await self.broadcast_event({
            "type": "calibrated",
            "energy_threshold": threshold,
        })
        return threshold

    async def start_active_listening(self) -> None:
        """Activate continuous microphone listening."""
        if self.is_listening:
            return
        self.is_listening = True
        await self.set_state("LISTENING", "🎙️ Microphone ON — Active listening & voice 'stop' interruption ready")
        self._listen_task = asyncio.create_task(self._continuous_listen_loop())

    async def stop_active_listening(self) -> None:
        """Deactivate continuous microphone listening."""
        self.is_listening = False
        if self._listen_task:
            self._listen_task.cancel()
            try:
                await self._listen_task
            except asyncio.CancelledError:
                pass
            self._listen_task = None
        await self.set_state("IDLE", "🔇 Microphone OFF — Active listening paused")

    async def _continuous_listen_loop(self) -> None:
        """Background loop that captures speech segments via VAD and triggers Magnus."""
        sample_rate = 16000
        chunk_duration = 0.1
        chunk_samples = int(sample_rate * chunk_duration)

        while self.is_listening:
            # If Magnus is currently playing audio, the _barge_in_monitor_while_speaking handles the mic
            if self.is_currently_playing or (time.time() - self.last_spoke_time) < 0.25:
                await asyncio.sleep(0.08)
                continue

            await self.set_state("LISTENING", "🎙️ Listening on microphone (say 'Magnus' to wake, or 'Stop' anytime)...")

            try:
                wav_path, duration = await self._capture_single_utterance(sample_rate, chunk_samples)
            except asyncio.CancelledError:
                break
            except Exception as e:
                await self.broadcast_event({"type": "error", "message": f"Microphone stream error: {e}"})
                await asyncio.sleep(1.5)
                continue

            if not self.is_listening or not wav_path:
                continue

            await self.set_state("PROCESSING_S2S", "Transcribing microphone audio...")
            transcript = await self.engine.async_transcribe_wav(wav_path)

            if not transcript:
                wav_path.unlink(missing_ok=True)
                continue

            # First check if user spoke a stop / mic-off command
            was_stop = await self.handle_realtime_voice_command(transcript)
            if was_stop:
                wav_path.unlink(missing_ok=True)
                entry = {
                    "role": "room",
                    "speaker": "Voice Command",
                    "text": f"🛑 {transcript}",
                    "duration": duration,
                    "triggered_magnus": False,
                    "timestamp": time.strftime("%H:%M:%S"),
                }
                self.transcript_history.append(entry)
                await self.broadcast_event({"type": "room_heard", "entry": entry})
                continue

            woke = check_magnus_wake_phrase(transcript, self.settings)
            entry = {
                "role": "room",
                "speaker": "Room Microphone",
                "text": transcript,
                "duration": duration,
                "triggered_magnus": woke,
                "timestamp": time.strftime("%H:%M:%S"),
            }
            self.transcript_history.append(entry)
            await self.broadcast_event({
                "type": "room_heard",
                "entry": entry,
            })

            if woke and self.settings.listening_mode != "manual":
                self._active_turn_task = asyncio.create_task(
                    self._execute_magnus_response(
                        input_wav_path=wav_path,
                        heard_transcript=transcript,
                    )
                )
                try:
                    await self._active_turn_task
                except asyncio.CancelledError:
                    pass
            else:
                wav_path.unlink(missing_ok=True)

    async def _capture_single_utterance(
        self,
        sample_rate: int,
        chunk_samples: int,
    ) -> tuple[Optional[Path], float]:
        """Record a single utterance from the microphone using VAD while streaming RMS telemetry."""
        q: asyncio.Queue[np.ndarray] = asyncio.Queue()
        loop = asyncio.get_running_loop()

        def _audio_callback(indata: np.ndarray, frames: int, time_info: Any, status: Any) -> None:
            loop.call_soon_threadsafe(q.put_nowait, indata.copy())

        pre_buffer: List[np.ndarray] = []
        recorded: List[np.ndarray] = []
        speech_started = False
        silence_chunks = 0
        tick_counter = 0

        stream = sd.InputStream(
            samplerate=sample_rate,
            channels=1,
            dtype="int16",
            blocksize=chunk_samples,
            device=self.settings.mic_device_index,
            callback=_audio_callback,
        )

        with stream:
            while self.is_listening and not self.is_currently_playing:
                try:
                    chunk = await asyncio.wait_for(q.get(), timeout=0.4)
                except asyncio.TimeoutError:
                    continue

                rms = compute_rms(chunk)
                self.current_rms = round(rms, 4)
                tick_counter += 1

                if tick_counter % 2 == 0:
                    await self.broadcast_event({
                        "type": "mic_level",
                        "rms": self.current_rms,
                        "threshold": self.settings.energy_threshold,
                        "speech_active": speech_started,
                    })

                if not speech_started:
                    pre_buffer.append(chunk)
                    if len(pre_buffer) > 4:
                        pre_buffer.pop(0)

                    if rms >= self.settings.energy_threshold:
                        speech_started = True
                        recorded.extend(pre_buffer)
                        silence_chunks = 0
                        await self.set_state("RECORDING_SPEECH", "🔴 Hearing voice — recording until pause...")
                else:
                    recorded.append(chunk)
                    if rms < self.settings.energy_threshold:
                        silence_chunks += 1
                    else:
                        silence_chunks = 0

                    max_silence = max(1, int(self.settings.silence_duration / 0.1))
                    max_total = max(1, int(self.settings.max_record_seconds / 0.1))
                    if silence_chunks >= max_silence or len(recorded) >= max_total:
                        break

        if not recorded:
            return None, 0.0

        fd, tmp_wav = tempfile.mkstemp(suffix=".wav", prefix="magnus_mic_")
        os.close(fd)
        dest = Path(tmp_wav)
        full_audio = np.concatenate(recorded, axis=0)
        duration = round(len(full_audio) / float(sample_rate), 2)
        sf.write(str(dest), full_audio, sample_rate, subtype="PCM_16")
        return dest, duration

    async def _execute_magnus_response(
        self,
        input_wav_path: Optional[Path] = None,
        heard_transcript: str = "",
        text_prompt_override: Optional[str] = None,
    ) -> Optional[SpeechToSpeechResult]:
        """Run Gemini Speech-to-Speech generation and play Magnus's voice out loud."""
        self.speech_interrupted = False
        await self.set_state(
            "PROCESSING_S2S",
            f"🧠 Magnus thinking via {self.settings.gemini_model}...",
        )

        try:
            result = await self.engine.process_speech_to_speech_turn(
                settings=self.settings,
                history=self.transcript_history,
                input_wav_path=input_wav_path,
                heard_transcript=heard_transcript,
                text_prompt_override=text_prompt_override,
            )
        finally:
            if input_wav_path and input_wav_path.exists():
                input_wav_path.unlink(missing_ok=True)

        if self.speech_interrupted:
            if result.audio_file_path:
                result.audio_file_path.unlink(missing_ok=True)
            return None

        self.turns_completed += 1
        magnus_entry = {
            "role": "magnus",
            "speaker": self.settings.agent_name,
            "text": result.spoken_text,
            "heard": result.heard_transcript,
            "next_agent": result.next_agent,
            "engine_used": result.engine_used,
            "timestamp": time.strftime("%H:%M:%S"),
        }
        self.transcript_history.append(magnus_entry)

        await self.set_state(
            "SPEAKING",
            f"🔊 {self.settings.agent_name} speaking (Say 'STOP' or click Stop to interrupt!)",
        )
        await self.broadcast_event({
            "type": "magnus_speaking",
            "entry": magnus_entry,
            "api_status": self.engine.last_api_status,
        })

        if result.audio_file_path:
            await self.play_audio_on_speaker(result.audio_file_path, cleanup=True)

        if not self.speech_interrupted:
            next_state = "LISTENING" if self.is_listening else "IDLE"
            await self.set_state(next_state, f"✔ Finished speaking — passed mic to {result.next_agent}")
        return result

    async def trigger_manual_turn(self, custom_prompt: Optional[str] = None) -> Optional[SpeechToSpeechResult]:
        """Immediately trigger Magnus to speak from the Web UI."""
        last_room = ""
        for item in reversed(self.transcript_history):
            if item.get("role") == "room":
                last_room = item.get("text", "")
                break
        self._active_turn_task = asyncio.create_task(
            self._execute_magnus_response(
                heard_transcript=last_room,
                text_prompt_override=custom_prompt,
            )
        )
        return await self._active_turn_task
