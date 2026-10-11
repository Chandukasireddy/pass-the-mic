"""Speech-to-Text (STT) and Acoustic Microphone Listener for pass-the-mic.

Allows every laptop in the room to listen through its physical microphone and
transcribe spoken voices into text using:
  1. Groq Whisper (`whisper-large-v3-turbo` — sub-second free tier)
  2. Google Web Speech API (via `SpeechRecognition` — 100% free, zero API keys!)
  3. OpenAI Whisper (`whisper-1` — optional fallback)
"""

from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path
from typing import Callable, Optional, Tuple
from dotenv import load_dotenv
import speech_recognition as sr

from .audio import async_record_from_mic, record_from_mic

load_dotenv()
logger = logging.getLogger("pass_the_mic.stt")


class STTClient:
    """Multi-backend Speech-to-Text client for transcribing microphone audio."""

    def __init__(
        self,
        provider: Optional[str] = None,
        groq_api_key: Optional[str] = None,
        openai_api_key: Optional[str] = None,
        groq_stt_model: Optional[str] = None,
    ):
        self.groq_api_key = groq_api_key or os.getenv("GROQ_API_KEY")
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        self.groq_stt_model = groq_stt_model or os.getenv("GROQ_STT_MODEL", "whisper-large-v3-turbo")

        requested = (provider or os.getenv("STT_PROVIDER", "auto")).lower()
        if requested == "auto":
            if self.groq_api_key and not self.groq_api_key.startswith("your_"):
                self.provider = "groq"
            elif self.openai_api_key and not self.openai_api_key.startswith("your_"):
                self.provider = "openai"
            else:
                # 100% Free, keyless Google Web Speech API via SpeechRecognition
                self.provider = "google"
        else:
            self.provider = requested

        self._groq_client = None
        self._openai_client = None
        self._recognizer = sr.Recognizer()
        self._init_clients()

    def _init_clients(self) -> None:
        if self.provider == "groq" and self.groq_api_key:
            try:
                from groq import Groq
                self._groq_client = Groq(api_key=self.groq_api_key)
            except Exception as e:
                logger.warning(f"Failed to init Groq STT client: {e}. Falling back to Google Free STT.")
                self.provider = "google"
        elif self.provider == "openai" and self.openai_api_key:
            try:
                from openai import OpenAI
                self._openai_client = OpenAI(api_key=self.openai_api_key)
            except Exception as e:
                logger.warning(f"Failed to init OpenAI STT client: {e}. Falling back to Google Free STT.")
                self.provider = "google"

    def transcribe_file(self, audio_path: str | Path) -> str:
        """Transcribe a recorded WAV/MP3 audio file to text."""
        path = Path(audio_path)
        if not path.exists():
            return ""

        # 1. Try Groq Whisper first if configured
        if self.provider == "groq" and self._groq_client:
            try:
                with open(path, "rb") as f:
                    transcription = self._groq_client.audio.transcriptions.create(
                        file=(path.name, f.read()),
                        model=self.groq_stt_model,
                        language="en",
                        response_format="text",
                    )
                text = str(transcription).strip()
                if text:
                    return text
            except Exception as e:
                logger.warning(f"Groq Whisper STT failed ({e}), falling back to Google Free STT.")

        # 2. Try OpenAI Whisper if configured
        if self.provider == "openai" and self._openai_client:
            try:
                with open(path, "rb") as f:
                    transcription = self._openai_client.audio.transcriptions.create(
                        file=f,
                        model="whisper-1",
                        language="en",
                    )
                text = getattr(transcription, "text", str(transcription)).strip()
                if text:
                    return text
            except Exception as e:
                logger.warning(f"OpenAI Whisper STT failed ({e}), falling back to Google Free STT.")

        # 3. Zero-Cost, Keyless Google Web Speech API via SpeechRecognition
        try:
            with sr.AudioFile(str(path)) as source:
                audio_data = self._recognizer.record(source)
            text = self._recognizer.recognize_google(audio_data, language="en-US")
            return str(text).strip()
        except sr.UnknownValueError:
            # Audio contained silence or unintelligible background noise
            return ""
        except Exception as e:
            logger.debug(f"SpeechRecognition error: {e}")
            return ""

    async def async_transcribe_file(self, audio_path: str | Path) -> str:
        """Asynchronously transcribe an audio file without blocking the event loop."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.transcribe_file, audio_path)

    def listen_and_transcribe(
        self,
        energy_threshold: float = 0.015,
        silence_duration: float = 1.8,
        max_duration: float = 25.0,
        wait_timeout: float = 30.0,
        fixed_duration: Optional[float] = None,
        device: Optional[int] = None,
        on_speech_start: Optional[Callable[[], None]] = None,
    ) -> Tuple[str, float]:
        """Record speech from the laptop's microphone and transcribe it to text.
        
        Returns:
            Tuple of (transcribed_text, audio_duration_seconds).
        """
        wav_path, duration = record_from_mic(
            energy_threshold=energy_threshold,
            silence_duration=silence_duration,
            max_duration=max_duration,
            wait_timeout=wait_timeout,
            fixed_duration=fixed_duration,
            device=device,
            on_speech_start=on_speech_start,
        )
        if not wav_path:
            return "", 0.0

        try:
            text = self.transcribe_file(wav_path)
            return text, duration
        finally:
            try:
                if wav_path.exists():
                    wav_path.unlink()
            except Exception:
                pass

    async def async_listen_and_transcribe(
        self,
        energy_threshold: float = 0.015,
        silence_duration: float = 1.8,
        max_duration: float = 25.0,
        wait_timeout: float = 30.0,
        fixed_duration: Optional[float] = None,
        device: Optional[int] = None,
        on_speech_start: Optional[Callable[[], None]] = None,
    ) -> Tuple[str, float]:
        """Asynchronously record speech from the microphone and transcribe it to text."""
        wav_path, duration = await async_record_from_mic(
            energy_threshold=energy_threshold,
            silence_duration=silence_duration,
            max_duration=max_duration,
            wait_timeout=wait_timeout,
            fixed_duration=fixed_duration,
            device=device,
            on_speech_start=on_speech_start,
        )
        if not wav_path:
            return "", 0.0

        try:
            text = await self.async_transcribe_file(wav_path)
            return text, duration
        finally:
            try:
                if wav_path.exists():
                    wav_path.unlink()
            except Exception:
                pass


def get_stt_client(provider: Optional[str] = None) -> STTClient:
    """Convenience factory to create an STTClient instance."""
    return STTClient(provider=provider)
