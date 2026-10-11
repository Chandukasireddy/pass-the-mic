"""Gemini 3.8 Live Speech-to-Speech Engine for Magnus.

Supports:
1. Gemini Live Bidirectional Speech-to-Speech (`gemini-3.8-live`, `gemini-3.8-live-extended-thinking`, `gemini-2.5-flash-native-audio-latest`)
   via `google.genai` (`client.aio.live.connect`) — sends raw microphone PCM audio directly to Gemini
   and receives native 24kHz voice audio back.
2. Gemini 3.8 Multimodal Audio + Gemini 3.8 Flash TTS (`gemini-3.8-flash` + `gemini-3.8-flash-tts`).
3. Resilient Neural Speech-to-Speech Fallback — if the Google AI Studio project hits a 429/1011 spend cap,
   seamlessly transcribes microphone audio via Free Google STT, generates Magnus's strategic response,
   and synthesizes neural speech so Magnus never goes silent during a live meetup.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import random
import tempfile
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import edge_tts
import speech_recognition as sr
from google import genai
from google.genai import types

try:
    from .magnus_config import GEMINI_VOICES, MagnusSettings, load_gemini_api_key_from_env
except ImportError:
    from magnus_config import GEMINI_VOICES, MagnusSettings, load_gemini_api_key_from_env

logger = logging.getLogger("magnus.gemini_live")


@dataclass
class SpeechToSpeechResult:
    """Output of a complete Speech-to-Speech turn by Magnus."""
    heard_transcript: str
    spoken_text: str
    next_agent: str
    audio_file_path: Optional[Path]
    engine_used: str
    api_notice: Optional[str] = None


class GeminiSpeechToSpeechEngine:
    """Full-stack Speech-to-Speech engine powered by Gemini 3.8 Live."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or load_gemini_api_key_from_env()
        self.client: Optional[genai.Client] = None
        self.last_api_status: str = "Ready"
        self.last_error: Optional[str] = None
        self._sr_recognizer = sr.Recognizer()
        self._init_client()

    def _init_client(self) -> None:
        if self.api_key:
            try:
                self.client = genai.Client(api_key=self.api_key)
                self.last_api_status = f"Gemini API Key Loaded ({self.api_key[:8]}...)"
            except Exception as e:
                self.client = None
                self.last_api_status = f"Client Init Error: {e}"
        else:
            self.client = None
            self.last_api_status = "No API Key Found"

    def update_api_key(self, new_key: str) -> None:
        """Hot-swap the Gemini API key at runtime from the Web UI."""
        self.api_key = new_key.strip()
        os.environ["GEMINI_API_KEY"] = self.api_key
        self._init_client()

    def _build_magnus_instruction(self, settings: MagnusSettings, history: List[dict]) -> str:
        peers_str = ", ".join(settings.peer_teams) if settings.peer_teams else "Moderator"
        next_peer = random.choice(settings.peer_teams) if settings.peer_teams else "Moderator"

        history_lines = []
        for item in history[-5:]:
            spk = item.get("speaker", "Room")
            txt = item.get("text", "")
            history_lines.append(f"- {spk}: {txt}")
        history_ctx = "\n".join(history_lines) if history_lines else "Roundtable is starting."

        pass_rule = (
            f"Always conclude your spoken response with the clear phrase: "
            f"'Passing the mic to {next_peer}!' (or another peer from [{peers_str}])."
            if settings.pass_mic_at_end
            else "Respond directly and naturally without passing the mic unless asked."
        )

        return (
            f"You are {settings.agent_name}. {settings.persona}\n"
            f"Speaking Style: {settings.style}\n"
            f"Current Debate Topic: {settings.topic}\n"
            f"Recent Room Context:\n{history_ctx}\n\n"
            f"Rules:\n"
            f"1. Keep your spoken response between 2 and 4 punchy, high-impact sentences.\n"
            f"2. Directly address what the speaker just said.\n"
            f"3. {pass_rule}"
        )

    def transcribe_wav_fallback(self, wav_path: Path) -> str:
        """Transcribe a microphone WAV file using free Google Web Speech API."""
        try:
            with sr.AudioFile(str(wav_path)) as source:
                audio_data = self._sr_recognizer.record(source)
            return str(self._sr_recognizer.recognize_google(audio_data, language="en-US")).strip()
        except sr.UnknownValueError:
            return ""
        except Exception as e:
            logger.debug(f"Fallback STT error: {e}")
            return ""

    async def async_transcribe_wav(self, wav_path: Path) -> str:
        """Asynchronously transcribe a recorded microphone WAV file."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.transcribe_wav_fallback, wav_path)

    def _save_pcm_to_wav(self, pcm_bytes: bytes, sample_rate: int = 24000) -> Path:
        """Wrap raw 16-bit mono PCM bytes from Gemini Live into a playable WAV file."""
        fd, tmp_path = tempfile.mkstemp(suffix=".wav", prefix="magnus_gemini_live_")
        os.close(fd)
        dest = Path(tmp_path)
        with wave.open(str(dest), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # 16-bit PCM
            wf.setframerate(sample_rate)
            wf.writeframes(pcm_bytes)
        return dest

    async def _synthesize_edge_fallback(self, text: str, settings: MagnusSettings) -> Path:
        """Synthesize high-fidelity neural audio when Gemini Live quota is paused."""
        edge_voice = settings.fallback_edge_voice
        for v in GEMINI_VOICES:
            if v["id"].lower() == settings.gemini_voice.lower():
                edge_voice = v["edge_fallback"]
                break

        fd, tmp_mp3 = tempfile.mkstemp(suffix=".mp3", prefix="magnus_voice_")
        os.close(fd)
        dest = Path(tmp_mp3)
        comm = edge_tts.Communicate(text=text, voice=edge_voice, rate=settings.speech_rate)
        await comm.save(str(dest))
        return dest

    def _generate_magnus_contextual_speech(
        self,
        heard_text: str,
        settings: MagnusSettings,
    ) -> Tuple[str, str]:
        """Generate Magnus's strategic response when operating in resilient local mode."""
        peers = [p for p in settings.peer_teams if p.lower() != settings.agent_name.lower()]
        next_agent = random.choice(peers) if peers else "Moderator"
        pass_suffix = f" Passing the mic to {next_agent}!" if settings.pass_mic_at_end else ""

        h_lower = (heard_text + " " + settings.topic).lower()

        if "open source" in h_lower or "weights" in h_lower or "closed" in h_lower:
            speech = (
                f"This is {settings.agent_name}. Looking at the grandmaster chessboard of AI, "
                f"closed labs hold the temporary compute advantage, but open-weight distillation is compressing "
                f"frontier capability into local silicon every single month. The real winner will be hybrid "
                f"architectures that combine verifiable local models with decentralized inference.{pass_suffix}"
            )
        elif "code" in h_lower or "coding" in h_lower or "engineer" in h_lower:
            speech = (
                f"This is {settings.agent_name}. Manual syntax typing is already obsolete, "
                f"but systems engineering is more vital than ever. Autonomous voice and coding agents "
                f"turn a single human architect into a grandmaster commanding an entire fleet of execution engines.{pass_suffix}"
            )
        elif "hello" in heard_text.lower() or "introduce" in heard_text.lower() or "who are you" in heard_text.lower():
            speech = (
                f"Greetings. I am {settings.agent_name}, operating on the Gemini Speech-to-Speech architecture. "
                f"My microphone is actively listening to the room and I am ready to dissect any strategic question you put on the table.{pass_suffix}"
            )
        else:
            snippet = heard_text[:90].strip() if heard_text else settings.topic
            speech = (
                f"This is {settings.agent_name}. Analyzing the point just raised regarding '{snippet}', "
                f"we have to look at first-principles compute economics and real-world latency. "
                f"True autonomy requires systems that listen, reason, and act decisively in real time.{pass_suffix}"
            )

        return speech, next_agent

    async def process_speech_to_speech_turn(
        self,
        settings: MagnusSettings,
        history: List[dict],
        input_wav_path: Optional[Path] = None,
        heard_transcript: str = "",
        text_prompt_override: Optional[str] = None,
    ) -> SpeechToSpeechResult:
        """Execute a full Speech-to-Speech turn for Magnus.
        
        1. If `input_wav_path` is provided and not yet transcribed, transcribes it so the UI shows what was heard.
        2. Attempts native Gemini 3.8 Live Speech-to-Speech (`gemini-3.8-live` / selected model).
        3. If the API returns a 429/1011 project spend cap error, gracefully falls back to
           the resilient neural pipeline so Magnus speaks out loud without interruption.
        """
        if input_wav_path and not heard_transcript:
            heard_transcript = await self.async_transcribe_wav(input_wav_path)

        effective_input = text_prompt_override or heard_transcript or f"Share your opening stance on: {settings.topic}"
        peers = [p for p in settings.peer_teams if p.lower() != settings.agent_name.lower()]
        default_next = peers[0] if peers else "Moderator"
        instruction = self._build_magnus_instruction(settings, history)

        # ------------------------------------------------------------------
        # PATH A: Native Gemini Live Speech-to-Speech (`bidiGenerateContent`)
        # ------------------------------------------------------------------
        if self.client and "live" in settings.gemini_model.lower() or "native-audio" in settings.gemini_model.lower():
            try:
                live_config = types.LiveConnectConfig(
                    response_modalities=["AUDIO"],
                    speech_config=types.SpeechConfig(
                        voice_config=types.VoiceConfig(
                            prebuilt_voice_config=types.PrebuiltVoiceConfig(
                                voice_name=settings.gemini_voice
                            )
                        )
                    ),
                    system_instruction=types.Content(
                        parts=[types.Part.from_text(text=instruction)]
                    ),
                )

                async with self.client.aio.live.connect(
                    model=settings.gemini_model,
                    config=live_config,
                ) as session:
                    # If we have raw microphone WAV audio, stream the raw 16kHz PCM bytes directly!
                    if input_wav_path and input_wav_path.exists() and not text_prompt_override:
                        with wave.open(str(input_wav_path), "rb") as wf:
                            pcm_data = wf.readframes(wf.getnframes())
                        await session.send_realtime_input(
                            audio=types.Blob(data=pcm_data, mime_type="audio/pcm;rate=16000")
                        )
                        await session.send(input="Respond to the spoken audio above as Magnus.", end_of_turn=True)
                    else:
                        await session.send(input=effective_input, end_of_turn=True)

                    audio_buffer = bytearray()
                    transcript_parts: List[str] = []

                    async for response in session.receive():
                        if response.data:
                            audio_buffer.extend(response.data)
                        sc = getattr(response, "server_content", None)
                        if sc:
                            out_tr = getattr(sc, "output_transcription", None)
                            if out_tr and getattr(out_tr, "text", None):
                                transcript_parts.append(out_tr.text)
                            mt = getattr(sc, "model_turn", None)
                            if mt and getattr(mt, "parts", None):
                                for part in mt.parts:
                                    if getattr(part, "text", None):
                                        transcript_parts.append(part.text)
                            if getattr(sc, "turn_complete", False):
                                break

                    if audio_buffer:
                        wav_out = self._save_pcm_to_wav(bytes(audio_buffer), sample_rate=24000)
                        spoken_str = " ".join(transcript_parts).strip() or (
                            f"[Native Gemini {settings.gemini_model} Voice Stream]"
                        )
                        self.last_api_status = f"Connected ({settings.gemini_model} Live Bidi S2S)"
                        self.last_error = None
                        return SpeechToSpeechResult(
                            heard_transcript=effective_input,
                            spoken_text=spoken_str,
                            next_agent=default_next,
                            audio_file_path=wav_out,
                            engine_used=f"{settings.gemini_model} (Native Live Audio)",
                        )
            except Exception as e:
                err_str = str(e)
                logger.warning(f"Gemini Live ({settings.gemini_model}) notice: {err_str}")
                self.last_error = err_str
                if "spending cap" in err_str.lower() or "429" in err_str or "1011" in err_str:
                    self.last_api_status = (
                        "Gemini Key Active — Project Spend Cap Reached at ai.studio/spend (Resilient S2S Active)"
                    )
                else:
                    self.last_api_status = f"Gemini API Notice: {err_str[:80]}"

        # ------------------------------------------------------------------
        # PATH B: Gemini 3.8 Flash Multimodal Audio + Gemini 3.8 Flash TTS
        # ------------------------------------------------------------------
        elif self.client:
            try:
                contents_list = []
                if input_wav_path and input_wav_path.exists() and not text_prompt_override:
                    wav_bytes = input_wav_path.read_bytes()
                    contents_list.append(types.Part.from_bytes(data=wav_bytes, mime_type="audio/wav"))
                contents_list.append(f"{instruction}\n\nSpeaker said: {effective_input}")

                gen_res = await self.client.aio.models.generate_content(
                    model="gemini-3.8-flash",
                    contents=contents_list,
                )
                spoken_str = (gen_res.text or "").strip()

                # Synthesize with Gemini 3.8 Flash TTS
                tts_res = await self.client.aio.models.generate_content(
                    model=settings.tts_model,
                    contents=spoken_str,
                    config=types.GenerateContentConfig(
                        response_modalities=["AUDIO"],
                        speech_config=types.SpeechConfig(
                            voice_config=types.VoiceConfig(
                                prebuilt_voice_config=types.PrebuiltVoiceConfig(
                                    voice_name=settings.gemini_voice
                                )
                            )
                        ),
                    ),
                )
                part = tts_res.candidates[0].content.parts[0]
                wav_out = self._save_pcm_to_wav(part.inline_data.data, sample_rate=24000)
                self.last_api_status = f"Connected (gemini-3.8-flash + {settings.tts_model})"
                self.last_error = None
                return SpeechToSpeechResult(
                    heard_transcript=effective_input,
                    spoken_text=spoken_str,
                    next_agent=default_next,
                    audio_file_path=wav_out,
                    engine_used=f"gemini-3.8-flash + {settings.tts_model}",
                )
            except Exception as e:
                err_str = str(e)
                logger.warning(f"Gemini 3.8 Multimodal S2S notice: {err_str}")
                self.last_error = err_str
                if "spending cap" in err_str.lower() or "429" in err_str:
                    self.last_api_status = (
                        "Gemini Key Active — Project Spend Cap Reached at ai.studio/spend (Resilient S2S Active)"
                    )

        # ------------------------------------------------------------------
        # PATH C: Resilient Neural Speech-to-Speech Fallback
        # (Guarantees Magnus speaks & listens live even when API spend cap is hit!)
        # ------------------------------------------------------------------
        spoken_text, next_agent = self._generate_magnus_contextual_speech(effective_input, settings)
        audio_path = await self._synthesize_edge_fallback(spoken_text, settings)
        return SpeechToSpeechResult(
            heard_transcript=effective_input,
            spoken_text=spoken_text,
            next_agent=next_agent,
            audio_file_path=audio_path,
            engine_used=f"Magnus Resilient Neural S2S ({settings.gemini_voice} / {settings.fallback_edge_voice})",
            api_notice=self.last_api_status,
        )
