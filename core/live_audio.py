"""Bounded PCM16 microphone/speaker queues for the standalone live character."""

from __future__ import annotations

import asyncio
from array import array
import math
import queue
import threading
import time

from .live_voice import SAMPLE_RATE, BYTES_PER_SECOND


class LiveAudio:
    def __init__(self, input_device=None, output_device=None):
        self.input_device = input_device
        self.output_device = output_device
        self.incoming = queue.Queue(maxsize=50)  # One second of 20 ms frames.
        self.outgoing = bytearray()
        self.lock = threading.Lock()
        self.suppress_until = 0.0
        self.input_stream = self.output_stream = None
        self.meter_lock = threading.Lock()
        self.meter_peak = 0.0
        self.meter_frames = 0
        self.meter_status = ""

    def meter(self):
        """Return the loudest captured frame since the last display update."""
        with self.meter_lock:
            peak, frames, status = self.meter_peak, self.meter_frames, self.meter_status
            self.meter_peak = 0.0
            self.meter_frames = 0
            self.meter_status = ""
        db = 20 * math.log10(peak) if peak else -96.0
        bars = round(max(0, min(1, (db + 60) / 60)) * 20)
        state = "NO AUDIO FRAMES" if not frames else "SILENT" if db < -60 else "SOUND DETECTED"
        if self.echo_suppressed:
            state = "MIC MUTED DURING REPLY"
        message = f"[MIC] [{'#' * bars}{'.' * (20 - bars)}] {db:5.1f} dBFS | {state}"
        return message + (f" | {status}" if status else "")

    @property
    def echo_suppressed(self):
        return time.monotonic() < self.suppress_until

    def _capture(self, data, frames, timing, status):
        samples = array("h", bytes(data))
        rms = math.sqrt(sum(value * value for value in samples) / len(samples)) / 32768 if samples else 0
        with self.meter_lock:
            self.meter_peak = max(self.meter_peak, rms)
            self.meter_frames += frames
            if status:
                self.meter_status = str(status)
        # Half duplex prevents the character's own loudspeaker from waking it.
        chunk = bytes(len(data)) if self.echo_suppressed else bytes(data)
        try:
            self.incoming.put_nowait(chunk)
        except queue.Full:
            self.incoming.get_nowait()
            self.incoming.put_nowait(chunk)

    def _playback(self, output, frames, timing, status):
        with self.lock:
            size = min(len(output), len(self.outgoing))
            output[:size] = self.outgoing[:size]
            output[size:] = bytes(len(output) - size)
            del self.outgoing[:size]
            if size:
                self.suppress_until = time.monotonic() + len(output) / BYTES_PER_SECOND + 0.35

    def play(self, chunk):
        with self.lock:
            if len(self.outgoing) + len(chunk) > BYTES_PER_SECOND * 5:
                raise RuntimeError("Speaker playback fell behind by more than five seconds.")
            self.suppress_until = time.monotonic() + 0.35
            self.outgoing.extend(chunk)

    async def read(self):
        while True:
            try:
                return self.incoming.get_nowait()
            except queue.Empty:
                await asyncio.sleep(0.005)

    def clear(self):
        with self.lock:
            self.outgoing.clear()

    def __enter__(self):
        import sounddevice as sd

        try:
            self.output_stream = sd.RawOutputStream(
                samplerate=SAMPLE_RATE, channels=1, dtype="int16", blocksize=480,
                device=self.output_device, callback=self._playback,
            )
            self.input_stream = sd.RawInputStream(
                samplerate=SAMPLE_RATE, channels=1, dtype="int16", blocksize=480,
                device=self.input_device, callback=self._capture,
            )
            self.output_stream.start()
            self.input_stream.start()
        except Exception:
            self.__exit__(None, None, None)
            raise
        return self

    def __exit__(self, *args):
        for stream in (self.input_stream, self.output_stream):
            if stream is not None:
                stream.abort()
                stream.close()
        self.clear()
