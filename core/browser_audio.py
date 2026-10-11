"""Browser-backed microphone/speaker for the live character.

A local page captures the mic with getUserMedia and plays replies, so the
browser's own microphone permission is used instead of the terminal's.
Exposes the same interface as LiveAudio.
"""

from __future__ import annotations

import asyncio
import json
from http import HTTPStatus
from pathlib import Path
import queue
import time

from .live_audio import LiveAudio
from .live_voice import BYTES_PER_SECOND

PAGE = Path(__file__).with_name("browser_audio.html")


class BrowserAudio(LiveAudio):
    def __init__(self, port=8765, host="127.0.0.1"):
        super().__init__()
        self.host = host
        self.port = port
        self.server = None
        self.client = None
        self.connected = asyncio.Event()
        self.lost = False
        self.out = asyncio.Queue()
        self.play_end = 0.0
        self.commands = asyncio.Queue()  # "start" / "stop" clicked in the page.
        self.status = {"type": "status", "state": "idle"}

    @property
    def url(self):
        return f"http://{self.host}:{self.port}/"

    # --- audio interface used by stream_session -------------------------

    async def read(self):
        while True:
            try:
                return self.incoming.get_nowait()
            except queue.Empty:
                if self.lost:
                    raise RuntimeError("Browser tab disconnected; stopping the session.")
                await asyncio.sleep(0.005)

    def play(self, chunk):
        now = time.monotonic()
        self.play_end = max(self.play_end, now) + len(chunk) / BYTES_PER_SECOND
        if self.play_end - now > 5:
            raise RuntimeError("Speaker playback fell behind by more than five seconds.")
        self.suppress_until = self.play_end + 0.35
        self.out.put_nowait(bytes(chunk))

    def clear(self):
        self.play_end = 0.0
        keep = []
        while not self.out.empty():
            item = self.out.get_nowait()
            if isinstance(item, dict):  # Status updates must survive a flush.
                keep.append(item)
        for item in keep:
            self.out.put_nowait(item)
        self.out.put_nowait(None)  # Tells the page to drop scheduled audio.

    def set_status(self, state, **extra):
        """Tell the page the agent state: idle, connecting, live or stopping."""
        self.status = {"type": "status", "state": state, **extra}
        self.out.put_nowait(self.status)

    def next_command(self):
        try:
            return self.commands.get_nowait()
        except asyncio.QueueEmpty:
            return None

    # --- local server ----------------------------------------------------

    def _process_request(self, connection, request):
        if request.path == "/ws":
            origin = request.headers.get("Origin")
            allowed = {f"http://127.0.0.1:{self.port}", f"http://localhost:{self.port}"}
            if origin not in allowed:
                return connection.respond(HTTPStatus.FORBIDDEN, "Forbidden\n")
            return None
        if request.path in ("/", "/index.html"):
            response = connection.respond(HTTPStatus.OK, PAGE.read_text(encoding="utf-8"))
            response.headers["Content-Type"] = "text/html; charset=utf-8"
            response.headers["Cache-Control"] = "no-store"
            return response
        return connection.respond(HTTPStatus.NOT_FOUND, "Not found\n")

    async def _handle(self, ws):
        if self.client is not None:
            await ws.close(1008, "Another tab is already connected")
            return
        self.client = ws
        self.connected.set()
        self.out.put_nowait(self.status)
        sender = asyncio.create_task(self._send_loop(ws))
        try:
            async for message in ws:
                if isinstance(message, bytes):
                    if len(message) % 2 == 0:
                        self._capture(message, len(message) // 2, None, None)
                    continue
                try:
                    command = json.loads(message)
                except ValueError:
                    continue
                if isinstance(command, dict) and command.get("type") == "agent" \
                        and command.get("action") in ("start", "stop"):
                    self.commands.put_nowait(command["action"])
        finally:
            sender.cancel()
            self.client = None
            self.lost = True

    async def _send_loop(self, ws):
        while True:
            item = await self.out.get()
            if item is None:
                item = {"type": "clear"}
            await ws.send(json.dumps(item) if isinstance(item, dict) else item)

    async def wait_connected(self, timeout=180):
        await asyncio.wait_for(self.connected.wait(), timeout)

    async def __aenter__(self):
        from websockets.asyncio.server import serve

        self.server = await serve(
            self._handle, self.host, self.port,
            process_request=self._process_request, compression=None,
        )
        self.port = self.server.sockets[0].getsockname()[1]
        return self

    async def __aexit__(self, *args):
        self.clear()
        if self.server is not None:
            self.server.close()
            await self.server.wait_closed()
