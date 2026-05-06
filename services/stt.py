import asyncio
import logging
from typing import Awaitable, Callable

from deepgram import AsyncDeepgramClient

log = logging.getLogger(__name__)


class DeepgramSTT:
    def __init__(self, api_key: str):
        self._client = AsyncDeepgramClient(api_key=api_key)
        self._audio_q: asyncio.Queue[bytes | None] = asyncio.Queue(maxsize=200)
        self._task: asyncio.Task | None = None

    async def start(self, on_final: Callable[[str], Awaitable[None]]) -> None:
        self._task = asyncio.create_task(self._run(on_final))

    async def _run(self, on_final: Callable[[str], Awaitable[None]]) -> None:
        try:
            async with self._client.listen.v2.connect(
                model="flux-general-en",
                eot_threshold=0.7,
                eager_eot_threshold=0.5,
                eot_timeout_ms=5000,
            ) as socket:
                log.info("Deepgram Flux connected")

                async def _sender():
                    while True:
                        chunk = await self._audio_q.get()
                        if chunk is None:
                            await socket.send_close_stream()
                            break
                        await socket.send_media(chunk)

                send_task = asyncio.create_task(_sender())

                async for msg in socket:
                    if getattr(msg, "type", None) != "TurnInfo":
                        continue
                    if getattr(msg, "event", None) == "EndOfTurn":
                        text = getattr(msg, "transcript", "").strip()
                        if text:
                            asyncio.create_task(on_final(text))

                await asyncio.gather(send_task, return_exceptions=True)
                log.info("Deepgram Flux closed")
        except Exception:
            log.exception("Deepgram STT error")

    async def send(self, audio: bytes) -> None:
        try:
            self._audio_q.put_nowait(audio)
        except asyncio.QueueFull:
            pass  # drop oldest indirectly by skipping; prevents unbounded growth

    async def finish(self) -> None:
        await self._audio_q.put(None)
        if self._task:
            try:
                await asyncio.wait_for(self._task, timeout=5.0)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                self._task.cancel()
