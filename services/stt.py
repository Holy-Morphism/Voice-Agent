import asyncio
import logging
from typing import Awaitable, Callable

from deepgram import AsyncDeepgramClient

log = logging.getLogger(__name__)

_SAMPLE_RATE = 16000


def _get(msg, key: str, default=None):
    """Works for both Pydantic models and plain dicts."""
    if isinstance(msg, dict):
        return msg.get(key, default)
    return getattr(msg, key, default)


class DeepgramSTT:
    def __init__(self, api_key: str):
        self._client = AsyncDeepgramClient(api_key=api_key)
        self._audio_q: asyncio.Queue[bytes | None] = asyncio.Queue(maxsize=200)
        self._task: asyncio.Task | None = None

    async def start(
        self,
        on_update: Callable[[str], Awaitable[None]],
        on_final: Callable[[str], Awaitable[None]],
    ) -> None:
        self._task = asyncio.create_task(self._run(on_update, on_final))

    async def _run(
        self,
        on_update: Callable[[str], Awaitable[None]],
        on_final: Callable[[str], Awaitable[None]],
    ) -> None:
        try:
            async with self._client.listen.v2.connect(
                model="flux-general-en",
                encoding="linear16",
                sample_rate=_SAMPLE_RATE,
                eot_threshold=0.6,
                eager_eot_threshold=0.5,
                eot_timeout_ms=2000,
            ) as connection:
                log.info("Deepgram Flux connected (linear16 %dHz)", _SAMPLE_RATE)

                async def _sender():
                    chunks_sent = 0
                    try:
                        while True:
                            chunk = await self._audio_q.get()
                            if chunk is None:
                                await connection.send_close_stream()
                                log.info("Deepgram sender closed  chunks=%d", chunks_sent)
                                break
                            await connection.send_media(chunk)
                            chunks_sent += 1
                            if chunks_sent == 1:
                                log.info("Deepgram: first PCM chunk sent (%d bytes)", len(chunk))
                            elif chunks_sent % 50 == 0:
                                log.info("Deepgram: sent %d chunks", chunks_sent)
                    except Exception:
                        log.exception("Deepgram sender error")

                send_task = asyncio.create_task(_sender())

                async for msg in connection:
                    t          = _get(msg, "type")
                    event      = _get(msg, "event")
                    transcript = (_get(msg, "transcript") or "").strip()
                    confidence = _get(msg, "end_of_turn_confidence", 0.0)

                    log.info(
                        "Deepgram msg: type=%-10s event=%-16s confidence=%.2f  %r",
                        t, event, confidence,
                        (transcript[:60] + "…") if len(transcript) > 60 else transcript,
                    )

                    if t != "TurnInfo":
                        continue
                    if event in ("Update", "StartOfTurn") and transcript:
                        asyncio.create_task(on_update(transcript))
                    elif event in ("EndOfTurn", "EagerEndOfTurn") and transcript:
                        asyncio.create_task(on_final(transcript))

                results = await asyncio.gather(send_task, return_exceptions=True)
                for r in results:
                    if isinstance(r, Exception):
                        log.error("Deepgram sender task failed: %s", r)
                log.info("Deepgram Flux closed")
        except Exception:
            log.exception("Deepgram STT error")

    async def send(self, audio: bytes) -> None:
        try:
            self._audio_q.put_nowait(audio)
        except asyncio.QueueFull:
            log.warning("Deepgram audio queue full — dropping frame")

    async def finish(self) -> None:
        await self._audio_q.put(None)
        if self._task:
            try:
                await asyncio.wait_for(self._task, timeout=5.0)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                self._task.cancel()
