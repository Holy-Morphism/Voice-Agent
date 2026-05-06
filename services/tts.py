import asyncio
import base64
import json
import logging
from typing import AsyncGenerator, AsyncIterator

import websockets

log = logging.getLogger(__name__)

_WS_URL = (
    "wss://api.elevenlabs.io/v1/text-to-speech/{voice_id}/stream-input"
    "?model_id=eleven_flash_v2_5"
    "&output_format=mp3_44100_128"
    "&optimize_streaming_latency=3"
)


class ElevenLabsTTS:
    def __init__(self, api_key: str, voice_id: str):
        self._api_key = api_key
        self._voice_id = voice_id

    async def stream(self, text_gen: AsyncIterator[str]) -> AsyncGenerator[bytes, None]:
        url = _WS_URL.format(voice_id=self._voice_id)
        audio_q: asyncio.Queue[bytes | None] = asyncio.Queue(maxsize=64)

        async with websockets.connect(
            url,
            additional_headers={"xi-api-key": self._api_key},
            ping_interval=None,
        ) as ws:
            await ws.send(
                json.dumps({
                    "text": " ",
                    "voice_settings": {
                        "stability": 0.4,
                        "similarity_boost": 0.8,
                        "style": 0.35,
                        "use_speaker_boost": True,
                    },
                    "generation_config": {
                        "chunk_length_schedule": [120, 160, 250, 290],
                    },
                })
            )

            async def _sender():
                try:
                    async for chunk in text_gen:
                        if chunk:
                            await ws.send(json.dumps({"text": chunk}))
                    await ws.send(json.dumps({"text": ""}))
                except Exception:
                    log.exception("ElevenLabs sender error")

            async def _receiver():
                try:
                    async for raw in ws:
                        msg = json.loads(raw)
                        if msg.get("audio"):
                            await audio_q.put(base64.b64decode(msg["audio"]))
                        if msg.get("isFinal"):
                            break
                except Exception:
                    log.exception("ElevenLabs receiver error")
                finally:
                    await audio_q.put(None)

            send_task = asyncio.create_task(_sender())
            recv_task = asyncio.create_task(_receiver())

            while True:
                chunk = await audio_q.get()
                if chunk is None:
                    break
                yield chunk

            await asyncio.gather(send_task, recv_task, return_exceptions=True)
