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
)


class ElevenLabsTTS:
    def __init__(self, api_key: str, voice_id: str):
        self._api_key = api_key
        self._voice_id = voice_id

    async def stream(self, text_gen: AsyncIterator[str]) -> AsyncGenerator[bytes, None]:
        url = _WS_URL.format(voice_id=self._voice_id)
        log.info("ElevenLabs: connecting  voice_id=%s", self._voice_id)

        if not self._api_key:
            log.error("ElevenLabs: ELEVENLABS_API_KEY / ELEVEN_LABS_API_KEY is not set")
            return

        try:
            async with websockets.connect(
                url,
                additional_headers={"xi-api-key": self._api_key},
                ping_interval=None,
            ) as ws:
                log.info("ElevenLabs: WebSocket connected")
                audio_q: asyncio.Queue[bytes | None] = asyncio.Queue(maxsize=64)

                bos = {
                    "text": " ",
                    "xi_api_key": self._api_key,
                    "voice_settings": {
                        "stability": 0.4,
                        "similarity_boost": 0.8,
                        "use_speaker_boost": True,
                    },
                    "generation_config": {
                        "chunk_length_schedule": [120, 160, 250, 290],
                    },
                }
                await ws.send(json.dumps(bos))
                log.info("ElevenLabs: sent BOS")

                async def _sender():
                    sent_chars = 0
                    try:
                        async for chunk in text_gen:
                            if chunk:
                                sent_chars += len(chunk)
                                log.debug("ElevenLabs: sending text (%d chars so far)", sent_chars)
                                await ws.send(json.dumps({"text": chunk}))
                        await ws.send(json.dumps({"text": ""}))
                        log.info("ElevenLabs: sent EOS  (total %d chars)", sent_chars)
                    except Exception:
                        log.exception("ElevenLabs: sender error")

                async def _receiver():
                    chunks_received = 0
                    bytes_received = 0
                    try:
                        async for raw in ws:
                            msg = json.loads(raw)
                            if err := msg.get("error"):
                                log.error("ElevenLabs: server error: %s", err)
                            if msg.get("audio"):
                                data = base64.b64decode(msg["audio"])
                                chunks_received += 1
                                bytes_received += len(data)
                                log.debug(
                                    "ElevenLabs: audio chunk #%d  %d bytes  (%d total)",
                                    chunks_received, len(data), bytes_received,
                                )
                                await audio_q.put(data)
                            if msg.get("isFinal"):
                                log.info(
                                    "ElevenLabs: isFinal  chunks=%d  bytes=%d",
                                    chunks_received, bytes_received,
                                )
                                break
                    except websockets.exceptions.ConnectionClosedError as e:
                        log.error("ElevenLabs: connection closed unexpectedly: %s", e)
                    except Exception:
                        log.exception("ElevenLabs: receiver error")
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
                log.info("ElevenLabs: stream complete")

        except websockets.exceptions.InvalidStatus as e:
            log.error(
                "ElevenLabs: WebSocket handshake rejected  status=%s  body=%s",
                e.response.status_code,
                e.response.body[:200] if hasattr(e.response, "body") else "",
            )
        except Exception:
            log.exception("ElevenLabs: unexpected error")
