import asyncio
import json
import logging
import os

from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket
from starlette.websockets import WebSocketDisconnect
from fastapi.responses import FileResponse, Response

from services.llm import VoiceAgent
from services.stt import DeepgramSTT
from services.tts import ElevenLabsTTS

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger(__name__)

app = FastAPI(title="Voice Agent")


@app.get("/")
async def index():
    return FileResponse("frontend/index.html")


@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return FileResponse("frontend/favicon.ico") if os.path.exists("frontend/favicon.ico") else Response(status_code=204)


@app.websocket("/ws")
async def session(ws: WebSocket):
    await ws.accept()
    log.info("session opened")

    stt = DeepgramSTT(os.environ["DEEPGRAM_API_KEY"])
    llm = VoiceAgent(os.environ["OPENAI_API_KEY"])
    tts = ElevenLabsTTS(
        api_key=os.environ.get("ELEVENLABS_API_KEY") or os.environ.get("ELEVEN_LABS_API_KEY", ""),
        voice_id=os.environ.get("ELEVENLABS_VOICE_ID", ""),
    )

    lock = asyncio.Lock()

    async def handle_utterance(text: str) -> None:
        if lock.locked():
            log.info("busy — skipping: %s", text)
            return
        async with lock:
            try:
                log.info("user: %s", text)
                await ws.send_json({"type": "transcript", "text": text})
                await ws.send_json({"type": "state", "state": "thinking"})

                async def text_stream():
                    async for chunk in llm.generate_stream(text):
                        await ws.send_json({"type": "response_text", "text": chunk})
                        yield chunk

                await ws.send_json({"type": "state", "state": "speaking"})
                async for audio in tts.stream(text_stream()):
                    await ws.send_bytes(audio)

                await ws.send_json({"type": "audio_done"})
                await ws.send_json({"type": "state", "state": "listening"})
            except Exception:
                log.exception("pipeline error")
                await ws.send_json({"type": "state", "state": "listening"})

    async def handle_update(text: str) -> None:
        try:
            await ws.send_json({"type": "live_transcript", "text": text})
        except Exception:
            pass

    await stt.start(on_update=handle_update, on_final=handle_utterance)

    chunks_received = 0
    try:
        while True:
            msg = await ws.receive()
            if msg["type"] == "websocket.disconnect":
                log.info("session closed by client  audio_chunks_received=%d", chunks_received)
                break
            if msg.get("bytes"):
                chunks_received += 1
                if chunks_received == 1:
                    log.info("first audio chunk from browser (%d bytes)", len(msg["bytes"]))
                elif chunks_received % 50 == 0:
                    log.info("audio chunks received from browser: %d", chunks_received)
                await stt.send(msg["bytes"])
    except WebSocketDisconnect:
        log.info("session closed by client")
    except Exception:
        log.exception("session error")
    finally:
        await stt.finish()
        log.info("session cleaned up")
