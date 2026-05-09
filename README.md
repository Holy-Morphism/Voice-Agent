# Voice Agent

A real-time voice assistant: Deepgram STT → LangChain agent (GPT-4o + web search) → ElevenLabs TTS, all streamed end-to-end.

## Prerequisites

- Python 3.14+
- [uv](https://docs.astral.sh/uv/getting-started/installation/)

## Environment variables

Create a `.env` file in the project root before running either way:

```env
OPENAI_API_KEY=sk-...
DEEPGRAM_API_KEY=...
ELEVEN_LABS_API_KEY=sk_...
ELEVENLABS_VOICE_ID=...     # e.g. JBFqnCBsd6RMkjVDRZzb
```

API keys:
- OpenAI — [platform.openai.com/api-keys](https://platform.openai.com/api-keys)
- Deepgram — [console.deepgram.com](https://console.deepgram.com)
- ElevenLabs — [elevenlabs.io](https://elevenlabs.io) (Settings → API Keys; Voice ID from the voice library)

---

## Local setup

**1. Install dependencies**

```bash
uv sync
```

**2. Run**

```bash
uv run fastapi dev main.py
```

Open [http://localhost:8000](http://localhost:8000) and allow microphone access.

---

## Docker

**Build and start**

```bash
docker compose up --build
```

**Start (after first build)**

```bash
docker compose up
```

**Run in the background**

```bash
docker compose up -d
```

**Stop**

```bash
docker compose down
```

**Rebuild after code changes**

```bash
docker compose up --build --force-recreate
```

Open [http://localhost:8000](http://localhost:8000) and allow microphone access.
