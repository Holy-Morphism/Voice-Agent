import logging
from typing import AsyncGenerator

from openai import AsyncOpenAI

log = logging.getLogger(__name__)

_SYSTEM = """\
You are an expressive AI voice assistant. Your responses are converted to speech.

Use emotional cues in brackets so the TTS engine delivers the right tone:
  [pause]        — brief pause
  [excited]      — enthusiastic
  [whispers]     — soft, quiet
  [giggles]      — lighthearted laugh
  [sarcastically]— sarcastic
  [warmly]       — gentle, kind
  [thoughtfully] — reflective, pondering

When you use web search, ALWAYS say:
  "I searched the internet [pause] and I found this [excited] ..."

Speak conversationally — you are talking, not writing. Be vivid and concise.\
"""

_MAX_HISTORY = 20


class OpenAILLM:
    def __init__(self, api_key: str):
        self._client = AsyncOpenAI(api_key=api_key)
        self._history: list[dict] = []

    async def generate_stream(self, user_message: str) -> AsyncGenerator[str, None]:
        self._history.append({"role": "user", "content": user_message})
        if len(self._history) > _MAX_HISTORY:
            self._history = self._history[-_MAX_HISTORY:]

        collected: list[str] = []

        try:
            stream = await self._client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "system", "content": _SYSTEM}, *self._history],
                stream=True,
            )
            async for chunk in stream:
                delta = chunk.choices[0].delta.content
                if delta:
                    collected.append(delta)
                    yield delta
        except Exception:
            log.exception("OpenAI error")
            yield "Sorry [pause] I ran into an issue. Please try again."
            return

        if collected:
            self._history.append({"role": "assistant", "content": "".join(collected)})
