import logging
from typing import AsyncGenerator

from google import genai
from google.genai import types

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

_MAX_HISTORY = 20  # keep last 10 turns (user + model pairs)


class GeminiLLM:
    def __init__(self, api_key: str):
        self._client = genai.Client(api_key=api_key)
        self._history: list[types.Content] = []

    async def generate_stream(self, user_message: str) -> AsyncGenerator[str, None]:
        self._history.append(
            types.Content(role="user", parts=[types.Part(text=user_message)])
        )
        if len(self._history) > _MAX_HISTORY:
            self._history = self._history[-_MAX_HISTORY:]

        collected: list[str] = []

        try:
            async for chunk in self._client.aio.models.generate_content_stream(
                model="gemini-2.0-flash",
                contents=self._history,
                config=types.GenerateContentConfig(
                    system_instruction=_SYSTEM,
                    tools=[types.Tool(google_search=types.GoogleSearch())],
                    temperature=1.1,
                ),
            ):
                if chunk.text:
                    collected.append(chunk.text)
                    yield chunk.text
        except Exception:
            log.exception("Gemini error")
            yield "Sorry [pause] I ran into an issue. Please try again."
            return

        if collected:
            self._history.append(
                types.Content(
                    role="model",
                    parts=[types.Part(text="".join(collected))],
                )
            )
