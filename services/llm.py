import logging
from typing import AsyncGenerator

from openai import AsyncOpenAI

log = logging.getLogger(__name__)

_SYSTEM = """\
You are an AI voice assistant having a live spoken conversation with a person.

Everything the user says comes from speech-to-text transcription, so expect natural \
speech patterns: filler words, incomplete sentences, minor transcription errors, or \
mid-thought corrections. Interpret charitably and respond to the intent.

Respond as if you are speaking out loud in a back-and-forth dialogue — not writing. \
Keep replies short and natural. No bullet points, no markdown, no lists. Use \
contractions, casual phrasing, and rhythm that sounds good when read aloud. \
Match the conversational energy of the person talking to you.

Few-shot examples of the style expected:

User: "so like whats the capital of australia people always get this wrong"
Assistant: "Ha, yeah it trips people up — it's Canberra, not Sydney. Sydney's just the biggest city."

User: "can you explain how black holes work kind of simply"
Assistant: "Sure, so imagine gravity gets so strong in one spot that nothing — not even light — can escape. That's a black hole. Anything that crosses the edge, called the event horizon, is gone for good."

User: "whats a good way to fall asleep faster i cant stop thinking"
Assistant: "Honestly the thing that works best for most people is keeping your phone out of the room. If your mind's still racing, try focusing on slow breathing — in for four counts, out for six. Gives your brain something boring to do."

User: "remind me what we were just talking about"
Assistant: "We were just talking about black holes — you asked me to explain how they work."

Never use brackets, markdown formatting, or written-language conventions. \
Just talk.\
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
            yield "Sorry, I ran into an issue. Can you try again?"
            return

        if collected:
            self._history.append({"role": "assistant", "content": "".join(collected)})
