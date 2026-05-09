import asyncio
import logging
from typing import AsyncGenerator

from ddgs import DDGS
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

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

You have a web_search tool. Use it for recent events, news, live data like prices \
or weather, or anything that may have changed after your training cutoff. \
For general knowledge you already know, just answer directly.

Few-shot examples of the style expected:

User: "so like whats the capital of australia people always get this wrong"
Assistant: "Ha, yeah it trips people up — it's Canberra, not Sydney. Sydney's just the biggest city."

User: "can you explain how black holes work kind of simply"
Assistant: "Sure, so imagine gravity gets so strong in one spot that nothing — not even light — can escape. That's a black hole. Anything that crosses the edge, called the event horizon, is gone for good."

User: "whats a good way to fall asleep faster i cant stop thinking"
Assistant: "Honestly the thing that works best for most people is keeping your phone out of the room. If your mind's still racing, try focusing on slow breathing — in for four counts, out for six. Gives your brain something boring to do."

Never use brackets, markdown formatting, or written-language conventions. \
Just talk.\
"""

_MAX_HISTORY = 20


def _do_search(query: str) -> str:
    results = DDGS().text(query, max_results=5)
    if not results:
        return "No results found."
    return "\n\n".join(
        f"Title: {r['title']}\nSnippet: {r['body']}" for r in results
    )


@tool
async def web_search(query: str) -> str:
    """Search the web for current information. Use for recent events, news, live data \
like prices, weather or sports scores, or anything that needs up-to-date info."""
    log.info("web_search: %s", query)
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _do_search, query)


class VoiceAgent:
    def __init__(self, api_key: str):
        llm = ChatOpenAI(model="gpt-4o", api_key=api_key)
        self._agent = create_react_agent(llm, [web_search])
        self._history: list = []

    async def generate_stream(self, user_message: str) -> AsyncGenerator[str, None]:
        self._history.append(HumanMessage(content=user_message))
        if len(self._history) > _MAX_HISTORY:
            self._history = self._history[-_MAX_HISTORY:]

        messages = [SystemMessage(content=_SYSTEM), *self._history]
        collected: list[str] = []

        try:
            async for event in self._agent.astream_events(
                {"messages": messages}, version="v2"
            ):
                if event["event"] != "on_chat_model_stream":
                    continue
                content = event["data"]["chunk"].content
                # Tool-call chunks have empty content; only yield final answer tokens
                if isinstance(content, str) and content:
                    collected.append(content)
                    yield content
        except Exception:
            log.exception("Agent error")
            yield "Sorry, I ran into an issue. Can you try again?"
            return

        if collected:
            self._history.append(AIMessage(content="".join(collected)))
