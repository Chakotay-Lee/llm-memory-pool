import asyncio
import time
import uuid
from openai import AsyncOpenAI
from .pool import MemoryPool
from .types import MemoryEntry, Turn
from .llm import chat_complete, parse_json

MAIN_MODEL = "gemma-4-31b-it"
MEMORY_MODEL = "gemma-4-26b-a4b-it"


def extract_keywords(text: str) -> list[str]:
    import re
    words = re.split(r"[\s,。、!?！？\n]+", text.lower())
    return [w for w in words if len(w) > 2][:10]


class MemoryWorker:
    def __init__(
        self,
        pool: MemoryPool,
        client: AsyncOpenAI,
        memory_model: str = MEMORY_MODEL,
        min_turns_to_compress: int = 2,
        on_event=None,
    ):
        self.pool = pool
        self.client = client
        self.memory_model = memory_model
        self.min_turns_to_compress = min_turns_to_compress
        self.on_event = on_event or (lambda kind, data: None)

        self._pending: list[Turn] = []
        self._compressing = False
        self._tasks: set[asyncio.Task] = set()   # keep strong refs to prevent GC cancellation

    def _spawn(self, coro) -> None:
        task = asyncio.create_task(coro)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    def add_turn(self, turn: Turn) -> None:
        self._pending.append(turn)
        if len(self._pending) >= self.min_turns_to_compress:
            batch = self._pending[:]
            self._pending.clear()
            self._spawn(self._compress(batch))

    async def get_memory(self, user_message: str, budget: int = 800) -> list[MemoryEntry]:
        entries = self.pool.retrieve(user_message, budget)
        self.on_event("memory_inject", {
            "count": len(entries),
            "poolSize": self.pool.size,
        })
        return entries

    async def _compress(self, turns: list[Turn]) -> None:
        if self._compressing:
            self._pending.extend(turns)
            return
        self._compressing = True
        self.on_event("compress_start", {"turnCount": len(turns)})

        turn_text = "\n".join(
            f"{'User' if t.role == 'user' else 'AI'}: {t.content}" for t in turns
        )

        try:
            strat_raw = await chat_complete(self.client, self.memory_model, [
                {"role": "system", "content": "You are a memory compression assistant. Output JSON only, no markdown."},
                {"role": "user", "content": (
                    f"Analyze these conversation turns. Identify key topics and keywords.\n\n{turn_text}\n\n"
                    "PRIORITY: Always extract user-stated facts first (name, role, project, goals, preferences) "
                    "before summarizing AI responses.\n\n"
                    "Output JSON:\n{\"strategy\":\"one sentence\",\"keywords\":[\"kw1\",\"kw2\"]}"
                )},
            ])

            try:
                strat = parse_json(strat_raw)
                keywords = strat.get("keywords", [])
                strategy = strat.get("strategy", "general compression")
            except Exception:
                keywords = []
                strategy = "general compression"

            comp_raw = await chat_complete(self.client, self.memory_model, [
                {"role": "system", "content": "You are a memory compression assistant. Output JSON only, no markdown."},
                {"role": "user", "content": (
                    f"Strategy: {strategy}\nKeywords: {', '.join(keywords)}\n\n"
                    f"Conversation:\n{turn_text}\n\n"
                    "Rules for memory entries:\n"
                    "1. FIRST extract any user-stated facts: name, identity, role, project, preferences.\n"
                    "2. Store as 'User stated: ...' — never infer or paraphrase user identity.\n"
                    "3. Then summarize key topics from the AI responses (max 50 words each).\n"
                    "4. Only store session-specific info, not general world knowledge.\n\n"
                    "Output JSON:\n{\"entries\":[{\"keyword\":\"topic\",\"content\":\"summary\",\"relatedKeywords\":[\"rel1\"]}]}"
                )},
            ])

            parsed = parse_json(comp_raw)
            entries_data = parsed.get("entries", []) if isinstance(parsed, dict) else []
            evicted_count = 0

            for e in entries_data:
                entry = MemoryEntry(
                    keyword=e["keyword"],
                    content=e["content"],
                    related_keywords=e.get("relatedKeywords", []),
                    created_at_turn=turns[-1].index,
                )
                evicted = self.pool.upsert(entry)
                if evicted:
                    evicted_count += 1
                    self.on_event("pool_evict", {"keyword": evicted})

            self.on_event("compress_done", {
                "entriesAdded": len(entries_data),
                "evicted": evicted_count,
                "poolSize": self.pool.size,
            })

        except Exception as err:
            self.on_event("compress_error", {"error": str(err)})
        finally:
            self._compressing = False
            if len(self._pending) >= self.min_turns_to_compress:
                batch = self._pending[:]
                self._pending.clear()
                self._spawn(self._compress(batch))
