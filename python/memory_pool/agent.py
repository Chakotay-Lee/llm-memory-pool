import os
import json
from pathlib import Path
from openai import AsyncOpenAI
from typing import AsyncIterator, Callable
from .pool import MemoryPool
from .types import MemoryEntry, Turn
from .worker import MemoryWorker, extract_keywords
from .llm import stream_chat_complete, chat_complete

BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai"
DEFAULT_MAIN_MODEL = "gemma-4-31b-it"
DEFAULT_MEMORY_MODEL = "gemma-4-26b-a4b-it"


class MemoryAgent:
    """High-level agent combining LLM chat with async memory compression."""

    def __init__(
        self,
        api_key: str | None = None,
        main_model: str = DEFAULT_MAIN_MODEL,
        memory_model: str = DEFAULT_MEMORY_MODEL,
        base_url: str = BASE_URL,
        max_raw_turns: int = 4,
        min_turns_to_compress: int = 2,
        max_pool_entries: int = 1000,
        context_window: int | None = None,
        pool_ratio: float = 0.15,
        inject_budget: int = 800,
        pool_file: str | None = None,
        on_event: Callable[[str, dict], None] | None = None,
    ):
        self.main_model = main_model
        self.max_raw_turns = max_raw_turns
        self.inject_budget = inject_budget
        self.pool_file = Path(pool_file) if pool_file else None
        self.on_event = on_event or (lambda kind, data: None)

        self.client = AsyncOpenAI(
            api_key=api_key or os.environ.get("GEMINI_API_KEY", ""),
            base_url=base_url,
        )
        self.pool = MemoryPool(
            max_entries=max_pool_entries,
            context_window=context_window,
            pool_ratio=pool_ratio,
        )
        self.worker = MemoryWorker(
            pool=self.pool,
            client=self.client,
            memory_model=memory_model,
            min_turns_to_compress=min_turns_to_compress,
            on_event=self._handle_event,
        )

        self._turns: list[Turn] = []
        self._turn_index = 0
        self._tasks: set = set()   # strong refs to prevent GC cancellation

    def _spawn(self, coro) -> None:
        import asyncio
        task = asyncio.create_task(coro)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    def _handle_event(self, kind: str, data: dict) -> None:
        self.on_event(kind, data)
        if kind == "compress_done" and self.pool_file:
            self._spawn(self._save_pool())

    async def load_pool(self) -> None:
        if not self.pool_file or not self.pool_file.exists():
            return
        try:
            entries = json.loads(self.pool_file.read_text())
            self.pool.load([MemoryEntry.from_dict(e) for e in entries])
            self.on_event("pool_state", {"restored": self.pool.size, "source": str(self.pool_file)})
        except Exception as err:
            self.on_event("compress_error", {"error": f"pool load failed: {err}"})

    async def _save_pool(self) -> None:
        if not self.pool_file:
            return
        try:
            self.pool_file.write_text(
                json.dumps([e.to_dict() for e in self.pool.get_all()], indent=2)
            )
        except Exception:
            pass

    def _build_messages(self, user_message: str, memory_text: str) -> list[dict]:
        recent = self._turns[-self.max_raw_turns:]
        messages = [{"role": "system", "content": "You are a helpful assistant."}]
        for t in recent[:-1]:
            messages.append({"role": t.role, "content": t.content})
        content = f"## Conversation Memory\n{memory_text}\n\n---\n\n{user_message}" if memory_text else user_message
        messages.append({"role": "user", "content": content})
        return messages

    async def chat_stream(self, user_message: str) -> AsyncIterator[str]:
        user_turn = Turn(role="user", content=user_message, index=self._turn_index)
        self._turn_index += 1
        self._turns.append(user_turn)
        self.on_event("turn_user", {"index": user_turn.index, "chars": len(user_message)})

        mem_entries = await self.worker.get_memory(user_message, self.inject_budget)
        memory_text = "\n".join(f"[{e.keyword}] {e.content}" for e in mem_entries)

        messages = self._build_messages(user_message, memory_text)
        self.on_event("context_built", {
            "rawTurns": min(len(self._turns), self.max_raw_turns),
            "memoryEntries": len(mem_entries),
            "totalMessages": len(messages),
        })

        full_text = ""
        async for chunk in stream_chat_complete(self.client, self.main_model, messages):
            full_text += chunk
            yield chunk

        asst_turn = Turn(role="assistant", content=full_text, index=self._turn_index)
        self._turn_index += 1
        self._turns.append(asst_turn)
        self.on_event("turn_assistant", {"index": asst_turn.index, "chars": len(full_text)})

        self.worker.add_turn(user_turn)
        self.worker.add_turn(asst_turn)

    async def chat(self, user_message: str) -> str:
        result = ""
        async for chunk in self.chat_stream(user_message):
            result += chunk
        return result
