import math
import time
from .types import MemoryEntry


class MemoryPool:
    def __init__(self, max_entries: int = 100):
        self.max_entries = max_entries
        self._entries: list[MemoryEntry] = []

    @property
    def size(self) -> int:
        return len(self._entries)

    def _score(self, entry: MemoryEntry, keywords: list[str]) -> float:
        recency = 1 / (1 + (time.time() - entry.last_used) / 60)
        freq = math.log1p(entry.use_count) * 0.5
        kws = [entry.keyword.lower()] + [k.lower() for k in entry.related_keywords]
        relevance = 2.0 if any(k in kw or kw in k for k in keywords for kw in kws) else 0.0
        return recency + freq + relevance

    def upsert(self, entry: MemoryEntry) -> str | None:
        existing = next(
            (e for e in self._entries if e.keyword.lower() == entry.keyword.lower()), None
        )
        if existing:
            existing.content = entry.content
            existing.related_keywords = entry.related_keywords
            existing.last_used = time.time()
            existing.use_count += 1
            return None

        evicted_keyword = None
        if len(self._entries) >= self.max_entries:
            worst = min(self._entries, key=lambda e: self._score(e, []))
            evicted_keyword = worst.keyword
            self._entries.remove(worst)

        self._entries.append(entry)
        return evicted_keyword

    def retrieve(self, keywords: list[str], budget: int = 800) -> list[MemoryEntry]:
        scored = sorted(self._entries, key=lambda e: self._score(e, keywords), reverse=True)
        result = []
        remaining = budget
        for entry in scored:
            if remaining <= 0:
                break
            entry.last_used = time.time()
            entry.use_count += 1
            result.append(entry)
            remaining -= len(entry.content)
        return result

    def get_all(self) -> list[MemoryEntry]:
        return list(self._entries)

    def load(self, entries: list[MemoryEntry]) -> None:
        for entry in entries:
            self.upsert(entry)
