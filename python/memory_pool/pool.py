import math
import re
from .types import MemoryEntry

USE_COUNT_SHRINK_THRESHOLD = 10_000


class MemoryPool:
    """
    Keyword-indexed memory pool with:
    - Inverted index for fast candidate retrieval
    - Reversed matching: entry keywords searched in query (not query words in entry)
    - useCount-based scoring (no wall-clock time dependency)
    - Dynamic sizing by context window character budget
    - useCount overflow protection via halving
    """

    def __init__(
        self,
        max_entries: int = 1000,
        context_window: int | None = None,
        pool_ratio: float = 0.15,
    ):
        # If context_window is provided, cap pool by total character budget
        self.max_entries = max_entries
        self.max_pool_chars: int | None = int(context_window * pool_ratio) if context_window else None

        self._entries: list[MemoryEntry] = []
        self._total_chars: int = 0
        self._index: dict[str, set[str]] = {}   # keyword token → {entry_id, ...}

    @property
    def size(self) -> int:
        return len(self._entries)

    # ── Tokenisation ────────────────────────────────────────────────────────

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return [w for w in re.split(r'[\s\-_,。、!?！？\n]+', text.lower()) if len(w) > 1]

    # ── Inverted index maintenance ───────────────────────────────────────────

    def _index_entry(self, entry: MemoryEntry) -> None:
        for kw in [entry.keyword] + entry.related_keywords:
            for token in self._tokenize(kw):
                self._index.setdefault(token, set()).add(entry.id)

    def _deindex_entry(self, entry: MemoryEntry) -> None:
        for kw in [entry.keyword] + entry.related_keywords:
            for token in self._tokenize(kw):
                if token in self._index:
                    self._index[token].discard(entry.id)
                    if not self._index[token]:
                        del self._index[token]

    # ── Scoring ──────────────────────────────────────────────────────────────

    def _score(
        self,
        entry: MemoryEntry,
        query_lower: str,
        total_use_count: int,
        max_use_count: int,
    ) -> float:
        # Recency proxy: relative share of total accesses
        recency = entry.use_count / (total_use_count or 1)

        # Frequency: log-normalised against pool maximum
        freq = math.log1p(entry.use_count) / math.log1p(max_use_count or 1)

        # Relevance: entry keywords searched inside query string (reversed direction)
        all_kws = [entry.keyword.lower()] + [k.lower() for k in entry.related_keywords]
        relevance = 2.0 if any(kw in query_lower for kw in all_kws) else 0.0

        return recency + freq + relevance

    # ── Capacity helpers ─────────────────────────────────────────────────────

    def _is_full(self) -> bool:
        if self.max_pool_chars is not None:
            return self._total_chars >= self.max_pool_chars
        return len(self._entries) >= self.max_entries

    def _shrink_counts(self) -> None:
        """Halve all useCount values when any entry exceeds the overflow threshold."""
        if any(e.use_count > USE_COUNT_SHRINK_THRESHOLD for e in self._entries):
            for e in self._entries:
                e.use_count = max(1, e.use_count // 2)

    # ── Public API ───────────────────────────────────────────────────────────

    def upsert(self, entry: MemoryEntry) -> str | None:
        """Insert or update an entry. Returns evicted keyword if eviction occurred."""
        existing = next(
            (e for e in self._entries if e.keyword.lower() == entry.keyword.lower()), None
        )
        if existing:
            self._deindex_entry(existing)
            self._total_chars -= len(existing.content)
            existing.content = entry.content
            existing.related_keywords = entry.related_keywords
            existing.use_count += 1
            self._total_chars += len(existing.content)
            self._index_entry(existing)
            return None

        evicted_keyword = None
        if self._is_full():
            self._shrink_counts()
            total = sum(e.use_count for e in self._entries)
            max_uc = max((e.use_count for e in self._entries), default=1)
            worst = min(self._entries, key=lambda e: self._score(e, '', total, max_uc))
            evicted_keyword = worst.keyword
            self._deindex_entry(worst)
            self._total_chars -= len(worst.content)
            self._entries.remove(worst)

        self._entries.append(entry)
        self._total_chars += len(entry.content)
        self._index_entry(entry)
        return evicted_keyword

    def retrieve(self, query: str, budget: int = 800) -> list[MemoryEntry]:
        """Return highest-scoring entries whose keywords appear in query, within budget chars."""
        if not self._entries:
            return []

        query_lower = query.lower()
        total = sum(e.use_count for e in self._entries)
        max_uc = max(e.use_count for e in self._entries)

        # Inverted index → candidate entry IDs
        candidate_ids: set[str] = set()
        for token in self._tokenize(query_lower):
            if token in self._index:
                candidate_ids.update(self._index[token])

        # Fall back to full scan if index yields no candidates
        candidates = (
            [e for e in self._entries if e.id in candidate_ids]
            if candidate_ids else self._entries
        )

        scored = sorted(
            candidates,
            key=lambda e: self._score(e, query_lower, total, max_uc),
            reverse=True,
        )

        result, remaining = [], budget
        for entry in scored:
            if remaining <= 0:
                break
            entry.use_count += 1
            result.append(entry)
            remaining -= len(entry.content)
        return result

    def get_all(self) -> list[MemoryEntry]:
        return list(self._entries)

    def load(self, entries: list[MemoryEntry]) -> None:
        for entry in entries:
            self.upsert(entry)
