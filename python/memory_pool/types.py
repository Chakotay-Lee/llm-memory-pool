from dataclasses import dataclass, field
from typing import Literal
import uuid
import time


@dataclass
class Turn:
    role: Literal["user", "assistant"]
    content: str
    index: int = 0


@dataclass
class MemoryEntry:
    keyword: str
    content: str
    related_keywords: list[str] = field(default_factory=list)
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    last_used: float = field(default_factory=time.time)
    use_count: int = 1
    created_at_turn: int = 0

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "keyword": self.keyword,
            "content": self.content,
            "relatedKeywords": self.related_keywords,
            "lastUsed": self.last_used,
            "useCount": self.use_count,
            "createdAtTurn": self.created_at_turn,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "MemoryEntry":
        return cls(
            id=d.get("id", str(uuid.uuid4())),
            keyword=d["keyword"],
            content=d["content"],
            related_keywords=d.get("relatedKeywords", []),
            last_used=d.get("lastUsed", time.time()),
            use_count=d.get("useCount", 1),
            created_at_turn=d.get("createdAtTurn", 0),
        )
