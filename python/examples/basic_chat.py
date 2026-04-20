"""Basic example: console chat with memory pool persistence."""
import asyncio
import os
from dotenv import load_dotenv

load_dotenv()

import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))

from memory_pool import MemoryAgent


def on_event(kind: str, data: dict) -> None:
    color = {
        "compress_start": "\033[33m",
        "compress_done":  "\033[32m",
        "compress_error": "\033[31m",
        "memory_inject":  "\033[34m",
        "pool_evict":     "\033[35m",
        "pool_state":     "\033[36m",
        "context_built":  "\033[33m",
    }.get(kind, "\033[90m")
    print(f"{color}[{kind}] {data}\033[0m", flush=True)


async def main():
    agent = MemoryAgent(
        api_key=os.environ.get("GEMINI_API_KEY"),
        max_raw_turns=4,
        min_turns_to_compress=2,
        pool_file="./pool.json",
        on_event=on_event,
    )
    await agent.load_pool()

    print("Memory Pool Chat — type 'quit' to exit, 'pool' to inspect pool\n")

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not user_input:
            continue
        if user_input.lower() == "quit":
            break
        if user_input.lower() == "pool":
            entries = agent.pool.get_all()
            print(f"\n── Pool ({len(entries)} entries) ──")
            for e in sorted(entries, key=lambda x: -x.use_count):
                print(f"  [{e.keyword}] {e.content[:80]}  (used×{e.use_count})")
            continue

        print("\nAssistant: ", end="", flush=True)
        async for chunk in agent.chat_stream(user_input):
            print(chunk, end="", flush=True)
        print()


if __name__ == "__main__":
    asyncio.run(main())
