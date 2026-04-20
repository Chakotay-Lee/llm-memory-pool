"""FastAPI server with SSE streaming and memory pool dashboard."""
import asyncio
import json
import os
from contextlib import asynccontextmanager
from pathlib import Path
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

load_dotenv()

from memory_pool import MemoryAgent

POOL_FILE = Path("./pool.json")

# Per-client SSE queues — each connected browser gets its own copy of events
_sse_clients: set[asyncio.Queue] = set()


def on_event(kind: str, data: dict) -> None:
    payload = json.dumps({"kind": kind, "data": data})
    for q in _sse_clients:
        try:
            q.put_nowait(payload)
        except asyncio.QueueFull:
            pass


agent = MemoryAgent(
    api_key=os.environ.get("GEMINI_API_KEY"),
    pool_file=str(POOL_FILE),
    on_event=on_event,
)

HTML_FILE = Path(__file__).parent / "public" / "index.html"


@asynccontextmanager
async def lifespan(app: FastAPI):
    await agent.load_pool()
    yield


app = FastAPI(title="LLM Memory Pool", lifespan=lifespan)


@app.get("/", response_class=HTMLResponse)
async def root():
    if HTML_FILE.exists():
        return HTML_FILE.read_text()
    return (
        "<h1>LLM Memory Pool</h1>"
        "<p>Endpoints: <code>POST /chat</code> &nbsp; "
        "<code>GET /events</code> &nbsp; "
        "<code>GET /debug/pool</code> &nbsp; "
        "<code>GET /debug/turns</code></p>"
    )


class ChatRequest(BaseModel):
    message: str


@app.post("/chat")
async def chat_endpoint(req: ChatRequest):
    async def generate():
        async for chunk in agent.chat_stream(req.message):
            yield chunk

    return StreamingResponse(generate(), media_type="text/plain; charset=utf-8")


@app.get("/events")
async def events_endpoint(request: Request):
    queue: asyncio.Queue = asyncio.Queue(maxsize=200)
    _sse_clients.add(queue)

    async def stream():
        try:
            while True:
                if await request.is_disconnected():
                    break
                try:
                    payload = await asyncio.wait_for(queue.get(), timeout=30)
                    yield {"data": payload}
                except asyncio.TimeoutError:
                    yield {"data": json.dumps({"kind": "ping", "data": {}})}
        finally:
            _sse_clients.discard(queue)

    return EventSourceResponse(stream())


@app.get("/debug/pool")
async def debug_pool():
    return {
        "size": agent.pool.size,
        "entries": [e.to_dict() for e in agent.pool.get_all()],
        "pendingTurns": len(agent.worker._pending),
        "isCompressing": agent.worker._compressing,
    }


@app.get("/debug/turns")
async def debug_turns():
    return {
        "turns": [
            {"role": t.role, "index": t.index, "chars": len(t.content)}
            for t in agent._turns
        ]
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
