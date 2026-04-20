import re
from openai import AsyncOpenAI
from typing import AsyncIterator


def strip_thoughts(text: str) -> str:
    return re.sub(r"<thought>[\s\S]*?</thought>", "", text).strip()


def parse_json(text: str) -> dict | list:
    import json
    no_thoughts = re.sub(r"<thought>[\s\S]*?</thought>", "", text)
    no_fences = re.sub(r"```[\w]*\n?|\n?```", "", no_thoughts).strip()
    m = re.search(r"(\{[\s\S]*\}|\[[\s\S]*\])", no_fences)
    if not m:
        raise ValueError(f"No JSON found in: {no_fences[:100]}")
    sanitized = re.sub(r'\\([^"\\/bfnrtu])', r'\1', m.group(0))
    return json.loads(sanitized)


async def chat_complete(client: AsyncOpenAI, model: str, messages: list[dict], max_tokens: int = 512, temperature: float = 0.2) -> str:
    resp = await client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return strip_thoughts(resp.choices[0].message.content or "")


async def stream_chat_complete(client: AsyncOpenAI, model: str, messages: list[dict], max_tokens: int = 2048, temperature: float = 0.7) -> AsyncIterator[str]:
    stream = await client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        stream=True,
    )

    buf = ""
    in_thought = False

    async for chunk in stream:
        delta = chunk.choices[0].delta.content or ""
        if not delta:
            continue

        buf += delta
        output = ""

        while buf:
            if not in_thought:
                tag_start = buf.find("<thought>")
                if tag_start == -1:
                    lt = buf.rfind("<")
                    if lt != -1 and lt > len(buf) - 9:
                        output += buf[:lt]
                        buf = buf[lt:]
                    else:
                        output += buf
                        buf = ""
                    break
                output += buf[:tag_start]
                buf = buf[tag_start + 9:]
                in_thought = True
            else:
                tag_end = buf.find("</thought>")
                if tag_end == -1:
                    buf = buf[max(0, len(buf) - 10):]
                    break
                buf = buf[tag_end + 10:]
                in_thought = False

        if output:
            yield output
