import os
from typing import Awaitable, Callable
from openai import AsyncOpenAI

MODEL_GPT_4 = "gpt-4-turbo"


async def stream_openai_response(
    messages, api_key: str, callback: Callable[[str], Awaitable[None]]
):
    client = AsyncOpenAI(api_key=api_key)

    model = MODEL_GPT_4

    # Base parameters
    params = {
        "model": model,
        "messages": messages,
        "stream": True,
        "timeout": 600,
        "max_tokens": 4096,
        "temperature": 0
    }

    completion = await client.chat.completions.create(**params)
    full_response = ""
    async for chunk in completion:
        content = chunk.choices[0].delta.content or ""
        full_response += content
        await callback(content)

    return full_response
