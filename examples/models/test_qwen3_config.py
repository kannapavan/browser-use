"""
Quick test to verify browser-use connects to local qwen3.6-35b via LiteLLM at http://localhost:4000/v1

Usage:
    cd browser-use
    python3 examples/models/test_qwen3_config.py
"""

from dotenv import load_dotenv
import os

load_dotenv()

from browser_use.llm.litellm.chat import ChatLiteLLM

# The OpenAI client requires api_key to be set, even for local servers.
# Pass a dummy key — your proxy at localhost:4000 just ignores it.
os.environ['OPENAI_API_KEY'] = 'sk-notneeded-local-server'

llm = ChatLiteLLM(
    model='openai/qwen3.6-35b',
    api_key='sk-notneeded-local-server',  # required by OpenAI client; ignored by local proxy
    api_base='http://localhost:4000/v1',
    temperature=0.0,
    max_tokens=4096,
    drop_params=True,
)

print(f"Model:   {llm.model}")
print(f"Provider:{llm.provider}")
print(f"Name:    {llm.name}")
print(f"Base:    {llm.api_base}")
print()

import asyncio
from browser_use.llm.messages import UserMessage

async def test():
    result = await llm.ainvoke(
        messages=[UserMessage(content='Explain browser-use in one sentence.')],
    )
    print(f"Response: {result.completion}")
    print(f"Tokens:   prompt={result.usage.prompt_tokens}, completion={result.usage.completion_tokens}")
    print("✅ Connection working!")

asyncio.run(test())
