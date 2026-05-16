"""
Test browser-use with local qwen3.6-35b via LiteLLM proxy at http://localhost:4000/v1

Run:
    cd browser-use
    python examples/models/test_local_qwen.py
"""

from dotenv import load_dotenv
import asyncio
from browser_use import Agent, Browser

# Load .env (LITE_DEFAULT_API_KEY, LITE_DEFAULT_BASE_URL, DEFAULT_LLM)
load_dotenv()

from browser_use.llm.models import get_llm_by_name

# Create LLM from env vars
print('🔌 Connecting to local LLM at http://localhost:4000/v1 ...')
llm = get_llm_by_name('lite_qwen3_6_35b')
print(f'✅ Connected! Model: {llm.model}, Provider: {llm.provider}')
print(f'   API Base: {llm.api_base}')

async def main():
    # Simple test: ask the agent to search Google and return a result
    task = 'Go to Google and search for "browser-use AI agent"'

    print(f'\n🚀 Running agent with task: "{task}"')
    print('=' * 70)

    agent = Agent(
        task=task,
        llm=llm,
        max_actions_per_step=1,
        use_vision=True,
    )

    history = await agent.run()
    print(f'\n✅ Done! Total steps: {len(history.history)}')

if __name__ == '__main__':
    asyncio.run(main())
