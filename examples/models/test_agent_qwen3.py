"""
"
Quick test: browser-use Agent with local qwen3.6-35b via LiteLLM.

Run:
    python3 test_agent_qwen3.py                                          # uses default website & task
    python3 test_agent_qwen3.py https://google.com "Search for browser-use"
    python3 test_agent_qwen3.py https://github.com "List top 5 trending repos"
"""

from dotenv import load_dotenv
import os
import sys

load_dotenv()
os.environ['OPENAI_API_KEY'] = 'sk-notneeded-local-server'

# --- CLI arguments ---
BASE_URL = sys.argv[1] if len(sys.argv) > 1 else None
TASK_INSTRUCTION = os.getenv('BROWSER_USE_TASK')
TASK_EXTRA = sys.argv[2] if len(sys.argv) > 2 else None

from browser_use import Agent, ChatLiteLLM, BrowserProfile

# --- Build full task from URL + instruction ---
def build_task(base_url, instruction):
    """Combine URL + instruction into a full agent task."""
    if base_url and instruction:
        return f'Go to {base_url} and {instruction}'
    elif base_url:
        return f'Go to {base_url} and tell me what you find on the page'
    elif instruction:
        return instruction
    else:
        return 'Go to https://example.com and tell me the page title'

llm = ChatLiteLLM(
    model='openai/qwen3.6-35b',
    api_key='sk-notneeded-local-server',
    api_base='http://localhost:4000/v1',
    max_tokens=4096,
    drop_params=True,
)

agent = Agent(
    task=build_task(BASE_URL, TASK_EXTRA if TASK_EXTRA else TASK_INSTRUCTION),
    llm=llm,
    max_actions_per_step=1,
    use_vision=False,  # Saves tokens — we just want to test structured output
    browser_profile=BrowserProfile(
        headless=True,
        args=[
            '--no-sandbox',
            '--disable-gpu',
            '--disable-dev-shm-usage',
        ],
    ),
)

print('🚀 Running Agent...')
result = agent.run_sync()
print(f'Steps taken: {len(result.history.history)}')
if result.history.history:
    print(f'Last response: {result.history.history[-1].model_output}')
print('✅ Agent round-trip complete!')
