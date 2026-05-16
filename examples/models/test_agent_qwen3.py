"""
Quick test: browser-use Agent with local qwen3.6-35b via LiteLLM.

Auto-detects display: runs UI browser if a display is available (DISPLAY/WAYLAND_DISPLAY env),
otherwise falls back to headless mode.

Run:
    python3 test_agent_qwen3.py                                          # detects display automatically
    python3 test_agent_qwen3.py https://google.com "Search for browser-use"
    python3 test_agent_qwen3.py https://github.com "List top 5 trending repos"
    python3 test_agent_qwen3.py https://example.com "Get the heading" headless  # force headless
    python3 test_agent_qwen3.py https://example.com "Get the heading" ui        # force UI
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
FORCE_MODE = sys.argv[3].lower() if len(sys.argv) > 3 else None  # 'ui' or 'headless'

from browser_use import Agent, ChatLiteLLM, BrowserProfile

# --- Build task from URL + instruction ---
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

# --- Auto-detect display: UI browser if display exists, headless otherwise ---
def has_display():
    """Check if a graphical display is available."""
    return bool(os.environ.get('DISPLAY') or os.environ.get('WAYLAND_DISPLAY'))

# --- Determine browser mode ---
if FORCE_MODE == 'ui':
    HEADLESS = False
elif FORCE_MODE == 'headless':
    HEADLESS = True
else:
    HEADLESS = not has_display()

browser_mode = 'UI (interactive)' if not HEADLESS else 'Headless (non-interactive)'

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
    max_actions_per_step=2,
    use_vision=False,  # Set True if using UI mode with screenshots
    browser_profile=BrowserProfile(
        headless=HEADLESS,
        args=[] if not HEADLESS else [
            '--no-sandbox',
            '--disable-gpu',
            '--disable-dev-shm-usage',
        ],
    ),
)

print('=== Browser-Use Agent ===')
print(f'  Mode:         {browser_mode}')
print(f'  Task:         {agent.task}')
print(f'  LLM:          qwen3.6-35b via LiteLLM')
print(f'  URL:          {BASE_URL or "default"}')
print()

print('🚀 Running Agent...')
result = agent.run_sync()
print(f'  Steps:        {len(result.history.history)}')
if result.history.history:
    last = result.history.history[-1]
    if last.model_output:
        print(f'  Final output: {last.model_output}')
print('✅ Agent round-trip complete!')
