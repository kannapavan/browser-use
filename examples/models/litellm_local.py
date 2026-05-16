"""
Local LLM via LiteLLM - works with Ollama, vLLM, LMStudio, text-generation-webui,
and any OpenAI-compatible server.

Setup:

  pip install browser-use[litellm]  # or: pip install litellm

Example models to try locally:
  - ollama pull llama3.2:3b          (~2GB)
  - ollama pull mistral:v2           (~4GB)
  - ollama pull qwen2.5:7b           (~5GB)

Usage:
  From this file (direct instantiation):
    python -m examples.models.litellm_local

  Via config.env / config.json (get_llm_by_name):
    Export LITE_DEFAULT_API_KEY (optional - many local servers don't need it):
      export LITE_DEFAULT_API_KEY="sk-notneeded"
    Export LITE_DEFAULT_BASE_URL (optional - for servers that need it):
      export LITE_DEFAULT_BASE_URL="http://localhost:1234/v1"
    Then use:
      llm = get_llm_by_name('lite_ollama_llama32')
"""

from dotenv import load_dotenv

from browser_use import Agent, ChatLiteLLM

load_dotenv()

# =====================================================================
# Option 1: Direct ChatLiteLLM instantiation (recommended for this demo)
# =====================================================================

# Ollama: no api_key or api_base needed (uses default http://localhost:11434)
llm = ChatLiteLLM(model='ollama/llama3.2:3b', temperature=0.0, max_tokens=2048)

# ---------------------------------------------------------------------
# Option 2: LMStudio / text-generation-webui (OpenAI-compatible server)
# ---------------------------------------------------------------------
# llm = ChatLiteLLM(
#     model='openai/Qwen2.5-7B-Instruct',  # model name as seen by OpenAI endpoint
#     api_base='http://localhost:1234/v1',
#     api_key='ls-xxxxx',   # optional, many local servers don't require auth
#     provider_override='openai',  # force OpenAI protocol
# )
# ---------------------------------------------------------------------
# Option 3: vLLM OpenAI-compatible API
# ---------------------------------------------------------------------
# llm = ChatLiteLLM(
#     model='Meta-Llama-3.1-8B-Instruct',
#     api_base='http://localhost:8000/v1',
#     provider_override='openai',
# )
# ---------------------------------------------------------------------
# Option 4: Cloud providers via litellm proxy
# ---------------------------------------------------------------------
# llm = ChatLiteLLM(
#     model='openrouter/meta-llama/llama-3.1-8b-instruct',
#     api_key='sk-or-xxxxx',
# )
# ---------------------------------------------------------------------
# Option 5: Anyscale endpoint
# ---------------------------------------------------------------------
# llm = ChatLiteLLM(
#     model='anyscale/meta-llama/Llama-3.1-8B-Instruct',
#     api_key='any-xxxxx',
# )

# =====================================================================
# Run the agent
# =====================================================================

agent = Agent(task='Find the founders of browser-use on Google', llm=llm, use_vision=True, max_actions_per_step=1)
agent.run_sync()
