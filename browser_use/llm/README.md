# Browser Use LLMs

We officially support the following LLMs:

- OpenAI
- Anthropic
- Google
- Groq
- Ollama
- DeepSeek

- Mistral

- LiteLLM (local + cloud models via [litellm](https://github.com/BerriAI/litellm))
  - Local models (no API key needed): `ChatLiteLLM(model='ollama/llama3.2')`
  - Cloud models: `ChatLiteLLM(model='openrouter/meta-llama/llama-3.1-8b-instruct')`
  - Providers: Ollama, vLLM, LMStudio, text-generation-webui, OpenRouter, Anyscale, AnyScale, local OpenAI-compatible servers, and any litellm-supported provider
  - Install: `pip install browser-use[litellm]`
  - Example: `examples/models/litellm_local.py`

- Cerebras


## Migrating from LangChain

Because of how we implemented the LLMs, we can technically support anything. If you want to use a LangChain model, you can use the `ChatLangchain` (NOT OFFICIALLY SUPPORTED) class.

You can find all the details in the [LangChain example](/examples/models/langchain/example.py). We suggest you grab that code and use it as a reference.
