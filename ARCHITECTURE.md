# AI Agent HA - Architecture

## Overview

This Home Assistant custom component provides AI-powered assistance using a modular plugin architecture.

## Structure

```
custom_components/ai_agent_ha/
├── __init__.py              # HA integration setup
├── agent_compat.py          # Public interface (AiAgentHaAgent)
│
├── providers/               # AI Provider Plugins
│   ├── registry.py          # ProviderRegistry + AIProvider ABC
│   ├── base_client.py       # BaseHTTPClient with retry logic
│   ├── openai.py            # OpenAI GPT models
│   ├── gemini.py            # Google Gemini
│   ├── anthropic.py         # Anthropic Claude
│   ├── groq.py              # Groq (fast inference)
│   ├── openrouter.py        # OpenRouter (multi-provider)
│   └── local.py             # Ollama / local LLMs
│
├── managers/                # Domain Logic
│   ├── entity_manager.py    # Entity state operations
│   ├── registry_manager.py  # HA registry operations
│   ├── automation_manager.py # Automation CRUD
│   ├── dashboard_manager.py # Dashboard CRUD
│   └── control_manager.py   # Service calls, entity control
│
├── core/                    # Orchestration
│   ├── agent.py             # Slim Agent orchestrator
│   ├── query_processor.py   # Query processing pipeline
│   ├── response_parser.py   # AI response parsing
│   └── conversation.py      # Conversation history
│
├── tools/                   # AI Tools (function calling)
├── rag/                     # RAG / Semantic Search
└── ...
```

## Adding a New AI Provider

1. Create `providers/new_provider.py`:

```python
from .registry import ProviderRegistry
from .base_client import BaseHTTPClient

@ProviderRegistry.register("new_provider")
class NewProvider(BaseHTTPClient):
    API_URL = "https://api.example.com/v1/chat"

    @property
    def supports_tools(self) -> bool:
        return True

    def _build_headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.config['token']}"}

    def _build_payload(self, messages, **kwargs):
        return {"model": self.config.get("model"), "messages": messages}

    def _extract_response(self, data) -> str:
        return data["choices"][0]["message"]["content"]
```

2. Add import in `providers/__init__.py`:
```python
from . import new_provider  # noqa: F401
```

3. Done! No changes to core code needed.

## Available Providers

| Provider | Registration | Tool Support |
|----------|--------------|--------------|
| OpenAI | `@ProviderRegistry.register("openai")` | Yes |
| Gemini | `@ProviderRegistry.register("gemini")` | Yes |
| Anthropic | `@ProviderRegistry.register("anthropic")` | Yes |
| Groq | `@ProviderRegistry.register("groq")` | Yes |
| OpenRouter | `@ProviderRegistry.register("openrouter")` | Yes |
| Local/Ollama | `@ProviderRegistry.register("local")` | No |

## Design Patterns

- **Registry Pattern** - Dynamic provider registration
- **Strategy Pattern** - Interchangeable AI providers
- **Template Method** - BaseHTTPClient with hooks
- **Facade Pattern** - Managers simplify HA operations
- **Dependency Injection** - Managers injected into Agent

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run only new architecture tests
pytest tests/test_providers/ tests/test_managers/ tests/test_core/ -v

# Check coverage
pytest --cov=custom_components.ai_agent_ha --cov-report=html
```
