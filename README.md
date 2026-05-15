# Week 3 Project 3: Build an Ask-the-Web Agent (Perplexity-style)

## Files

- `main.py` - complete Week 3 flow in one runnable script
- `requirements.txt` - dependencies
- `agent.py` / `tools.py` - earlier scaffold versions (optional)

## Setup

```bash
cd project_3_ask_the_web_agent
pip install -r requirements.txt
```

## Run

```bash
python main.py
```

## Ollama Requirement (matching solution notebook)

This script uses OpenAI-compatible Ollama endpoint:
- base URL: `http://localhost:11434/v1`
- API key: `ollama` (dummy)
- default model: `gemma3:1b`


```bash
ollama run gemma3:1b
```

## Notes

- The weather tool is intentionally offline-friendly and returns dummy weather text.
- `search_web` retrieves live web results through DuckDuckGo.
