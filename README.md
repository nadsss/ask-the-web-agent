# Week 3 Project 3: Build an Ask-the-Web Agent (Perplexity-style)

This implementation follows your provided `ask_the_web_agent_solution.ipynb` flow:

1. Build a tool (`get_current_weather`) 
2. Prompt an LLM to emit `TOOL_CALL:{...}`
3. Parse tool calls and execute in Python
4. Auto-generate JSON schema from function signatures
5. Add a real web-search tool (`search_web`) using DuckDuckGo

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

Start Ollama first and ensure the model exists:

```bash
ollama run gemma3:1b
```

## Notes

- The weather tool is intentionally offline-friendly and returns dummy weather text.
- `search_web` retrieves live web results through DuckDuckGo.
- You can swap model names inside `main.py`.
