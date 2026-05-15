from __future__ import annotations

import inspect
import json
import re
from typing import Any, Dict, Callable

from openai import OpenAI
from ddgs import DDGS


# ---------------------------------------------------------
# Step 1: Tool implementations
# ---------------------------------------------------------
def get_current_weather(city: str, unit: str = "celsius") -> str:
    """Return the current weather as a human-readable sentence."""
    # offline-friendly dummy weather function (as in solution notebook)
    symbol = "C" if unit.lower().startswith("c") else "F"
    return f"It is 23°{symbol} and sunny in {city}."


def search_web(query: str, max_results: int = 8) -> str:
    """Return top web results for query as formatted lines."""
    results = []
    with DDGS() as ddgs:
        for r in ddgs.text(query, max_results=max_results):
            title = r.get("title", "")
            href = r.get("href", "")
            results.append(f"- {title} — {href}")
    return "\n".join(results) if results else "No search results found."


# ---------------------------------------------------------
# Step 2: Prompt setup (manual tool-calling pattern)
# ---------------------------------------------------------
SYSTEM_PROMPT = (
    "You are an assistant that can call tools. "
    "When the user asks something requiring external/fresh data, respond ONLY with: "
    'TOOL_CALL:{"name": <tool_name>, "args": { ... }}'
)

TOOLS_SPEC = """
You can call these tools:
- name: get_current_weather
  description: Return current weather for a city.
  arguments:
    city: string
    unit: "celsius" | "fahrenheit" (optional, default "celsius")

- name: search_web
  description: Search the web and return top results.
  arguments:
    query: string
    max_results: integer (optional)
""".strip()


def get_client() -> OpenAI:
    # Ollama OpenAI-compatible endpoint (as used in provided solution)
    return OpenAI(api_key="ollama", base_url="http://localhost:11434/v1")


def call_llm_for_tool(model: str, user_question: str) -> str:
    client = get_client()
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT + "\n\n" + TOOLS_SPEC},
            {"role": "user", "content": user_question},
        ],
        temperature=0,
    )
    return response.choices[0].message.content or ""


def parse_and_execute_tool(raw: str) -> str:
    pattern = r"TOOL_CALL:\s*({.*})"
    m = re.search(pattern, raw, flags=re.DOTALL)
    if not m:
        return "No tool call detected."

    call_json = json.loads(m.group(1))
    fn_name = call_json["name"]
    args = call_json.get("args", {})

    tools: Dict[str, Callable[..., Any]] = {
        "get_current_weather": get_current_weather,
        "search_web": search_web,
    }

    if fn_name not in tools:
        return f"Unknown tool requested: {fn_name}"

    result = tools[fn_name](**args)
    return f"Calling tool `{fn_name}` with args {args}\nResult:\n{result}"


# ---------------------------------------------------------
# Step 3: Automatic schema generation
# ---------------------------------------------------------
def to_schema(fn: Callable[..., Any]) -> Dict[str, Any]:
    sig = inspect.signature(fn)
    properties = {}
    required = []

    for name, param in sig.parameters.items():
        ann = param.annotation
        if ann is int:
            t = "integer"
        elif ann is float:
            t = "number"
        elif ann is bool:
            t = "boolean"
        else:
            t = "string"

        properties[name] = {
            "type": t,
            "description": f"Argument {name}",
        }

        if param.default is inspect._empty:
            required.append(name)

    return {
        "name": fn.__name__,
        "description": (fn.__doc__ or "").strip().split("\n")[0],
        "parameters": {
            "type": "object",
            "properties": properties,
            "required": required,
        },
    }


def demo_schema_prompt(model: str, question: str) -> str:
    client = get_client()
    schemas = [to_schema(get_current_weather), to_schema(search_web)]

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "system", "name": "tool_spec", "content": json.dumps(schemas)},
        {"role": "user", "content": question},
    ]

    resp = client.chat.completions.create(model=model, messages=messages, temperature=0)
    return resp.choices[0].message.content or ""


def run_cli() -> None:
    print("Week 3 - Ask-the-Web Agent (manual TOOL_CALL version)")
    print("Requires Ollama running at http://localhost:11434 with a model like gemma3:1b")
    print("Type 'exit' to quit.\n")

    model = "gemma3:1b"

    while True:
        q = input("Question: ").strip()
        if q.lower() in {"exit", "quit"}:
            break

        try:
            raw = call_llm_for_tool(model=model, user_question=q)
            print("\nModel output:\n", raw)
            print("\nTool execution:\n", parse_and_execute_tool(raw))

            print("\nSchema-guided call (optional demo):")
            raw2 = demo_schema_prompt(model=model, question=q)
            print(raw2)
            print("\n" + "-" * 60 + "\n")
        except Exception as e:
            print(f"Error: {e}\n")


if __name__ == "__main__":
    run_cli()
