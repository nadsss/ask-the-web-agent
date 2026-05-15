from __future__ import annotations

import json
import os
from typing import Any, Dict, List

from openai import OpenAI

from tools import web_search, fetch_url_text


SYSTEM_PROMPT = """You are Ask-the-Web Agent.
You answer user questions by searching the web and grounding answers in retrieved evidence.
When needed, call tools to search and read pages before finalizing.
Always include concise citations as URLs in the final answer.
"""


def get_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set.")
    return OpenAI(api_key=api_key)


def tool_schemas() -> List[Dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": "web_search",
                "description": "Search the web for relevant pages.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "max_results": {"type": "integer", "default": 5},
                    },
                    "required": ["query"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "fetch_url_text",
                "description": "Fetch and clean text from a given URL.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "url": {"type": "string"},
                        "timeout": {"type": "integer", "default": 12},
                        "max_chars": {"type": "integer", "default": 4000},
                    },
                    "required": ["url"],
                },
            },
        },
    ]


def call_tool(name: str, args: Dict[str, Any]) -> Any:
    if name == "web_search":
        return web_search(**args)
    if name == "fetch_url_text":
        return fetch_url_text(**args)
    raise ValueError(f"Unknown tool: {name}")


def ask_the_web(question: str, model: str = "gpt-4o-mini", max_loops: int = 6) -> str:
    client = get_client()

    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]

    for _ in range(max_loops):
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tool_schemas(),
            tool_choice="auto",
            temperature=0.2,
        )

        msg = response.choices[0].message

        if msg.tool_calls:
            messages.append(
                {
                    "role": "assistant",
                    "content": msg.content or "",
                    "tool_calls": [tc.model_dump() for tc in msg.tool_calls],
                }
            )

            for tc in msg.tool_calls:
                tool_name = tc.function.name
                tool_args = json.loads(tc.function.arguments or "{}")
                try:
                    result = call_tool(tool_name, tool_args)
                    tool_payload = json.dumps(result, ensure_ascii=False)[:12000]
                except Exception as e:
                    tool_payload = json.dumps({"error": str(e)})

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": tool_payload,
                    }
                )
            continue

        # Final answer (no more tool calls)
        return msg.content or "No response generated."

    return "I reached the maximum reasoning loops without finalizing an answer."


def main():
    print("Ask-the-Web Agent (type 'exit' to quit)")
    while True:
        q = input("\nQuestion: ").strip()
        if q.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break

        try:
            ans = ask_the_web(q)
            print("\nAnswer:\n")
            print(ans)
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    main()
