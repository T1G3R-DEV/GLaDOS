import requests
from typing import List, Optional, Dict, Any

def query_ollama(
    prompt: str,
    model: str = "deepseek-r1:8b",
    stream: bool = False,
    context: Optional[List[int]] = None,
    options: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Send a prompt to the local Ollama server with optional context and options.
    
    :param prompt: Input prompt.
    :param model: Ollama model to use.
    :param stream: If True, enable streaming (not implemented here).
    :param context: List of token IDs to continue the conversation.
    :param options: Dictionary of generation options (e.g., temperature).
    :return: Dictionary containing response text and updated context.
    """
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": stream,
    }
    if context:
        payload["context"] = context
    if options:
        payload["options"] = options

    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()

        data = response.json()
        return {
            "response": data["response"],
            "context": data.get("context"),  # Updated context for next call
        }

    except requests.RequestException as e:
        print("Error communicating with Ollama:", e)
        return {"response": "", "context": None}

if __name__ == "__main__":
    ctx = None  # Start without any context

    # First question starts a session
    result = query_ollama("Tell me a story about a dragon.", context=ctx)
    print("A:", result["response"])

    # Update context so Ollama remembers what was said
    ctx = result["context"]

    # Follow-up question in the same session
    result = query_ollama("What happened next?", context=ctx)
    print("A:", result["response"])
    ctx = result["context"]

    # Another follow-up with options (e.g., increase creativity)
    result = query_ollama(
        "Make the ending more dramatic.",
        context=ctx,
        options={"temperature": 1.2},
    )
    print("A:", result["response"])
