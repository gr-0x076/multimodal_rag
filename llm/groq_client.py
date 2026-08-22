"""
ContextMesh — Groq LLM Client

Sends grounded prompts to the Groq API and returns the model's response.
Reads GROQ_API_KEY from .env — never hardcoded.
"""

import os
import sys
from pathlib import Path

root_dir = str(Path(__file__).resolve().parent.parent)
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(root_dir, ".env"))
except ImportError:
    pass  # python-dotenv not installed; rely on env vars


DEFAULT_MODEL = "llama-3.3-70b-versatile"


def query_groq(
    system_prompt: str,
    user_prompt: str,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.2,
    max_tokens: int = 2048,
) -> str:
    """
    Send a grounded prompt to Groq and return the assistant's response text.

    Args:
        system_prompt:  System-level grounding instructions.
        user_prompt:    User message including evidence context + question.
        model:          Groq model identifier.
        temperature:    Sampling temperature (low = more deterministic).
        max_tokens:     Maximum response length.

    Returns:
        The assistant's response text.

    Raises:
        EnvironmentError:  If GROQ_API_KEY is not set.
        RuntimeError:      If the Groq API call fails.
    """
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "GROQ_API_KEY is not set. "
            "Add it to your .env file:\n"
            "  GROQ_API_KEY=gsk_..."
        )

    try:
        from groq import Groq
    except ImportError:
        raise ImportError(
            "The 'groq' package is required. Install it with:\n"
            "  pip install groq"
        )

    client = Groq(api_key=api_key)

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content

    except Exception as e:
        raise RuntimeError(f"Groq API call failed: {e}") from e


if __name__ == "__main__":
    # Quick connectivity test
    try:
        result = query_groq(
            system_prompt="You are a helpful assistant. Reply in one sentence.",
            user_prompt="Say hello and confirm you are working.",
        )
        print("Groq response:", result)
    except EnvironmentError as e:
        print(f"Setup needed: {e}")
    except RuntimeError as e:
        print(f"API error: {e}")
