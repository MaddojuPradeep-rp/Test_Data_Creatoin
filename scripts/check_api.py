"""OpenAI API health check utility.

Verifies that the OpenAI API key is valid and the required models are accessible.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv


def main() -> None:
    """Check OpenAI API connectivity and model access."""
    # Load .env from project root
    root = Path(__file__).resolve().parents[1]
    env_path = root / ".env"
    if env_path.exists():
        load_dotenv(env_path)

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY not found in environment or .env file.")
        print(f"  Looked for .env at: {env_path}")
        print("  Set it via: echo OPENAI_API_KEY=sk-your-key > .env")
        sys.exit(1)

    print(f"API key found: {api_key[:8]}...{api_key[-4:]}")

    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)

        # Test chat model
        print("\nTesting gpt-4o-mini...")
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Say 'hello' in one word."}],
            max_tokens=5,
        )
        print(f"  Chat model OK: {response.choices[0].message.content}")

        # Test embedding model
        print("\nTesting text-embedding-3-small...")
        emb_response = client.embeddings.create(
            input=["test"], model="text-embedding-3-small"
        )
        dim = len(emb_response.data[0].embedding)
        print(f"  Embedding model OK: {dim} dimensions")

        print("\nAll checks passed!")

    except Exception as exc:
        print(f"\nERROR: API check failed: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
