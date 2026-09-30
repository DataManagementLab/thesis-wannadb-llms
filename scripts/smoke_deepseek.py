import argparse
import os
import sys

from openai import OpenAI, OpenAIError

DEFAULT_BASE_URL = "http://10.0.21.72:13505/v1"


def main() -> int:
    parser = argparse.ArgumentParser(description="Send one short request to check model name, URL and key.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--api-key-env", default=None, help="environment variable that holds the API key")
    parser.add_argument("--provider", default=None, help="OpenRouter only: the provider to pin")
    args = parser.parse_args()

    api_key = os.environ.get(args.api_key_env) if args.api_key_env else None
    if args.api_key_env and not api_key:
        print(f"Environment variable {args.api_key_env} is not set.")
        return 1
    client = OpenAI(base_url=args.base_url, api_key=api_key or "not-needed", timeout=120)
    extra_body = {"provider": {"order": [args.provider], "allow_fallbacks": False}} if args.provider else None

    try:
        response = client.chat.completions.create(
            model=args.model,
            messages=[{"role": "user", "content": "Reply with exactly: ok"}],
            max_tokens=20,
            temperature=0,
            extra_body=extra_body,
        )
    except OpenAIError as e:
        print(f"FAILED: {type(e).__name__}: {e}")
        return 1

    print("Model:", response.model)
    if getattr(response, "provider", None):
        print("Provider:", response.provider)
    print("Response:", response.choices[0].message.content)
    print("Usage:", response.usage)
    return 0


if __name__ == "__main__":
    sys.exit(main())
