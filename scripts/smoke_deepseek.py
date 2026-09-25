
import sys

from openai import APIConnectionError, APITimeoutError, OpenAI

BASE_URL = "http://10.0.21.72:13505/v1"
MODEL = "unsloth/DeepSeek-V4-Flash-0731"
TIMEOUT_SECONDS = 15


def main() -> int:
    client = OpenAI(base_url=BASE_URL, api_key="not-needed", timeout=TIMEOUT_SECONDS)

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": "Reply with exactly: ok"}],
            max_tokens=20,
            temperature=0,
        )
    except (APIConnectionError, APITimeoutError) as e:
        print(f"UNREACHABLE: could not reach {BASE_URL} within {TIMEOUT_SECONDS}s.")
        print(f"  {type(e).__name__}: {e}")
        return 1

    print("Response:", response.choices[0].message.content)
    print("Usage:", response.usage)
    return 0


if __name__ == "__main__":
    sys.exit(main())
