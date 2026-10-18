"""A simple streaming AI chatbot for the terminal, powered by Claude."""

import os
import sys

import anthropic

MODEL = os.getenv("CHATBOT_MODEL", "claude-sonnet-5-5")
SYSTEM_PROMPT = os.getenv(
    "CHATBOT_SYSTEM_PROMPT",
    "You are a friendly, concise assistant.",
)


def main() -> None:
    if not os.getenv("ANTHROPIC_API_KEY"):
        sys.exit("Missing ANTHROPIC_API_KEY. See the README for setup.")

    client = anthropic.Anthropic()
    history = []

    print("Chatbot ready. Type 'quit' to exit, 'reset' to clear the conversation.\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not user_input:
            continue
        if user_input.lower() in {"quit", "exit"}:
            break
        if user_input.lower() == "reset":
            history.clear()
            print("(conversation cleared)\n")
            continue

        history.append({"role": "user", "content": user_input})

        try:
            print("Bot: ", end="", flush=True)
            reply = ""
            with client.messages.stream(
                model=MODEL,
                max_tokens=1024,
                system=SYSTEM_PROMPT,
                messages=history,
            ) as stream:
                for text in stream.text_stream:
                    print(text, end="", flush=True)
                    reply += text
            print("\n")
            history.append({"role": "assistant", "content": reply})
        except anthropic.APIError as err:
            history.pop()  # drop the failed user turn so the history stays valid
            print(f"\n[API error: {err}]\n")


if __name__ == "__main__":
    main()
