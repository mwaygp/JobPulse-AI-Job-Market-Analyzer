# AI Chatbot

A simple terminal chatbot built on the Claude API. Streams replies and remembers the conversation until you reset it.

## Setup

```bash
git clone <your-repo-url>
cd ai-chatbot
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Get an API key from https://console.anthropic.com, then set it:

```bash
export ANTHROPIC_API_KEY="your-key-here"   # Windows (PowerShell): $env:ANTHROPIC_API_KEY="your-key-here"
```

## Run

```bash
python chatbot.py
```

Commands: `reset` clears the conversation, `quit` exits.

## Customize

Set environment variables to change behavior without editing code:

- `CHATBOT_SYSTEM_PROMPT` – the bot's personality or instructions
- `CHATBOT_MODEL` – which Claude model to use (default: `claude-sonnet-5-5`)

## Security

Never commit your API key. `.env` is already in `.gitignore`.
