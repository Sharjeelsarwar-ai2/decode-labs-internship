# Decode Labs — Project 1: Rule-Based AI Chatbot

A simple rule-based chatbot built with Python. No machine learning, no AI API — every reply comes from an explicit `if-else` rule in `chatbot.py`.

## How to run

```bash
python chatbot.py
```

Or the web version (same rules, imported from `chatbot.py`):

```bash
streamlit run app.py
```

Type `bye`, `goodbye`, `exit` or `quit` to stop.

## What it can do

- Greetings: `hello`, `hi`, `hey`, `salam`
- Exit commands: `bye`, `goodbye`, `exit`, `quit`, `stop`
- Remember your name: `my name is Sharjeel`
- Answer: `how are you`, `what is your name`, `who made you`, `decode labs`
- Help: `help` / `what can you do`
- Real time and date from the computer clock: `time`, `date`
- Jokes: `tell me a joke`
- Fallback for unknown input: suggests `help` instead of crashing

## Example

```
DecodeBot: Hello! I'm a rule-based chatbot. Type 'bye' to exit.
You: HELLO!!!
DecodeBot: Hello! How can I help you today?
You: my name is sharjeel
DecodeBot: Nice to meet you, Sharjeel! I will remember your name.
You: hello
DecodeBot: Hello, Sharjeel! How can I help you today?
You: bye
DecodeBot: Goodbye! It was nice talking to you.
```

## How it works

1. `normalize_input()` cleans the input (`HELLO!!!` → `hello`) so one rule covers many spellings.
2. `get_response()` checks rules in order — specific first, fallback last — and returns a reply.
3. `main()` runs a `while True` loop: read input, decide, reply, repeat, until an exit word breaks the loop.
