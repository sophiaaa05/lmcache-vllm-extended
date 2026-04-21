#!/usr/bin/env python3
import os
import argparse
from transformers import AutoTokenizer
import chat_session

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
IP = "127.0.0.1"
PORT = 8000
SYSTEM_PROMPT = (
    "You are a helpful assistant. I will now give you a document and "
    "please answer my question afterwards based on the content in document"
)


def read_context_files(paths):
    contexts = []
    for path in paths:
        with open(path, "r") as f:
            contexts.append(f.read())
    return contexts


def print_divider():
    print("-" * 60)


def main():
    parser = argparse.ArgumentParser(description="CLI chat client for LMCache vLLM")
    parser.add_argument("--ip", default=IP)
    parser.add_argument("--port", type=int, default=PORT)
    parser.add_argument("--model", default=MODEL_NAME)
    parser.add_argument("--context", nargs="*", default=[], metavar="FILE",
                        help="Text files to prepend as context")
    parser.add_argument("--system-prompt", default=SYSTEM_PROMPT)
    args = parser.parse_args()

    print(f"Loading tokenizer for {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    session = chat_session.ChatSession(args.ip, args.port)

    contexts = read_context_files(args.context)
    session.set_context([args.system_prompt] + contexts)

    context_text = session.get_context()
    num_tokens = len(tokenizer.encode(context_text))

    preview = context_text[:200].replace("\n", " ") + " ..." if len(context_text) > 200 else context_text
    print_divider()
    print(f"Context given to LLM: ({num_tokens} tokens)")
    print(preview)
    print_divider()
    print()
    print("Chat session started. Type 'exit' or Ctrl+C to quit.\n")

    while True:
        try:
            question = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye.")
            break

        if not question:
            continue
        if question.lower() in ("exit", "quit"):
            print("Goodbye.")
            break

        print("Assistant: ", end="", flush=True)
        for chunk in session.chat(question):
            print(chunk, end="", flush=True)
        print()


if __name__ == "__main__":
    main()
