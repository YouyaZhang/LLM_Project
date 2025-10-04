from __future__ import annotations
import argparse

from models.model_loader import load_model_from_config
from utils.config import load_yaml
from inference.pipeline import InferencePipeline


def main():
    parser = argparse.ArgumentParser(description="CLI chat for Med Assistant")
    parser.add_argument("--cfg", default="configs/model.yaml")
    parser.add_argument("--prompt", default=None, help="single-turn user prompt (if omitted, enters REPL)")
    args = parser.parse_args()

    cfg = load_yaml(args.cfg)
    runtime = load_model_from_config(args.cfg)

    pipe = InferencePipeline(
        tokenizer=runtime.tokenizer,
        model=runtime.model,
        gen_cfg={
            "max_new_tokens": cfg.get("max_new_tokens", 512),
            "temperature": cfg.get("temperature", 0.7),
            "top_p": cfg.get("top_p", 0.9),
            "repetition_penalty": cfg.get("repetition_penalty", 1.05),
        },
        system_prompt=cfg.get("system_prompt"),
    )

    if args.prompt:
        out = pipe.generate([{"role": "user", "content": args.prompt}])
        print("\nAssistant:\n", out)
        return

    print("\nEntering chat REPL. Type /exit to quit.\n")
    history = []
    while True:
        try:
            user = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break
        if user.lower() in {"/exit", ":q", "quit", "exit"}:
            print("Bye.")
            break
        resp = pipe.generate([{"role": "user", "content": user}])
        print("Assistant:\n", resp, "\n")
        history.append((user, resp))


if __name__ == "__main__":
    main()