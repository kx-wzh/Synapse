import argparse
import os
from pathlib import Path

from synapse.utils.llm import DEFAULT_OLLAMA_API_BASE, slugify_model_name


def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env", type=str, choices=["miniwob", "mind2web"], required=True
    )
    parser.add_argument("--mind2web_data_dir", type=str)
    parser.add_argument("--mind2web_top_k_elements", type=int, default=3)
    parser.add_argument("--embedding_model", type=str, default="qwen3-embedding:0.6b")
    parser.add_argument("--api_base", type=str, default=DEFAULT_OLLAMA_API_BASE)
    parser.add_argument("--api_key", type=str, default=None)
    return parser


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()

    current_path = os.getcwd()
    if args.env == "miniwob":
        from synapse.memory.miniwob.build_memory import build_memory

        memory_path = os.path.join(current_path, "synapse/memory/miniwob")
        build_memory(memory_path)
    else:
        from synapse.memory.mind2web.build_memory import build_memory

        memory_path = os.path.join(
            current_path,
            "synapse/memory/mind2web",
            slugify_model_name(args.embedding_model),
            f"top{args.mind2web_top_k_elements}",
        )
        Path(memory_path).mkdir(parents=True, exist_ok=True)
        build_memory(
            memory_path=memory_path,
            data_dir=args.mind2web_data_dir,
            top_k=args.mind2web_top_k_elements,
            embedding_model=args.embedding_model,
            api_base=args.api_base,
            api_key=args.api_key,
        )
