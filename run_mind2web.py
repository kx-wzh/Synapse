import json
import pickle
import logging
import argparse
import os
from pathlib import Path
from tqdm import tqdm

from synapse.envs.mind2web.env_utils import load_json
from synapse.agents.mind2web import eval_sample, get_mind2web_log_dir
from synapse.utils.llm import DEFAULT_OLLAMA_API_BASE, slugify_model_name

logger = logging.getLogger("synapse")
logger.setLevel(logging.INFO)


def ensure_logger_handler():
    if any(getattr(existing, "_synapse_cli_handler", False) for existing in logger.handlers):
        return
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    handler._synapse_cli_handler = True
    logger.addHandler(handler)


def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    # 252, 177, 912
    parser.add_argument(
        "--benchmark",
        type=str,
        choices=["test_task", "test_website", "test_domain"],
        required=True,
    )
    parser.add_argument("--previous_top_k_elements", type=int, default=3)
    parser.add_argument("--top_k_elements", type=int, default=5)
    parser.add_argument("--retrieve_top_k", type=int, default=3)
    parser.add_argument("--chat_model", type=str, default="qwen3.5:4b")
    parser.add_argument("--embedding_model", type=str, default="qwen3-embedding:0.6b")
    parser.add_argument("--api_base", type=str, default=DEFAULT_OLLAMA_API_BASE)
    parser.add_argument("--api_key", type=str, default=None)
    parser.add_argument("--max_context_tokens", type=int, default=131072)
    parser.add_argument("--temperature", type=float, default=1)
    parser.add_argument("--no_memory", action="store_true", default=False)
    parser.add_argument("--no_trajectory", action="store_true", default=False)
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--end_idx", type=int, default=None)

    return parser


def configure_runtime_args(args, current_path):
    args.model = args.chat_model
    args.memory_path = os.path.join(
        current_path,
        "synapse/memory/mind2web",
        slugify_model_name(args.embedding_model),
        f"top{args.previous_top_k_elements}",
    )
    args.log_dir = os.path.join(current_path, "results/mind2web")
    return args


def _mean(values):
    return sum(values) / len(values) if values else 0.0


def summarize_benchmark_results(result_dir: Path, task_ids: list[int]) -> dict:
    element_acc = []
    action_f1 = []
    step_success = []
    success = []
    token_totals = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    completed_task_ids = []

    for task_id in task_ids:
        result_path = result_dir / f"{task_id}.json"
        if not result_path.exists():
            logger.warning("Skipping missing result file: %s", result_path)
            continue

        conversation = json.loads(result_path.read_text())
        if not conversation:
            logger.warning("Skipping empty result file: %s", result_path)
            continue

        summary = conversation[-1]
        completed_task_ids.append(task_id)
        element_acc.extend(summary.get("element_acc", []))
        action_f1.extend(summary.get("action_f1", []))
        step_success.extend(summary.get("step_success", []))
        success.extend(summary.get("success", []))

        token_stats = summary.get("token_stats", {})
        for key in token_totals:
            token_totals[key] += token_stats.get(key, 0)

    return {
        "num_examples": len(completed_task_ids),
        "task_ids": completed_task_ids,
        "element_acc": _mean(element_acc),
        "action_f1": _mean(action_f1),
        "step_success": _mean(step_success),
        "success_rate": _mean(success),
        "token_totals": token_totals,
    }


def write_benchmark_summary(args, result_dir: Path, task_ids: list[int]) -> Path:
    summary = summarize_benchmark_results(result_dir, task_ids)
    summary.update(
        {
            "benchmark": getattr(args, "benchmark", None),
            "chat_model": getattr(args, "chat_model", None),
            "embedding_model": getattr(args, "embedding_model", None),
            "start_idx": min(task_ids) if task_ids else None,
            "end_idx": max(task_ids) + 1 if task_ids else None,
        }
    )

    summary_path = result_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    logger.info("Benchmark summary written to %s", summary_path)
    logger.info(
        "Summary: examples=%s element_acc=%.4f action_f1=%.4f step_success=%.4f success_rate=%.4f",
        summary["num_examples"],
        summary["element_acc"],
        summary["action_f1"],
        summary["step_success"],
        summary["success_rate"],
    )
    return summary_path


ensure_logger_handler()


def main():
    parser = create_parser()
    args = parser.parse_args()
    current_path = os.getcwd()
    args = configure_runtime_args(args, current_path)

    # Evaluate test set
    samples = load_json(args.data_dir, args.benchmark)

    # add prediction scores and ranks to candidates
    with open(os.path.join(args.data_dir, "scores_all_data.pkl"), "rb") as f:
        candidate_results = pickle.load(f)
    candidate_scores = candidate_results["scores"]
    candidate_ranks = candidate_results["ranks"]
    for sample in samples:
        for s, act_repr in zip(sample["actions"], sample["action_reprs"]):
            sample_id = f"{sample['annotation_id']}_{s['action_uid']}"
            for candidates in [s["pos_candidates"], s["neg_candidates"]]:
                for candidate in candidates:
                    candidate_id = candidate["backend_node_id"]
                    candidate["score"] = candidate_scores[sample_id][candidate_id]
                    candidate["rank"] = candidate_ranks[sample_id][candidate_id]

    if args.end_idx is None:
        args.end_idx = len(samples)
    task_ids = list(range(args.start_idx, args.end_idx))
    for i in tqdm(task_ids):
        eval_sample(i, args, samples[i])
    write_benchmark_summary(args, get_mind2web_log_dir(args), task_ids)


if __name__ == "__main__":
    main()
