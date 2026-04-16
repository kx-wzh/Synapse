import unittest
from argparse import Namespace
import json
import tempfile
from pathlib import Path

import build_memory
import run_mind2web
from synapse.utils.llm import slugify_model_name


class CliEntrypointTests(unittest.TestCase):
    def test_build_memory_parser_requires_env(self):
        with self.assertRaises(SystemExit):
            build_memory.create_parser().parse_args([])

    def test_build_memory_parser_accepts_ollama_embedding_flags(self):
        args = build_memory.create_parser().parse_args(
            [
                "--env",
                "mind2web",
                "--mind2web_data_dir",
                "/tmp/Mind2Web/data",
                "--embedding_model",
                "qwen3-embedding:0.6b",
                "--api_base",
                "http://localhost:11434/v1",
            ]
        )
        self.assertEqual(args.embedding_model, "qwen3-embedding:0.6b")
        self.assertEqual(args.api_base, "http://localhost:11434/v1")

    def test_build_memory_mind2web_requires_data_dir(self):
        parser = build_memory.create_parser()
        with self.assertRaises(SystemExit):
            build_memory.parse_args_with_validation(parser, ["--env", "mind2web"])

    def test_run_mind2web_parser_requires_data_dir_and_benchmark(self):
        parser = run_mind2web.create_parser()
        with self.assertRaises(SystemExit):
            parser.parse_args([])
        with self.assertRaises(SystemExit):
            parser.parse_args(["--benchmark", "test_domain"])
        with self.assertRaises(SystemExit):
            parser.parse_args(["--data_dir", "/tmp/Mind2Web/data"])

    def test_run_mind2web_parser_accepts_ollama_chat_flags(self):
        args = run_mind2web.create_parser().parse_args(
            [
                "--data_dir",
                "/tmp/Mind2Web/data",
                "--benchmark",
                "test_domain",
                "--chat_model",
                "qwen3.5:4b",
                "--embedding_model",
                "qwen3-embedding:0.6b",
                "--max_context_tokens",
                "32768",
            ]
        )
        self.assertEqual(args.chat_model, "qwen3.5:4b")
        self.assertEqual(args.embedding_model, "qwen3-embedding:0.6b")
        self.assertEqual(args.max_context_tokens, 32768)

    def test_run_mind2web_runtime_wiring_sets_model_alias(self):
        args = Namespace(
            chat_model="qwen3.5:4b",
            embedding_model="qwen3-embedding:0.6b",
            previous_top_k_elements=7,
        )
        updated = run_mind2web.configure_runtime_args(args, "/repo")
        self.assertEqual(updated.model, updated.chat_model)

    def test_run_mind2web_runtime_wiring_builds_slugged_memory_path(self):
        embedding_model = "qwen3-embedding:0.6b"
        args = Namespace(
            chat_model="qwen3.5:4b",
            embedding_model=embedding_model,
            previous_top_k_elements=7,
        )
        updated = run_mind2web.configure_runtime_args(args, "/repo")
        self.assertEqual(
            updated.memory_path,
            f"/repo/synapse/memory/mind2web/{slugify_model_name(embedding_model)}/top7",
        )

    def test_run_mind2web_summarize_benchmark_results_aggregates_metrics(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            result_dir = Path(tmpdir)
            (result_dir / "0.json").write_text(
                json.dumps(
                    [
                        {"role": "user", "content": "x"},
                        {
                            "element_acc": [1, 0],
                            "action_f1": [1.0, 0.5],
                            "step_success": [1, 0],
                            "success": [0],
                            "token_stats": {
                                "prompt_tokens": 10,
                                "completion_tokens": 2,
                                "total_tokens": 12,
                            },
                        },
                    ]
                )
            )
            (result_dir / "1.json").write_text(
                json.dumps(
                    [
                        {"role": "user", "content": "y"},
                        {
                            "element_acc": [1],
                            "action_f1": [0.25],
                            "step_success": [1],
                            "success": [1],
                            "token_stats": {
                                "prompt_tokens": 20,
                                "completion_tokens": 4,
                                "total_tokens": 24,
                            },
                        },
                    ]
                )
            )

            summary = run_mind2web.summarize_benchmark_results(result_dir, [0, 1])

        self.assertEqual(summary["num_examples"], 2)
        self.assertAlmostEqual(summary["element_acc"], 2 / 3)
        self.assertAlmostEqual(summary["action_f1"], 7 / 12)
        self.assertAlmostEqual(summary["step_success"], 2 / 3)
        self.assertAlmostEqual(summary["success_rate"], 0.5)
        self.assertEqual(
            summary["token_totals"],
            {"prompt_tokens": 30, "completion_tokens": 6, "total_tokens": 36},
        )

    def test_run_mind2web_write_benchmark_summary_persists_summary_json(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            result_dir = Path(tmpdir)
            (result_dir / "0.json").write_text(
                json.dumps(
                    [
                        {
                            "element_acc": [1],
                            "action_f1": [1.0],
                            "step_success": [1],
                            "success": [1],
                            "token_stats": {
                                "prompt_tokens": 3,
                                "completion_tokens": 1,
                                "total_tokens": 4,
                            },
                        }
                    ]
                )
            )
            args = Namespace(log_dir="/unused")

            summary_path = run_mind2web.write_benchmark_summary(
                args=args,
                result_dir=result_dir,
                task_ids=[0],
            )

            self.assertEqual(summary_path, result_dir / "summary.json")
            persisted = json.loads(summary_path.read_text())

        self.assertEqual(persisted["num_examples"], 1)
        self.assertAlmostEqual(persisted["success_rate"], 1.0)


if __name__ == "__main__":
    unittest.main()
