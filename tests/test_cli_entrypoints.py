import unittest
from argparse import Namespace

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


if __name__ == "__main__":
    unittest.main()
