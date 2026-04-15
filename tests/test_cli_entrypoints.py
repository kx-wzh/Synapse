import unittest

import build_memory
import run_mind2web


class CliEntrypointTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
