import unittest
from pathlib import Path
from types import SimpleNamespace

from synapse.agents.mind2web import (
    build_response_record,
    get_mind2web_log_dir,
    normalize_action_response,
)


class Mind2WebAgentTests(unittest.TestCase):
    def test_normalize_action_response_prefers_backticks(self):
        response = "Reasoning first\n`CLICK [42]`\nFinal answer"
        self.assertEqual(normalize_action_response(response), "CLICK [42]")

    def test_normalize_action_response_falls_back_to_plain_text(self):
        response = "Next action: TYPE [17] [Boston] </s>"
        self.assertEqual(normalize_action_response(response), "TYPE [17] [Boston]")

    def test_get_mind2web_log_dir_uses_chat_model_slug_and_mode(self):
        args = SimpleNamespace(
            log_dir="/tmp/results/mind2web",
            chat_model="qwen3.5:9b",
            benchmark="test_domain",
            no_memory=False,
            no_trajectory=False,
        )
        self.assertEqual(
            get_mind2web_log_dir(args),
            Path("/tmp/results/mind2web/qwen3.5-9b/test_domain/with_memory"),
        )

    def test_build_response_record_captures_model_metadata(self):
        args = SimpleNamespace(
            chat_model="qwen3.5:4b",
            embedding_model="qwen3-embedding:0.6b",
            api_base="http://localhost:11434/v1",
        )
        record = build_response_record(
            args=args,
            message=[{"role": "user", "content": "Next action:"}],
            raw_response="`CLICK [4]`",
            normalized_action="CLICK [4]",
            info={"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
            error_type=None,
        )
        self.assertEqual(record["chat_model"], "qwen3.5:4b")
        self.assertEqual(record["embedding_model"], "qwen3-embedding:0.6b")
        self.assertEqual(record["normalized_action"], "CLICK [4]")
        self.assertNotIn("error_type", record)


if __name__ == "__main__":
    unittest.main()
