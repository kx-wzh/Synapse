import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from synapse.agents.mind2web import (
    build_response_record,
    eval_sample,
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

    @patch("synapse.agents.mind2web.get_target_obs_and_act")
    def test_eval_sample_logs_structured_record_when_ground_truth_missing(
        self, get_target_obs_and_act_mock
    ):
        get_target_obs_and_act_mock.return_value = ("obs", "CLICK [4]")
        sample = {
            "confirmed_task": "book something",
            "action_reprs": ["click candidate"],
            "actions": [{"pos_candidates": [], "neg_candidates": []}],
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            args = SimpleNamespace(
                log_dir=tmpdir,
                chat_model="qwen3.5:4b",
                embedding_model="qwen3-embedding:0.6b",
                api_base="http://localhost:11434/v1",
                api_key="ollama",
                benchmark="test_domain",
                no_memory=True,
                no_trajectory=True,
                top_k_elements=5,
                temperature=0.0,
                max_context_tokens=32768,
            )
            eval_sample(task_id=0, args=args, sample=sample)

            with open(get_mind2web_log_dir(args) / "0.json", "r") as handle:
                conversation = json.load(handle)

        record = conversation[0]
        self.assertIsInstance(record, dict)
        self.assertEqual(record["error_type"], "ground_truth_not_in_cleaned_html")
        self.assertEqual(record["chat_model"], "qwen3.5:4b")
        self.assertEqual(record["embedding_model"], "qwen3-embedding:0.6b")
        self.assertEqual(record["api_base"], "http://localhost:11434/v1")
        self.assertEqual(
            record["token_stats"],
            {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        )

    @patch("synapse.agents.mind2web.generate_response")
    @patch("synapse.agents.mind2web.num_tokens_from_messages")
    @patch("synapse.agents.mind2web.get_top_k_obs")
    @patch("synapse.agents.mind2web.get_target_obs_and_act")
    def test_eval_sample_logs_structured_record_for_context_limit_overflow(
        self,
        get_target_obs_and_act_mock,
        get_top_k_obs_mock,
        num_tokens_from_messages_mock,
        generate_response_mock,
    ):
        get_target_obs_and_act_mock.return_value = ("obs", "CLICK [4]")
        get_top_k_obs_mock.return_value = ("<html>content</html>", ["4"])
        num_tokens_from_messages_mock.return_value = 42
        sample = {
            "confirmed_task": "book something",
            "action_reprs": ["click candidate"],
            "actions": [
                {
                    "pos_candidates": [{"rank": 0, "backend_node_id": "4"}],
                    "neg_candidates": [],
                }
            ],
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            args = SimpleNamespace(
                log_dir=tmpdir,
                chat_model="qwen3.5:4b",
                embedding_model="qwen3-embedding:0.6b",
                api_base="http://localhost:11434/v1",
                api_key="ollama",
                benchmark="test_domain",
                no_memory=True,
                no_trajectory=True,
                top_k_elements=5,
                temperature=0.0,
                max_context_tokens=10,
            )
            eval_sample(task_id=1, args=args, sample=sample)

            with open(get_mind2web_log_dir(args) / "1.json", "r") as handle:
                conversation = json.load(handle)

        generate_response_mock.assert_not_called()
        record = conversation[0]
        self.assertIsInstance(record, dict)
        self.assertEqual(record["error_type"], "context_limit_exceeded")
        self.assertEqual(record["chat_model"], "qwen3.5:4b")
        self.assertEqual(record["embedding_model"], "qwen3-embedding:0.6b")
        self.assertEqual(record["api_base"], "http://localhost:11434/v1")
        self.assertEqual(
            record["token_stats"],
            {"prompt_tokens": 42, "completion_tokens": 0, "total_tokens": 42},
        )

    @patch("synapse.agents.mind2web.generate_response")
    @patch("synapse.agents.mind2web.num_tokens_from_messages")
    @patch("synapse.agents.mind2web.get_top_k_obs")
    @patch("synapse.agents.mind2web.get_target_obs_and_act")
    @patch("synapse.agents.mind2web.load_memory")
    def test_eval_sample_success_path_normalizes_ollama_plain_text_response(
        self,
        load_memory_mock,
        get_target_obs_and_act_mock,
        get_top_k_obs_mock,
        num_tokens_from_messages_mock,
        generate_response_mock,
    ):
        raw_response = "Next action: CLICK [4] </s>"
        get_target_obs_and_act_mock.return_value = ("target obs", "CLICK [4]")
        get_top_k_obs_mock.side_effect = [
            ("target obs html", ["4"]),
            ("current obs html", ["4"]),
        ]
        num_tokens_from_messages_mock.return_value = 16
        generate_response_mock.return_value = (
            raw_response,
            {"prompt_tokens": 11, "completion_tokens": 3, "total_tokens": 14},
        )
        sample = {
            "confirmed_task": "book something",
            "action_reprs": ["click candidate"],
            "actions": [
                {
                    "pos_candidates": [{"rank": 0, "backend_node_id": "4"}],
                    "neg_candidates": [],
                }
            ],
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            memory_path = Path(tmpdir) / "memory"
            memory_path.mkdir()
            with open(memory_path / "exemplars.json", "w") as handle:
                json.dump(
                    [
                        [
                            {"role": "user", "content": "demo observation"},
                            {"role": "assistant", "content": "`CLICK [1]`"},
                        ]
                    ],
                    handle,
                )
            args = SimpleNamespace(
                log_dir=tmpdir,
                memory_path=str(memory_path),
                chat_model="qwen3.5:4b",
                embedding_model="mxbai-embed-large:latest",
                api_base="http://localhost:11434/v1",
                api_key="ollama",
                benchmark="test_domain",
                no_memory=True,
                no_trajectory=False,
                retrieve_top_k=1,
                previous_top_k_elements=5,
                top_k_elements=5,
                temperature=0.0,
                max_context_tokens=32768,
            )
            eval_sample(task_id=2, args=args, sample=sample)

            with open(get_mind2web_log_dir(args) / "2.json", "r") as handle:
                conversation = json.load(handle)

        load_memory_mock.assert_called_once_with(
            str(memory_path),
            embedding_model="mxbai-embed-large:latest",
            api_base="http://localhost:11434/v1",
            api_key="ollama",
        )
        generate_response_mock.assert_called_once()
        generate_kwargs = generate_response_mock.call_args.kwargs
        self.assertEqual(generate_kwargs["model"], "qwen3.5:4b")
        self.assertEqual(generate_kwargs["api_base"], "http://localhost:11434/v1")
        self.assertEqual(generate_kwargs["api_key"], "ollama")

        record = conversation[0]
        self.assertEqual(record["raw_response"], raw_response)
        self.assertEqual(record["normalized_action"], "CLICK [4]")
        self.assertNotIn("error_type", record)
        self.assertEqual(record["token_stats"]["total_tokens"], 14)

        self.assertEqual(conversation[1]["pred_act"], "CLICK [4]")
        self.assertEqual(conversation[1]["target_act"], "CLICK [4]")
        self.assertNotIn("</s>", conversation[1]["pred_act"])

        summary = conversation[2]
        self.assertEqual(summary["element_acc"], [1])
        self.assertEqual(summary["step_success"], [1])
        self.assertEqual(summary["success"], [1])
        self.assertAlmostEqual(summary["action_f1"][0], 1.0)
        self.assertEqual(
            summary["token_stats"],
            {"prompt_tokens": 11, "completion_tokens": 3, "total_tokens": 14},
        )


if __name__ == "__main__":
    unittest.main()
