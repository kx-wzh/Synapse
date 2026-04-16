import unittest
from unittest.mock import Mock, patch

from synapse.utils.llm import (
    DEFAULT_OLLAMA_API_BASE,
    build_openai_compatible_client,
    extract_from_response,
    generate_response,
    num_tokens_from_messages,
    resolve_api_key,
    slugify_model_name,
)


class LlmUtilsTests(unittest.TestCase):
    def test_default_ollama_api_base_constant(self):
        self.assertEqual(DEFAULT_OLLAMA_API_BASE, "http://localhost:11434/v1")

    def test_build_openai_compatible_client_initializes_real_client(self):
        client = build_openai_compatible_client(
            api_base=DEFAULT_OLLAMA_API_BASE,
            api_key="ollama",
        )
        try:
            self.assertIsNotNone(client.chat)
        finally:
            client.close()

    def test_slugify_model_name_replaces_provider_separators(self):
        self.assertEqual(slugify_model_name("qwen3.5:9b"), "qwen3.5-9b")
        self.assertEqual(
            slugify_model_name("some-org/qwen3.5:4b"),
            "some-org-qwen3.5-4b",
        )

    def test_resolve_api_key_falls_back_to_default_ollama_key(self):
        with patch.dict("os.environ", {}, clear=True):
            self.assertEqual(resolve_api_key(None), "ollama")

    def test_resolve_api_key_ignores_whitespace_env_value(self):
        with patch.dict("os.environ", {"OPENAI_API_KEY": "   "}, clear=True):
            self.assertEqual(resolve_api_key(None), "ollama")

    @patch("synapse.utils.llm.OpenAI")
    def test_build_client_ignores_whitespace_env_base_urls(self, openai_cls):
        with patch.dict(
            "os.environ",
            {
                "OPENAI_API_BASE": "   ",
                "OLLAMA_API_BASE": "",
                "OPENAI_API_KEY": "   ",
            },
            clear=True,
        ):
            build_openai_compatible_client(api_base=None, api_key=None)

        openai_cls.assert_called_once_with(
            base_url=DEFAULT_OLLAMA_API_BASE,
            api_key="ollama",
        )

    def test_num_tokens_from_messages_accepts_unknown_models(self):
        count = num_tokens_from_messages(
            [{"role": "user", "content": "Observation: `<html />`"}],
            "qwen3.5:4b",
        )
        self.assertGreater(count, 0)

    @patch("synapse.utils.llm.OpenAI")
    def test_generate_response_uses_openai_compatible_base_url(self, openai_cls):
        fake_response = Mock()
        fake_response.choices = [Mock(message=Mock(content="`CLICK [7]`"))]
        fake_response.usage.prompt_tokens = 11
        fake_response.usage.completion_tokens = 2
        fake_response.usage.total_tokens = 13

        fake_client = Mock()
        fake_client.chat.completions.create.return_value = fake_response
        openai_cls.return_value = fake_client

        message, info = generate_response(
            messages=[{"role": "user", "content": "Next action:"}],
            model="qwen3.5:4b",
            temperature=0.0,
            api_base=DEFAULT_OLLAMA_API_BASE,
            api_key="ollama",
            stop_tokens=["Task:"],
        )

        openai_cls.assert_called_once_with(
            base_url=DEFAULT_OLLAMA_API_BASE,
            api_key="ollama",
        )
        fake_client.chat.completions.create.assert_called_once_with(
            model="qwen3.5:4b",
            messages=[{"role": "user", "content": "Next action:"}],
            temperature=0.0,
            stop=["Task:"],
        )
        self.assertEqual(message, "`CLICK [7]`")
        self.assertEqual(
            info,
            {
                "prompt_tokens": 11,
                "completion_tokens": 2,
                "total_tokens": 13,
            },
        )

    @patch("synapse.utils.llm.OpenAI")
    def test_generate_response_defaults_usage_to_zero_when_missing(self, openai_cls):
        fake_response = type("Response", (), {})()
        fake_response.choices = [Mock(message=Mock(content="`CLICK [8]`"))]

        fake_client = Mock()
        fake_client.chat.completions.create.return_value = fake_response
        openai_cls.return_value = fake_client

        message, info = generate_response(
            messages=[{"role": "user", "content": "Next action:"}],
            model="qwen3.5:4b",
            temperature=0.0,
            api_base=DEFAULT_OLLAMA_API_BASE,
            api_key="ollama",
        )

        self.assertEqual(message, "`CLICK [8]`")
        self.assertEqual(
            info,
            {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            },
        )

    @patch("synapse.utils.llm.OpenAI")
    def test_generate_response_raises_for_malformed_empty_response(self, openai_cls):
        fake_response = type("Response", (), {})()
        fake_response.choices = []

        fake_client = Mock()
        fake_client.chat.completions.create.return_value = fake_response
        openai_cls.return_value = fake_client

        with self.assertRaisesRegex(
            ValueError,
            "Malformed chat completion response: missing choices\\[0\\]\\.message\\.content",
        ):
            generate_response(
                messages=[{"role": "user", "content": "Next action:"}],
                model="qwen3.5:4b",
                temperature=0.0,
                api_base=DEFAULT_OLLAMA_API_BASE,
                api_key="ollama",
            )

    def test_extract_from_response_keeps_existing_backtick_behavior(self):
        response = "Reasoning\n`TYPE [17] [Boston]`\nDone"
        self.assertEqual(extract_from_response(response, "`"), "TYPE [17] [Boston]")

    def test_extract_from_response_supports_triple_backticks_default(self):
        response = "Reasoning\n```action\nCLICK [7]\n```\nDone"
        self.assertEqual(extract_from_response(response), "CLICK [7]")


if __name__ == "__main__":
    unittest.main()
