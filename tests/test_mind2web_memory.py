import json
import os
import pickle
import tempfile
import unittest
from unittest.mock import Mock, patch

from synapse.memory.mind2web.build_memory import (
    build_memory,
    load_memory,
    retrieve_exemplar_name,
    save_memory_metadata,
)
from synapse.utils.embeddings import OpenAICompatibleEmbeddings


class Mind2WebMemoryTests(unittest.TestCase):
    def test_save_memory_metadata_writes_expected_fields(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            save_memory_metadata(
                memory_path=tmpdir,
                embedding_model="qwen3-embedding:0.6b",
                api_base="http://localhost:11434/v1",
                top_k=3,
                num_exemplars=12,
            )

            with open(os.path.join(tmpdir, "memory_meta.json"), "r") as handle:
                metadata = json.load(handle)

        self.assertEqual(metadata["embedding_model"], "qwen3-embedding:0.6b")
        self.assertEqual(metadata["api_base"], "http://localhost:11434/v1")
        self.assertEqual(metadata["top_k_elements"], 3)
        self.assertEqual(metadata["num_exemplars"], 12)
        self.assertIn("created_at", metadata)

    @patch("synapse.memory.mind2web.build_memory.FAISS")
    @patch("synapse.memory.mind2web.build_memory.OpenAICompatibleEmbeddings")
    def test_load_memory_uses_ollama_embeddings(self, embedding_cls, faiss_cls):
        embedding_instance = Mock()
        embedding_cls.return_value = embedding_instance
        faiss_cls.load_local.return_value = "fake-memory"

        memory = load_memory(
            memory_path="/tmp/memory",
            embedding_model="qwen3-embedding:0.6b",
            api_base="http://localhost:11434/v1",
            api_key="ollama",
        )

        embedding_cls.assert_called_once_with(
            model="qwen3-embedding:0.6b",
            api_base="http://localhost:11434/v1",
            api_key="ollama",
        )
        faiss_cls.load_local.assert_called_once_with("/tmp/memory", embedding_instance)
        self.assertEqual(memory, "fake-memory")

    @patch("synapse.memory.mind2web.build_memory.FAISS")
    @patch("synapse.memory.mind2web.build_memory.OpenAICompatibleEmbeddings")
    @patch("synapse.memory.mind2web.build_memory.get_top_k_obs")
    @patch("synapse.memory.mind2web.build_memory.get_target_obs_and_act")
    @patch("synapse.memory.mind2web.build_memory.load_json")
    def test_build_memory_creates_memory_path_before_writes(
        self,
        load_json_mock,
        get_target_obs_and_act_mock,
        get_top_k_obs_mock,
        embedding_cls,
        faiss_cls,
    ):
        sample = {
            "website": "website",
            "domain": "domain",
            "subdomain": "subdomain",
            "confirmed_task": "task",
            "annotation_id": "annotation",
            "actions": [
                {
                    "action_uid": "action",
                    "pos_candidates": [{"backend_node_id": "candidate"}],
                    "neg_candidates": [],
                }
            ],
            "action_reprs": ["click button"],
        }
        load_json_mock.return_value = [sample]
        get_target_obs_and_act_mock.return_value = ("obs", "target action")
        get_top_k_obs_mock.return_value = ("top-k obs", None)

        embedding_cls.return_value = Mock()
        fake_memory = Mock()
        faiss_cls.from_texts.return_value = fake_memory

        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = os.path.join(tmpdir, "data")
            os.makedirs(data_dir, exist_ok=True)
            with open(os.path.join(data_dir, "scores_all_data.pkl"), "wb") as handle:
                pickle.dump(
                    {
                        "scores": {"annotation_action": {"candidate": 1.0}},
                        "ranks": {"annotation_action": {"candidate": 1}},
                    },
                    handle,
                )

            memory_path = os.path.join(tmpdir, "new-memory-path")
            build_memory(memory_path=memory_path, data_dir=data_dir)

            self.assertTrue(os.path.isdir(memory_path))
            self.assertTrue(os.path.exists(os.path.join(memory_path, "exemplars.json")))
            self.assertTrue(os.path.exists(os.path.join(memory_path, "memory_meta.json")))

    def test_retrieve_exemplar_name_annotation_uses_integer_indices(self):
        self.assertEqual(
            retrieve_exemplar_name.__annotations__["return"],
            tuple[list[int], list[float]],
        )

    @patch("synapse.utils.embeddings.OpenAI")
    def test_embed_query_raises_clear_error_when_no_vectors(self, openai_cls):
        mock_client = Mock()
        mock_client.embeddings.create.return_value = Mock(data=[])
        openai_cls.return_value = mock_client

        embedding = OpenAICompatibleEmbeddings(
            model="qwen3-embedding:0.6b",
            api_base="http://localhost:11434/v1",
            api_key="ollama",
        )

        with self.assertRaisesRegex(
            ValueError, "Embeddings API returned no vectors for query."
        ):
            embedding.embed_query("query text")


if __name__ == "__main__":
    unittest.main()
