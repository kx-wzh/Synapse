from openai import OpenAI

from synapse.utils.llm import DEFAULT_OLLAMA_API_BASE, resolve_api_key


class OpenAICompatibleEmbeddings:
    def __init__(
        self,
        model: str,
        api_base: str = DEFAULT_OLLAMA_API_BASE,
        api_key: str | None = None,
    ):
        self.model = model
        self.client = OpenAI(
            base_url=api_base,
            api_key=resolve_api_key(api_key),
        )

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        response = self.client.embeddings.create(model=self.model, input=texts)
        return [item.embedding for item in response.data]

    def embed_query(self, text: str) -> list[float]:
        vectors = self.embed_documents([text])
        if not vectors:
            raise ValueError("Embeddings API returned no vectors for query.")
        return vectors[0]
