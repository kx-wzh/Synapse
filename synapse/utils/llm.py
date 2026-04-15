import logging
import re
import os
import inspect
import backoff
try:
    import tiktoken
except ImportError:
    tiktoken = None
from openai import (
    OpenAI,
    APIConnectionError,
    APIError,
    RateLimitError,
)

logger = logging.getLogger("main")

DEFAULT_OLLAMA_API_BASE = "http://localhost:11434/v1"
DEFAULT_OLLAMA_API_KEY = "ollama"


def _normalize_non_empty(value: str | None) -> str | None:
    if value is None:
        return None
    stripped = value.strip()
    return stripped or None


def resolve_api_key(explicit_api_key: str | None) -> str:
    normalized_explicit = _normalize_non_empty(explicit_api_key)
    if normalized_explicit is not None:
        return normalized_explicit

    normalized_env_key = _normalize_non_empty(os.environ.get("OPENAI_API_KEY"))
    if normalized_env_key is not None:
        return normalized_env_key

    return DEFAULT_OLLAMA_API_KEY


def slugify_model_name(model_name: str) -> str:
    return model_name.replace("/", "-").replace(":", "-")


def build_openai_compatible_client(
    api_base: str | None = None,
    api_key: str | None = None,
) -> OpenAI:
    base_url = _normalize_non_empty(api_base)
    if base_url is None:
        base_url = _normalize_non_empty(os.environ.get("OPENAI_API_BASE"))
    if base_url is None:
        base_url = _normalize_non_empty(os.environ.get("OLLAMA_API_BASE"))
    if base_url is None:
        base_url = DEFAULT_OLLAMA_API_BASE
    return OpenAI(base_url=base_url, api_key=resolve_api_key(api_key))


def num_tokens_from_messages(messages, model):
    """Return the number of tokens used by a list of messages.
    Borrowed from https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb
    """
    encoding = None
    if tiktoken is not None:
        try:
            encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            logger.warning("Model %s not found. Using cl100k_base encoding.", model)
            try:
                encoding = tiktoken.get_encoding("cl100k_base")
            except Exception:
                encoding = None
    if encoding is None:
        # Keep token counting available in offline environments.
        class _FallbackEncoding:
            @staticmethod
            def encode(value):
                return value.encode("utf-8")

        encoding = _FallbackEncoding()
    if model == "gpt-3.5-turbo-0301":
        tokens_per_message = (
            4  # every message follows <|start|>{role/name}\n{content}<|end|>\n
        )
        tokens_per_name = -1  # if there's a name, the role is omitted
    else:
        tokens_per_message = 3
        tokens_per_name = 1
    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":
                num_tokens += tokens_per_name
    num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
    return num_tokens


MAX_TOKENS = {
    "gpt-3.5-turbo-0301": 4097,
    "gpt-3.5-turbo-0613": 4097,
    "gpt-3.5-turbo-16k-0613": 16385,
    "gpt-3.5-turbo-1106": 16385,
}


@backoff.on_exception(
    backoff.constant,
    (APIError, RateLimitError, APIConnectionError),
    interval=10,
)
def generate_response(
    messages: list[dict[str, str]],
    model: str,
    temperature: float,
    api_base: str | None = None,
    api_key: str | None = None,
    stop_tokens: list[str] | None = None,
) -> tuple[str, dict[str, int]]:
    """Send a request to an OpenAI-compatible chat API."""

    logger.info(
        f"Send a request to the language model from {inspect.stack()[1].function}"
    )

    client = build_openai_compatible_client(api_base=api_base, api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        stop=stop_tokens if stop_tokens else None,
    )
    choices = getattr(response, "choices", None)
    if not choices:
        raise ValueError(
            "Malformed chat completion response: missing choices[0].message.content"
        )
    message_content = getattr(getattr(choices[0], "message", None), "content", None)
    if message_content is None:
        raise ValueError(
            "Malformed chat completion response: missing choices[0].message.content"
        )
    message = message_content
    usage = getattr(response, "usage", None)
    info = {
        "prompt_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
        "completion_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
        "total_tokens": int(getattr(usage, "total_tokens", 0) or 0),
    }

    return message, info


def extract_from_response(response: str, backtick="```") -> str:
    if backtick == "```":
        # Matches anything between ```<optional label>\n and \n```
        pattern = r"```(?:[a-zA-Z]*)\n?(.*?)\n?```"
    elif backtick == "`":
        pattern = r"`(.*?)`"
    else:
        raise ValueError(f"Unknown backtick: {backtick}")
    match = re.search(
        pattern, response, re.DOTALL
    )  # re.DOTALL makes . match also newlines
    if match:
        extracted_string = match.group(1)
    else:
        extracted_string = ""

    return extracted_string
