"""Model inference clients (mock + HTTP llama-server) — Phase E."""

from inference.base import InferenceBackend
from inference.http_client import HttpInference, create_inference
from inference.mock import MockInference
from inference.prompts import PROMPT_VERSION
from inference.server import LlamaServerManager

__all__ = [
    "InferenceBackend",
    "MockInference",
    "HttpInference",
    "LlamaServerManager",
    "create_inference",
    "PROMPT_VERSION",
]
