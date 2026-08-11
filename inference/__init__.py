"""Model inference clients (mock + HTTP llama-server)."""

from inference.base import InferenceBackend
from inference.mock import MockInference

__all__ = ["InferenceBackend", "MockInference"]
