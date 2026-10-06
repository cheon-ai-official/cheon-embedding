"""Cheon Embedding: one vector per text, for search, RAG and similarity.

Queries are written after a one-line task instruction; documents are embedded
as they are. Every vector has unit length, so the dot product of two vectors is
their cosine similarity.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from transformers import AutoModel, AutoTokenizer

DEFAULT_MODEL = "cheonai/cheon-embedding-0.6b-v1"

# The model card's task description for web search, the usual one for search queries.
WEB_SEARCH = "Given a web search query, retrieve relevant passages that answer the query"


def default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class CheonEmbedding:
    """A Cheon Embedding model loaded from the Hugging Face Hub.

    Args:
        model: Hub repository id or local directory of the model.
        device: Where to run it ("cuda", "mps", "cpu", ...). Picked automatically when omitted.
        dtype: Weight precision. The released weights are float32.
        revision: Hub branch, tag or commit to pin the model to.
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        *,
        device: str | None = None,
        dtype: torch.dtype = torch.float32,
        revision: str | None = None,
    ) -> None:
        self.model_name = model
        self.device = torch.device(device or default_device())
        self.tokenizer = AutoTokenizer.from_pretrained(model, revision=revision)
        self.model = AutoModel.from_pretrained(model, revision=revision, trust_remote_code=True, dtype=dtype)
        self.model.to(self.device).eval()
        self.dimensions: int = self.model.config.hidden_size
        self.max_tokens: int = self.model.config.max_seq_length

    @staticmethod
    def _check(texts: Sequence[str], what: str) -> list[str]:
        if isinstance(texts, str) or len(texts) == 0:
            raise ValueError(f"{what} must be a non-empty list of strings")
        for position, text in enumerate(texts):
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"{what}[{position}] has no text")
        return list(texts)

    def embed_queries(
        self, queries: Sequence[str], instruction: str = WEB_SEARCH, batch_size: int = 32
    ) -> torch.Tensor:
        """One unit vector per query, in input order, each query written after `instruction`."""
        if not isinstance(instruction, str) or not instruction.strip():
            raise ValueError("instruction must be a non-empty string")
        texts = self._check(queries, "queries")
        return self.model.encode(texts, self.tokenizer, instruction=instruction, batch_size=batch_size)

    def embed_documents(self, documents: Sequence[str], batch_size: int = 32) -> torch.Tensor:
        """One unit vector per document, in input order. Documents take no instruction."""
        texts = self._check(documents, "documents")
        return self.model.encode(texts, self.tokenizer, batch_size=batch_size)
