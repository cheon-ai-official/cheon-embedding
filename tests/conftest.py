import pytest

from cheon_embedding import CheonEmbedding

# The model card's usage example: a search query and a passage that answers it.
QUERY = "How do I renew my passport online?"
DOCUMENT = "Passports can be renewed online through the government portal."


@pytest.fixture(scope="session")
def embedding() -> CheonEmbedding:
    return CheonEmbedding(device="cpu")
