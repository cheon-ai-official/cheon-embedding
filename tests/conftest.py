import pytest

from cheon_embedding import CheonEmbedding

# The model card's usage example: a search query, its passages and their order by cosine
# similarity (the three answers first, the boiling point last).
QUERY = "How do I renew my passport online?"
PASSAGES = [
    "A lost or stolen passport should be reported to the authority that issued it.",
    "Passports can be renewed online through the government portal.",
    "Water boils at 100 degrees Celsius at sea level.",
    "여권 재발급은 정부 포털에서 온라인으로 신청할 수 있습니다.",
    "Los pasaportes se pueden renovar en línea a través del portal del gobierno.",
]
RANKING = [1, 4, 3, 0, 2]
DOCUMENT = PASSAGES[1]


@pytest.fixture(scope="session")
def embedding() -> CheonEmbedding:
    return CheonEmbedding(device="cpu")
