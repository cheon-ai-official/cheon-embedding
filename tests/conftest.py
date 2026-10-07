import pytest

from cheon_embedding import CheonEmbedding

# The model card's usage example: a search query, its passages and their order by cosine
# similarity (the five answers, one per language, first; the unrelated two last).
QUERY = "How do I renew my passport online?"
PASSAGES = [
    "여권을 잃어버렸다면 바로 분실 신고를 해야 합니다.",
    "Passports can be renewed online through the government portal.",
    "富士山は日本で一番高い山です。",
    "여권 재발급은 정부 포털에서 온라인으로 신청할 수 있습니다.",
    "يمكن تجديد جواز السفر عبر الإنترنت من خلال البوابة الحكومية.",
    "القاهرة هي عاصمة مصر.",
    "护照可以通过政府门户网站在线申请换发。",
    "パスポートの更新は、政府のポータルサイトからオンラインで申請できます。",
]
RANKING = [1, 6, 4, 7, 3, 0, 5, 2]
DOCUMENT = PASSAGES[1]


@pytest.fixture(scope="session")
def embedding() -> CheonEmbedding:
    return CheonEmbedding(device="cpu")
