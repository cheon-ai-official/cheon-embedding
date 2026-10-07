"""
Embed and Score - A Query and Its Passages
==========================================
Start here. A query is embedded after a one-line task instruction, passages as
they are, and the dot product of two unit vectors is their cosine similarity.
Sorting the passages by it ranks them for the query.

The example is the model card's own: a search query in English, five passages
that answer it in English, Chinese, Arabic, Japanese and Korean, one about
passports that does not, and two unrelated.
"""

from cheon_embedding import WEB_SEARCH, CheonEmbedding

# ---------------------------------------------------------------------------
# Load the model
# ---------------------------------------------------------------------------
embedding = CheonEmbedding()  # cheonai/cheon-embedding-0.6b-v1 on CUDA, MPS or CPU

query = "How do I renew my passport online?"
passages = [
    "여권을 잃어버렸다면 바로 분실 신고를 해야 합니다.",
    "Passports can be renewed online through the government portal.",
    "富士山は日本で一番高い山です。",
    "여권 재발급은 정부 포털에서 온라인으로 신청할 수 있습니다.",
    "يمكن تجديد جواز السفر عبر الإنترنت من خلال البوابة الحكومية.",
    "القاهرة هي عاصمة مصر.",
    "护照可以通过政府门户网站在线申请换发。",
    "パスポートの更新は、政府のポータルサイトからオンラインで申請できます。",
]

# ---------------------------------------------------------------------------
# Embed, score and rank
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    queries = embedding.embed_queries([query], instruction=WEB_SEARCH)
    scores = (queries @ embedding.embed_documents(passages).T)[0]
    print(f"query: {query}\n")
    for rank, index in enumerate(scores.argsort(descending=True).tolist(), start=1):
        print(f"{rank}. {scores[index]:.4f}  {passages[index]}")
    print(f"\n({embedding.model_name} on {embedding.device}, {embedding.dimensions} dimensions)")
