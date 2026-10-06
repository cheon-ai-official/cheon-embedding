"""
Embed and Score - A Query and Its Passages
==========================================
Start here. A query is embedded after a one-line task instruction, passages as
they are, and the dot product of two unit vectors is their cosine similarity.
Sorting the passages by it ranks them for the query.

The example is the model card's own: a search query, three passages that answer
it in different languages, one about passports that does not, and one unrelated.
"""

from cheon_embedding import WEB_SEARCH, CheonEmbedding

# ---------------------------------------------------------------------------
# Load the model
# ---------------------------------------------------------------------------
embedding = CheonEmbedding()  # cheonai/cheon-embedding-0.6b-v1 on CUDA, MPS or CPU

query = "How do I renew my passport online?"
passages = [
    "A lost or stolen passport should be reported to the authority that issued it.",
    "Passports can be renewed online through the government portal.",
    "Water boils at 100 degrees Celsius at sea level.",
    "여권 재발급은 정부 포털에서 온라인으로 신청할 수 있습니다.",
    "Los pasaportes se pueden renovar en línea a través del portal del gobierno.",
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
