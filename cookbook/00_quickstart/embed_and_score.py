"""
Embed and Score - A Query and a Passage
=======================================
Start here. A query is embedded after a one-line task instruction, a passage as
it is, and the dot product of the two unit vectors is their cosine similarity.

The example is the model card's own: a search query and a passage that answers it.
"""

from cheon_embedding import WEB_SEARCH, CheonEmbedding

# ---------------------------------------------------------------------------
# Load the model
# ---------------------------------------------------------------------------
embedding = CheonEmbedding()  # cheonai/cheon-embedding-0.6b-v1 on CUDA, MPS or CPU

query = "How do I renew my passport online?"
passage = "Passports can be renewed online through the government portal."

# ---------------------------------------------------------------------------
# Embed and score
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    queries = embedding.embed_queries([query], instruction=WEB_SEARCH)
    passages = embedding.embed_documents([passage])
    score = (queries @ passages.T).item()
    print(f"query:   {query}")
    print(f"passage: {passage}\n")
    print(f"cosine similarity {score:.4f}")
    print(f"({embedding.model_name} on {embedding.device}, {embedding.dimensions} dimensions)")
