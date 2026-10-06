# Quickstart

Embed a query and a passage, and score how well the passage answers it.

## Start Here

From the repository root:

```bash
pip install -e .
python cookbook/00_quickstart/embed_and_score.py
```

The minimum:

```python
from cheon_embedding import CheonEmbedding

embedding = CheonEmbedding()
queries = embedding.embed_queries(["How do I renew my passport online?"])
passages = embedding.embed_documents(["Passports can be renewed online through the government portal."])
scores = queries @ passages.T
```

`embed_queries` writes each query after a one-line task instruction (web search
unless you pass another); `embed_documents` takes passages as they are. Both
return one unit vector per text in input order, so the dot product is the
cosine similarity.

## Recipes

| # | Recipe | What you learn | What it prints |
|---:|:---|:---|:---|
| 01 | [`embed_and_score.py`](embed_and_score.py) | Queries with an instruction, passages without, one score | The model card's query and passage and their cosine similarity |
