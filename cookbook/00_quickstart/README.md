# Quickstart

Embed a query and five passages, and rank the passages by how well they answer it.

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
passages = embedding.embed_documents([
    "A lost or stolen passport should be reported to the authority that issued it.",
    "Passports can be renewed online through the government portal.",
    "Water boils at 100 degrees Celsius at sea level.",
    "여권 재발급은 정부 포털에서 온라인으로 신청할 수 있습니다.",
    "Los pasaportes se pueden renovar en línea a través del portal del gobierno.",
])
scores = queries @ passages.T
print(scores.argsort(descending=True))  # tensor([[1, 4, 3, 0, 2]])
```

`embed_queries` writes each query after a one-line task instruction (web search
unless you pass another); `embed_documents` takes passages as they are. Both
return one unit vector per text in input order, so the dot product is the
cosine similarity, and sorting the passages by it ranks them for the query. Of
the five, three answer the question in different languages, one is about
passports but does not answer it, and one is unrelated; the answers come first
and the unrelated sentence last.

## Recipes

| # | Recipe | What you learn | What it prints |
|---:|:---|:---|:---|
| 01 | [`embed_and_score.py`](embed_and_score.py) | Queries with an instruction, passages without, a ranking | The model card's query and passages, best first, with their cosine similarity |
