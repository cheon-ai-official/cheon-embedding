# Quickstart

Embed a query and eight passages in five languages, and rank the passages by how well they answer it.

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
    "여권을 잃어버렸다면 바로 분실 신고를 해야 합니다.",
    "Passports can be renewed online through the government portal.",
    "富士山は日本で一番高い山です。",
    "여권 재발급은 정부 포털에서 온라인으로 신청할 수 있습니다.",
    "يمكن تجديد جواز السفر عبر الإنترنت من خلال البوابة الحكومية.",
    "القاهرة هي عاصمة مصر.",
    "护照可以通过政府门户网站在线申请换发。",
    "パスポートの更新は、政府のポータルサイトからオンラインで申請できます。",
])
scores = queries @ passages.T
print(scores.argsort(descending=True))  # tensor([[1, 6, 4, 7, 3, 0, 5, 2]])
```

`embed_queries` writes each query after a one-line task instruction (web search
unless you pass another); `embed_documents` takes passages as they are. Both
return one unit vector per text in input order, so the dot product is the
cosine similarity, and sorting the passages by it ranks them for the query. Of
the eight, five answer the question, in English, Chinese, Arabic, Japanese and
Korean; one is about passports but does not answer it, and two are unrelated.
The five answers come first whatever their language, and the unrelated two last.

## Recipes

| # | Recipe | What you learn | What it prints |
|---:|:---|:---|:---|
| 01 | [`embed_and_score.py`](embed_and_score.py) | Queries with an instruction, passages without, a ranking | The model card's query and passages, best first, with their cosine similarity |
