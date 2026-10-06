# Cheon Embedding Cookbook

A few recipes for 체온 임베딩 (Cheon Embedding). Copy, paste, run.

Every recipe runs on its own and on the model card's examples. From the
repository root:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu   # skip on a CUDA machine
pip install -e .
python cookbook/00_quickstart/embed_and_score.py
```

The first run downloads the model (2.4 GB).

## Where to Start

**New to the model?** Start with [00_quickstart](./00_quickstart): embed a query
and a passage and score them.

| Folder | Recipe | What it shows |
| --- | --- | --- |
| [00_quickstart](./00_quickstart) | [`embed_and_score.py`](./00_quickstart/embed_and_score.py) | A query with its instruction, a passage, their cosine similarity |
