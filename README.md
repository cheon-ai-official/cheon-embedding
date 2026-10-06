# Cheon Embedding (체온 임베딩)

Multilingual text embeddings for search and RAG, from [CHEON:AI](https://cheon.ai.kr).

This repository has a cookbook of runnable recipes and the small Python
package they share. The weights are on Hugging Face.

| Model | Base | Parameters | Dimensions | Context | License |
| --- | --- | --- | --- | --- | --- |
| [cheon-embedding-0.6b-v1](https://huggingface.co/cheonai/cheon-embedding-0.6b-v1) | harrier-oss-v1-0.6b | 596.0M | 1,024 | 32,768 tokens | CC-BY-NC-4.0 |

## Benchmarks

Average main score on MMTEB (Multilingual, v2), over the same 128 of its 131
tasks. Cheon Embedding was scored with the official `mteb` harness (float32,
up to 32,768 tokens); the other scores are from the official MTEB results
repository. Ranks are among all models there that report all 128 tasks; the
remaining three tasks are being scored.

| Rank | Model | Parameters | Average |
| ---: | --- | --- | ---: |
| 1 | microsoft/harrier-oss-v1-27b | 27B | 75.29 |
| 2 | tencent/KaLM-Embedding-Gemma3-12B-2511 | 12B | 73.32 |
| **3** | **Cheon Embedding** | **0.6B** | **71.73** |
| 4 | Qwen/Qwen3-Embedding-8B | 8B | 71.56 |
| 5 | Bytedance/Seed1.6-embedding-1215 | undisclosed | 71.23 |
| 6 | Qwen/Qwen3-Embedding-4B | 4B | 70.40 |
| 7 | nvidia/llama-embed-nemotron-8b | 8B | 70.38 |
| 8 | microsoft/harrier-oss-v1-0.6b | 0.6B | 69.97 |
| 10 | google/gemini-embedding-001 | undisclosed | 69.29 |
| 21 | Qwen/Qwen3-Embedding-0.6B | 0.6B | 65.24 |

At 0.6B parameters only two much larger models score higher. It is ahead of
Qwen3-Embedding-8B, a model about 13 times its size, and of Gemini Embedding,
and it leads the 8B model on bitext mining, classification, multilabel
classification and retrieval.

By task type (same 128 tasks):

| Task type | Tasks | Cheon Embedding | harrier-oss-v1-0.6b (base) | Qwen3-Embedding-8B |
|---|---|---|---|---|
| Bitext mining | 13 | **83.36** | 82.85 | 80.89 |
| Classification | 43 | **76.25** | 73.88 | 74.00 |
| Multilabel classification | 4 | **43.92** | 31.61 | 34.63 |
| Retrieval | 17 | **72.20** | 71.01 | 70.89 |
| Clustering | 16 | 55.13 | 54.00 | **57.65** |
| Pair classification | 11 | 83.35 | 82.07 | **86.40** |
| Reranking | 5 | 74.20 | 73.25 | **76.40** |
| Semantic similarity (STS) | 16 | 77.71 | 77.09 | **81.08** |
| Instruction reranking | 3 | 0.86 | 0.81 | **10.06** |

## Cookbook

| Recipe | What it shows |
| --- | --- |
| [00_quickstart/embed_and_score.py](cookbook/00_quickstart/embed_and_score.py) | A query with its instruction and five passages, ranked by cosine similarity |

```bash
git clone https://github.com/cheon-ai-official/cheon-embedding
cd cheon-embedding
pip install torch --index-url https://download.pytorch.org/whl/cpu   # skip on a CUDA machine
pip install -e .
python cookbook/00_quickstart/embed_and_score.py
```

The first run downloads the weights (2.4 GB, float32). More in
[cookbook/README.md](cookbook/README.md).

## Quick start

```python
from cheon_embedding import CheonEmbedding

embedding = CheonEmbedding()  # cheonai/cheon-embedding-0.6b-v1 on CUDA, MPS or CPU

queries = embedding.embed_queries(["How do I renew my passport online?"])
passages = embedding.embed_documents([
    "A lost or stolen passport should be reported to the authority that issued it.",
    "Passports can be renewed online through the government portal.",
    "Water boils at 100 degrees Celsius at sea level.",
    "여권 재발급은 정부 포털에서 온라인으로 신청할 수 있습니다.",
    "Los pasaportes se pueden renovar en línea a través del portal del gobierno.",
])
scores = queries @ passages.T  # cosine similarity: the vectors are unit length
print(scores.argsort(descending=True))  # tensor([[1, 4, 3, 0, 2]]): the three answers first, the boiling point last
```

`embed_queries` writes each query after a one-line task instruction: web
search by default (`cheon_embedding.WEB_SEARCH`), or pass your own, such as
"Retrieve semantically similar text". `embed_documents` takes passages as they
are. Both return one 1,024-dimensional unit vector per text, in input order,
for texts of up to 32,768 tokens.

Without the package, the model loads with `transformers` alone; the
[model card](https://huggingface.co/cheonai/cheon-embedding-0.6b-v1) shows how.

## Development

```bash
pip install -e ".[dev]"
ruff check .
pytest
```

The tests load the model from Hugging Face and run on CPU.

## License

The code in this repository and the model weights are released under
[CC-BY-NC-4.0](LICENSE): research and evaluation, not commercial use. For a
commercial license or hosted API access, [contact us](https://cheon.ai.kr/contact).
