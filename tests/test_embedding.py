import pytest
import torch
from conftest import DOCUMENT, PASSAGES, QUERY, RANKING

from cheon_embedding import WEB_SEARCH


def test_vectors_have_the_model_size_and_unit_length(embedding):
    vectors = torch.cat([embedding.embed_queries([QUERY]), embedding.embed_documents([DOCUMENT])])
    assert vectors.shape == (2, embedding.dimensions)
    assert torch.linalg.vector_norm(vectors, dim=-1).tolist() == pytest.approx([1.0, 1.0], abs=1e-4)


def test_same_input_gives_the_same_vectors(embedding):
    assert torch.allclose(embedding.embed_documents([DOCUMENT]), embedding.embed_documents([DOCUMENT]), atol=1e-6)


def test_a_batch_keeps_the_input_order(embedding):
    texts = [DOCUMENT, QUERY]
    together = embedding.embed_documents(texts)
    one_by_one = torch.cat([embedding.embed_documents([text]) for text in texts])
    assert torch.allclose(together, one_by_one, atol=1e-4)


def test_the_card_example_ranks_the_answers_first(embedding):
    scores = (embedding.embed_queries([QUERY]) @ embedding.embed_documents(PASSAGES).T)[0]
    assert scores.argsort(descending=True).tolist() == RANKING


def test_a_query_is_embedded_after_its_instruction(embedding):
    written = embedding.model.config.instruction_template.format(instruction=WEB_SEARCH) + QUERY
    assert torch.allclose(embedding.embed_queries([QUERY]), embedding.embed_documents([written]), atol=1e-5)


@pytest.mark.parametrize("texts", [[], "not a list", [DOCUMENT, "  "], [""]])
def test_rejects_unusable_input(embedding, texts):
    with pytest.raises(ValueError):
        embedding.embed_documents(texts)


def test_rejects_an_empty_instruction(embedding):
    with pytest.raises(ValueError):
        embedding.embed_queries([QUERY], instruction=" ")
