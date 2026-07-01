from app.rerank import CrossEncoderReranker
from app.vectorstore import Retrieved


class FakeModel:
    # returns a score per (query, text) pair; higher = more relevant
    def predict(self, pairs):
        return [len(text) for _, text in pairs]


def _cand(cid, text):
    return Retrieved(chunk_id=cid, doc_id="d", source="s", page=1, group="g", score=0.0, text=text)


def test_cross_encoder_sorts_by_model_score_desc():
    r = CrossEncoderReranker(model=FakeModel())
    out = r.rerank("q", [_cand("a", "short"), _cand("b", "much longer text")])
    assert [c.chunk_id for c in out] == ["b", "a"]
    assert out[0].rerank_score == len("much longer text")


def test_cross_encoder_empty_candidates():
    assert CrossEncoderReranker(model=FakeModel()).rerank("q", []) == []
