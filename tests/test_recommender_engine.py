from __future__ import annotations

import json
from types import SimpleNamespace

import pandas as pd
import pytest

from recommender_engine.recommender_engine import RecommenderEngine


def make_engine(sample_recipes_df, qdrant_client=None, openai_client=None) -> RecommenderEngine:
    from tests.conftest import FakeEmbeddingClient, FakeQdrantClient, FakeRepository

    return RecommenderEngine(
        FakeRepository(sample_recipes_df),
        qdrant_client or FakeQdrantClient(),
        openai_client=openai_client or FakeEmbeddingClient(),
        collection_name="test_recipes",
        load_environment=False,
    )


def test_normalize_overall_time_range_clamps_and_sorts(sample_recipes_df: pd.DataFrame) -> None:
    engine = make_engine(sample_recipes_df)

    assert engine._normalize_overall_time_range((-10, 30)) == (0.0, 30.0)
    assert engine._normalize_overall_time_range((60, 15)) == (15.0, 60.0)


def test_build_qdrant_filter_returns_none_without_conditions(sample_recipes_df: pd.DataFrame) -> None:
    engine = make_engine(sample_recipes_df)

    assert engine._build_qdrant_filter(None, None) is None


def test_build_qdrant_filter_builds_dietary_and_time_conditions(sample_recipes_df: pd.DataFrame) -> None:
    engine = make_engine(sample_recipes_df)

    result = engine._build_qdrant_filter(["is_vegan"], (45, 10))

    assert result is not None
    assert len(result.must) == 2
    assert result.must[0].key == "is_vegan"
    assert result.must[0].match.value == 1
    assert result.must[1].key == "overall_time"
    assert result.must[1].range.gte == 10.0
    assert result.must[1].range.lte == 45.0


def test_build_qdrant_filter_rejects_unknown_filter(sample_recipes_df: pd.DataFrame) -> None:
    engine = make_engine(sample_recipes_df)

    with pytest.raises(ValueError, match="Unsupported dietary filter"):
        engine._build_qdrant_filter(["is_low_sodium"], None)


def test_vector_search_fetches_hit_recipes_and_sorts_by_score(sample_recipes_df: pd.DataFrame) -> None:
    from tests.conftest import FakeQdrantClient

    qdrant_client = FakeQdrantClient(
        points=[
            SimpleNamespace(payload={"id": 2}, score=0.95),
            SimpleNamespace(payload={"id": 1}, score=0.75),
        ]
    )
    engine = make_engine(sample_recipes_df, qdrant_client=qdrant_client)

    result = engine.vector_search("salad", n_returned=2, dietary_filters=["is_vegan"])

    assert result["id"].tolist() == [2, 1]
    assert result["score"].tolist() == [0.95, 0.75]
    assert qdrant_client.query_calls[0]["collection_name"] == "test_recipes"
    assert qdrant_client.query_calls[0]["query_filter"].must[0].key == "is_vegan"


def test_vector_search_returns_empty_dataframe_when_qdrant_has_no_hits(sample_recipes_df: pd.DataFrame) -> None:
    engine = make_engine(sample_recipes_df)

    result = engine.vector_search("missing")

    assert result.empty


def test_build_retrieval_query_uses_completion_response(monkeypatch, sample_recipes_df: pd.DataFrame) -> None:
    engine = make_engine(sample_recipes_df)
    calls = []

    def fake_completion(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=" spicy noodles "))],
        )

    monkeypatch.setattr("recommender_engine.recommender_engine.completion", fake_completion)

    result = engine.build_retrieval_query("question", [{"role": "assistant", "content": "previous"}])

    assert result == "spicy noodles"
    assert calls[0]["temperature"] == 0.05
    assert calls[0]["max_tokens"] == 80


def test_recommend_returns_empty_result_when_context_is_empty(monkeypatch, sample_recipes_df: pd.DataFrame) -> None:
    engine = make_engine(sample_recipes_df)
    history = []
    monkeypatch.setattr(engine, "build_retrieval_query", lambda question, history: "retrieval")
    monkeypatch.setattr(engine, "fetch_context", lambda *args, **kwargs: "")

    result = engine.recommend("question", history)

    assert result.recommendations.empty
    assert result.retrieval_query == "retrieval"
    assert len(history) == 2
    assert history[0]["role"] == "user"


def test_recommend_ranks_recipes_and_updates_history(monkeypatch, sample_recipes_df: pd.DataFrame) -> None:
    engine = make_engine(sample_recipes_df)
    history = []
    monkeypatch.setattr(engine, "build_retrieval_query", lambda question, history: "retrieval")
    monkeypatch.setattr(engine, "fetch_context", lambda *args, **kwargs: "context")
    monkeypatch.setattr(
        engine,
        "_rank_recipes",
        lambda *args, **kwargs: {
            "ranked_recipes": [
                {"recipe_id": 1, "new_score": 0.9},
                {"recipe_id": 2, "new_score": 0.8},
            ],
            "first_place_justification": "first",
            "second_place_justification": "second",
            "third_place_justification": "third",
        },
    )

    result = engine.recommend("question", history)

    assert result.recommendations["id"].tolist() == [1, 2]
    assert result.recommendations["new_score"].tolist() == [0.9, 0.8]
    assert result.justifications["first_place"] == "first"
    assert len(history) == 2


def test_build_output_dataframe_returns_empty_when_repository_has_no_rows(sample_recipes_df: pd.DataFrame) -> None:
    from tests.conftest import FakeRepository

    engine = make_engine(sample_recipes_df)
    engine.recipes_repository = FakeRepository(pd.DataFrame(columns=sample_recipes_df.columns))

    result = engine._build_output_dataframe({"ranked_recipes": [{"recipe_id": 99, "new_score": 1.0}]})

    assert result.empty


def test_ensure_source_vector_collection_loaded_skips_when_count_matches(sample_recipes_df: pd.DataFrame) -> None:
    from tests.conftest import FakeQdrantClient

    qdrant_client = FakeQdrantClient(collection_exists=True, count=len(sample_recipes_df))
    engine = make_engine(sample_recipes_df, qdrant_client=qdrant_client)

    assert engine.ensure_source_vector_collection_loaded() == 0


def test_ensure_source_vector_collection_loaded_reloads_on_count_mismatch(
    monkeypatch,
    sample_recipes_df: pd.DataFrame,
) -> None:
    from tests.conftest import FakeQdrantClient

    qdrant_client = FakeQdrantClient(collection_exists=True, count=0)
    engine = make_engine(sample_recipes_df, qdrant_client=qdrant_client)
    monkeypatch.setattr(engine, "_load_source_collection", lambda: 2)

    assert engine.ensure_source_vector_collection_loaded() == 2


def test_rank_recipes_parses_completion_json(monkeypatch, sample_recipes_df: pd.DataFrame) -> None:
    engine = make_engine(sample_recipes_df)
    payload = {
        "ranked_recipes": [{"recipe_id": 1, "new_score": 0.9}],
        "first_place_justification": "first",
        "second_place_justification": "second",
        "third_place_justification": "third",
    }

    def fake_completion(**kwargs):
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(payload)))]
        )

    monkeypatch.setattr("recommender_engine.recommender_engine.completion", fake_completion)

    assert engine._rank_recipes("question", [], "system") == payload
