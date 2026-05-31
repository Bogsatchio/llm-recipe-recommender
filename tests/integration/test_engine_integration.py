from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from recipes_repository.recipes_repository import RecipesRepository
from recommender_engine.recommender_engine import RecommenderEngine


pytestmark = pytest.mark.integration


class FakeEmbeddingClient:
    def __init__(self, embedding: list[float]) -> None:
        self.embedding = embedding
        self.embeddings = self
        self.calls: list[dict[str, object]] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        input_value = kwargs["input"]
        if isinstance(input_value, list):
            data = [SimpleNamespace(embedding=self.embedding) for _ in input_value]
        else:
            data = [SimpleNamespace(embedding=self.embedding)]
        return SimpleNamespace(data=data)


def vector_1536(first: float = 0.0, second: float = 0.0, third: float = 0.0) -> list[float]:
    vector = [0.0] * 1536
    vector[0] = first
    vector[1] = second
    vector[2] = third
    return vector


def seed_engine_collection(
    qdrant_client: QdrantClient,
    collection_name: str,
) -> None:
    qdrant_client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
    )
    qdrant_client.upsert(
        collection_name=collection_name,
        points=[
            PointStruct(
                id=101,
                vector=vector_1536(first=1.0),
                payload={
                    "id": 101,
                    "overall_time": 25,
                    "is_vegan": 0,
                    "is_vegetarian": 1,
                    "is_gluten_free": 0,
                    "is_halal": 1,
                    "is_kosher": 0,
                    "keto_friendliness": 0,
                },
            ),
            PointStruct(
                id=102,
                vector=vector_1536(second=1.0),
                payload={
                    "id": 102,
                    "overall_time": 8,
                    "is_vegan": 1,
                    "is_vegetarian": 1,
                    "is_gluten_free": 1,
                    "is_halal": 1,
                    "is_kosher": 1,
                    "keto_friendliness": 1,
                },
            ),
            PointStruct(
                id=103,
                vector=vector_1536(third=1.0),
                payload={
                    "id": 103,
                    "overall_time": 120,
                    "is_vegan": 0,
                    "is_vegetarian": 0,
                    "is_gluten_free": 1,
                    "is_halal": 0,
                    "is_kosher": 0,
                    "keto_friendliness": 1,
                },
            ),
        ],
    )


def test_engine_recommends_with_real_mysql_and_qdrant_but_fake_llm(
    mysql_repository: RecipesRepository,
    qdrant_client: QdrantClient,
    qdrant_collection: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seed_engine_collection(qdrant_client, qdrant_collection)

    def fake_completion(**kwargs):
        if "response_format" not in kwargs:
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content="spicy noodles"))]
            )

        payload = {
            "ranked_recipes": [
                {"recipe_id": 101, "new_score": 0.98},
            ],
            "first_place_justification": "Best match for spicy noodles.",
            "second_place_justification": "No second recipe selected.",
            "third_place_justification": "No third recipe selected.",
        }
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(payload)))]
        )

    monkeypatch.setattr("recommender_engine.recommender_engine.completion", fake_completion)

    engine = RecommenderEngine(
        mysql_repository,
        qdrant_client,
        openai_client=FakeEmbeddingClient(vector_1536(first=1.0)),
        collection_name=qdrant_collection,
        retrieval_limit=3,
        prefetch_limit=3,
        score_threshold=0.0,
        load_environment=False,
    )
    history = []

    result = engine.recommend(
        "I want spicy noodles",
        history,
        dietary_filters=["is_vegetarian"],
        overall_time_range=(20, 30),
    )

    assert result.retrieval_query == "spicy noodles"
    assert result.recommendations["id"].tolist() == [101]
    assert result.recommendations["new_score"].tolist() == [0.98]
    assert result.justifications["first_place"] == "Best match for spicy noodles."
    assert len(history) == 2


def test_engine_returns_empty_result_when_real_qdrant_filter_has_no_hits(
    mysql_repository: RecipesRepository,
    qdrant_client: QdrantClient,
    qdrant_collection: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seed_engine_collection(qdrant_client, qdrant_collection)

    monkeypatch.setattr(
        "recommender_engine.recommender_engine.completion",
        lambda **kwargs: SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="spicy noodles"))]
        ),
    )

    engine = RecommenderEngine(
        mysql_repository,
        qdrant_client,
        openai_client=FakeEmbeddingClient(vector_1536(first=1.0)),
        collection_name=qdrant_collection,
        retrieval_limit=3,
        prefetch_limit=3,
        score_threshold=0.0,
        load_environment=False,
    )
    history = []

    result = engine.recommend(
        "I want spicy noodles",
        history,
        dietary_filters=["is_vegan"],
        overall_time_range=(20, 30),
    )

    assert result.recommendations.empty
    assert result.retrieval_query == "spicy noodles"
    assert len(history) == 2
