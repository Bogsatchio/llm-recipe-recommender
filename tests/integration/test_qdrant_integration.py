from __future__ import annotations

import pytest
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, FieldCondition, Filter, MatchValue, PointStruct, Range, VectorParams


pytestmark = pytest.mark.integration


def test_qdrant_vector_search_returns_nearest_payload(
    qdrant_client: QdrantClient,
    qdrant_collection: str,
) -> None:
    qdrant_client.create_collection(
        collection_name=qdrant_collection,
        vectors_config=VectorParams(size=3, distance=Distance.COSINE),
    )
    qdrant_client.upsert(
        collection_name=qdrant_collection,
        points=[
            PointStruct(id=101, vector=[1.0, 0.0, 0.0], payload={"id": 101, "name": "Spicy Noodles"}),
            PointStruct(id=102, vector=[0.0, 1.0, 0.0], payload={"id": 102, "name": "Green Salad"}),
        ],
    )

    result = qdrant_client.query_points(
        collection_name=qdrant_collection,
        query=[1.0, 0.0, 0.0],
        limit=1,
        with_payload=True,
    )

    assert result.points[0].payload["id"] == 101


def test_qdrant_filters_by_dietary_payload(
    qdrant_client: QdrantClient,
    qdrant_collection: str,
) -> None:
    qdrant_client.create_collection(
        collection_name=qdrant_collection,
        vectors_config=VectorParams(size=3, distance=Distance.COSINE),
    )
    qdrant_client.upsert(
        collection_name=qdrant_collection,
        points=[
            PointStruct(id=101, vector=[1.0, 0.0, 0.0], payload={"id": 101, "is_vegan": 0}),
            PointStruct(id=102, vector=[0.9, 0.1, 0.0], payload={"id": 102, "is_vegan": 1}),
        ],
    )

    result = qdrant_client.query_points(
        collection_name=qdrant_collection,
        query=[1.0, 0.0, 0.0],
        query_filter=Filter(
            must=[FieldCondition(key="is_vegan", match=MatchValue(value=1))]
        ),
        limit=5,
        with_payload=True,
    )

    assert [point.payload["id"] for point in result.points] == [102]


def test_qdrant_filters_by_combined_dietary_and_time_payload(
    qdrant_client: QdrantClient,
    qdrant_collection: str,
) -> None:
    qdrant_client.create_collection(
        collection_name=qdrant_collection,
        vectors_config=VectorParams(size=3, distance=Distance.COSINE),
    )
    qdrant_client.upsert(
        collection_name=qdrant_collection,
        points=[
            PointStruct(
                id=101,
                vector=[1.0, 0.0, 0.0],
                payload={"id": 101, "is_vegetarian": 1, "overall_time": 25},
            ),
            PointStruct(
                id=102,
                vector=[0.9, 0.1, 0.0],
                payload={"id": 102, "is_vegetarian": 1, "overall_time": 8},
            ),
            PointStruct(
                id=103,
                vector=[0.8, 0.2, 0.0],
                payload={"id": 103, "is_vegetarian": 0, "overall_time": 120},
            ),
        ],
    )

    result = qdrant_client.query_points(
        collection_name=qdrant_collection,
        query=[1.0, 0.0, 0.0],
        query_filter=Filter(
            must=[
                FieldCondition(key="is_vegetarian", match=MatchValue(value=1)),
                FieldCondition(key="overall_time", range=Range(gte=20, lte=30)),
            ]
        ),
        limit=5,
        with_payload=True,
    )

    assert [point.payload["id"] for point in result.points] == [101]
