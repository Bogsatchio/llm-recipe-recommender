from __future__ import annotations

from types import SimpleNamespace

import pandas as pd


class FakeEmbeddingClient:
    def __init__(self, embedding: list[float] | None = None) -> None:
        self.embedding = embedding or [0.1, 0.2, 0.3]
        self.calls: list[dict[str, object]] = []
        self.embeddings = self

    def create(self, **kwargs):
        self.calls.append(kwargs)
        input_value = kwargs["input"]
        if isinstance(input_value, list):
            data = [SimpleNamespace(embedding=self.embedding) for _ in input_value]
        else:
            data = [SimpleNamespace(embedding=self.embedding)]
        return SimpleNamespace(data=data)


class FakeRepository:
    def __init__(self, recipes: pd.DataFrame) -> None:
        self.recipes = recipes.copy()
        self.requested_ids: list[list[int]] = []

    def get_relevant_recipes_by_ids(self, recipe_ids):
        ids = list(recipe_ids)
        self.requested_ids.append(ids)
        return self.recipes[self.recipes["id"].isin(ids)].copy()

    def get_recipes(self):
        return self.recipes.copy()


class FakeQdrantClient:
    def __init__(self, points=None, collection_exists: bool = True, count: int = 0) -> None:
        self.points = points or []
        self.collection_exists_value = collection_exists
        self.count_value = count
        self.query_calls: list[dict[str, object]] = []
        self.deleted_collections: list[str] = []
        self.created_collections: list[dict[str, object]] = []
        self.upserts: list[dict[str, object]] = []

    def query_points(self, **kwargs):
        self.query_calls.append(kwargs)
        return SimpleNamespace(points=self.points)

    def collection_exists(self, collection_name: str) -> bool:
        return self.collection_exists_value

    def count(self, **kwargs):
        return SimpleNamespace(count=self.count_value)

    def delete_collection(self, collection_name: str) -> None:
        self.deleted_collections.append(collection_name)

    def create_collection(self, **kwargs) -> None:
        self.created_collections.append(kwargs)

    def upsert(self, **kwargs) -> None:
        self.upserts.append(kwargs)
