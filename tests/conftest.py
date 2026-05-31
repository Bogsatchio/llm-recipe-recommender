from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest


@pytest.fixture
def sample_recipes_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "id": 1,
                "name": "Spicy Noodles",
                "rating_value": 4.8,
                "rating_count": 25,
                "preparation_time": 10,
                "cooking_time": 15,
                "overall_time": 25,
                "category": '["dinner"]',
                "cuisine": "Thai",
                "ingredients": '["noodles", "chili"]',
                "ingredients_normalized": '["noodles", "chili"]',
                "cooking_methods": '["boil"]',
                "number_of_steps": 4,
                "nutrition": "{'Calories': '450', 'Protein': '18g'}",
                "is_vegan": 0,
                "is_vegetarian": 1,
                "is_gluten_free": 0,
                "is_halal": 1,
                "is_kosher": 0,
                "keto_friendliness": 0,
            },
            {
                "id": 2,
                "name": "Green Salad",
                "rating_value": 4.1,
                "rating_count": 40,
                "preparation_time": 8,
                "cooking_time": 0,
                "overall_time": 8,
                "category": '["lunch"]',
                "cuisine": "Mediterranean",
                "ingredients": '["lettuce", "cucumber"]',
                "ingredients_normalized": '["lettuce", "cucumber"]',
                "cooking_methods": '["raw"]',
                "number_of_steps": 3,
                "nutrition": "{'Calories': '120', 'Fat': '2g'}",
                "is_vegan": 1,
                "is_vegetarian": 1,
                "is_gluten_free": 1,
                "is_halal": 1,
                "is_kosher": 1,
                "keto_friendliness": 1,
            },
        ]
    )


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
