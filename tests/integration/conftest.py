from __future__ import annotations

import os
import uuid
from collections.abc import Iterator

import pandas as pd
import pytest
from qdrant_client import QdrantClient
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from recipes_repository.recipes_repository import RECIPES_MAIN_TABLE, RecipesRepository


TEST_DATABASE_URL = os.getenv(
    "TEST_DATABASE_URL",
    "mysql+pymysql://test:test@127.0.0.1:3307/recipe_app_test",
)
TEST_QDRANT_HOST = os.getenv("TEST_QDRANT_HOST", "127.0.0.1")
TEST_QDRANT_PORT = int(os.getenv("TEST_QDRANT_PORT", "6334"))


@pytest.fixture
def integration_recipes_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "id": 101,
                "name": "Spicy Noodles",
                "created_at": pd.Timestamp("2026-01-01 10:00:00"),
                "rating_value": 4.8,
                "rating_count": 25,
                "preparation_time": 10,
                "cooking_time": 15,
                "category": '["dinner"]',
                "cuisine": "Thai",
                "ingredients": '["noodles", "chili"]',
                "ingredients_raw": '["noodles", "fresh chili"]',
                "instructions": "Boil noodles and toss with chili.",
                "cooking_methods": '["boil"]',
                "implements": '["pot"]',
                "number_of_steps": 4,
                "nutrition": '{"Calories": "450", "Protein": "18g"}',
                "url": "https://example.test/spicy-noodles",
                "ingredients_normalized": '["noodles", "chili"]',
                "is_vegan": 0,
                "is_vegetarian": 1,
                "is_gluten_free": 0,
                "is_halal": 1,
                "is_kosher": 0,
                "keto_friendliness": 0,
            },
            {
                "id": 102,
                "name": "Green Salad",
                "created_at": pd.Timestamp("2026-01-02 10:00:00"),
                "rating_value": 4.1,
                "rating_count": 40,
                "preparation_time": 8,
                "cooking_time": 0,
                "category": '["lunch"]',
                "cuisine": "Mediterranean",
                "ingredients": '["lettuce", "cucumber"]',
                "ingredients_raw": '["lettuce", "cucumber"]',
                "instructions": "Chop vegetables and combine.",
                "cooking_methods": '["raw"]',
                "implements": '["knife"]',
                "number_of_steps": 3,
                "nutrition": '{"Calories": "120", "Fat": "2g"}',
                "url": "https://example.test/green-salad",
                "ingredients_normalized": '["lettuce", "cucumber"]',
                "is_vegan": 1,
                "is_vegetarian": 1,
                "is_gluten_free": 1,
                "is_halal": 1,
                "is_kosher": 1,
                "keto_friendliness": 1,
            },
            {
                "id": 103,
                "name": "Beef Stew",
                "created_at": pd.Timestamp("2026-01-03 10:00:00"),
                "rating_value": 4.6,
                "rating_count": 18,
                "preparation_time": 20,
                "cooking_time": 100,
                "category": '["dinner"]',
                "cuisine": "Polish",
                "ingredients": '["beef", "carrot"]',
                "ingredients_raw": '["beef", "carrot"]',
                "instructions": "Simmer beef and vegetables.",
                "cooking_methods": '["simmer"]',
                "implements": '["pot"]',
                "number_of_steps": 6,
                "nutrition": '{"Calories": "620", "Protein": "38g"}',
                "url": "https://example.test/beef-stew",
                "ingredients_normalized": '["beef", "carrot"]',
                "is_vegan": 0,
                "is_vegetarian": 0,
                "is_gluten_free": 1,
                "is_halal": 0,
                "is_kosher": 0,
                "keto_friendliness": 1,
            },
        ]
    )


@pytest.fixture
def mysql_engine() -> Iterator[Engine]:
    engine = create_engine(TEST_DATABASE_URL, pool_pre_ping=True)
    try:
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
    except Exception as exc:
        engine.dispose()
        pytest.skip(f"MySQL integration database is not available: {exc}")

    with engine.begin() as connection:
        connection.execute(text(f"DROP TABLE IF EXISTS {RECIPES_MAIN_TABLE}"))

    try:
        yield engine
    finally:
        with engine.begin() as connection:
            connection.execute(text(f"DROP TABLE IF EXISTS {RECIPES_MAIN_TABLE}"))
        engine.dispose()


@pytest.fixture
def mysql_repository(
    mysql_engine: Engine,
    integration_recipes_df: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
) -> RecipesRepository:
    monkeypatch.setattr(
        "recipes_repository.recipes_repository.pd.read_parquet",
        lambda path: integration_recipes_df,
    )
    repository = RecipesRepository(mysql_engine)
    repository.ensure_source_data_loaded("integration-fixture.parquet")
    return repository


@pytest.fixture
def qdrant_client() -> Iterator[QdrantClient]:
    client = QdrantClient(host=TEST_QDRANT_HOST, port=TEST_QDRANT_PORT)
    try:
        client.get_collections()
    except Exception as exc:
        client.close()
        pytest.skip(f"Qdrant integration database is not available: {exc}")

    try:
        yield client
    finally:
        client.close()


@pytest.fixture
def qdrant_collection(qdrant_client: QdrantClient) -> Iterator[str]:
    collection_name = f"recipes_test_{uuid.uuid4().hex}"
    try:
        yield collection_name
    finally:
        if qdrant_client.collection_exists(collection_name):
            qdrant_client.delete_collection(collection_name)


def vector_1536(first: float = 0.0, second: float = 0.0, third: float = 0.0) -> list[float]:
    vector = [0.0] * 1536
    vector[0] = first
    vector[1] = second
    vector[2] = third
    return vector
