from __future__ import annotations

from enum import Enum

import pandas as pd
from sqlalchemy import create_engine, text

from recipes_repository.recipes_repository import RECIPES_MAIN_TABLE, RecipesRepository


class SampleEnum(Enum):
    VALUE = "value"


def create_sqlite_repository() -> RecipesRepository:
    engine = create_engine("sqlite:///:memory:")
    with engine.begin() as connection:
        connection.execute(
            text(
                """
                CREATE TABLE recipes_main (
                    id INTEGER PRIMARY KEY,
                    name TEXT NOT NULL,
                    rating_value FLOAT,
                    rating_count INTEGER,
                    preparation_time INTEGER,
                    cooking_time INTEGER,
                    category TEXT,
                    cuisine TEXT,
                    ingredients TEXT,
                    ingredients_normalized TEXT,
                    cooking_methods TEXT,
                    number_of_steps INTEGER,
                    nutrition TEXT,
                    is_vegan INTEGER,
                    is_vegetarian INTEGER,
                    is_gluten_free INTEGER,
                    is_halal INTEGER,
                    is_kosher INTEGER,
                    keto_friendliness INTEGER
                )
                """
            )
        )
        connection.execute(
            text(
                """
                INSERT INTO recipes_main (
                    id, name, rating_value, rating_count, preparation_time, cooking_time,
                    category, cuisine, ingredients, ingredients_normalized, cooking_methods,
                    number_of_steps, nutrition, is_vegan, is_vegetarian, is_gluten_free,
                    is_halal, is_kosher, keto_friendliness
                )
                VALUES
                    (1, 'Noodles', 4.8, 25, 10, 15, '["dinner"]', 'Thai',
                     '["noodles"]', '["noodles"]', '["boil"]', 4, '{}',
                     0, 1, 0, 1, 0, 0),
                    (2, 'Salad', 4.1, 40, 8, 0, '["lunch"]', 'Greek',
                     '["lettuce"]', '["lettuce"]', '["raw"]', 3, '{}',
                     1, 1, 1, 1, 1, 1)
                """
            )
        )
    return RecipesRepository(engine)


def test_enum_helpers() -> None:
    repository = create_sqlite_repository()

    assert repository._enum_value(SampleEnum.VALUE) == "value"
    assert repository._enum_value("plain") == "plain"
    assert repository._enum_list(None) == []
    assert repository._enum_list([SampleEnum.VALUE, "plain"]) == ["value", "plain"]


def test_get_relevant_recipes_by_ids_empty_ids_returns_empty_dataframe() -> None:
    repository = create_sqlite_repository()

    result = repository.get_relevant_recipes_by_ids([])

    assert result.empty


def test_get_relevant_recipes_by_ids_returns_matching_rows() -> None:
    repository = create_sqlite_repository()

    result = repository.get_relevant_recipes_by_ids([2])

    assert result["id"].tolist() == [2]
    assert result["name"].tolist() == ["Salad"]
    assert result["overall_time"].tolist() == [8]


def test_get_recipes_returns_all_rows() -> None:
    repository = create_sqlite_repository()

    result = repository.get_recipes()

    assert result["id"].tolist() == [1, 2]


def test_ensure_source_data_loaded_returns_zero_when_table_has_rows() -> None:
    repository = create_sqlite_repository()

    assert repository.ensure_source_data_loaded() == 0


def test_ensure_source_data_loaded_loads_parquet_when_table_empty(monkeypatch) -> None:
    engine = create_engine("sqlite:///:memory:")
    repository = RecipesRepository(engine)
    repository._relevant_cols_recipes_sql = "SELECT id, name FROM recipes_main"
    loaded_df = pd.DataFrame({"id": [1, 2], "name": ["A", "B"]})

    monkeypatch.setattr(
        "recipes_repository.recipes_repository.RECIPES_MAIN_DDL_SQL",
        "CREATE TABLE recipes_main (id INTEGER PRIMARY KEY, name TEXT NOT NULL)",
    )
    monkeypatch.setattr(
        "recipes_repository.recipes_repository.pd.read_parquet",
        lambda path: loaded_df,
    )

    assert repository.ensure_source_data_loaded("fake.parquet") == 2

    with engine.connect() as connection:
        row_count = connection.execute(
            text(f"SELECT COUNT(*) FROM {RECIPES_MAIN_TABLE}")
        ).scalar_one()
    assert row_count == 2
