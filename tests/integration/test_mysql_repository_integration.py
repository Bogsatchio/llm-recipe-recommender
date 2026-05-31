from __future__ import annotations

import pandas as pd
import pytest
from sqlalchemy import text
from sqlalchemy.engine import Engine

from recipes_repository.recipes_repository import RECIPES_MAIN_TABLE, RecipesRepository


pytestmark = pytest.mark.integration


def test_mysql_repository_loads_source_data_idempotently(
    mysql_engine: Engine,
    integration_recipes_df: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "recipes_repository.recipes_repository.pd.read_parquet",
        lambda path: integration_recipes_df,
    )
    repository = RecipesRepository(mysql_engine)

    assert repository.ensure_source_data_loaded("integration-fixture.parquet") == 3
    assert repository.ensure_source_data_loaded("integration-fixture.parquet") == 0

    with mysql_engine.connect() as connection:
        row_count = connection.execute(
            text(f"SELECT COUNT(*) FROM {RECIPES_MAIN_TABLE}")
        ).scalar_one()

    assert row_count == 3


def test_mysql_repository_reads_expected_recipe_projection(
    mysql_repository: RecipesRepository,
) -> None:
    recipes = mysql_repository.get_recipes()

    assert recipes["id"].tolist() == [101, 102, 103]
    assert recipes.loc[recipes["id"] == 101, "overall_time"].item() == 25
    assert "ingredients_raw" not in recipes.columns
    assert "instructions" not in recipes.columns


def test_mysql_repository_fetches_relevant_recipes_by_ids(
    mysql_repository: RecipesRepository,
) -> None:
    result = mysql_repository.get_relevant_recipes_by_ids([102, 101])

    assert set(result["id"].tolist()) == {101, 102}
    assert result.loc[result["id"] == 102, "name"].item() == "Green Salad"
    assert result.loc[result["id"] == 102, "overall_time"].item() == 8


def test_mysql_repository_empty_id_list_returns_empty_dataframe(
    mysql_repository: RecipesRepository,
) -> None:
    assert mysql_repository.get_relevant_recipes_by_ids([]).empty
