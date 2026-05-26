from enum import Enum
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
from sqlalchemy import bindparam, text
from sqlalchemy.engine import Engine


RELEVANT_COLS_RECIPES_SQL = (
    Path(__file__).parent / "SQL" / "select_relevant_cols_recipes.sql"
).read_text()


class RecipesRepository:

    def __init__(self, engine: Engine) -> None:
        self.engine = engine
        self._relevant_cols_recipes_sql = RELEVANT_COLS_RECIPES_SQL

    def _enum_value(self, value: Any) -> Any:
        return value.value if isinstance(value, Enum) else value

    def _enum_list(self, values: Any) -> list:
        if values is None:
            return []
        return [self._enum_value(value) for value in values]

    def get_relevant_recipes_by_ids(self, recipe_ids: Iterable[int]) -> pd.DataFrame:
        ids = list(recipe_ids)
        if not ids:
            return pd.DataFrame()

        query = text(f"{self._relevant_cols_recipes_sql} WHERE id IN :ids").bindparams(
            bindparam("ids", expanding=True)
        )
        return pd.read_sql(query, self.engine, params={"ids": ids})
