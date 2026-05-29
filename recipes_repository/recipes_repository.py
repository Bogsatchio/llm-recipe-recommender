from enum import Enum
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import logging
from sqlalchemy import bindparam, text
from sqlalchemy import inspect


RELEVANT_COLS_RECIPES_SQL = (
    Path(__file__).parent / "SQL" / "select_relevant_cols_recipes.sql"
).read_text()
RECIPES_MAIN_DDL_SQL = (Path(__file__).parent / "SQL" / "ddl.sql").read_text()
RECIPES_MAIN_TABLE = "recipes_main"
RECIPES_MAIN_PARQUET_PATH = (
    Path(__file__).resolve().parents[1] / "data" / "gold" / "recipes_main.parquet"
)

logger = logging.getLogger(__name__)


class RecipesRepository:

    def __init__(self, engine) -> None:
        self.engine = engine
        self._relevant_cols_recipes_sql = RELEVANT_COLS_RECIPES_SQL

    def _enum_value(self, value: Any) -> Any:
        return value.value if isinstance(value, Enum) else value

    def _enum_list(self, values: Any) -> list:
        if values is None:
            return []
        return [self._enum_value(value) for value in values]

    def ensure_source_data_loaded(
        self,
        parquet_path: str | Path = RECIPES_MAIN_PARQUET_PATH,
    ) -> int:
        """Create and populate recipes_main when the table is missing or empty.

        Returns the number of rows loaded. If the table already contains data,
        returns 0.
        """
        if not inspect(self.engine).has_table(RECIPES_MAIN_TABLE):
            with self.engine.begin() as connection:
                connection.execute(text(RECIPES_MAIN_DDL_SQL))
                logger.info("Source table was not present, `recipes_main` table created.")

        with self.engine.connect() as connection:
            row_count = connection.execute(
                text(f"SELECT COUNT(*) FROM {RECIPES_MAIN_TABLE}")
            ).scalar_one()

        if row_count:
            logger.info("`recipes_main` table present and has data.")
            return 0

        parquet_path = Path(parquet_path)
        recipes_df = pd.read_parquet(parquet_path)
        recipes_df.to_sql(
            RECIPES_MAIN_TABLE,
            self.engine,
            if_exists="append",
            index=False,
            chunksize=1000,
            method="multi",
        )
        logger.info("Data reloaded in `recipes_main` table.")
        return len(recipes_df)

    def get_relevant_recipes_by_ids(self, recipe_ids: Iterable[int]) -> pd.DataFrame:
        ids = list(recipe_ids)
        if not ids:
            return pd.DataFrame()

        query = text(f"{self._relevant_cols_recipes_sql} WHERE id IN :ids").bindparams(
            bindparam("ids", expanding=True)
        )
        return pd.read_sql(query, self.engine, params={"ids": ids})

    def get_recipes(self):
        return pd.read_sql(self._relevant_cols_recipes_sql, self.engine)
