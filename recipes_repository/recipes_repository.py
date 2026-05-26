import json
from datetime import datetime
from enum import Enum
from typing import Any, Iterable, List, Optional
from pathlib import Path

import pandas as pd
from sqlalchemy import bindparam, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session


RELEVANT_COLS_RECIPES = text(Path("../SQL/select_relevant_cols_recipes.sql").read_text())


class RecipesRepository:

    def __init__(self, engine: Engine) -> None:
        self.engine = engine
        self._relevant_cols_sql = RELEVANT_COLS_RECIPES

    def _enum_value(self, value: Any) -> Any:
        return value.value if isinstance(value, Enum) else value

    def _enum_list(self, values: Any) -> list:
        if values is None:
            return []
        return [self._enum_value(value) for value in values]