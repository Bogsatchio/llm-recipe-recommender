from __future__ import annotations

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
