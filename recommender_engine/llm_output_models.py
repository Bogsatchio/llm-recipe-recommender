from pydantic import BaseModel, Field


DEFAULT_FINAL_RECOMMENDATION_COUNT = 7


class RankedRecipe(BaseModel):
    recipe_id: int = Field(description="Unique recipe identifier")
    new_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Re-evaluated relevance score from 0 to 1",
    )


class RankAndJustifications(BaseModel):
    ranked_recipes: list[RankedRecipe] = Field(
        min_length=DEFAULT_FINAL_RECOMMENDATION_COUNT,
        max_length=DEFAULT_FINAL_RECOMMENDATION_COUNT,
        description="Exactly 7 recipes ordered from highest new_score to lowest new_score",
    )
    first_place_justification: str = Field(
        max_length=300,
        description="Short justification for the best recipe",
    )
    second_place_justification: str = Field(
        max_length=300,
        description="Short justification for the second best recipe",
    )
    third_place_justification: str = Field(
        max_length=300,
        description="Short justification for the third best recipe",
    )