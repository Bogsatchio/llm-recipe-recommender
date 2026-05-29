import json
import ast
from dataclasses import dataclass
from typing import Any, MutableSequence, Sequence
import pandas as pd

from recommender_engine.prompts import (
    RECOMMENDATION_SYSTEM_PROMPT_TEMPLATE,
)


DIETARY_FILTER_LABELS = {
    "is_vegan": "vegan",
    "is_vegetarian": "vegetarian",
    "is_gluten_free": "gluten-free",
    "is_halal": "halal",
    "is_kosher": "kosher",
    "keto_friendliness": "keto-friendly",
}


ChatMessage = dict[str, str]


@dataclass(frozen=True)
class RecommendationResult:
    recommendations: pd.DataFrame
    retrieval_query: str
    justifications: dict[str, str]


def _extract_justifications(response_data: dict[str, Any]) -> dict[str, str]:
    return {
        "first_place": response_data["first_place_justification"],
        "second_place": response_data["second_place_justification"],
        "third_place": response_data["third_place_justification"],
    }


def _empty_recommendation_result(retrieval_query: str) -> RecommendationResult:
    message = "No matching recipes were found for this request and filter set."
    return RecommendationResult(
        recommendations=pd.DataFrame(),
        retrieval_query=retrieval_query,
        justifications={
            "first_place": message,
            "second_place": message,
            "third_place": message,
        },
    )


def _append_interaction_to_history(
    question: str,
    recommendations: pd.DataFrame,
    chat_history: MutableSequence[ChatMessage],
) -> None:
    chat_history.append({"role": "user", "content": question})
    chat_history.append(
        {"role": "assistant", "content": clean_reply(recommendations)}
    )


def build_system_prompt(context: str) -> str:
    return RECOMMENDATION_SYSTEM_PROMPT_TEMPLATE.format(context=context)


def enhance_query_with_filters(
    query: str,
    dietary_filters: Sequence[str] | None = None,
    overall_time_range: tuple[float, float] | None = None,
) -> str:
    prerequisites = []

    selected_dietary_filters = [
        DIETARY_FILTER_LABELS.get(field, field.replace("_", " "))
        for field in dietary_filters or []
        if field
    ]
    if selected_dietary_filters:
        prerequisites.append(
            "recipes must be " + ", ".join(selected_dietary_filters)
        )

    if overall_time_range is not None:
        min_time, max_time = overall_time_range
        min_time = max(0.0, float(min_time))
        max_time = max(0.0, float(max_time))
        if min_time > max_time:
            min_time, max_time = max_time, min_time

        prerequisites.append(
            f"Preparation time must be between "
            f"{min_time:g} and {max_time:g} minutes"
        )

    if not prerequisites:
        return query

    return f"{query}\n\nPREREQUISITES: {'; '.join(prerequisites)}"


def clean_reply(recommendations: pd.DataFrame) -> str:
    reply = (
        "Recipe recommendations were provided based on previous preferences. "
        "Here are the top 3 picks: \n"
    )
    if recommendations.empty:
        return reply + "No matching recipes were found."

    top_3 = "\n\n".join(
        recommendations.head(3).apply(
            build_semantic_representation,
            add_id=False,
            axis=1,
        ).tolist()
    )
    return reply + top_3


def safe_parse(value):
    """Parse JSON strings or return value as-is if already a list/dict."""
    if isinstance(value, (list, dict)):
        return value
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return None


def effort_label(total_minutes, preparation_minutes=0):
    total_minutes = 0 if pd.isna(total_minutes) else total_minutes
    preparation_minutes = 0 if pd.isna(preparation_minutes) else preparation_minutes
    if total_minutes <= 30 or preparation_minutes <= 10:  return "Quick"
    if total_minutes <= 60 or preparation_minutes <= 30:  return "Medium"
    return "Long"


def complexity_label(steps, total_minutes=0):
    steps = 0 if pd.isna(steps) else steps
    if steps <= 5:   return "Simple"
    if steps <= 10:  return "Moderate"
    return "High"


def nutrition_summary(nutrition_raw):
    # 1. Cleanly attempt to parse the data
    data = None
    try:
        data = ast.literal_eval(safe_parse(nutrition_raw))
    except (SyntaxError, ValueError):
        # Attempt the fix
        try:
            data = ast.literal_eval(safe_parse(nutrition_raw + "}"))
        except Exception:
            data = None
    except Exception:
        data = None

    # 2. Guard clause: Exit early if parsing failed
    if not data:
        return ""

    # 3. Logic: Define helper and build parts
    def extract_int(key):
        val = data.get(key, "")
        try:
            # Strips non-digits to cleanly handle strings like "15g" or "400 kcal"
            digits = "".join(filter(str.isdigit, str(val)))
            return int(digits) if digits else None
        except (ValueError, TypeError):
            return None

    parts = []

    # Calorie threshold checks
    calories = extract_int("Calories")
    if calories is not None:
        if calories < 200:
            parts.append("low calorie")
        elif calories < 500:
            parts.append("moderate calorie")
        else:
            parts.append("high calorie")

    # Simple high-threshold checks
    thresholds = {"Sugar": 15, "Sodium": 600, "Protein": 15}
    for key, limit in thresholds.items():
        val = extract_int(key)
        if val is not None and val > limit:
            parts.append(f"high {key.lower()}")

    # Specialized Fat check
    fat = extract_int("Fat")
    if fat is not None:
        if fat < 5:
            parts.append("low fat")
        elif fat > 30:
            parts.append("rich")

    return ", ".join(parts)


def build_semantic_representation(row, add_id=True):
    parts = []

    if add_id:
        parts.append(f"[RECIPE_ID: {row['id']}] - SCORE: {round(row['score'], 2)}")

    # name
    parts.append(f"Recipe: {row['name']}")

    # cuisine & category
    if row['cuisine']:
        parts.append(f"Cuisine: {row['cuisine']}")

    category = safe_parse(row['category'])
    if category:
        parts.append(f"Category: {', '.join(category)}")

    # cooking method
    methods = safe_parse(row['cooking_methods'])
    if methods:
        parts.append(f"Cooking method: {', '.join(methods)}")

    # ingredients (clean names only, not raw)
    ingredients = safe_parse(row['ingredients'])
    if ingredients:
        parts.append(f"Ingredients: {', '.join(ingredients)}")

    # dietary flags
    dietary_labels = {
        "is_vegan":        "vegan",
        "is_vegetarian":   "vegetarian",
        "is_gluten_free":  "gluten-free",
        "is_halal":        "halal",
        "is_kosher":       "kosher",
        "keto_friendliness": "keto-friendly",
    }

    active = [label for col, label in dietary_labels.items() if row.get(col)]
    if active:
        parts.append(f"Dietary: {', '.join(active)}")

    # effort
    prep  = 0 if pd.isna(row['preparation_time']) else row['preparation_time']
    cook  = 0 if pd.isna(row['cooking_time']) else row['cooking_time']
    total = row['overall_time']
    steps = 0 if pd.isna(row['number_of_steps']) else row['number_of_steps']
    if total > 0 or steps > 0:
        parts.append(
            f"Effort: {complexity_label(steps, total)} / {effort_label(total, prep)}"
            f"\nPreparation time: {total} min \nSteps: {steps}"
        )

    # nutrition
    nutrition_text = nutrition_summary(row['nutrition'])
    if nutrition_text:
        parts.append(f"Nutrition: {nutrition_text}")

    # rating  — only mention if meaningfully rated
    try:
        rating = float(row['rating_value'])
        count  = int(row['rating_count'])
        if rating >= 4.5 and count >= 20:
            parts.append(f"Highly rated ({rating}/5, {count} reviews)")
        elif rating >= 4.0 and count >= 20:
            parts.append(f"Well rated ({rating}/5)")
    except (TypeError, ValueError):
        pass

    return "\n".join(parts)



# QUERY = """
#     SELECT
#         id,
#         name,
#         rating_value,
#         rating_count,
#         preparation_time,
#         cooking_time,
#         preparation_time + cooking_time AS overall_time,
#         category,
#         cuisine,
#         ingredients,
#         --instructions,
#         cooking_methods,
#         number_of_steps,
#         nutrition,
#         is_vegan,
#         is_vegetarian,
#         is_gluten_free,
#         is_halal,
#         is_kosher,
#         keto_friendliness
#     FROM recipes_main
# """
