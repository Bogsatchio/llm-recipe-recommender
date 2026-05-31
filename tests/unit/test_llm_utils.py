from __future__ import annotations

import pandas as pd

from recommender_engine.llm_utils import (
    _append_interaction_to_history,
    _empty_recommendation_result,
    _extract_justifications,
    build_semantic_representation,
    clean_reply,
    complexity_label,
    effort_label,
    enhance_query_with_filters,
    nutrition_summary,
    safe_parse,
)


def test_extract_justifications_maps_response_fields() -> None:
    assert _extract_justifications(
        {
            "first_place_justification": "first",
            "second_place_justification": "second",
            "third_place_justification": "third",
        }
    ) == {
        "first_place": "first",
        "second_place": "second",
        "third_place": "third",
    }


def test_empty_recommendation_result_has_consistent_message() -> None:
    result = _empty_recommendation_result("query")

    assert result.retrieval_query == "query"
    assert result.recommendations.empty
    assert result.justifications["first_place"].startswith("No matching recipes")


def test_enhance_query_with_filters_appends_prerequisites() -> None:
    result = enhance_query_with_filters(
        "I want dinner",
        ["is_vegan", "keto_friendliness"],
        (45, 15),
    )

    assert result.startswith("I want dinner")
    assert "recipes must be vegan, keto-friendly" in result
    assert "Preparation time must be between 15 and 45 minutes" in result


def test_enhance_query_without_filters_returns_original_query() -> None:
    assert enhance_query_with_filters("plain query") == "plain query"


def test_safe_parse_handles_json_and_invalid_values() -> None:
    assert safe_parse('["a", "b"]') == ["a", "b"]
    assert safe_parse(["a"]) == ["a"]
    assert safe_parse("not json") is None


def test_effort_and_complexity_labels() -> None:
    assert effort_label(20, 15) == "Quick"
    assert effort_label(45, 20) == "Medium"
    assert effort_label(90, 45) == "Long"
    assert complexity_label(4) == "Simple"
    assert complexity_label(8) == "Moderate"
    assert complexity_label(12) == "High"


def test_nutrition_summary_extracts_useful_labels() -> None:
    summary = nutrition_summary("{'Calories': '650 kcal', 'Sugar': '20g', 'Fat': '35g'}")

    assert "high calorie" in summary
    assert "high sugar" in summary
    assert "rich" in summary


def test_clean_reply_empty_dataframe() -> None:
    assert clean_reply(pd.DataFrame()).endswith("No matching recipes were found.")


def test_build_semantic_representation_includes_core_recipe_fields(sample_recipes_df: pd.DataFrame) -> None:
    row = sample_recipes_df.iloc[0].copy()
    row["score"] = 0.873

    result = build_semantic_representation(row)

    assert "[RECIPE_ID: 1] - SCORE: 0.87" in result
    assert "Recipe: Spicy Noodles" in result
    assert "Cuisine: Thai" in result
    assert "Ingredients: noodles, chili" in result
    assert "Well rated" in result or "Highly rated" in result


def test_append_interaction_to_history_adds_user_and_assistant(sample_recipes_df: pd.DataFrame) -> None:
    history = []

    _append_interaction_to_history("question", sample_recipes_df, history)

    assert history[0] == {"role": "user", "content": "question"}
    assert history[1]["role"] == "assistant"
    assert "Recipe recommendations were provided" in history[1]["content"]
