from __future__ import annotations

import pandas as pd

from app import (
    DISPLAY_COLUMNS,
    OVERALL_TIME_SLIDER_MAX,
    _build_overall_time_range,
    _format_justifications,
    _format_table,
    _new_chat,
    _respond,
)
from recommender_engine.llm_utils import RecommendationResult


def test_format_table_returns_display_columns_and_limits_rows() -> None:
    recommendations = pd.DataFrame(
        {
            "name": [f"Recipe {index}" for index in range(10)],
            "rating_value": [4.5] * 10,
            "ingredients_normalized": ["[]"] * 10,
            "overall_time": [20] * 10,
            "extra": ["ignored"] * 10,
        }
    )

    result = _format_table(recommendations)

    assert list(result.columns) == DISPLAY_COLUMNS
    assert len(result) == 7


def test_format_table_empty_returns_display_columns() -> None:
    result = _format_table(pd.DataFrame())

    assert result.empty
    assert list(result.columns) == DISPLAY_COLUMNS


def test_format_justifications_for_empty_result_uses_first_place_message() -> None:
    result = RecommendationResult(
        recommendations=pd.DataFrame(),
        retrieval_query="query",
        justifications={
            "first_place": "No matches",
            "second_place": "Unused",
            "third_place": "Unused",
        },
    )

    assert _format_justifications(result) == "No matches"


def test_format_justifications_for_recommendations_formats_top_three() -> None:
    result = RecommendationResult(
        recommendations=pd.DataFrame({"id": [1]}),
        retrieval_query="query",
        justifications={
            "first_place": "Best reason",
            "second_place": "Second reason",
            "third_place": "Third reason",
        },
    )

    formatted = _format_justifications(result)

    assert "**1. Best match**\nBest reason" in formatted
    assert "**2. Second best**\nSecond reason" in formatted
    assert "**3. Third best**\nThird reason" in formatted


def test_build_overall_time_range() -> None:
    assert _build_overall_time_range(False, 5, 20) is None
    assert _build_overall_time_range(True, 5, 20) == (5, 20)


def test_new_chat_returns_default_ui_state() -> None:
    result = _new_chat()

    assert result[0] == ""
    assert result[1] == []
    assert result[2] == []
    assert list(result[3].columns) == DISPLAY_COLUMNS
    assert result[4] == ""
    assert result[5] == []
    assert result[6] == []
    assert result[7] is False
    assert result[8] == 0
    assert result[9] == OVERALL_TIME_SLIDER_MAX


class FakeEngine:
    def __init__(self, result: RecommendationResult | None = None, exc: Exception | None = None) -> None:
        self.result = result
        self.exc = exc
        self.calls = []

    def recommend(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        if self.exc:
            raise self.exc
        return self.result


def test_respond_blank_message_does_not_call_engine() -> None:
    engine = FakeEngine()
    ui_history = [{"role": "user", "content": "previous"}]
    engine_history = [{"role": "assistant", "content": "previous"}]

    result = _respond("   ", [], False, 0, 10, ui_history, engine_history, engine)

    assert result[0] == ""
    assert result[1] is ui_history
    assert result[2] is ui_history
    assert result[3].empty
    assert list(result[3].columns) == DISPLAY_COLUMNS
    assert result[4] == ""
    assert result[5] is engine_history
    assert engine.calls == []


def test_respond_success_updates_ui_outputs(sample_recipes_df: pd.DataFrame) -> None:
    recommendation_result = RecommendationResult(
        recommendations=sample_recipes_df,
        retrieval_query="spicy noodle query",
        justifications={
            "first_place": "Best",
            "second_place": "Second",
            "third_place": "Third",
        },
    )
    engine = FakeEngine(recommendation_result)
    engine_history = []

    _, chatbot, ui_history, table, retrieval_query, returned_history = _respond(
        "spicy noodles",
        ["is_vegetarian"],
        True,
        0,
        30,
        [],
        engine_history,
        engine,
    )

    assert chatbot == ui_history
    assert ui_history[0] == {"role": "user", "content": "spicy noodles"}
    assert "Found 2 recommendations" in ui_history[1]["content"]
    assert list(table.columns) == DISPLAY_COLUMNS
    assert retrieval_query == "**Retrieval query**\n\n`spicy noodle query`"
    assert returned_history is engine_history
    assert engine.calls[0][1]["dietary_filters"] == ["is_vegetarian"]
    assert engine.calls[0][1]["overall_time_range"] == (0, 30)


def test_respond_engine_exception_returns_error_message() -> None:
    engine = FakeEngine(exc=RuntimeError("service unavailable"))
    engine_history = [{"role": "user", "content": "old"}]

    _, chatbot, ui_history, table, retrieval_query, returned_history = _respond(
        "dinner",
        [],
        False,
        0,
        30,
        [],
        engine_history,
        engine,
    )

    assert chatbot == ui_history
    assert ui_history[-1]["content"] == "Recommendation failed: service unavailable"
    assert table.empty
    assert list(table.columns) == DISPLAY_COLUMNS
    assert retrieval_query == ""
    assert returned_history is engine_history
