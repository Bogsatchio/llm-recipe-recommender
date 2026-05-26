from __future__ import annotations

import gradio as gr
import pandas as pd

from recommender_engine import RecommendationResult, RecommenderEngine


DISPLAY_COLUMNS = ["name", "rating_value", "ingredients", "preparation_time"]


def _format_table(recommendations: pd.DataFrame) -> pd.DataFrame:
    if recommendations.empty:
        return pd.DataFrame(columns=DISPLAY_COLUMNS)

    table = recommendations.head(5).copy()
    existing_columns = [column for column in DISPLAY_COLUMNS if column in table.columns]
    return table[existing_columns]


def _format_justifications(result: RecommendationResult) -> str:
    justifications = result.justifications
    return "\n\n".join(
        [
            f"**1. Best match**\n{justifications['first_place']}",
            f"**2. Second best**\n{justifications['second_place']}",
            f"**3. Third best**\n{justifications['third_place']}",
        ]
    )


def _respond(
    message: str,
    ui_history: list[dict[str, str]],
    engine_history: list[dict[str, str]],
    engine: RecommenderEngine,
) -> tuple[
    str,
    list[dict[str, str]],
    list[dict[str, str]],
    pd.DataFrame,
    str,
    list[dict[str, str]],
]:
    if not message.strip():
        return (
            "",
            ui_history,
            ui_history,
            pd.DataFrame(columns=DISPLAY_COLUMNS),
            "",
            engine_history,
        )

    try:
        result = engine.recommend(message, engine_history)
    except Exception as exc:
        ui_history = [
            *ui_history,
            {"role": "user", "content": message},
            {"role": "assistant", "content": f"Recommendation failed: {exc}"},
        ]
        return (
            "",
            ui_history,
            ui_history,
            pd.DataFrame(columns=DISPLAY_COLUMNS),
            "",
            engine_history,
        )

    table = _format_table(result.recommendations)
    justifications = _format_justifications(result)

    reply = (
        f"Found {len(result.recommendations)} recommendations.\n\n"
        f"Retrieval query: `{result.retrieval_query}`"
    )
    ui_history = [
        *ui_history,
        {"role": "user", "content": message},
        {"role": "assistant", "content": reply},
    ]

    return "", ui_history, ui_history, table, justifications, engine_history


def build_app() -> gr.Blocks:
    engine = RecommenderEngine()

    with gr.Blocks(title="Recipe Recommender") as app:
        gr.Markdown("# Recipe Recommender")

        ui_history = gr.State([])
        engine_history = gr.State([])

        with gr.Row(equal_height=True):
            with gr.Column(scale=1):
                chatbot = gr.Chatbot(
                    label="Chat",
                    type="messages",
                    height=560,
                )
                message = gr.Textbox(
                    label="Ask for recipes",
                    placeholder="Something spicy and with noodles",
                    lines=2,
                )
                submit = gr.Button("Send", variant="primary")

            with gr.Column(scale=1):
                recommendations = gr.Dataframe(
                    headers=DISPLAY_COLUMNS,
                    label="Top 5 Recommendations",
                    interactive=False,
                    wrap=True,
                )
                justifications = gr.Markdown(label="Top 3 Justifications")

        submit.click(
            fn=lambda text, chat, history: _respond(text, chat, history, engine),
            inputs=[message, ui_history, engine_history],
            outputs=[
                message,
                chatbot,
                ui_history,
                recommendations,
                justifications,
                engine_history,
            ],
        )

        message.submit(
            fn=lambda text, chat, history: _respond(text, chat, history, engine),
            inputs=[message, ui_history, engine_history],
            outputs=[
                message,
                chatbot,
                ui_history,
                recommendations,
                justifications,
                engine_history,
            ],
        )

    return app


if __name__ == "__main__":
    build_app().launch()
