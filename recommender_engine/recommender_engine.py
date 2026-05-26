from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, MutableSequence, Sequence

import pandas as pd
from dotenv import load_dotenv
from litellm import completion
from openai import OpenAI

from qdrant_client import QdrantClient
from qdrant_client.models import Filter, Prefetch
from sqlalchemy.engine import Engine

from database import QD_RECIPES_COLLECTION, engine as default_sql_engine, qd_client
from recipes_repository.recipes_repository import RecipesRepository
from recommender_engine.llm_output_models import RankAndJustifications
from recommender_engine.llm_utils import build_semantic_representation
from recommender_engine.prompts import (
    RECOMMENDATION_SYSTEM_PROMPT_TEMPLATE,
    RETRIEVAL_QUERY_SYSTEM_PROMPT,
)


EMBEDDING_MODEL = "text-embedding-3-small"
REASONING_MODEL = "gpt-4.1-mini"
DEFAULT_RETRIEVAL_LIMIT = 20
DEFAULT_PREFETCH_LIMIT = 200
DEFAULT_SCORE_THRESHOLD = 0.3


ChatMessage = dict[str, str]


@dataclass(frozen=True)
class RecommendationResult:
    recommendations: pd.DataFrame
    retrieval_query: str
    justifications: dict[str, str]


class RecommenderEngine:
    def __init__(
        self,
        *,
        sql_engine: Engine = default_sql_engine,
        qdrant_client: QdrantClient = qd_client,
        openai_client: OpenAI | None = None,
        collection_name: str = QD_RECIPES_COLLECTION,
        embedding_model: str = EMBEDDING_MODEL,
        reasoning_model: str = REASONING_MODEL,
        retrieval_limit: int = DEFAULT_RETRIEVAL_LIMIT,
        prefetch_limit: int = DEFAULT_PREFETCH_LIMIT,
        score_threshold: float = DEFAULT_SCORE_THRESHOLD,
        load_environment: bool = True,
    ) -> None:
        if load_environment:
            load_dotenv(override=True)
            self.openai_client = OpenAI()
            print("Loaded env")
        else:
            print("ENV NOT LOADED")

        self.sql_engine = sql_engine
        self.recipes_repository = RecipesRepository(self.sql_engine)
        self.qdrant_client = qdrant_client
        self.openai_client = openai_client or OpenAI()
        self.collection_name = collection_name
        self.embedding_model = embedding_model
        self.reasoning_model = reasoning_model
        self.retrieval_limit = retrieval_limit
        self.prefetch_limit = prefetch_limit
        self.score_threshold = score_threshold

    def give_recommendations(
        self,
        question: str,
        chat_history: MutableSequence[ChatMessage] | None = None,
        *,
        update_history: bool = True,
    ) -> tuple[pd.DataFrame, str]:
        result = self.recommend(
            question=question,
            chat_history=chat_history,
            update_history=update_history,
        )
        return result.recommendations, result.retrieval_query

    def recommend(
        self,
        question: str,
        chat_history: MutableSequence[ChatMessage] | None = None,
        *,
        update_history: bool = True,
    ) -> RecommendationResult:
        history = chat_history if chat_history is not None else []
        retrieval_query = self.build_retrieval_query(question, history)
        context = self.fetch_context(retrieval_query)
        system_prompt = self._build_system_prompt(context)
        response_data = self._rank_recipes(question, history, system_prompt)

        output_df = self._build_output_dataframe(response_data)
        justifications = self._extract_justifications(response_data)

        if update_history and chat_history is not None:
            self._append_interaction_to_history(question, output_df, chat_history)

        return RecommendationResult(
            recommendations=output_df,
            retrieval_query=retrieval_query,
            justifications=justifications,
        )

    def build_retrieval_query(
        self,
        question: str,
        chat_history: Sequence[ChatMessage] | None = None,
    ) -> str:
        messages = [
            {"role": "system", "content": RETRIEVAL_QUERY_SYSTEM_PROMPT},
            *(chat_history or []),
            {"role": "user", "content": question},
        ]

        response = completion(
            model=self.reasoning_model,
            messages=messages,
            temperature=0.05,
            max_tokens=80,
        )

        return response.choices[0].message.content.strip()

    def fetch_context(self, query: str, n_returned: int = 10) -> str:
        recipes_df = self.vector_search(query, n_returned=n_returned)
        context_items = recipes_df.apply(build_semantic_representation, axis=1).tolist()
        return "".join(f"\n{recipe}\n" for recipe in context_items)

    def vector_search(self, query: str, n_returned: int = 10) -> pd.DataFrame:
        vector = self._embed_query(query)
        results = self.qdrant_client.query_points(
            collection_name=self.collection_name,
            prefetch=[
                Prefetch(
                    query=vector,
                    limit=self.prefetch_limit,
                )
            ],
            query=vector,
            query_filter=Filter(should=[]),
            limit=self.retrieval_limit,
            score_threshold=self.score_threshold,
            with_payload=True,
        )

        hits = {
            hit.payload["id"]: hit.score
            for hit in results.points
            if hit.payload and "id" in hit.payload
        }
        if not hits:
            return pd.DataFrame()

        recipes_df = self.recipes_repository.get_relevant_recipes_by_ids(hits.keys())#self._fetch_recipes_by_ids(hits.keys())
        recipes_df["score"] = recipes_df["id"].map(hits)
        return recipes_df.sort_values(by="score", ascending=False).head(n_returned)

    def clean_reply(self, recommendations: pd.DataFrame) -> str:
        reply = (
            "Recipe recommendations were provided based on previous preferences. "
            "Here are the top 3 picks: \n"
        )
        top_3 = "\n\n".join(
            recommendations.head(3).apply(
                build_semantic_representation,
                add_id=False,
                axis=1,
            ).tolist()
        )
        return reply + top_3

    def _embed_query(self, query: str) -> list[float]:
        query_vec = self.openai_client.embeddings.create(
            model=self.embedding_model,
            input=query,
        )
        return query_vec.data[0].embedding

    def _rank_recipes(
        self,
        question: str,
        chat_history: Sequence[ChatMessage],
        system_prompt: str,
    ) -> dict[str, Any]:
        response = completion(
            model=self.reasoning_model,
            messages=[
                {"role": "system", "content": system_prompt},
                *chat_history,
                {"role": "user", "content": question},
            ],
            response_format=RankAndJustifications,
        )
        return json.loads(response.choices[0].message.content)

    def _build_output_dataframe(self, response_data: dict[str, Any]) -> pd.DataFrame:
        ranked_recipes = response_data["ranked_recipes"]
        ids = [recipe["recipe_id"] for recipe in ranked_recipes]
        recipes_df = self.recipes_repository.get_relevant_recipes_by_ids(ids)#self._fetch_recipes_by_ids(ids)
        scores_df = pd.DataFrame(ranked_recipes)

        return (
            recipes_df.merge(
                scores_df,
                left_on="id",
                right_on="recipe_id",
                how="left",
            )
            .drop(columns="recipe_id")
            .sort_values(by="new_score", ascending=False)
            .reset_index(drop=True)
        )

    # def _fetch_recipes_by_ids(self, recipe_ids: Sequence[int]) -> pd.DataFrame:
    #     return self.recipes_repository.get_relevant_recipes_by_ids(recipe_ids)

    def _append_interaction_to_history(
        self,
        question: str,
        recommendations: pd.DataFrame,
        chat_history: MutableSequence[ChatMessage],
    ) -> None:
        chat_history.append({"role": "user", "content": question})
        chat_history.append(
            {"role": "assistant", "content": self.clean_reply(recommendations)}
        )

    def _build_system_prompt(self, context: str) -> str:
        return RECOMMENDATION_SYSTEM_PROMPT_TEMPLATE.format(context=context)

    def _extract_justifications(self, response_data: dict[str, Any]) -> dict[str, str]:
        return {
            "first_place": response_data["first_place_justification"],
            "second_place": response_data["second_place_justification"],
            "third_place": response_data["third_place_justification"],
        }
