from __future__ import annotations
import json
import ast
import logging
from typing import Any, MutableSequence, Sequence
from tqdm import tqdm

import pandas as pd
from dotenv import load_dotenv
from litellm import completion
from openai import OpenAI

from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue, Prefetch, Range, PointStruct, Distance, VectorParams

from database import QD_RECIPES_COLLECTION
from recipes_repository.recipes_repository import RecipesRepository
from recommender_engine.llm_output_models import RankAndJustifications
from recommender_engine.llm_utils import (
    ChatMessage,
    RecommendationResult,
    _append_interaction_to_history,
    _empty_recommendation_result,
    _extract_justifications,
    build_system_prompt,
    build_semantic_representation,
    clean_reply,
    enhance_query_with_filters,
)
from recommender_engine.prompts import RETRIEVAL_QUERY_SYSTEM_PROMPT



EMBEDDING_MODEL = "text-embedding-3-small"
REASONING_MODEL = "gpt-4.1-mini"
DEFAULT_RETRIEVAL_LIMIT = 20
DEFAULT_PREFETCH_LIMIT = 200
DEFAULT_SCORE_THRESHOLD = 0.3
QDRANT_BATCH_SIZE = 1_000
logger = logging.getLogger(__name__)
DIETARY_FILTER_FIELDS = frozenset(
    {
        "is_vegan",
        "is_vegetarian",
        "is_gluten_free",
        "is_halal",
        "is_kosher",
        "keto_friendliness",
    }
)


class RecommenderEngine:
    def __init__(
        self,
        recipes_repository: RecipesRepository,
        qdrant_client: QdrantClient,
        *,
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

        self.recipes_repository = recipes_repository
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
        dietary_filters: Sequence[str] | None = None,
        overall_time_range: tuple[float, float] | None = None,
        update_history: bool = True,
    ) -> tuple[pd.DataFrame, str]:
        result = self.recommend(
            question=question,
            chat_history=chat_history,
            dietary_filters=dietary_filters,
            overall_time_range=overall_time_range,
            update_history=update_history,
        )
        return result.recommendations, result.retrieval_query

    def recommend(
        self,
        question: str,
        chat_history: MutableSequence[ChatMessage] | None = None,
        *,
        dietary_filters: Sequence[str] | None = None,
        overall_time_range: tuple[float, float] | None = None,
        update_history: bool = True,
    ) -> RecommendationResult:
        history = chat_history if chat_history is not None else []
        question = enhance_query_with_filters(
            question,
            dietary_filters,
            overall_time_range,
        )
        retrieval_query = self.build_retrieval_query(question, history)
        context = self.fetch_context(
            retrieval_query,
            dietary_filters=dietary_filters,
            overall_time_range=overall_time_range,
        )
        if not context.strip():
            result = _empty_recommendation_result(retrieval_query)
            if update_history and chat_history is not None:
                _append_interaction_to_history(
                    question,
                    result.recommendations,
                    chat_history,
                )
            return result

        system_prompt = build_system_prompt(context)
        response_data = self._rank_recipes(question, history, system_prompt)

        output_df = self._build_output_dataframe(response_data)
        justifications = _extract_justifications(response_data)

        if update_history and chat_history is not None:
            _append_interaction_to_history(question, output_df, chat_history)

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

    def fetch_context(
        self,
        query: str,
        n_returned: int = 10,
        dietary_filters: Sequence[str] | None = None,
        overall_time_range: tuple[float, float] | None = None,
    ) -> str:
        recipes_df = self.vector_search(
            query,
            n_returned=n_returned,
            dietary_filters=dietary_filters,
            overall_time_range=overall_time_range,
        )
        if recipes_df.empty:
            return ""

        context_items = recipes_df.apply(build_semantic_representation, axis=1).tolist()
        return "".join(f"\n{recipe}\n" for recipe in context_items)

    def vector_search(
        self,
        query: str,
        n_returned: int = 10,
        dietary_filters: Sequence[str] | None = None,
        overall_time_range: tuple[float, float] | None = None,
    ) -> pd.DataFrame:
        vector = self._embed_query(query)
        query_filter = self._build_qdrant_filter(
            dietary_filters=dietary_filters,
            overall_time_range=overall_time_range,
        )
        print(query_filter)
        results = self.qdrant_client.query_points(
            collection_name=self.collection_name,
            prefetch=[
                Prefetch(
                    query=vector,
                    limit=self.prefetch_limit,
                )
            ],
            query=vector,
            query_filter=query_filter,
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

    def _build_qdrant_filter(
        self,
        dietary_filters: Sequence[str] | None,
        overall_time_range: tuple[float, float] | None,
    ) -> Filter | None:
        selected_filters = [field for field in dietary_filters or [] if field]
        invalid_filters = set(selected_filters) - DIETARY_FILTER_FIELDS
        if invalid_filters:
            invalid = ", ".join(sorted(invalid_filters))
            raise ValueError(f"Unsupported dietary filter field(s): {invalid}")

        must_conditions = [
            FieldCondition(
                key=field,
                match=MatchValue(value=1),
            )
            for field in selected_filters
        ]

        if overall_time_range is not None:
            min_time, max_time = self._normalize_overall_time_range(overall_time_range)
            must_conditions.append(
                FieldCondition(
                    key="overall_time",
                    range=Range(gte=min_time, lte=max_time),
                )
            )

        if not must_conditions:
            return None

        return Filter(must=must_conditions)

    def _normalize_overall_time_range(
        self,
        overall_time_range: tuple[float, float],
    ) -> tuple[float, float]:
        min_time, max_time = overall_time_range
        min_time = max(0.0, float(min_time))
        max_time = max(0.0, float(max_time))

        if min_time > max_time:
            min_time, max_time = max_time, min_time

        return min_time, max_time

    def clean_reply(self, recommendations: pd.DataFrame) -> str:
        return clean_reply(recommendations)

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
        if recipes_df.empty:
            return pd.DataFrame()

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

    def ensure_source_vector_collection_loaded(self) -> int:
        """Create and populate the source vector collection when it is missing.

        Returns the number of points loaded. If the collection already exists,
        returns 0.
        """
        if self.qdrant_client.collection_exists(QD_RECIPES_COLLECTION):
            logger.info("`%s` vector collection present.", QD_RECIPES_COLLECTION)
            return 0

        loaded_count = self._load_source_collection()
        logger.info(
            "Data loaded in `%s` vector collection.",
            QD_RECIPES_COLLECTION,
        )
        return loaded_count

    def _load_source_collection(self) -> int:
        def embed_batch(texts: list[str], batch_size: int = 500) -> list:
            """Embed a list of texts in batches, returns list of vectors."""
            all_embeddings = []

            for i in tqdm(range(0, len(texts), batch_size), desc="Embedding recipes"):
                batch = texts[i: i + batch_size]
                response = self.openai_client.embeddings.create(input=batch, model=self.embedding_model)
                all_embeddings.extend([item.embedding for item in response.data])

            return all_embeddings

        df = self.recipes_repository.get_recipes()
        df['embedding_text'] = df.apply(build_semantic_representation, add_id=False, axis=1)
        vectors = embed_batch(df['embedding_text'].tolist())
        df["payload"] = df.apply(
            lambda r: {
                "id": r["id"],
                #"ingredients": ast.literal_eval(r["ingredients"]),
                "category": ast.literal_eval(r["category"]),
                "cuisine": r["cuisine"],
                "overall_time": r["overall_time"],
                "is_vegan": r["is_vegan"],
                "is_vegetarian": r["is_vegetarian"],
                "is_gluten_free": r["is_gluten_free"],
                "is_halal": r["is_halal"],
                "is_kosher": r["is_kosher"],
                "keto_friendliness": r["keto_friendliness"],
            },
            axis=1
        )

        points = [
            PointStruct(
                id=int(df.iloc[i]["id"]),
                vector=vectors[i],
                payload=df.iloc[i]["payload"]
            )
            for i in range(len(df))
        ]

        if self.qdrant_client.collection_exists(QD_RECIPES_COLLECTION):
            self.qdrant_client.delete_collection(QD_RECIPES_COLLECTION)

        self.qdrant_client.create_collection(
            collection_name=QD_RECIPES_COLLECTION,
            vectors_config=VectorParams(
                size=1536,
                distance=Distance.COSINE
            )
        )

        for start in range(0, len(points), QDRANT_BATCH_SIZE):
            batch = points[start:start + QDRANT_BATCH_SIZE]

            self.qdrant_client.upsert(
                collection_name=QD_RECIPES_COLLECTION,
                points=batch
            )

        return len(points)
