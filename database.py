import logging
import os
import time

from qdrant_client import QdrantClient
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "mysql+pymysql://dev:dev@localhost:3306/recipe_app",
)
DATABASE_CONNECT_RETRIES = int(os.getenv("DATABASE_CONNECT_RETRIES", "10"))
DATABASE_CONNECT_RETRY_DELAY_SECONDS = int(
    os.getenv("DATABASE_CONNECT_RETRY_DELAY_SECONDS", "3")
)
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))

logger = logging.getLogger(__name__)

# setup qdrant client connection
QD_RECIPES_COLLECTION = "recipes"


def create_vector_database_client_with_retry() -> QdrantClient:
    last_error: Exception | None = None

    for attempt in range(1, DATABASE_CONNECT_RETRIES + 1):
        client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

        try:
            client.get_collections()
            logger.info("Vector database connection established.")
            return client
        except Exception as exc:
            last_error = exc
            client.close()

            if attempt == DATABASE_CONNECT_RETRIES:
                break

            logger.warning(
                "Vector database connection failed on attempt %s/%s. "
                "Retrying in %s seconds.",
                attempt,
                DATABASE_CONNECT_RETRIES,
                DATABASE_CONNECT_RETRY_DELAY_SECONDS,
            )
            time.sleep(DATABASE_CONNECT_RETRY_DELAY_SECONDS)

    raise RuntimeError(
        "Could not connect to the vector database after "
        f"{DATABASE_CONNECT_RETRIES} attempts."
    ) from last_error


def create_database_engine_with_retry() -> Engine:
    last_error: SQLAlchemyError | None = None

    for attempt in range(1, DATABASE_CONNECT_RETRIES + 1):
        engine = create_engine(
            DATABASE_URL,
            pool_size=20,
            max_overflow=20,
            pool_pre_ping=True,
        )

        try:
            with engine.connect() as connection:
                connection.execute(text("SELECT 1"))
            logger.info("Database connection established.")
            return engine
        except SQLAlchemyError as exc:
            last_error = exc
            engine.dispose()

            if attempt == DATABASE_CONNECT_RETRIES:
                break

            logger.warning(
                "Database connection failed on attempt %s/%s. Retrying in %s seconds.",
                attempt,
                DATABASE_CONNECT_RETRIES,
                DATABASE_CONNECT_RETRY_DELAY_SECONDS,
            )
            time.sleep(DATABASE_CONNECT_RETRY_DELAY_SECONDS)

    raise RuntimeError(
        "Could not connect to the database after "
        f"{DATABASE_CONNECT_RETRIES} attempts."
    ) from last_error
