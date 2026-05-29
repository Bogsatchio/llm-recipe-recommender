import logging
import time

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import sessionmaker
from qdrant_client import QdrantClient

DATABASE_URL = "mysql+pymysql://dev:dev@localhost:3306/recipe_app"
DATABASE_CONNECT_RETRIES = 10
DATABASE_CONNECT_RETRY_DELAY_SECONDS = 3

logger = logging.getLogger(__name__)

# setup qdrant client connection
QD_RECIPES_COLLECTION = "recipes"
qd_client = QdrantClient(host="localhost", port=6333)


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

