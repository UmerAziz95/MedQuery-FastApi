from functools import lru_cache
from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    app_name: str = "MedQuery RAG API"
    environment: str = "development"
    api_v1_prefix: str = "/api"

    database_url: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/medquery"

    jwt_secret_key: str = "change-me"
    jwt_algorithm: str = "HS256"
    jwt_access_token_expire_minutes: int = 60

    openai_api_key: str | None = None
    openai_base_url: str = "https://api.openai.com/v1"
    default_embedding_model: str = "text-embedding-3-small"
    default_chat_model: str = "gpt-4.1-mini"
    vector_dimension: int = 1536

    file_storage_path: str = "./storage"
    rag_background_ingest: bool = False


class EmbeddingModelInfo(BaseModel):
    name: str
    dimension: int


EMBEDDING_MODEL_DIMENSIONS = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
}


@lru_cache
def get_settings() -> Settings:
    return Settings()
