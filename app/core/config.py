from functools import lru_cache
from pydantic import BaseModel, Field
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

    # Static key: validation_alias prevents .env OPENAI_API_KEY from overriding. Replace below with your full key (51+ chars).
    openai_api_key: str | None = Field(
        default="",
        validation_alias="OPENAI_API_KEY_FROM_ENV",
    )
    openai_base_url: str = "https://api.openai.com/v1"
    default_embedding_model: str = "text-embedding-3-small"
    default_chat_model: str = "gpt-4.1-mini"
    vector_dimension: int = 384
    # Local embeddings (no API call): set use_local_embeddings=true and local_embedding_model (e.g. all-MiniLM-L6-v2).
    # Then set vector_dimension to match the model (384 for all-MiniLM-L6-v2) and run a migration if changing from 384.
    use_local_embeddings: bool = True
    local_embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    file_storage_path: str = "./storage"
    rag_background_ingest: bool = False


class EmbeddingModelInfo(BaseModel):
    name: str
    dimension: int


EMBEDDING_MODEL_DIMENSIONS = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "sentence-transformers/all-MiniLM-L6-v2": 384,
    "all-MiniLM-L6-v2": 384,
    "sentence-transformers/all-mpnet-base-v2": 768,
    "all-mpnet-base-v2": 768,
}


@lru_cache
def get_settings() -> Settings:
    return Settings()
