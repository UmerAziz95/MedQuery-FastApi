from pydantic import BaseModel, Field


class WorkspaceConfigUpdate(BaseModel):
    chunk_words: int = Field(300, example=300)
    overlap_words: int = Field(50, example=50)
    top_k: int = Field(5, example=5)
    similarity_threshold: float = Field(0.2, example=0.2)
    max_context_chars: int = Field(12000, example=12000)
    embedding_model: str = Field("text-embedding-3-small", example="text-embedding-3-small")
    chat_model_default: str = Field("gpt-4.1-mini", example="gpt-4.1-mini")
    chat_temperature_default: float = Field(0.2, example=0.2)
    chat_max_tokens_default: int = Field(600, example=600)


class WorkspaceConfigOut(WorkspaceConfigUpdate):
    class Config:
        from_attributes = True
