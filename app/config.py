from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_env: str = "dev"
    embedding_model: str = "all-MiniLM-L6-v2"
    chroma_persist_dir: str = "chroma_db"
    deepseek_api_key: str | None = None
    llm_model: str = "deepseek-chat"
    top_k: int = 5
    whisper_model: str = "base"
    whisper_device: str = "cpu"
    whisper_compute_type: str = "int8"


settings = Settings()
