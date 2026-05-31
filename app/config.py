from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_env: str = "dev"
    openai_api_key: str | None = None
    embedding_model: str = "all-MiniLM-L6-v2"
    chroma_persist_dir: str = "chroma_db"


settings = Settings()
