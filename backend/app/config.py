"""Application configuration and settings management."""

from functools import lru_cache
from typing import List

from pydantic import AnyUrl, BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class RedisDsn(AnyUrl):
    allowed_schemes = {"redis", "rediss"}


class Settings(BaseSettings):
    """Runtime configuration sourced from environment variables."""

    # Azure OpenAI Chat configuration (AI Foundry format)
    azure_openai_endpoint: str = Field(
        default="https://dehub.services.ai.azure.com",
        alias="AZURE_OPENAI_ENDPOINT",
        description="Azure OpenAI endpoint base URL for chat completions",
    )
    azure_openai_api_key: str = Field(..., alias="AZURE_OPENAI_API_KEY")
    azure_openai_deployment_name: str = Field(
        default="gpt-4o-mini",
        alias="AZURE_OPENAI_DEPLOYMENT_NAME",
        description="Azure OpenAI deployment name for chat completions",
    )
    azure_openai_api_version: str = Field(
        default="2024-05-01-preview",
        alias="AZURE_OPENAI_API_VERSION",
        description="API version for chat completions",
    )

    # Azure OpenAI Embeddings configuration (Cognitive Services format)
    azure_openai_embedding_endpoint: str = Field(
        default="https://dehub.cognitiveservices.azure.com",
        alias="AZURE_OPENAI_EMBEDDING_ENDPOINT",
        description="Azure OpenAI endpoint base URL for embeddings",
    )
    azure_openai_embedding_deployment: str = Field(
        default="text-embedding-ada-002",
        alias="AZURE_OPENAI_EMBEDDING_DEPLOYMENT",
        description="Azure OpenAI deployment name for embeddings",
    )
    azure_openai_embedding_api_version: str = Field(
        default="2023-05-15",
        alias="AZURE_OPENAI_EMBEDDING_API_VERSION",
        description="API version for embeddings",
    )

    # Azure FLUX Image Generation configuration
    azure_flux_deployment_name: str = Field(
        default="FLUX-1.1-pro",
        alias="AZURE_FLUX_DEPLOYMENT_NAME",
        description="Azure FLUX deployment name for image generation",
    )
    azure_flux_api_version: str = Field(
        default="2025-04-01-preview",
        alias="AZURE_FLUX_API_VERSION",
        description="API version for FLUX image generation",
    )

    # Azure Speech Services configuration
    azure_speech_key: str | None = Field(default=None, alias="AZURE_SPEECH_KEY")
    azure_speech_region: str = Field(
        default="eastus",
        alias="AZURE_SPEECH_REGION",
        description="Azure Speech Services region",
    )
    azure_speech_stt_endpoint: str = Field(
        default="https://eastus.stt.speech.microsoft.com",
        alias="AZURE_SPEECH_STT_ENDPOINT",
        description="Azure Speech Services STT endpoint",
    )
    azure_speech_tts_endpoint: str = Field(
        default="https://eastus.tts.speech.microsoft.com",
        alias="AZURE_SPEECH_TTS_ENDPOINT",
        description="Azure Speech Services TTS endpoint",
    )

    # TTS Configuration
    tts_enabled_by_default: bool = Field(
        default=False,
        alias="TTS_ENABLED_BY_DEFAULT",
        description="Enable TTS by default for all agent responses",
    )
    tts_voice: str = Field(
        default="en-US-JennyNeural",
        alias="TTS_VOICE",
        description="Default voice for text-to-speech synthesis",
    )
    tts_language: str = Field(
        default="en-US",
        alias="TTS_LANGUAGE",
        description="Default language for text-to-speech synthesis",
    )

    redis_url: RedisDsn = Field(default="redis://localhost:6379/0", alias="REDIS_URL")
    chroma_persist_path: str = Field(default="./storage/chroma", alias="CHROMA_PERSIST_PATH")
    allow_origins_str: str = Field(
        default="http://localhost:3000",
        alias="ALLOW_ORIGINS",
        description="Comma-separated list of allowed CORS origins (comma-separated string)",
    )

    @property
    def allow_origins(self) -> List[str]:
        """Parse comma-separated string into list of origins."""
        if not self.allow_origins_str:
            return ["http://localhost:3000"]
        origins = [origin.strip() for origin in self.allow_origins_str.split(",") if origin.strip()]
        return origins if origins else ["http://localhost:3000"]

    langfuse_public_key: str | None = Field(default=None, alias="LANGFUSE_PUBLIC_KEY")
    langfuse_secret_key: str | None = Field(default=None, alias="LANGFUSE_SECRET_KEY")
    langfuse_base_url: str | None = Field(default=None, alias="LANGFUSE_BASE_URL")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="allow",
        env_parse_none_str="",
        env_ignore_empty=True,
    )


class AppContext(BaseModel):
    """Application-level context shared across routers and agent graph."""

    settings: Settings
    agent_graph: object | None = None  # Will be set during startup


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached settings instance."""

    return Settings()

