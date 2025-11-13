"""Factory functions for creating Azure OpenAI clients."""

from __future__ import annotations

from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings

from app.config import Settings


def create_azure_chat_llm(settings: Settings, model: str | None = None, temperature: float = 0.0) -> AzureChatOpenAI:
    """Create an Azure OpenAI chat LLM client."""
    return AzureChatOpenAI(
        azure_endpoint=settings.azure_openai_endpoint,
        azure_deployment=model or settings.azure_openai_deployment_name,
        api_key=settings.azure_openai_api_key,
        api_version=settings.azure_openai_api_version,
        temperature=temperature,
    )


def create_azure_embeddings(settings: Settings, model: str | None = None) -> AzureOpenAIEmbeddings:
    """Create an Azure OpenAI embeddings client with separate endpoint configuration."""
    return AzureOpenAIEmbeddings(
        azure_endpoint=settings.azure_openai_embedding_endpoint,
        azure_deployment=model or settings.azure_openai_embedding_deployment,
        api_key=settings.azure_openai_api_key,
        api_version=settings.azure_openai_embedding_api_version,
    )

