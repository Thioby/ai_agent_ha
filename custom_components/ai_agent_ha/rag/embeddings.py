"""Embedding providers for RAG system.

This module provides embedding generation using available AI providers.
Supports Gemini OAuth (preferred), OpenAI, and Gemini API key as fallbacks.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import aiohttp

if TYPE_CHECKING:
    from homeassistant.config_entries import ConfigEntry
    from homeassistant.core import HomeAssistant

_LOGGER = logging.getLogger(__name__)

# Embedding dimensions for different providers
EMBEDDING_DIMENSIONS = {
    "gemini": 768,  # text-embedding-004
    "openai": 1536,  # text-embedding-3-small
}


class EmbeddingError(Exception):
    """Raised when embedding generation fails."""


@dataclass
class EmbeddingProvider(ABC):
    """Abstract base class for embedding providers."""

    @abstractmethod
    async def get_embeddings(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a list of texts.

        Args:
            texts: List of text strings to embed.

        Returns:
            List of embedding vectors.

        Raises:
            EmbeddingError: If embedding generation fails.
        """

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return the dimension of the embeddings."""

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return the name of the embedding provider."""


@dataclass
class GeminiOAuthEmbeddings(EmbeddingProvider):
    """Gemini embeddings using existing OAuth token from Gemini CLI.

    Uses the same OAuth authentication as the main GeminiOAuthClient.
    """

    hass: HomeAssistant
    config_entry: ConfigEntry
    model: str = "text-embedding-004"
    _session: aiohttp.ClientSession | None = field(default=None, repr=False)

    ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models/{model}:embedContent"

    @property
    def dimension(self) -> int:
        return 768

    @property
    def provider_name(self) -> str:
        return "gemini_oauth"

    async def _get_valid_token(self) -> str:
        """Get a valid OAuth access token, refreshing if necessary."""
        import time
        from ..gemini_oauth import refresh_token

        # OAuth tokens are stored in the nested "gemini_oauth" dict
        oauth_data = dict(self.config_entry.data.get("gemini_oauth", {}))

        # Check if token is expired or about to expire (within 5 minutes)
        expires_at = oauth_data.get("expires_at", 0)
        if time.time() >= expires_at - 300:
            _LOGGER.debug("Gemini OAuth token expired or expiring, refreshing...")
            refresh = oauth_data.get("refresh_token")
            if not refresh:
                raise EmbeddingError("No refresh token available for Gemini OAuth")

            async with aiohttp.ClientSession() as session:
                new_tokens = await refresh_token(session, refresh)

            # Update oauth_data and persist to config entry
            oauth_data.update(new_tokens)
            self.hass.config_entries.async_update_entry(
                self.config_entry,
                data={**self.config_entry.data, "gemini_oauth": oauth_data},
            )
            return new_tokens["access_token"]

        return oauth_data["access_token"]

    async def get_embeddings(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings using Gemini OAuth."""
        if not texts:
            return []

        try:
            token = await self._get_valid_token()
            embeddings = []

            async with aiohttp.ClientSession() as session:
                for text in texts:
                    url = self.ENDPOINT.format(model=self.model)
                    headers = {
                        "Authorization": f"Bearer {token}",
                        "Content-Type": "application/json",
                    }
                    payload = {
                        "content": {"parts": [{"text": text}]},
                        "taskType": "RETRIEVAL_DOCUMENT",
                    }

                    async with session.post(
                        url, headers=headers, json=payload
                    ) as resp:
                        if resp.status != 200:
                            error_text = await resp.text()
                            _LOGGER.error(
                                "Gemini embedding failed: %s - %s",
                                resp.status,
                                error_text,
                            )
                            raise EmbeddingError(
                                f"Gemini embedding failed: {resp.status}"
                            )

                        data = await resp.json()
                        embedding = data.get("embedding", {}).get("values", [])
                        if not embedding:
                            raise EmbeddingError("No embedding in Gemini response")
                        embeddings.append(embedding)

            _LOGGER.debug(
                "Generated %d embeddings with Gemini OAuth", len(embeddings)
            )
            return embeddings

        except EmbeddingError:
            raise
        except Exception as e:
            _LOGGER.error("Gemini OAuth embedding error: %s", e)
            raise EmbeddingError(f"Gemini OAuth embedding failed: {e}") from e


@dataclass
class GeminiApiKeyEmbeddings(EmbeddingProvider):
    """Gemini embeddings using API key."""

    api_key: str
    model: str = "text-embedding-004"

    ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models/{model}:embedContent"

    @property
    def dimension(self) -> int:
        return 768

    @property
    def provider_name(self) -> str:
        return "gemini"

    async def get_embeddings(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings using Gemini API key."""
        if not texts:
            return []

        try:
            embeddings = []

            async with aiohttp.ClientSession() as session:
                for text in texts:
                    url = f"{self.ENDPOINT.format(model=self.model)}?key={self.api_key}"
                    headers = {"Content-Type": "application/json"}
                    payload = {
                        "content": {"parts": [{"text": text}]},
                        "taskType": "RETRIEVAL_DOCUMENT",
                    }

                    async with session.post(
                        url, headers=headers, json=payload
                    ) as resp:
                        if resp.status != 200:
                            error_text = await resp.text()
                            _LOGGER.error(
                                "Gemini API embedding failed: %s - %s",
                                resp.status,
                                error_text,
                            )
                            raise EmbeddingError(
                                f"Gemini API embedding failed: {resp.status}"
                            )

                        data = await resp.json()
                        embedding = data.get("embedding", {}).get("values", [])
                        if not embedding:
                            raise EmbeddingError("No embedding in Gemini response")
                        embeddings.append(embedding)

            _LOGGER.debug(
                "Generated %d embeddings with Gemini API key", len(embeddings)
            )
            return embeddings

        except EmbeddingError:
            raise
        except Exception as e:
            _LOGGER.error("Gemini API embedding error: %s", e)
            raise EmbeddingError(f"Gemini API embedding failed: {e}") from e


@dataclass
class OpenAIEmbeddings(EmbeddingProvider):
    """OpenAI embeddings using text-embedding-3-small."""

    api_key: str
    model: str = "text-embedding-3-small"

    ENDPOINT = "https://api.openai.com/v1/embeddings"

    @property
    def dimension(self) -> int:
        return 1536

    @property
    def provider_name(self) -> str:
        return "openai"

    async def get_embeddings(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings using OpenAI API."""
        if not texts:
            return []

        try:
            async with aiohttp.ClientSession() as session:
                headers = {
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                }
                payload = {
                    "model": self.model,
                    "input": texts,
                }

                async with session.post(
                    self.ENDPOINT, headers=headers, json=payload
                ) as resp:
                    if resp.status != 200:
                        error_text = await resp.text()
                        _LOGGER.error(
                            "OpenAI embedding failed: %s - %s",
                            resp.status,
                            error_text,
                        )
                        raise EmbeddingError(
                            f"OpenAI embedding failed: {resp.status}"
                        )

                    data = await resp.json()
                    embeddings_data = data.get("data", [])
                    if not embeddings_data:
                        raise EmbeddingError("No embeddings in OpenAI response")

                    # Sort by index and extract embeddings
                    embeddings_data.sort(key=lambda x: x.get("index", 0))
                    embeddings = [e["embedding"] for e in embeddings_data]

            _LOGGER.debug("Generated %d embeddings with OpenAI", len(embeddings))
            return embeddings

        except EmbeddingError:
            raise
        except Exception as e:
            _LOGGER.error("OpenAI embedding error: %s", e)
            raise EmbeddingError(f"OpenAI embedding failed: {e}") from e


def create_embedding_provider(
    hass: HomeAssistant,
    config: dict[str, Any],
    config_entry: ConfigEntry | None = None,
) -> EmbeddingProvider:
    """Create an embedding provider based on available configuration.

    Fallback chain:
    1. OpenAI (if configured) - text-embedding-3-small (most reliable)
    2. Gemini API key (if configured) - text-embedding-004

    Note: Gemini OAuth tokens from Code Assist do NOT have scopes for
    the embeddings API (generativelanguage.googleapis.com), so we skip it.

    Args:
        hass: Home Assistant instance.
        config: Configuration dictionary with API keys.
        config_entry: Optional config entry for OAuth providers.

    Returns:
        Configured EmbeddingProvider instance.

    Raises:
        EmbeddingError: If no embedding provider could be configured.
    """
    # 1. Try OpenAI API key (most reliable for embeddings)
    openai_token = config.get("openai_token")
    if openai_token and openai_token.startswith("sk-"):
        _LOGGER.info("Using OpenAI for embeddings")
        return OpenAIEmbeddings(api_key=openai_token)

    # 2. Try Gemini API key
    gemini_token = config.get("gemini_token")
    if gemini_token:
        _LOGGER.info("Using Gemini API key for embeddings")
        return GeminiApiKeyEmbeddings(api_key=gemini_token)

    raise EmbeddingError(
        "No embedding provider available. Configure OpenAI API key or Gemini API key. "
        "Note: Gemini OAuth (Code Assist) tokens don't support embeddings API."
    )


async def get_embedding_for_query(
    provider: EmbeddingProvider,
    query: str,
) -> list[float]:
    """Get embedding for a single query text.

    This is a convenience function for search queries where
    we need a single embedding.

    Args:
        provider: The embedding provider to use.
        query: The query text to embed.

    Returns:
        The embedding vector for the query.
    """
    embeddings = await provider.get_embeddings([query])
    if not embeddings:
        raise EmbeddingError("Failed to generate query embedding")
    return embeddings[0]
