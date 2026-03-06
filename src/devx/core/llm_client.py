"""Unified LLM client supporting OpenAI and Anthropic.

The client abstracts provider differences behind a single ``complete()``
interface so callers never deal with provider-specific SDKs directly.
Retry logic uses *tenacity* with exponential backoff.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from devx.core.config import LLMConfig

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LLMResponse:
    """Normalized response from any LLM provider."""

    content: str
    model: str
    usage: dict[str, int] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def input_tokens(self) -> int:
        return self.usage.get("input_tokens", self.usage.get("prompt_tokens", 0))

    @property
    def output_tokens(self) -> int:
        return self.usage.get("output_tokens", self.usage.get("completion_tokens", 0))


class LLMClient:
    """Provider-agnostic LLM client.

    Supports OpenAI and Anthropic through their respective Python SDKs.
    Provider is selected via ``LLMConfig.provider``.

    Example::

        config = LLMConfig(provider="openai", api_key="sk-...")
        client = LLMClient(config)
        response = await client.complete("Explain this code", system="You are a reviewer.")
    """

    def __init__(self, config: LLMConfig | None = None) -> None:
        self._config = config or LLMConfig()
        self._openai_client: Any = None
        self._anthropic_client: Any = None

    def _get_openai(self) -> Any:
        """Lazily initialize the OpenAI async client."""
        if self._openai_client is None:
            from openai import AsyncOpenAI

            self._openai_client = AsyncOpenAI(
                api_key=self._config.api_key or None,
                timeout=float(self._config.timeout_seconds),
            )
        return self._openai_client

    def _get_anthropic(self) -> Any:
        """Lazily initialize the Anthropic async client."""
        if self._anthropic_client is None:
            from anthropic import AsyncAnthropic

            self._anthropic_client = AsyncAnthropic(
                api_key=self._config.api_key or None,
                timeout=float(self._config.timeout_seconds),
            )
        return self._anthropic_client

    @retry(
        retry=retry_if_exception_type(Exception),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )
    async def complete(
        self,
        prompt: str,
        *,
        system: str = "",
        temperature: float | None = None,
        max_tokens: int | None = None,
        model: str | None = None,
    ) -> LLMResponse:
        """Send a completion request to the configured LLM provider.

        Args:
            prompt: User message / prompt text.
            system: Optional system message.
            temperature: Override default temperature.
            max_tokens: Override default max tokens.
            model: Override default model identifier.

        Returns:
            Normalized LLMResponse.

        Raises:
            ValueError: If provider is not supported.
        """
        temp = temperature if temperature is not None else self._config.temperature
        tokens = max_tokens or self._config.max_tokens
        mdl = model or self._config.model

        if self._config.provider == "openai":
            return await self._complete_openai(
                prompt,
                system=system,
                temperature=temp,
                max_tokens=tokens,
                model=mdl,
            )
        if self._config.provider == "anthropic":
            return await self._complete_anthropic(
                prompt,
                system=system,
                temperature=temp,
                max_tokens=tokens,
                model=mdl,
            )

        msg = f"Unsupported LLM provider: {self._config.provider}"
        raise ValueError(msg)

    async def _complete_openai(
        self,
        prompt: str,
        *,
        system: str,
        temperature: float,
        max_tokens: int,
        model: str,
    ) -> LLMResponse:
        """Send request via OpenAI SDK."""
        client = self._get_openai()
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        response = await client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        choice = response.choices[0]
        usage = {}
        if response.usage:
            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
            }

        return LLMResponse(
            content=choice.message.content or "",
            model=response.model,
            usage=usage,
            raw=response.model_dump() if hasattr(response, "model_dump") else {},
        )

    async def _complete_anthropic(
        self,
        prompt: str,
        *,
        system: str,
        temperature: float,
        max_tokens: int,
        model: str,
    ) -> LLMResponse:
        """Send request via Anthropic SDK."""
        client = self._get_anthropic()
        kwargs: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            kwargs["system"] = system

        response = await client.messages.create(**kwargs)
        content = ""
        for block in response.content:
            if hasattr(block, "text"):
                content += block.text

        usage = {
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
        }

        return LLMResponse(
            content=content,
            model=response.model,
            usage=usage,
            raw=response.model_dump() if hasattr(response, "model_dump") else {},
        )
