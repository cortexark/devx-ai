"""Configuration management for devx-ai.

Settings are loaded from environment variables (with DEVX_ prefix) and can be
overridden by a local ``devx.yaml`` file.  Pydantic Settings validates every
value at startup so misconfigurations surface immediately.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class LLMConfig(BaseSettings):
    """LLM provider configuration."""

    model_config = SettingsConfigDict(env_prefix="DEVX_LLM_")

    provider: str = Field(default="openai", description="openai | anthropic")
    model: str = Field(default="gpt-4o", description="Model identifier")
    api_key: str = Field(default="", description="API key (never hardcoded)")
    temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    max_tokens: int = Field(default=4096, ge=1)
    timeout_seconds: int = Field(default=60, ge=1)

    @field_validator("provider")
    @classmethod
    def validate_provider(cls, v: str) -> str:
        allowed = {"openai", "anthropic"}
        if v not in allowed:
            msg = f"provider must be one of {allowed}, got '{v}'"
            raise ValueError(msg)
        return v


class GitHubConfig(BaseSettings):
    """GitHub API configuration."""

    model_config = SettingsConfigDict(env_prefix="DEVX_GITHUB_")

    token: str = Field(default="", description="GitHub personal access token")
    base_url: str = Field(default="https://api.github.com")
    rate_limit_buffer: int = Field(
        default=100, ge=0, description="Stop requests when remaining < buffer"
    )


class MetricsConfig(BaseSettings):
    """Metrics collection configuration."""

    model_config = SettingsConfigDict(env_prefix="DEVX_METRICS_")

    window_days: int = Field(default=30, ge=1, description="Default DORA window")
    cache_ttl_seconds: int = Field(default=300, ge=0)


class Settings(BaseSettings):
    """Top-level application settings.

    Load order (later wins):
      1. Field defaults
      2. Environment variables (DEVX_ prefix)
      3. ``devx.yaml`` in working directory
    """

    model_config = SettingsConfigDict(env_prefix="DEVX_")

    llm: LLMConfig = Field(default_factory=LLMConfig)
    github: GitHubConfig = Field(default_factory=GitHubConfig)
    metrics: MetricsConfig = Field(default_factory=MetricsConfig)
    log_level: str = Field(default="INFO")
    debug: bool = Field(default=False)

    @classmethod
    def from_yaml(cls, path: str | Path = "devx.yaml") -> Settings:
        """Load settings from a YAML file, merged with env vars.

        Args:
            path: Path to the YAML configuration file.

        Returns:
            Fully validated Settings instance.
        """
        config_path = Path(path)
        overrides: dict[str, Any] = {}
        if config_path.exists():
            with config_path.open() as f:
                raw = yaml.safe_load(f)
                if isinstance(raw, dict):
                    overrides = raw
        return cls(**overrides)

    @classmethod
    def default(cls) -> Settings:
        """Return settings with all defaults (useful for testing)."""
        return cls()
