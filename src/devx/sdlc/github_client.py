"""GitHub API wrapper with rate limiting and retry logic.

Wraps the GitHub REST API using ``httpx`` for async HTTP.  Includes
automatic rate-limit awareness: the client checks ``X-RateLimit-Remaining``
headers and pauses before hitting the limit.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any

import httpx

from devx.core.config import GitHubConfig

logger = logging.getLogger(__name__)


@dataclass
class RateLimitInfo:
    """Current GitHub API rate limit state."""

    limit: int
    remaining: int
    reset_timestamp: int
    used: int


class GitHubClient:
    """Async GitHub REST API client with rate-limit awareness.

    Example::

        config = GitHubConfig(token="ghp_...")
        async with GitHubClient(config) as gh:
            pr = await gh.get_pull_request("owner/repo", 42)
            diff = await gh.get_pull_request_diff("owner/repo", 42)
    """

    def __init__(self, config: GitHubConfig | None = None) -> None:
        self._config = config or GitHubConfig()
        self._client: httpx.AsyncClient | None = None

    async def __aenter__(self) -> GitHubClient:
        self._client = httpx.AsyncClient(
            base_url=self._config.base_url,
            headers=self._default_headers(),
            timeout=30.0,
        )
        return self

    async def __aexit__(self, *args: Any) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None

    def _default_headers(self) -> dict[str, str]:
        """Build default request headers."""
        headers = {
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        if self._config.token:
            headers["Authorization"] = f"Bearer {self._config.token}"
        return headers

    def _ensure_client(self) -> httpx.AsyncClient:
        """Ensure HTTP client is initialized."""
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self._config.base_url,
                headers=self._default_headers(),
                timeout=30.0,
            )
        return self._client

    async def _request(
        self,
        method: str,
        path: str,
        *,
        json: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> httpx.Response:
        """Execute an API request with rate-limit checking.

        Raises:
            httpx.HTTPStatusError: On 4xx/5xx responses.
            RuntimeError: If rate limit is exhausted.
        """
        client = self._ensure_client()

        response = await client.request(
            method, path, json=json, params=params, headers=headers or {}
        )

        # Check rate limit
        rate_info = self._parse_rate_limit(response)
        if rate_info and rate_info.remaining < self._config.rate_limit_buffer:
            wait_seconds = max(rate_info.reset_timestamp - self._current_timestamp(), 1)
            logger.warning(
                "Rate limit low (%d remaining). Waiting %ds.",
                rate_info.remaining,
                wait_seconds,
            )
            await asyncio.sleep(min(wait_seconds, 60))  # Cap wait at 60s

        response.raise_for_status()
        return response

    # -----------------------------------------------------------------------
    # Pull Requests
    # -----------------------------------------------------------------------

    async def get_pull_request(self, repo: str, pr_number: int) -> dict[str, Any]:
        """Fetch pull request metadata.

        Args:
            repo: Repository in ``owner/repo`` format.
            pr_number: PR number.

        Returns:
            PR data as a dictionary.
        """
        response = await self._request("GET", f"/repos/{repo}/pulls/{pr_number}")
        return response.json()  # type: ignore[no-any-return]

    async def get_pull_request_diff(self, repo: str, pr_number: int) -> str:
        """Fetch the raw diff for a pull request.

        Args:
            repo: Repository in ``owner/repo`` format.
            pr_number: PR number.

        Returns:
            Unified diff as a string.
        """
        response = await self._request(
            "GET",
            f"/repos/{repo}/pulls/{pr_number}",
            headers={"Accept": "application/vnd.github.diff"},
        )
        return response.text

    async def get_pull_request_files(
        self, repo: str, pr_number: int
    ) -> list[dict[str, Any]]:
        """List files changed in a pull request.

        Args:
            repo: Repository in ``owner/repo`` format.
            pr_number: PR number.

        Returns:
            List of file change objects.
        """
        response = await self._request(
            "GET", f"/repos/{repo}/pulls/{pr_number}/files"
        )
        return response.json()  # type: ignore[no-any-return]

    async def add_labels(
        self, repo: str, issue_number: int, labels: list[str]
    ) -> list[dict[str, Any]]:
        """Add labels to an issue or pull request.

        Args:
            repo: Repository in ``owner/repo`` format.
            issue_number: Issue or PR number.
            labels: Label names to add.

        Returns:
            Updated labels list.
        """
        response = await self._request(
            "POST",
            f"/repos/{repo}/issues/{issue_number}/labels",
            json={"labels": labels},
        )
        return response.json()  # type: ignore[no-any-return]

    # -----------------------------------------------------------------------
    # Issues
    # -----------------------------------------------------------------------

    async def get_issue(self, repo: str, issue_number: int) -> dict[str, Any]:
        """Fetch issue metadata.

        Args:
            repo: Repository in ``owner/repo`` format.
            issue_number: Issue number.

        Returns:
            Issue data as a dictionary.
        """
        response = await self._request(
            "GET", f"/repos/{repo}/issues/{issue_number}"
        )
        return response.json()  # type: ignore[no-any-return]

    async def list_issues(
        self,
        repo: str,
        *,
        state: str = "open",
        labels: str = "",
        per_page: int = 30,
    ) -> list[dict[str, Any]]:
        """List repository issues.

        Args:
            repo: Repository in ``owner/repo`` format.
            state: Filter by state (open, closed, all).
            labels: Comma-separated label names to filter by.
            per_page: Results per page (max 100).

        Returns:
            List of issue objects.
        """
        params: dict[str, Any] = {"state": state, "per_page": min(per_page, 100)}
        if labels:
            params["labels"] = labels
        response = await self._request(
            "GET", f"/repos/{repo}/issues", params=params
        )
        return response.json()  # type: ignore[no-any-return]

    # -----------------------------------------------------------------------
    # Deployments (for DORA metrics)
    # -----------------------------------------------------------------------

    async def list_deployments(
        self,
        repo: str,
        *,
        environment: str = "production",
        per_page: int = 100,
    ) -> list[dict[str, Any]]:
        """List deployments for a repository.

        Args:
            repo: Repository in ``owner/repo`` format.
            environment: Filter by deployment environment.
            per_page: Results per page.

        Returns:
            List of deployment objects.
        """
        response = await self._request(
            "GET",
            f"/repos/{repo}/deployments",
            params={"environment": environment, "per_page": per_page},
        )
        return response.json()  # type: ignore[no-any-return]

    async def list_deployment_statuses(
        self, repo: str, deployment_id: int
    ) -> list[dict[str, Any]]:
        """List statuses for a deployment.

        Args:
            repo: Repository in ``owner/repo`` format.
            deployment_id: Deployment ID.

        Returns:
            List of deployment status objects.
        """
        response = await self._request(
            "GET", f"/repos/{repo}/deployments/{deployment_id}/statuses"
        )
        return response.json()  # type: ignore[no-any-return]

    # -----------------------------------------------------------------------
    # Rate Limit Helpers
    # -----------------------------------------------------------------------

    async def get_rate_limit(self) -> RateLimitInfo:
        """Fetch current rate limit status.

        Returns:
            RateLimitInfo with current limits.
        """
        response = await self._request("GET", "/rate_limit")
        data = response.json()
        core = data.get("resources", {}).get("core", {})
        return RateLimitInfo(
            limit=core.get("limit", 0),
            remaining=core.get("remaining", 0),
            reset_timestamp=core.get("reset", 0),
            used=core.get("used", 0),
        )

    @staticmethod
    def _parse_rate_limit(response: httpx.Response) -> RateLimitInfo | None:
        """Extract rate limit info from response headers."""
        try:
            return RateLimitInfo(
                limit=int(response.headers.get("X-RateLimit-Limit", "0")),
                remaining=int(response.headers.get("X-RateLimit-Remaining", "0")),
                reset_timestamp=int(response.headers.get("X-RateLimit-Reset", "0")),
                used=int(response.headers.get("X-RateLimit-Used", "0")),
            )
        except (ValueError, TypeError):
            return None

    @staticmethod
    def _current_timestamp() -> int:
        """Get current Unix timestamp."""
        import time
        return int(time.time())
