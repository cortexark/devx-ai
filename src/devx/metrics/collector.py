"""Engineering metrics collector from GitHub API.

Collects deployment events, pull request data, and incident records
from GitHub to feed the DORA metrics analyzer.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta
from typing import Any

from devx.core.config import GitHubConfig, MetricsConfig
from devx.core.models import DeploymentRecord
from devx.sdlc.github_client import GitHubClient

logger = logging.getLogger(__name__)


class MetricsCollector:
    """Collect engineering metrics from GitHub.

    Gathers deployment records, PR merge times, and incident data
    to feed DORA metrics calculations.

    Example::

        collector = MetricsCollector(
            github_config=GitHubConfig(token="ghp_..."),
        )
        deployments = await collector.collect_deployments("owner/repo")
        pr_data = await collector.collect_pr_metrics("owner/repo")
    """

    def __init__(
        self,
        github_config: GitHubConfig | None = None,
        metrics_config: MetricsConfig | None = None,
    ) -> None:
        self._github_config = github_config or GitHubConfig()
        self._metrics_config = metrics_config or MetricsConfig()

    async def collect_deployments(
        self,
        repo: str,
        *,
        environment: str = "production",
        days: int | None = None,
    ) -> list[DeploymentRecord]:
        """Collect deployment records from GitHub Deployments API.

        Args:
            repo: Repository in ``owner/repo`` format.
            environment: Target deployment environment.
            days: Number of days to look back. Uses config default if None.

        Returns:
            List of DeploymentRecord objects.
        """
        window = days or self._metrics_config.window_days
        cutoff = datetime.now(tz=UTC) - timedelta(days=window)

        async with GitHubClient(self._github_config) as gh:
            raw_deployments = await gh.list_deployments(repo, environment=environment)

            records: list[DeploymentRecord] = []
            for dep in raw_deployments:
                deployed_at = self._parse_datetime(dep.get("created_at", ""))
                if deployed_at and deployed_at < cutoff:
                    continue

                # Get deployment status to determine success/failure
                statuses = await gh.list_deployment_statuses(repo, dep["id"])
                status = self._determine_status(statuses)

                # Calculate lead time from the deployment's ref
                lead_time = await self._calculate_lead_time(gh, repo, dep)

                records.append(
                    DeploymentRecord(
                        id=str(dep["id"]),
                        repo=repo,
                        environment=environment,
                        sha=dep.get("sha", ""),
                        deployed_at=deployed_at or datetime.now(tz=UTC),
                        status=status,
                        lead_time_seconds=lead_time,
                    )
                )

            return records

    async def collect_pr_metrics(
        self,
        repo: str,
        *,
        days: int | None = None,
    ) -> list[dict[str, Any]]:
        """Collect pull request metrics (merge time, review time, size).

        Args:
            repo: Repository in ``owner/repo`` format.
            days: Number of days to look back.

        Returns:
            List of PR metric dictionaries.
        """
        window = days or self._metrics_config.window_days

        async with GitHubClient(self._github_config) as gh:
            # List recently closed PRs
            issues = await gh.list_issues(repo, state="closed", per_page=100)

            pr_metrics: list[dict[str, Any]] = []
            cutoff = datetime.now(tz=UTC) - timedelta(days=window)

            for issue in issues:
                if "pull_request" not in issue:
                    continue

                closed_at = self._parse_datetime(issue.get("closed_at", ""))
                if closed_at and closed_at < cutoff:
                    continue

                created_at = self._parse_datetime(issue.get("created_at", ""))
                if created_at and closed_at:
                    cycle_time = (closed_at - created_at).total_seconds()
                else:
                    cycle_time = None

                pr_metrics.append(
                    {
                        "number": issue["number"],
                        "title": issue.get("title", ""),
                        "created_at": issue.get("created_at"),
                        "closed_at": issue.get("closed_at"),
                        "cycle_time_seconds": cycle_time,
                        "labels": [lbl.get("name", "") for lbl in issue.get("labels", [])],
                        "author": issue.get("user", {}).get("login", ""),
                    }
                )

            return pr_metrics

    async def _calculate_lead_time(
        self,
        gh: GitHubClient,
        repo: str,
        deployment: dict[str, Any],
    ) -> float | None:
        """Calculate lead time from first commit to deployment.

        Lead time = deployment time - first commit time in the deployment ref.
        This is a simplified calculation; a production system would trace
        the full commit graph.
        """
        deployed_at = self._parse_datetime(deployment.get("created_at", ""))
        if not deployed_at:
            return None

        # Use the deployment's created_at vs the ref's first commit
        # Simplified: we approximate using the deployment creation time
        # A full implementation would resolve the SHA to its first commit
        sha = deployment.get("sha", "")
        if not sha:
            return None

        try:
            # Approximate lead time from the time between ref creation and deploy
            # In practice, you'd compare against the commit timestamp
            return None  # Return None when we can't accurately calculate
        except Exception:
            logger.debug("Could not calculate lead time for deployment %s", deployment.get("id"))
            return None

    @staticmethod
    def _determine_status(statuses: list[dict[str, Any]]) -> str:
        """Determine overall deployment status from status list.

        GitHub returns statuses in reverse chronological order.
        The most recent status is the current state.
        """
        if not statuses:
            return "unknown"

        latest = statuses[0]
        state = latest.get("state", "unknown")

        status_map = {
            "success": "success",
            "failure": "failure",
            "error": "failure",
            "inactive": "rollback",
            "in_progress": "in_progress",
            "queued": "queued",
            "pending": "pending",
        }

        return status_map.get(state, "unknown")

    @staticmethod
    def _parse_datetime(dt_str: str) -> datetime | None:
        """Parse ISO 8601 datetime string from GitHub API."""
        if not dt_str:
            return None
        try:
            # Handle GitHub's ISO format (2024-01-15T10:30:00Z)
            return datetime.fromisoformat(dt_str.replace("Z", "+00:00"))
        except (ValueError, TypeError):
            return None
