"""DORA metrics analysis engine.

Calculates the four DORA metrics from deployment and incident data:
  1. Deployment Frequency
  2. Lead Time for Changes
  3. Change Failure Rate
  4. Mean Time to Recovery (MTTR)

References the DORA State of DevOps research for benchmark classifications.
"""

from __future__ import annotations

import statistics
from datetime import datetime
from typing import Any

from devx.core.models import DeploymentRecord, DORAMetrics


class DORAAnalyzer:
    """Calculate and analyze DORA metrics.

    Example::

        analyzer = DORAAnalyzer()
        metrics = analyzer.calculate(deployments, window_days=30)
        print(f"Deploy freq: {metrics.deployment_frequency}/day")
        print(f"Rating: {metrics.deployment_frequency_rating}")

        trend = analyzer.trend(current_metrics, previous_metrics)
        print(f"Frequency trend: {trend['deployment_frequency']}")
    """

    def calculate(
        self,
        deployments: list[DeploymentRecord],
        *,
        window_days: int = 30,
        incidents: list[dict[str, Any]] | None = None,
    ) -> DORAMetrics:
        """Calculate DORA metrics from deployment records.

        Args:
            deployments: List of deployment records within the window.
            window_days: Measurement window in days.
            incidents: Optional list of incident records with
                ``created_at`` and ``resolved_at`` datetime fields.

        Returns:
            DORAMetrics snapshot.
        """
        if not deployments:
            return DORAMetrics(
                deployment_frequency=0.0,
                lead_time_seconds=0.0,
                change_failure_rate=0.0,
                mttr_seconds=0.0,
                window_days=window_days,
            )

        deploy_freq = self._deployment_frequency(deployments, window_days)
        lead_time = self._median_lead_time(deployments)
        failure_rate = self._change_failure_rate(deployments)
        mttr = self._mean_time_to_recovery(deployments, incidents or [])

        return DORAMetrics(
            deployment_frequency=round(deploy_freq, 4),
            lead_time_seconds=round(lead_time, 2),
            change_failure_rate=round(failure_rate, 4),
            mttr_seconds=round(mttr, 2),
            window_days=window_days,
        )

    def trend(
        self,
        current: DORAMetrics,
        previous: DORAMetrics,
    ) -> dict[str, dict[str, Any]]:
        """Compare two DORA metrics snapshots and return trends.

        Args:
            current: Current metrics snapshot.
            previous: Previous metrics snapshot (earlier period).

        Returns:
            Dictionary with metric name -> {value, previous, change, direction}.
        """
        trends: dict[str, dict[str, Any]] = {}

        metrics_to_compare = [
            ("deployment_frequency", True),     # Higher is better
            ("lead_time_seconds", False),        # Lower is better
            ("change_failure_rate", False),       # Lower is better
            ("mttr_seconds", False),             # Lower is better
        ]

        for metric_name, higher_is_better in metrics_to_compare:
            curr_val = getattr(current, metric_name)
            prev_val = getattr(previous, metric_name)

            if prev_val == 0:
                pct_change = 100.0 if curr_val > 0 else 0.0
            else:
                pct_change = ((curr_val - prev_val) / prev_val) * 100

            if higher_is_better:
                if pct_change > 0:
                    direction = "improving"
                elif pct_change < 0:
                    direction = "declining"
                else:
                    direction = "stable"
            else:
                if pct_change < 0:
                    direction = "improving"
                elif pct_change > 0:
                    direction = "declining"
                else:
                    direction = "stable"

            trends[metric_name] = {
                "value": curr_val,
                "previous": prev_val,
                "change_percent": round(pct_change, 2),
                "direction": direction,
            }

        return trends

    def team_comparison(
        self,
        team_metrics: dict[str, DORAMetrics],
    ) -> dict[str, dict[str, Any]]:
        """Compare DORA metrics across teams.

        Args:
            team_metrics: Dictionary mapping team name to DORAMetrics.

        Returns:
            Comparison summary with rankings per metric.
        """
        if not team_metrics:
            return {}

        comparison: dict[str, dict[str, Any]] = {}
        metric_names = [
            "deployment_frequency",
            "lead_time_seconds",
            "change_failure_rate",
            "mttr_seconds",
        ]

        for metric_name in metric_names:
            values = {
                team: getattr(metrics, metric_name)
                for team, metrics in team_metrics.items()
            }

            # Sort: higher is better for deploy freq, lower for everything else
            reverse = metric_name == "deployment_frequency"
            ranked = sorted(values.items(), key=lambda x: x[1], reverse=reverse)

            comparison[metric_name] = {
                "rankings": [{"team": t, "value": v} for t, v in ranked],
                "average": round(statistics.mean(values.values()), 4) if values else 0.0,
                "best_team": ranked[0][0] if ranked else None,
            }

        return comparison

    @staticmethod
    def _deployment_frequency(
        deployments: list[DeploymentRecord],
        window_days: int,
    ) -> float:
        """Calculate deployments per day.

        Only counts successful deployments.
        """
        successful = [d for d in deployments if d.status == "success"]
        if window_days <= 0:
            return 0.0
        return len(successful) / window_days

    @staticmethod
    def _median_lead_time(deployments: list[DeploymentRecord]) -> float:
        """Calculate median lead time in seconds.

        Lead time = time from first commit to production deployment.
        Uses the lead_time_seconds field from DeploymentRecord.
        """
        lead_times = [
            d.lead_time_seconds
            for d in deployments
            if d.lead_time_seconds is not None and d.lead_time_seconds > 0
        ]

        if not lead_times:
            return 0.0

        return statistics.median(lead_times)

    @staticmethod
    def _change_failure_rate(deployments: list[DeploymentRecord]) -> float:
        """Calculate the fraction of deployments that caused failures.

        Change failure rate = failed deploys / total deploys.
        """
        if not deployments:
            return 0.0

        total = len(deployments)
        failures = sum(
            1 for d in deployments if d.status in ("failure", "rollback")
        )

        return failures / total

    @staticmethod
    def _mean_time_to_recovery(
        deployments: list[DeploymentRecord],
        incidents: list[dict[str, Any]],
    ) -> float:
        """Calculate mean time to recovery in seconds.

        If incidents are provided, uses incident resolution time.
        Otherwise, estimates from the gap between a failed deployment
        and the next successful one.
        """
        # Method 1: Use explicit incident data
        if incidents:
            recovery_times: list[float] = []
            for incident in incidents:
                created = incident.get("created_at")
                resolved = incident.get("resolved_at")
                if isinstance(created, datetime) and isinstance(resolved, datetime):
                    recovery_times.append(
                        (resolved - created).total_seconds()
                    )
            if recovery_times:
                return statistics.mean(recovery_times)

        # Method 2: Estimate from deployment gaps
        sorted_deps = sorted(deployments, key=lambda d: d.deployed_at)
        recovery_times_est: list[float] = []

        for i, dep in enumerate(sorted_deps):
            if dep.status in ("failure", "rollback"):
                # Find next successful deployment
                for next_dep in sorted_deps[i + 1:]:
                    if next_dep.status == "success":
                        recovery = (
                            next_dep.deployed_at - dep.deployed_at
                        ).total_seconds()
                        recovery_times_est.append(recovery)
                        break

        if recovery_times_est:
            return statistics.mean(recovery_times_est)

        return 0.0
