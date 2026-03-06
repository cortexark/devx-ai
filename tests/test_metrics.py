"""Tests for engineering metrics: DORA analysis and dashboard."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from fastapi.testclient import TestClient

from devx.core.models import DeploymentRecord, DORAMetrics
from devx.metrics.analyzer import DORAAnalyzer
from devx.metrics.dashboard import MetricsStore, app, get_store

# ---------------------------------------------------------------------------
# DORAAnalyzer
# ---------------------------------------------------------------------------


class TestDORAAnalyzer:
    def test_calculate_with_deployments(self, sample_deployments):
        analyzer = DORAAnalyzer()
        metrics = analyzer.calculate(sample_deployments, window_days=30)

        assert metrics.deployment_frequency > 0
        assert metrics.change_failure_rate > 0  # We have one failed deploy
        assert metrics.window_days == 30

    def test_calculate_empty_deployments(self):
        analyzer = DORAAnalyzer()
        metrics = analyzer.calculate([], window_days=30)

        assert metrics.deployment_frequency == 0.0
        assert metrics.lead_time_seconds == 0.0
        assert metrics.change_failure_rate == 0.0
        assert metrics.mttr_seconds == 0.0

    def test_deployment_frequency(self, sample_deployments):
        analyzer = DORAAnalyzer()
        metrics = analyzer.calculate(sample_deployments, window_days=30)
        # 4 successful deploys / 30 days
        expected = 4 / 30
        assert abs(metrics.deployment_frequency - expected) < 0.01

    def test_change_failure_rate(self, sample_deployments):
        analyzer = DORAAnalyzer()
        metrics = analyzer.calculate(sample_deployments, window_days=30)
        # 1 failure out of 5 deployments = 0.2
        assert abs(metrics.change_failure_rate - 0.2) < 0.01

    def test_lead_time_median(self, sample_deployments):
        analyzer = DORAAnalyzer()
        metrics = analyzer.calculate(sample_deployments, window_days=30)
        # Lead times: 3600, 7200, 1800, 5400, 10800
        # Median of all with lead_time > 0 = 5400
        assert metrics.lead_time_seconds == 5400.0

    def test_mttr_estimation(self, sample_deployments):
        analyzer = DORAAnalyzer()
        metrics = analyzer.calculate(sample_deployments, window_days=30)
        # The failed deploy (dep-3, day -5) should have recovery from dep-4 (day -4)
        # MTTR = ~1 day (86400 seconds)
        assert metrics.mttr_seconds > 0

    def test_trend_improving(self):
        analyzer = DORAAnalyzer()
        current = DORAMetrics(
            deployment_frequency=2.0,
            lead_time_seconds=3600,
            change_failure_rate=0.05,
            mttr_seconds=1800,
        )
        previous = DORAMetrics(
            deployment_frequency=1.0,
            lead_time_seconds=7200,
            change_failure_rate=0.10,
            mttr_seconds=3600,
        )

        trend = analyzer.trend(current, previous)

        assert trend["deployment_frequency"]["direction"] == "improving"
        assert trend["lead_time_seconds"]["direction"] == "improving"
        assert trend["change_failure_rate"]["direction"] == "improving"
        assert trend["mttr_seconds"]["direction"] == "improving"

    def test_trend_declining(self):
        analyzer = DORAAnalyzer()
        current = DORAMetrics(
            deployment_frequency=0.5,
            lead_time_seconds=14400,
            change_failure_rate=0.20,
            mttr_seconds=7200,
        )
        previous = DORAMetrics(
            deployment_frequency=2.0,
            lead_time_seconds=3600,
            change_failure_rate=0.05,
            mttr_seconds=1800,
        )

        trend = analyzer.trend(current, previous)

        assert trend["deployment_frequency"]["direction"] == "declining"
        assert trend["lead_time_seconds"]["direction"] == "declining"

    def test_trend_stable(self):
        analyzer = DORAAnalyzer()
        metrics = DORAMetrics(
            deployment_frequency=1.0,
            lead_time_seconds=3600,
            change_failure_rate=0.1,
            mttr_seconds=1800,
        )

        trend = analyzer.trend(metrics, metrics)

        assert trend["deployment_frequency"]["direction"] == "stable"
        assert trend["deployment_frequency"]["change_percent"] == 0.0

    def test_team_comparison(self):
        analyzer = DORAAnalyzer()
        teams = {
            "platform": DORAMetrics(
                deployment_frequency=3.0,
                lead_time_seconds=3600,
                change_failure_rate=0.05,
                mttr_seconds=900,
            ),
            "payments": DORAMetrics(
                deployment_frequency=1.0,
                lead_time_seconds=7200,
                change_failure_rate=0.10,
                mttr_seconds=3600,
            ),
            "mobile": DORAMetrics(
                deployment_frequency=0.5,
                lead_time_seconds=14400,
                change_failure_rate=0.15,
                mttr_seconds=7200,
            ),
        }

        comparison = analyzer.team_comparison(teams)

        assert comparison["deployment_frequency"]["best_team"] == "platform"
        assert len(comparison["deployment_frequency"]["rankings"]) == 3

    def test_team_comparison_empty(self):
        analyzer = DORAAnalyzer()
        assert analyzer.team_comparison({}) == {}

    def test_mttr_with_incidents(self):
        now = datetime.now(tz=UTC)
        analyzer = DORAAnalyzer()
        deployments = [
            DeploymentRecord(
                id="d1",
                repo="r",
                sha="a",
                deployed_at=now,
                status="success",
                lead_time_seconds=3600,
            )
        ]
        incidents = [
            {
                "created_at": now - timedelta(hours=2),
                "resolved_at": now - timedelta(hours=1),
            },
            {
                "created_at": now - timedelta(hours=4),
                "resolved_at": now - timedelta(hours=3),
            },
        ]

        metrics = analyzer.calculate(deployments, window_days=30, incidents=incidents)
        # Each incident took 1 hour to resolve -> mean = 3600s
        assert metrics.mttr_seconds == 3600.0


# ---------------------------------------------------------------------------
# Dashboard API
# ---------------------------------------------------------------------------


class TestDashboardAPI:
    @pytest.fixture
    def client(self):
        """FastAPI test client."""
        return TestClient(app)

    @pytest.fixture(autouse=True)
    def reset_store(self):
        """Reset the global store before each test."""
        store = get_store()
        store.deployments = []
        store.dora_snapshots = []
        store.team_metrics = {}
        yield

    def test_health_check(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["version"] == "0.1.0"

    def test_get_dora_no_data(self, client):
        response = client.get("/api/v1/dora")
        assert response.status_code == 404

    def test_get_dora_with_snapshot(self, client):
        store = get_store()
        store.add_dora_snapshot(
            DORAMetrics(
                deployment_frequency=2.0,
                lead_time_seconds=3600,
                change_failure_rate=0.05,
                mttr_seconds=1800,
            )
        )

        response = client.get("/api/v1/dora")
        assert response.status_code == 200
        data = response.json()
        assert data["metrics"]["deployment_frequency"] == 2.0
        assert data["ratings"]["deployment_frequency"] == "elite"

    def test_record_deployment(self, client):
        now = datetime.now(tz=UTC)
        response = client.post(
            "/api/v1/deployments",
            json={
                "id": "dep-1",
                "repo": "org/app",
                "sha": "abc123",
                "deployed_at": now.isoformat(),
            },
        )
        assert response.status_code == 201
        assert response.json()["status"] == "recorded"

    def test_list_deployments_empty(self, client):
        response = client.get("/api/v1/deployments")
        assert response.status_code == 200
        assert response.json()["total"] == 0

    def test_list_deployments_with_data(self, client):
        store = get_store()
        now = datetime.now(tz=UTC)
        store.add_deployment(
            DeploymentRecord(
                id="dep-1",
                repo="org/app",
                sha="abc123",
                deployed_at=now,
            )
        )
        store.add_deployment(
            DeploymentRecord(
                id="dep-2",
                repo="org/other",
                sha="def456",
                deployed_at=now - timedelta(hours=1),
            )
        )

        response = client.get("/api/v1/deployments")
        assert response.status_code == 200
        assert response.json()["total"] == 2

    def test_list_deployments_filter_repo(self, client):
        store = get_store()
        now = datetime.now(tz=UTC)
        store.add_deployment(
            DeploymentRecord(
                id="dep-1",
                repo="org/app",
                sha="abc",
                deployed_at=now,
            )
        )
        store.add_deployment(
            DeploymentRecord(
                id="dep-2",
                repo="org/other",
                sha="def",
                deployed_at=now,
            )
        )

        response = client.get("/api/v1/deployments?repo=org/app")
        assert response.status_code == 200
        assert response.json()["total"] == 1

    def test_store_dora_snapshot(self, client):
        response = client.post(
            "/api/v1/dora/snapshot",
            json={
                "deployment_frequency": 1.5,
                "lead_time_seconds": 7200,
                "change_failure_rate": 0.1,
                "mttr_seconds": 3600,
            },
        )
        assert response.status_code == 201

    def test_get_team_comparison_empty(self, client):
        response = client.get("/api/v1/teams")
        assert response.status_code == 200


# ---------------------------------------------------------------------------
# MetricsStore
# ---------------------------------------------------------------------------


class TestMetricsStore:
    def test_add_and_get_deployment(self):
        store = MetricsStore()
        now = datetime.now(tz=UTC)
        dep = DeploymentRecord(
            id="d1",
            repo="org/app",
            sha="abc",
            deployed_at=now,
        )
        store.add_deployment(dep)
        assert len(store.get_deployments()) == 1

    def test_get_latest_dora_empty(self):
        store = MetricsStore()
        assert store.get_latest_dora() is None

    def test_get_latest_dora(self):
        store = MetricsStore()
        m1 = DORAMetrics(
            deployment_frequency=1.0,
            lead_time_seconds=3600,
            change_failure_rate=0.1,
            mttr_seconds=1800,
        )
        m2 = DORAMetrics(
            deployment_frequency=2.0,
            lead_time_seconds=1800,
            change_failure_rate=0.05,
            mttr_seconds=900,
        )
        store.add_dora_snapshot(m1)
        store.add_dora_snapshot(m2)
        latest = store.get_latest_dora()
        assert latest is not None
        assert latest.deployment_frequency == 2.0

    def test_deployments_sorted_by_date(self):
        store = MetricsStore()
        now = datetime.now(tz=UTC)
        store.add_deployment(
            DeploymentRecord(
                id="old",
                repo="r",
                sha="a",
                deployed_at=now - timedelta(days=2),
            )
        )
        store.add_deployment(
            DeploymentRecord(
                id="new",
                repo="r",
                sha="b",
                deployed_at=now,
            )
        )
        deployments = store.get_deployments()
        assert deployments[0].id == "new"  # Most recent first
