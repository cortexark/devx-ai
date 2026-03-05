"""FastAPI dashboard endpoints for engineering metrics.

Provides REST API endpoints for DORA metrics, deployment history,
and team comparisons.  Designed to feed a frontend dashboard or
be consumed by monitoring tools.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from devx.core.models import DeploymentRecord, DORAMetrics

app = FastAPI(
    title="devx-ai Metrics Dashboard",
    description="Engineering metrics API powered by DORA framework",
    version="0.1.0",
)


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = "ok"
    version: str = "0.1.0"
    timestamp: datetime = Field(default_factory=lambda: datetime.now(tz=UTC))


class DORAResponse(BaseModel):
    """DORA metrics API response."""

    metrics: DORAMetrics
    ratings: dict[str, str] = Field(default_factory=dict)


class TrendResponse(BaseModel):
    """Trend comparison API response."""

    current: DORAMetrics
    previous: DORAMetrics
    trends: dict[str, dict[str, Any]] = Field(default_factory=dict)


class DeploymentListResponse(BaseModel):
    """Deployment history API response."""

    deployments: list[DeploymentRecord] = Field(default_factory=list)
    total: int = 0


# ---------------------------------------------------------------------------
# In-memory store (replaced with real data source in production)
# ---------------------------------------------------------------------------


class MetricsStore:
    """Simple in-memory store for demo/testing purposes.

    In production, this would be backed by a database or metrics
    aggregation service.
    """

    def __init__(self) -> None:
        self.deployments: list[DeploymentRecord] = []
        self.dora_snapshots: list[DORAMetrics] = []
        self.team_metrics: dict[str, DORAMetrics] = {}

    def add_deployment(self, record: DeploymentRecord) -> None:
        """Record a deployment event."""
        self.deployments.append(record)

    def add_dora_snapshot(self, metrics: DORAMetrics) -> None:
        """Store a DORA metrics snapshot."""
        self.dora_snapshots.append(metrics)

    def get_latest_dora(self) -> DORAMetrics | None:
        """Get the most recent DORA snapshot."""
        if not self.dora_snapshots:
            return None
        return self.dora_snapshots[-1]

    def get_deployments(
        self,
        *,
        repo: str | None = None,
        limit: int = 50,
    ) -> list[DeploymentRecord]:
        """Get deployment records, optionally filtered by repo."""
        records = self.deployments
        if repo:
            records = [d for d in records if d.repo == repo]
        return sorted(records, key=lambda d: d.deployed_at, reverse=True)[:limit]


# Global store instance
_store = MetricsStore()


def get_store() -> MetricsStore:
    """Get the global metrics store. Override in tests via dependency injection."""
    return _store


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Health check endpoint."""
    return HealthResponse()


@app.get("/api/v1/dora", response_model=DORAResponse)
async def get_dora_metrics(
    window_days: int = Query(default=30, ge=1, le=365, description="Measurement window"),
) -> DORAResponse:
    """Get current DORA metrics.

    Returns the most recent DORA metrics snapshot with
    DORA benchmark ratings.
    """
    store = get_store()
    metrics = store.get_latest_dora()

    if not metrics:
        raise HTTPException(
            status_code=404,
            detail="No DORA metrics available. Run a collection first.",
        )

    ratings = {
        "deployment_frequency": metrics.deployment_frequency_rating,
        "lead_time": metrics.lead_time_rating,
    }

    return DORAResponse(metrics=metrics, ratings=ratings)


@app.get("/api/v1/deployments", response_model=DeploymentListResponse)
async def list_deployments(
    repo: str | None = Query(default=None, description="Filter by repository"),
    limit: int = Query(default=50, ge=1, le=200, description="Max results"),
) -> DeploymentListResponse:
    """List deployment history.

    Returns deployment records sorted by most recent first.
    """
    store = get_store()
    deployments = store.get_deployments(repo=repo, limit=limit)

    return DeploymentListResponse(
        deployments=deployments,
        total=len(deployments),
    )


@app.post("/api/v1/deployments", status_code=201)
async def record_deployment(record: DeploymentRecord) -> dict[str, str]:
    """Record a new deployment event.

    This endpoint is called by CI/CD pipelines to report deployments.
    """
    store = get_store()
    store.add_deployment(record)
    return {"status": "recorded", "id": record.id}


@app.get("/api/v1/teams", response_model=dict[str, Any])
async def get_team_comparison() -> dict[str, Any]:
    """Compare DORA metrics across teams.

    Returns rankings and averages for each DORA metric.
    """
    store = get_store()
    if not store.team_metrics:
        return {"teams": {}, "message": "No team metrics available."}

    from devx.metrics.analyzer import DORAAnalyzer

    analyzer = DORAAnalyzer()
    comparison = analyzer.team_comparison(store.team_metrics)
    return {"teams": store.team_metrics, "comparison": comparison}


@app.post("/api/v1/dora/snapshot", status_code=201)
async def store_dora_snapshot(metrics: DORAMetrics) -> dict[str, str]:
    """Store a DORA metrics snapshot.

    Called after metrics collection and analysis to persist results.
    """
    store = get_store()
    store.add_dora_snapshot(metrics)
    return {"status": "stored", "calculated_at": str(metrics.calculated_at)}
