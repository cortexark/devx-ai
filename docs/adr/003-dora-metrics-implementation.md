# ADR 003: DORA Metrics Implementation

## Status
Accepted

## Date
2026-03-01

## Context

The DORA (DevOps Research and Assessment) framework defines four key metrics that predict software delivery performance:

1. **Deployment Frequency** -- How often code is deployed to production
2. **Lead Time for Changes** -- Time from commit to production
3. **Change Failure Rate** -- Percentage of deployments causing failures
4. **Mean Time to Recovery (MTTR)** -- Time to restore service after failure

We need to calculate these from GitHub data (deployments, pull requests, incidents) and present them through a dashboard API. The challenge is that GitHub's data model does not map cleanly to DORA concepts.

## Decision

### Data Sources

| DORA Metric | GitHub Data Source | Calculation |
|-------------|-------------------|-------------|
| Deployment Frequency | Deployments API | `count(successful_deploys) / window_days` |
| Lead Time | Deployment SHA + Commit timestamps | `median(deploy_time - first_commit_time)` |
| Change Failure Rate | Deployment Statuses | `count(failed or rollback) / count(total)` |
| MTTR | Deployment gaps or Incidents | `mean(next_success_time - failure_time)` |

### Key Design Decisions

**1. Deployment-centric, not PR-centric.** We track actual deployments, not PR merges. Teams that merge to main but deploy weekly have different dynamics than teams with continuous deployment. The Deployments API gives us the ground truth.

**2. MTTR estimation from deployment gaps.** Most GitHub repos do not have a formal incident management system. We estimate MTTR by measuring the time between a failed deployment and the next successful one. When explicit incident data is available (created_at/resolved_at), we use that instead.

**3. Window-based calculation.** All metrics are calculated over a configurable window (default: 30 days). This smooths out noise and matches the DORA research methodology. The `DORAMetrics` model stores `window_days` for reproducibility.

**4. DORA benchmark ratings.** We classify each metric according to the State of DevOps benchmarks:

| Rating | Deploy Freq | Lead Time | CFR | MTTR |
|--------|------------|-----------|-----|------|
| Elite | >= 1/day | < 1 day | < 5% | < 1 hour |
| High | 1/week - 1/day | 1 day - 1 week | 5-15% | < 1 day |
| Medium | 1/month - 1/week | 1 week - 1 month | 15-30% | < 1 week |
| Low | < 1/month | > 1 month | > 30% | > 1 week |

**5. Trend analysis.** We compare current and previous period snapshots to show direction (improving, stable, declining). This matters more than absolute numbers for team retrospectives.

### What Actually Matters for AI/ML Teams

DORA metrics were designed for traditional software delivery. For AI teams, additional considerations:

- **Model deployment frequency** may be lower than code deployment frequency. A team might deploy code daily but retrain/deploy models weekly. Consider tracking both.
- **Lead time** for ML includes data pipeline time, training time, and validation time -- not just code review and CI.
- **Change failure rate** for models includes model quality regressions (accuracy drops, bias increases) not just service outages.
- **MTTR** for model incidents may include rollback to a previous model version, which is architecturally different from code rollbacks.

We recommend teams track DORA for their code delivery pipeline AND separate metrics for their ML pipeline. devx-ai focuses on the code delivery side.

### Limitations

1. **Lead time accuracy.** Calculating true lead time requires tracing from the first commit in a feature branch to its production deployment. GitHub's Deployments API gives us the SHA but not the full commit graph. We approximate by using the deployment's associated commit timestamp.

2. **Incident data.** Without a dedicated incident management integration (PagerDuty, Opsgenie), MTTR is estimated from deployment patterns. This misses incidents that don't result in a deployment (e.g., configuration changes, database issues).

3. **Multi-service architectures.** DORA metrics per-repo may not reflect system-level performance. A repo with high deployment frequency may be a leaf service with low blast radius.

## Consequences

- `MetricsCollector` gathers raw data from GitHub.
- `DORAAnalyzer` computes metrics and trends.
- `dashboard.py` exposes metrics via FastAPI REST API.
- Trend analysis enables retrospective conversations, not just snapshots.
- The architecture supports adding more data sources (PagerDuty, Datadog) as collectors.

## Alternatives Considered

| Approach | Pros | Cons |
|----------|------|------|
| PR-based metrics only | Simpler, no Deployments API needed | Doesn't reflect actual delivery |
| Full Sleuth/LinearB integration | More accurate | Vendor dependency, API cost |
| Git log analysis only | No API needed | No deployment or incident data |
