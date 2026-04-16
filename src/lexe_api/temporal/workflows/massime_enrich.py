"""MassimeEnrichWorkflow — weekly enrichment of kb.massime.source_url via ItalGiure.

Sprint 27 T9 S7.1. Scheduled Sunday 02:00 UTC (cron: "0 2 * * 0").

Pipeline:
1. get_pending_massime(batch_size): fetch kb.massime rows with source_url IS NULL
   AND is_active = true AND numero IS NOT NULL AND anno IS NOT NULL, ordered by id.
2. enrich_massime_batch(batch): call ItalGiure resolver (via lexe-core internal
   endpoint or local Solr) for each massima, update kb.massime.source_url when
   a Tier 1 URL is returned.

The workflow loops until get_pending_massime returns empty (no more pending)
OR max_iterations reached (safety cap). Each iteration processes one batch.

Migration 081 provides ``idx_massime_source_url_null`` for efficient lookup.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import timedelta

from temporalio import workflow
from temporalio.common import RetryPolicy

with workflow.unsafe.imports_passed_through():
    from lexe_api.temporal.task_queues import TaskQueues

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class MassimeEnrichParams:
    """Input parameters for MassimeEnrichWorkflow.

    Attributes:
        batch_size: max massime per batch (tradeoff: SLA vs activity timeout).
        max_iterations: safety cap on total iterations (prevents runaway).
        triggered_by: "schedule" | "manual" | "backfill".
    """

    batch_size: int = 500
    max_iterations: int = 200  # 200 * 500 = 100_000 massime cap per run
    triggered_by: str = "schedule"


@dataclass
class MassimeEnrichResult:
    """Aggregated result across all iterations of MassimeEnrichWorkflow."""

    total_batches: int = 0
    total_fetched: int = 0
    total_resolved: int = 0
    total_unresolved: int = 0
    total_errors: int = 0
    errors: list[str] = field(default_factory=list)
    stopped_reason: str = "no_more_pending"
    success: bool = True


# ---------------------------------------------------------------------------
# Retry policies
# ---------------------------------------------------------------------------

_DB_RETRY = RetryPolicy(
    initial_interval=timedelta(seconds=2),
    backoff_coefficient=2.0,
    maximum_interval=timedelta(minutes=2),
    maximum_attempts=5,
)

_ENRICH_RETRY = RetryPolicy(
    initial_interval=timedelta(seconds=5),
    backoff_coefficient=2.0,
    maximum_interval=timedelta(minutes=5),
    maximum_attempts=3,  # italgiure can be flaky; accept some unresolved
)


# ---------------------------------------------------------------------------
# Workflow
# ---------------------------------------------------------------------------


@workflow.defn(name="MassimeEnrichWorkflow")
class MassimeEnrichWorkflow:
    """Weekly batch enrichment of kb.massime.source_url via ItalGiure resolver.

    Schedule: Sunday 02:00 UTC (``cron: 0 2 * * 0``) — low-traffic window,
    after the Saturday nightly KB sync (kb_sync workflow, daily 03:00 UTC).
    """

    @workflow.run
    async def run(self, params: MassimeEnrichParams) -> MassimeEnrichResult:
        result = MassimeEnrichResult()
        workflow.logger.info(
            "[MassimeEnrich] start triggered_by=%s batch_size=%d max_iter=%d",
            params.triggered_by, params.batch_size, params.max_iterations,
        )

        for iteration in range(params.max_iterations):
            # Step 1: fetch pending batch
            batch = await workflow.execute_activity(
                "get_pending_massime_activity",
                {"batch_size": params.batch_size},
                task_queue=TaskQueues.KB_SYNC.value,
                start_to_close_timeout=timedelta(minutes=2),
                retry_policy=_DB_RETRY,
            )
            if not batch:
                result.stopped_reason = "no_more_pending"
                workflow.logger.info(
                    "[MassimeEnrich] no more pending after %d iterations",
                    iteration,
                )
                break

            result.total_batches += 1
            result.total_fetched += len(batch)

            # Step 2: enrich batch (HTTP to italgiure + DB update)
            try:
                batch_result = await workflow.execute_activity(
                    "enrich_massime_batch_activity",
                    {"massime": batch},
                    task_queue=TaskQueues.KB_SYNC.value,
                    start_to_close_timeout=timedelta(minutes=10),
                    retry_policy=_ENRICH_RETRY,
                )
                result.total_resolved += batch_result.get("resolved", 0)
                result.total_unresolved += batch_result.get("unresolved", 0)
            except Exception as exc:  # noqa: BLE001
                result.total_errors += 1
                err = f"iter={iteration} batch_size={len(batch)} err={exc}"
                result.errors.append(err)
                workflow.logger.warning("[MassimeEnrich] enrich failed: %s", err)
                if result.total_errors >= 5:
                    result.stopped_reason = "too_many_errors"
                    result.success = False
                    break
        else:
            # for-else: executed if loop exhausts max_iterations without break
            result.stopped_reason = "max_iterations_reached"

        workflow.logger.info(
            "[MassimeEnrich] done batches=%d fetched=%d resolved=%d unresolved=%d errors=%d stop=%s",
            result.total_batches, result.total_fetched,
            result.total_resolved, result.total_unresolved,
            result.total_errors, result.stopped_reason,
        )
        return result
