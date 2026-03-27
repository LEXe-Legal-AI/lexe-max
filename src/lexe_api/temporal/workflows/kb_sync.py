"""KBSyncWorkflow - Nightly sync of KB normativa from OpenData sources.

Orchestrates the full sync pipeline:
1. Fetch list of updated acts since last sync
2. For each act, fetch updated articles
3. Upsert normativa rows in KB
4. Rechunk updated articles
5. Reembed affected chunks
6. Notify sync complete with metrics
"""

import logging
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any

from temporalio import workflow
from temporalio.common import RetryPolicy

with workflow.unsafe.imports_passed_through():
    from lexe_api.temporal.task_queues import TaskQueues

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataclasses for workflow params / results
# ---------------------------------------------------------------------------


@dataclass
class KBSyncParams:
    """Input parameters for KB sync workflow."""

    collections: list[str] = field(default_factory=list)
    full_sync: bool = False
    triggered_by: str = "schedule"


@dataclass
class KBSyncResult:
    """Result of a KB sync workflow run."""

    acts_checked: int = 0
    acts_updated: int = 0
    articles_upserted: int = 0
    chunks_created: int = 0
    embeddings_generated: int = 0
    errors: list[str] = field(default_factory=list)
    success: bool = True


# ---------------------------------------------------------------------------
# Retry policies
# ---------------------------------------------------------------------------

_FETCH_RETRY = RetryPolicy(
    initial_interval=timedelta(seconds=5),
    backoff_coefficient=2.0,
    maximum_interval=timedelta(minutes=5),
    maximum_attempts=5,
)

_DB_RETRY = RetryPolicy(
    initial_interval=timedelta(seconds=2),
    backoff_coefficient=2.0,
    maximum_interval=timedelta(minutes=2),
    maximum_attempts=5,
)

_EMBED_RETRY = RetryPolicy(
    initial_interval=timedelta(seconds=3),
    backoff_coefficient=2.0,
    maximum_interval=timedelta(minutes=10),
    maximum_attempts=3,
)


# ---------------------------------------------------------------------------
# Workflow
# ---------------------------------------------------------------------------


@workflow.defn
class KBSyncWorkflow:
    """Nightly KB sync workflow.

    Fetches updated acts from OpenData, upserts articles,
    rechunks, and regenerates embeddings for affected content.
    """

    def __init__(self) -> None:
        self._progress: dict[str, Any] = {
            "step": "initialized",
            "acts_checked": 0,
            "acts_updated": 0,
            "articles_upserted": 0,
            "chunks_created": 0,
            "embeddings_generated": 0,
        }

    @workflow.query
    def get_progress(self) -> dict[str, Any]:
        """Query current sync progress."""
        return self._progress

    @workflow.run
    async def run(self, params: dict[str, Any]) -> dict[str, Any]:
        """Execute the full KB sync pipeline.

        Args:
            params: Dict with keys matching KBSyncParams fields.

        Returns:
            Dict with keys matching KBSyncResult fields.
        """
        collections = params.get("collections", [])
        full_sync = params.get("full_sync", False)
        triggered_by = params.get("triggered_by", "schedule")

        workflow.logger.info(
            f"KB sync started: triggered_by={triggered_by}, "
            f"full_sync={full_sync}, collections={collections}"
        )

        result = KBSyncResult()

        # ------------------------------------------------------------------
        # Step 1: Fetch list of updated acts
        # ------------------------------------------------------------------
        self._progress["step"] = "fetch_updated_acts"

        try:
            updated_acts: list[dict[str, Any]] = await workflow.execute_activity(
                "fetch_updated_acts_activity",
                args=[collections, full_sync],
                task_queue=TaskQueues.KB_SYNC.value,
                start_to_close_timeout=timedelta(minutes=10),
                heartbeat_timeout=timedelta(minutes=2),
                retry_policy=_FETCH_RETRY,
            )
        except Exception as e:
            workflow.logger.error(f"Failed to fetch updated acts: {e}")
            result.success = False
            result.errors.append(f"fetch_updated_acts failed: {e}")
            return self._to_dict(result)

        result.acts_checked = len(updated_acts)
        self._progress["acts_checked"] = result.acts_checked

        if not updated_acts:
            workflow.logger.info("No updated acts found, sync complete")
            self._progress["step"] = "complete"
            return self._to_dict(result)

        # ------------------------------------------------------------------
        # Step 2: For each act, fetch updated articles
        # ------------------------------------------------------------------
        self._progress["step"] = "fetch_act_articles"

        all_articles: list[dict[str, Any]] = []
        for act in updated_acts:
            try:
                articles: list[dict[str, Any]] = await workflow.execute_activity(
                    "fetch_act_articles_activity",
                    args=[act],
                    task_queue=TaskQueues.KB_SYNC.value,
                    start_to_close_timeout=timedelta(minutes=15),
                    heartbeat_timeout=timedelta(minutes=3),
                    retry_policy=_FETCH_RETRY,
                )
                all_articles.extend(articles)
                result.acts_updated += 1
                self._progress["acts_updated"] = result.acts_updated
            except Exception as e:
                workflow.logger.warning(
                    f"Failed to fetch articles for act {act.get('urn', 'unknown')}: {e}"
                )
                result.errors.append(
                    f"fetch_act_articles failed for {act.get('urn', 'unknown')}: {e}"
                )

        if not all_articles:
            workflow.logger.info("No articles to upsert after fetching")
            self._progress["step"] = "complete"
            return self._to_dict(result)

        # ------------------------------------------------------------------
        # Step 3: Upsert normativa rows
        # ------------------------------------------------------------------
        self._progress["step"] = "upsert_normativa"

        try:
            upsert_count: int = await workflow.execute_activity(
                "upsert_normativa_activity",
                args=[all_articles],
                task_queue=TaskQueues.KB_SYNC.value,
                start_to_close_timeout=timedelta(minutes=30),
                heartbeat_timeout=timedelta(minutes=5),
                retry_policy=_DB_RETRY,
            )
            result.articles_upserted = upsert_count
            self._progress["articles_upserted"] = upsert_count
        except Exception as e:
            workflow.logger.error(f"Failed to upsert normativa: {e}")
            result.success = False
            result.errors.append(f"upsert_normativa failed: {e}")
            return self._to_dict(result)

        # ------------------------------------------------------------------
        # Step 4: Rechunk updated articles
        # ------------------------------------------------------------------
        self._progress["step"] = "rechunk_articles"

        try:
            article_ids = [a["id"] for a in all_articles if "id" in a]
            chunks_created: int = await workflow.execute_activity(
                "rechunk_articles_activity",
                args=[article_ids],
                task_queue=TaskQueues.KB_SYNC.value,
                start_to_close_timeout=timedelta(minutes=30),
                heartbeat_timeout=timedelta(minutes=5),
                retry_policy=_DB_RETRY,
            )
            result.chunks_created = chunks_created
            self._progress["chunks_created"] = chunks_created
        except Exception as e:
            workflow.logger.error(f"Failed to rechunk articles: {e}")
            result.success = False
            result.errors.append(f"rechunk_articles failed: {e}")
            return self._to_dict(result)

        # ------------------------------------------------------------------
        # Step 5: Reembed affected chunks
        # ------------------------------------------------------------------
        self._progress["step"] = "reembed_chunks"

        try:
            embeddings_count: int = await workflow.execute_activity(
                "reembed_chunks_activity",
                args=[article_ids],
                task_queue=TaskQueues.KB_SYNC.value,
                start_to_close_timeout=timedelta(hours=2),
                heartbeat_timeout=timedelta(minutes=10),
                retry_policy=_EMBED_RETRY,
            )
            result.embeddings_generated = embeddings_count
            self._progress["embeddings_generated"] = embeddings_count
        except Exception as e:
            workflow.logger.error(f"Failed to reembed chunks: {e}")
            result.success = False
            result.errors.append(f"reembed_chunks failed: {e}")
            return self._to_dict(result)

        # ------------------------------------------------------------------
        # Step 6: Notify sync complete
        # ------------------------------------------------------------------
        self._progress["step"] = "notify"

        try:
            await workflow.execute_activity(
                "notify_sync_complete_activity",
                args=[self._to_dict(result)],
                task_queue=TaskQueues.KB_SYNC.value,
                start_to_close_timeout=timedelta(seconds=30),
                retry_policy=RetryPolicy(maximum_attempts=2),
            )
        except Exception as e:
            # Notification failure is non-fatal
            workflow.logger.warning(f"Notification failed (non-fatal): {e}")

        self._progress["step"] = "complete"
        workflow.logger.info(
            f"KB sync complete: {result.acts_updated} acts, "
            f"{result.articles_upserted} articles, "
            f"{result.chunks_created} chunks, "
            f"{result.embeddings_generated} embeddings"
        )

        return self._to_dict(result)

    @staticmethod
    def _to_dict(result: KBSyncResult) -> dict[str, Any]:
        """Convert KBSyncResult to a plain dict for Temporal serialization."""
        return {
            "acts_checked": result.acts_checked,
            "acts_updated": result.acts_updated,
            "articles_upserted": result.articles_upserted,
            "chunks_created": result.chunks_created,
            "embeddings_generated": result.embeddings_generated,
            "errors": result.errors,
            "success": result.success,
        }
