"""LEXE KB Temporal Client.

Provides a high-level interface for scheduling and triggering
KB nightly sync workflows via Temporal.
"""

import logging
import os
from datetime import datetime, timedelta

from temporalio.client import (
    Client,
    Schedule,
    ScheduleActionStartWorkflow,
    ScheduleCalendarSpec,
    ScheduleIntervalSpec,
    ScheduleRange,
    ScheduleSpec,
    WorkflowHandle,
)
from temporalio.common import RetryPolicy

from lexe_api.temporal.task_queues import TaskQueues

logger = logging.getLogger(__name__)


class TemporalConfig:
    """Temporal connection configuration."""

    def __init__(
        self,
        address: str = "lexe-temporal:7233",
        namespace: str = "default",
        tls: bool = False,
    ) -> None:
        self.address = address
        self.namespace = namespace
        self.tls = tls

    @classmethod
    def from_env(cls) -> "TemporalConfig":
        """Create configuration from environment variables."""
        return cls(
            address=os.getenv("TEMPORAL_HOST", "lexe-temporal:7233"),
            namespace=os.getenv("TEMPORAL_NAMESPACE", "default"),
            tls=os.getenv("TEMPORAL_TLS", "false").lower() == "true",
        )


class TemporalClient:
    """High-level Temporal client for LEXE KB operations.

    Provides methods for scheduling nightly KB sync and
    triggering manual sync runs.

    Example:
        ```python
        client = await TemporalClient.connect()

        # Schedule nightly sync at 02:00 UTC
        await client.schedule_kb_sync()

        # Manual trigger
        handle = await client.trigger_kb_sync(collections=["codici", "leggi"])
        result = await handle.result()
        ```
    """

    def __init__(self, client: Client, config: TemporalConfig) -> None:
        self._client = client
        self._config = config

    @classmethod
    async def connect(cls, config: TemporalConfig | None = None) -> "TemporalClient":
        """Connect to Temporal server.

        Args:
            config: Optional configuration. Uses environment if not provided.

        Returns:
            Connected TemporalClient instance.
        """
        if config is None:
            config = TemporalConfig.from_env()

        logger.info(f"Connecting to Temporal at {config.address}")

        client = await Client.connect(
            config.address,
            namespace=config.namespace,
            tls=config.tls,
        )

        logger.info(f"Connected to Temporal namespace: {config.namespace}")
        return cls(client, config)

    async def close(self) -> None:
        """Close the Temporal client connection."""
        logger.info("Temporal client closed")

    @property
    def native_client(self) -> Client:
        """Get the underlying Temporal client."""
        return self._client

    async def schedule_kb_sync(
        self,
        schedule_id: str = "lexe-kb-nightly-sync",
        interval_hours: int = 24,
        collections: list[str] | None = None,
    ) -> str:
        """Create or update the nightly KB sync schedule.

        Args:
            schedule_id: Unique schedule identifier.
            interval_hours: Hours between sync runs (default 24).
            collections: Optional list of collections to sync.
                         If None, syncs all collections.

        Returns:
            The schedule ID.
        """
        from lexe_api.temporal.workflows.kb_sync import KBSyncWorkflow

        params = {
            "collections": collections or [],
            "full_sync": False,
            "triggered_by": "schedule",
        }

        try:
            await self._client.create_schedule(
                schedule_id,
                Schedule(
                    action=ScheduleActionStartWorkflow(
                        KBSyncWorkflow.run,
                        args=[params],
                        id=f"kb-sync-scheduled",
                        task_queue=TaskQueues.KB_SYNC.value,
                        retry_policy=RetryPolicy(
                            initial_interval=timedelta(seconds=10),
                            backoff_coefficient=2.0,
                            maximum_interval=timedelta(minutes=30),
                            maximum_attempts=3,
                        ),
                    ),
                    spec=ScheduleSpec(
                        intervals=[
                            ScheduleIntervalSpec(every=timedelta(hours=interval_hours)),
                        ],
                    ),
                ),
            )
            logger.info(
                f"Created KB sync schedule '{schedule_id}' "
                f"every {interval_hours}h"
            )
        except Exception as e:
            if "already exists" in str(e).lower():
                logger.info(f"Schedule '{schedule_id}' already exists, skipping creation")
            else:
                raise

        return schedule_id

    async def schedule_massime_enrich(
        self,
        schedule_id: str = "lexe-massime-weekly-enrich",
        batch_size: int = 500,
    ) -> str:
        """Create or update the weekly massime enrichment schedule.

        Runs Sunday 02:00 UTC — low-traffic window after the Saturday
        KB nightly sync (03:00 UTC). See MassimeEnrichWorkflow docstring.

        Args:
            schedule_id: Unique schedule identifier.
            batch_size: Rows per activity batch (default 500).

        Returns:
            The schedule ID.
        """
        from lexe_api.temporal.workflows.massime_enrich import (
            MassimeEnrichParams,
            MassimeEnrichWorkflow,
        )

        params = MassimeEnrichParams(
            batch_size=batch_size,
            triggered_by="schedule",
        )

        try:
            await self._client.create_schedule(
                schedule_id,
                Schedule(
                    action=ScheduleActionStartWorkflow(
                        MassimeEnrichWorkflow.run,
                        args=[params],
                        id="massime-enrich-scheduled",
                        task_queue=TaskQueues.KB_SYNC.value,
                        retry_policy=RetryPolicy(
                            initial_interval=timedelta(seconds=10),
                            backoff_coefficient=2.0,
                            maximum_interval=timedelta(minutes=30),
                            maximum_attempts=3,
                        ),
                    ),
                    spec=ScheduleSpec(
                        # cron "0 2 * * 0" — Sunday 02:00 UTC
                        calendars=[
                            ScheduleCalendarSpec(
                                hour=(ScheduleRange(start=2, end=2, step=1),),
                                minute=(ScheduleRange(start=0, end=0, step=1),),
                                day_of_week=(ScheduleRange(start=0, end=0, step=1),),
                            ),
                        ],
                    ),
                ),
            )
            logger.info(
                f"Created massime enrich schedule '{schedule_id}' "
                f"(Sunday 02:00 UTC, batch_size={batch_size})"
            )
        except Exception as e:
            if "already exists" in str(e).lower():
                logger.info(
                    f"Schedule '{schedule_id}' already exists, skipping creation"
                )
            else:
                raise

        return schedule_id

    async def trigger_massime_enrich(
        self,
        batch_size: int = 500,
        max_iterations: int = 200,
    ) -> WorkflowHandle:
        """Manually trigger a massime enrichment workflow run.

        Args:
            batch_size: Rows per activity batch.
            max_iterations: Safety cap on total iterations.

        Returns:
            Workflow handle for monitoring/cancel.
        """
        from lexe_api.temporal.workflows.massime_enrich import (
            MassimeEnrichParams,
            MassimeEnrichWorkflow,
        )

        params = MassimeEnrichParams(
            batch_size=batch_size,
            max_iterations=max_iterations,
            triggered_by="manual",
        )
        handle = await self._client.start_workflow(
            MassimeEnrichWorkflow.run,
            args=[params],
            id=f"massime-enrich-manual-{int(datetime.now().timestamp())}",
            task_queue=TaskQueues.KB_SYNC.value,
        )
        logger.info(f"Triggered massime enrich workflow: {handle.id}")
        return handle

    async def trigger_kb_sync(
        self,
        collections: list[str] | None = None,
        full_sync: bool = False,
    ) -> WorkflowHandle:
        """Manually trigger a KB sync workflow.

        Args:
            collections: Optional list of collections to sync.
                         If None, syncs all collections.
            full_sync: If True, re-syncs everything regardless of last sync time.

        Returns:
            WorkflowHandle for tracking the workflow.
        """
        from lexe_api.temporal.workflows.kb_sync import KBSyncWorkflow

        import hashlib
        from datetime import UTC, datetime

        # Deterministic workflow ID for idempotency within the same minute
        ts = datetime.now(UTC).strftime("%Y%m%d-%H%M")
        suffix = hashlib.sha256(ts.encode()).hexdigest()[:8]
        workflow_id = f"kb-sync-manual-{suffix}"

        params = {
            "collections": collections or [],
            "full_sync": full_sync,
            "triggered_by": "manual",
        }

        handle = await self._client.start_workflow(
            KBSyncWorkflow.run,
            args=[params],
            id=workflow_id,
            task_queue=TaskQueues.KB_SYNC.value,
            retry_policy=RetryPolicy(
                initial_interval=timedelta(seconds=10),
                backoff_coefficient=2.0,
                maximum_interval=timedelta(minutes=30),
                maximum_attempts=3,
            ),
        )

        logger.info(f"Triggered manual KB sync: {workflow_id}")
        return handle


# ---------------------------------------------------------------------------
# Global client singleton
# ---------------------------------------------------------------------------

_temporal_client: TemporalClient | None = None


async def get_temporal_client() -> TemporalClient:
    """Get or create the global Temporal client."""
    global _temporal_client
    if _temporal_client is None:
        _temporal_client = await TemporalClient.connect()
    return _temporal_client


async def close_temporal_client() -> None:
    """Close the global Temporal client."""
    global _temporal_client
    if _temporal_client is not None:
        await _temporal_client.close()
        _temporal_client = None
