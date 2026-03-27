"""LEXE KB Temporal Worker.

Entrypoint for running the Temporal worker that processes KB sync workflows.

Usage:
    # Run worker
    python -m lexe_api.temporal.worker

    # With custom Temporal address
    python -m lexe_api.temporal.worker --address localhost:7233
"""

import asyncio
import logging
import signal
import sys
from argparse import ArgumentParser

from temporalio.client import Client
from temporalio.worker import Worker

from lexe_api.temporal.activities.kb_fetch import (
    fetch_act_articles_activity,
    fetch_updated_acts_activity,
)
from lexe_api.temporal.activities.kb_process import (
    notify_sync_complete_activity,
    rechunk_articles_activity,
    reembed_chunks_activity,
    upsert_normativa_activity,
)
from lexe_api.temporal.task_queues import TaskQueues
from lexe_api.temporal.workflows.kb_sync import KBSyncWorkflow

logger = logging.getLogger(__name__)

# All activities registered on the single KB_SYNC queue
ALL_ACTIVITIES = [
    fetch_updated_acts_activity,
    fetch_act_articles_activity,
    upsert_normativa_activity,
    rechunk_articles_activity,
    reembed_chunks_activity,
    notify_sync_complete_activity,
]

ALL_WORKFLOWS = [KBSyncWorkflow]


async def run_worker(
    address: str = "lexe-temporal:7233",
    namespace: str = "default",
) -> None:
    """Run the LEXE KB Temporal worker.

    Args:
        address: Temporal server address.
        namespace: Temporal namespace.
    """
    logger.info(f"Connecting to Temporal at {address}, namespace: {namespace}")

    client = await Client.connect(address, namespace=namespace)

    worker = Worker(
        client,
        task_queue=TaskQueues.KB_SYNC.value,
        workflows=ALL_WORKFLOWS,
        activities=ALL_ACTIVITIES,
        max_concurrent_workflow_tasks=10,
        max_concurrent_activities=20,
    )

    # Setup graceful shutdown
    shutdown_event = asyncio.Event()

    def signal_handler() -> None:
        logger.info("Received shutdown signal")
        shutdown_event.set()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            loop.add_signal_handler(sig, signal_handler)
        except NotImplementedError:
            # Windows doesn't support add_signal_handler
            pass

    logger.info(f"Starting KB sync worker on queue '{TaskQueues.KB_SYNC.value}'...")

    worker_task = asyncio.create_task(worker.run())

    # Wait for shutdown signal
    await shutdown_event.wait()

    logger.info("Shutting down worker...")
    worker_task.cancel()

    try:
        await worker_task
    except asyncio.CancelledError:
        pass

    logger.info("Worker shut down gracefully")


def main() -> None:
    """Main entrypoint for the worker."""
    parser = ArgumentParser(description="LEXE KB Temporal Worker")
    parser.add_argument(
        "--address",
        default="lexe-temporal:7233",
        help="Temporal server address",
    )
    parser.add_argument(
        "--namespace",
        default="default",
        help="Temporal namespace",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Log level",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    asyncio.run(
        run_worker(
            address=args.address,
            namespace=args.namespace,
        )
    )


if __name__ == "__main__":
    main()
