"""Temporal task queue definitions for LEXE KB sync."""

from enum import Enum


class TaskQueues(str, Enum):
    """Task queue names for LEXE KB Temporal workers.

    Single queue for KB sync operations. Can be expanded
    if independent scaling is needed for specific activity types.
    """

    KB_SYNC = "lexe-kb-sync"
