"""PostgreSQL-backed checkpoint store for experiment orchestration.

This module exposes a concurrency-safe API to register and claim experiment
runs while keeping a durable, single source of truth that multiple workers can
share. PostgreSQL is used as the storage backend so that jobs can be safely
coordinated across machines.
"""

from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from typing import (
    Awaitable,
    Callable,
    Deque,
    Dict,
    Iterable,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
)

from rclpy.node import Node

import psycopg
from psycopg.rows import dict_row
from psycopg import sql


@dataclass(frozen=True)
class ExperimentJobDescriptor:
    """Descriptor for a unique experiment configuration."""

    algorithm: str
    experiment_identifier: str
    experiment_tag: str
    run_id: int


@dataclass
class ExperimentJob:
    """Concrete job persisted in the database."""

    id: int
    algorithm: str
    experiment_identifier: str
    experiment_tag: str
    run_id: int
    status: str
    result: Optional[str]
    worker_id: Optional[str]
    updated_at: str

    def descriptor(self) -> ExperimentJobDescriptor:
        return ExperimentJobDescriptor(
            algorithm=self.algorithm,
            experiment_identifier=self.experiment_identifier,
            experiment_tag=self.experiment_tag,
            run_id=self.run_id,
        )


@dataclass(frozen=True)
class ExperimentJobHandle:
    """In-memory handle that identifies a job to execute."""

    job_id: Optional[int]
    experiment_tag: str
    run_id: int


ActiveJobInterruptedCallback = Callable[[ExperimentJobHandle, str], Awaitable[None]]
CompletedRunsLoader = Callable[[], Set[Tuple[str, int]]]


class ExperimentCheckpointStore:
    """PostgreSQL-backed store that tracks experiment job state."""

    def __init__(
        self,
        dsn: str,
        *,
        connect_timeout_sec: float = 5.0,
    ) -> None:
        if not dsn:
            raise ValueError(
                "A PostgreSQL DSN must be provided to initialise the checkpoint store"
            )
        self._dsn = dsn
        self._connect_timeout = int(max(connect_timeout_sec, 0.0))
        self._ensure_schema()

    # --- public API -----------------------------------------------------
    def register_jobs(self, jobs: Iterable[ExperimentJobDescriptor]) -> None:
        """Insert job descriptors into the database if they are missing."""

        rows = [
            {
                "algorithm": descriptor.algorithm,
                "experiment_identifier": descriptor.experiment_identifier,
                "experiment_tag": descriptor.experiment_tag,
                "run_id": descriptor.run_id,
                "ts": self._timestamp(),
            }
            for descriptor in jobs
        ]
        if not rows:
            return
        with psycopg.connect(
            self._dsn,
            autocommit=True,
            connect_timeout=self._connect_timeout,
        ) as conn:
            with conn.cursor() as cur:
                cur.executemany(
                    """
                    INSERT INTO jobs (
                        algorithm,
                        experiment_identifier,
                        experiment_tag,
                        run_id,
                        status,
                        created_at,
                        updated_at,
                        heartbeat_at
                    ) VALUES (%(algorithm)s, %(experiment_identifier)s, %(experiment_tag)s, %(run_id)s,
                              'PENDING', %(ts)s, %(ts)s, NULL)
                    ON CONFLICT (algorithm, experiment_identifier, experiment_tag, run_id)
                    DO NOTHING
                    """,
                    rows,
                )

    def recover_stale_jobs(self, *, worker_hint: Optional[str] = None) -> int:
        """Mark jobs that were running as failed to recover from crashes.

        Args:
            worker_hint: Optional identifier added to recovery logs.

        Returns:
            Number of jobs transitioned from RUNNING to FAILED.
        """

        recovery_note = "Recovered after unclean shutdown"
        if worker_hint:
            recovery_note += f" ({worker_hint})"
        with psycopg.connect(
            self._dsn,
            autocommit=True,
            connect_timeout=self._connect_timeout,
        ) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE jobs
                       SET status = 'FAILED',
                           updated_at = %(ts)s,
                           result = COALESCE(result, %(note)s),
                           heartbeat_at = NULL
                     WHERE status = 'RUNNING'
                    """,
                    {"ts": self._timestamp(), "note": recovery_note},
                )
                return cur.rowcount or 0

    def claim_next_job(
        self,
        *,
        algorithm: str,
        experiment_identifier: str,
        worker_id: str,
    ) -> Optional[ExperimentJob]:
        """Atomically claim the next PENDING job for execution."""

        with psycopg.connect(
            self._dsn,
            autocommit=False,
            connect_timeout=self._connect_timeout,
        ) as conn:
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute(
                    """
                    SELECT id, algorithm, experiment_identifier, experiment_tag, run_id,
                           status, result, worker_id, updated_at
                      FROM jobs
                     WHERE algorithm = %(algorithm)s
                       AND experiment_identifier = %(experiment_identifier)s
                       AND status = 'PENDING'
                  ORDER BY run_id, id
                     FOR UPDATE SKIP LOCKED
                     LIMIT 1
                    """,
                    {
                        "algorithm": algorithm,
                        "experiment_identifier": experiment_identifier,
                    },
                )
                row = cur.fetchone()
                if row is None:
                    conn.rollback()
                    return None
                job_id = row["id"]
                now_ts = self._timestamp()
                cur.execute(
                    """
                    UPDATE jobs
                       SET status = 'RUNNING',
                           worker_id = %(worker_id)s,
                           updated_at = %(ts)s,
                           heartbeat_at = %(ts)s
                     WHERE id = %(job_id)s
                    """,
                    {"worker_id": worker_id, "ts": now_ts, "job_id": job_id},
                )
                conn.commit()
        row.update({"status": "RUNNING", "worker_id": worker_id, "updated_at": now_ts})
        return ExperimentJob(**row)

    def get_job(self, descriptor: ExperimentJobDescriptor) -> Optional[ExperimentJob]:
        params: Dict[str, object] = {
            "algorithm": descriptor.algorithm,
            "experiment_identifier": descriptor.experiment_identifier,
            "experiment_tag": descriptor.experiment_tag,
            "run_id": descriptor.run_id,
        }
        with psycopg.connect(
            self._dsn,
            autocommit=True,
            connect_timeout=self._connect_timeout,
        ) as conn:
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute(
                    """
                    SELECT id, algorithm, experiment_identifier, experiment_tag, run_id,
                           status, result, worker_id, updated_at
                      FROM jobs
                     WHERE algorithm = %(algorithm)s
                       AND experiment_identifier = %(experiment_identifier)s
                       AND experiment_tag = %(experiment_tag)s
                       AND run_id = %(run_id)s
                    """,
                    params,
                )
                row = cur.fetchone()
        if row is None:
            return None
        return ExperimentJob(**row)

    def has_incomplete_jobs(
        self,
        *,
        algorithm: str,
        experiment_identifier: str,
        include_running: bool = True,
    ) -> bool:
        """Return True when any job still needs work.

        Args:
            algorithm: Algorithm identifier.
            experiment_identifier: Experiment identifier.
            include_running: When True (default), treat RUNNING jobs as incomplete.

        Returns:
            True if a matching job has status PENDING (or RUNNING when requested).
        """

        statuses: Sequence[str]
        if include_running:
            statuses = ("PENDING", "RUNNING", "FAILED")
        else:
            statuses = ("PENDING", "FAILED")
        with psycopg.connect(
            self._dsn,
            autocommit=True,
            connect_timeout=self._connect_timeout,
        ) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT 1
                      FROM jobs
                     WHERE algorithm = %(algorithm)s
                       AND experiment_identifier = %(experiment_identifier)s
                       AND status = ANY(%(statuses)s)
                     LIMIT 1
                    """,
                    {
                        "algorithm": algorithm,
                        "experiment_identifier": experiment_identifier,
                        "statuses": list(statuses),
                    },
                )
                return cur.fetchone() is not None

    def refresh_job_heartbeat(self, job_id: int, worker_id: str) -> bool:
        """Update the heartbeat timestamp for a RUNNING job.

        Returns True when the heartbeat was updated; False if the job is no longer
        owned by the worker (e.g. requeued or finished).
        """

        now_ts = self._timestamp()
        with psycopg.connect(
            self._dsn,
            autocommit=True,
            connect_timeout=self._connect_timeout,
        ) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE jobs
                       SET heartbeat_at = %(ts)s,
                           updated_at = %(ts)s
                     WHERE id = %(job_id)s
                       AND worker_id = %(worker_id)s
                       AND status = 'RUNNING'
                    """,
                    {"ts": now_ts, "job_id": job_id, "worker_id": worker_id},
                )
                return cur.rowcount == 1

    def requeue_stale_jobs(
        self,
        *,
        algorithm: str,
        experiment_identifier: str,
        heartbeat_timeout_sec: float,
    ) -> int:
        """Move RUNNING jobs without recent heartbeats back to PENDING."""

        timeout = max(heartbeat_timeout_sec, 0.0)
        threshold_dt = datetime.now(timezone.utc) - timedelta(seconds=timeout)
        threshold = threshold_dt.isoformat()
        note = (
            f"Requeued after missing heartbeat for {timeout:.1f}s"
            if timeout > 0
            else "Requeued after missing heartbeat"
        )
        with psycopg.connect(
            self._dsn,
            autocommit=True,
            connect_timeout=self._connect_timeout,
        ) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE jobs
                       SET status = 'PENDING',
                           worker_id = NULL,
                           result = COALESCE(result, %(note)s),
                           updated_at = %(ts)s,
                           heartbeat_at = NULL
                     WHERE algorithm = %(algorithm)s
                       AND experiment_identifier = %(experiment_identifier)s
                       AND status = 'RUNNING'
                       AND (heartbeat_at IS NULL OR heartbeat_at < %(threshold)s)
                    """,
                    {
                        "algorithm": algorithm,
                        "experiment_identifier": experiment_identifier,
                        "threshold": threshold,
                        "note": note,
                        "ts": self._timestamp(),
                    },
                )
                return cur.rowcount or 0

    def force_requeue_job(
        self,
        job_id: int,
        *,
        reason: Optional[str] = None,
        expected_worker_id: Optional[str] = None,
    ) -> bool:
        """Force a RUNNING job back to PENDING immediately.

        Args:
            job_id: Identifier of the job to requeue.
            reason: Optional note to persist as the job result.
            expected_worker_id: When provided, only requeue the job if the
                worker currently assigned matches this value.

        Returns:
            True when the job was transitioned to PENDING, False otherwise.
        """

        note = reason or "Requeued by watchdog request"
        params: Dict[str, object] = {
            "job_id": job_id,
            "note": note,
            "ts": self._timestamp(),
        }
        worker_clause = ""
        if expected_worker_id is not None:
            params["worker_id"] = expected_worker_id
            worker_clause = " AND worker_id = %(worker_id)s"
        query = (
            """
            UPDATE jobs
               SET status = 'PENDING',
                   worker_id = NULL,
                   result = COALESCE(result, %(note)s),
                   updated_at = %(ts)s,
                   heartbeat_at = NULL
             WHERE id = %(job_id)s
               AND status = 'RUNNING'
            """
            + worker_clause
        )
        with psycopg.connect(
            self._dsn,
            autocommit=True,
            connect_timeout=self._connect_timeout,
        ) as conn:
            with conn.cursor() as cur:
                cur.execute(query, params)
                return cur.rowcount == 1

    def mark_job_done(self, job_id: int, result: str) -> None:
        """Mark a job as successfully completed."""

        with psycopg.connect(
            self._dsn,
            autocommit=True,
            connect_timeout=self._connect_timeout,
        ) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE jobs
                       SET status = 'DONE',
                           result = %(result)s,
                           updated_at = %(ts)s,
                           heartbeat_at = NULL
                     WHERE id = %(job_id)s
                    """,
                    {"result": result, "ts": self._timestamp(), "job_id": job_id},
                )

    def mark_job_failed(self, job_id: int, error_message: str) -> None:
        """Mark a job as failed with the given error description."""

        with psycopg.connect(
            self._dsn,
            autocommit=True,
            connect_timeout=self._connect_timeout,
        ) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE jobs
                       SET status = 'FAILED',
                           result = %(msg)s,
                           updated_at = %(ts)s,
                           heartbeat_at = NULL
                     WHERE id = %(job_id)s
                    """,
                    {"msg": error_message, "ts": self._timestamp(), "job_id": job_id},
                )

    def list_jobs(
        self,
        *,
        algorithm: str,
        experiment_identifier: str,
        statuses: Optional[Sequence[str]] = None,
    ) -> List[ExperimentJob]:
        query = """
            SELECT id, algorithm, experiment_identifier, experiment_tag, run_id,
                   status, result, worker_id, updated_at
              FROM jobs
             WHERE algorithm = %(algorithm)s
               AND experiment_identifier = %(experiment_identifier)s
        """
        params: Dict[str, object] = {
            "algorithm": algorithm,
            "experiment_identifier": experiment_identifier,
        }
        if statuses:
            query += " AND status = ANY(%(statuses)s)"
            params["statuses"] = list(statuses)
        query += " ORDER BY run_id, id"
        with psycopg.connect(
            self._dsn,
            autocommit=True,
            connect_timeout=self._connect_timeout,
        ) as conn:
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute(sql.SQL(query), params)
                rows = cur.fetchall()
        return [ExperimentJob(**row) for row in rows]

    def reset_jobs(
        self,
        *,
        algorithm: str,
        experiment_identifier: str,
    ) -> int:
        """Reset all jobs to PENDING and clear results for a given descriptor."""

        with psycopg.connect(
            self._dsn,
            autocommit=True,
            connect_timeout=self._connect_timeout,
        ) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE jobs
                       SET status = 'PENDING',
                           result = NULL,
                           worker_id = NULL,
                                                     updated_at = %(ts)s,
                                                     heartbeat_at = NULL
                     WHERE algorithm = %(algorithm)s
                       AND experiment_identifier = %(experiment_identifier)s
                    """,
                    {
                        "algorithm": algorithm,
                        "experiment_identifier": experiment_identifier,
                        "ts": self._timestamp(),
                    },
                )
                return cur.rowcount or 0

    def requeue_failed_jobs(
        self,
        *,
        algorithm: Optional[str] = None,
        experiment_identifier: Optional[str] = None,
    ) -> int:
        """Mark FAILED jobs as PENDING so they can be re-run."""

        clauses: List[sql.SQL] = [sql.SQL("status = 'FAILED'")]
        params: Dict[str, object] = {"ts": self._timestamp()}
        if algorithm is not None:
            clauses.append(sql.SQL("algorithm = %(algorithm)s"))
            params["algorithm"] = algorithm
        if experiment_identifier is not None:
            clauses.append(sql.SQL("experiment_identifier = %(experiment_identifier)s"))
            params["experiment_identifier"] = experiment_identifier
        where_clause = sql.SQL(" AND ").join(clauses)

        query = sql.SQL(
            """
            UPDATE jobs
               SET status = 'PENDING',
                   result = NULL,
                   worker_id = NULL,
                   updated_at = %(ts)s,
                   heartbeat_at = NULL
             WHERE {}
        """
        ).format(where_clause)

        with psycopg.connect(
            self._dsn,
            autocommit=True,
            connect_timeout=self._connect_timeout,
        ) as conn:
            with conn.cursor() as cur:
                cur.execute(query, params)
                return cur.rowcount or 0

    # --- internals --------------------------------------------------
    def _ensure_schema(self) -> None:
        with psycopg.connect(
            self._dsn,
            autocommit=True,
            connect_timeout=self._connect_timeout,
        ) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS jobs (
                        id SERIAL PRIMARY KEY,
                        algorithm TEXT NOT NULL,
                        experiment_identifier TEXT NOT NULL,
                        experiment_tag TEXT NOT NULL,
                        run_id INTEGER NOT NULL,
                        status TEXT NOT NULL CHECK (status IN ('PENDING', 'RUNNING', 'DONE', 'FAILED')),
                        result TEXT,
                        worker_id TEXT,
                        created_at TIMESTAMPTZ NOT NULL,
                        updated_at TIMESTAMPTZ NOT NULL,
                        heartbeat_at TIMESTAMPTZ,
                        UNIQUE (algorithm, experiment_identifier, experiment_tag, run_id)
                    )
                    """,
                )
                cur.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_jobs_pending
                        ON jobs (algorithm, experiment_identifier, run_id)
                     WHERE status = 'PENDING'
                    """,
                )
                cur.execute(
                    """
                    ALTER TABLE jobs
                    ADD COLUMN IF NOT EXISTS heartbeat_at TIMESTAMPTZ
                    """,
                )
                cur.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_jobs_running_heartbeat
                        ON jobs (algorithm, experiment_identifier, heartbeat_at)
                     WHERE status = 'RUNNING'
                    """,
                )

    @staticmethod
    def _timestamp() -> str:
        return datetime.now(timezone.utc).isoformat()


class ExperimentJobCoordinator:
    """Coordinate experiment job lifecycle with optional checkpointing."""

    def __init__(
        self,
        *,
        node: Node,
        algorithm_name: str,
        experiment_identifier: str,
        worker_id: str,
        checkpoint_dsn: Optional[str],
        checkpoint_connect_timeout_sec: float,
        heartbeat_interval_sec: float,
        heartbeat_timeout_sec: float,
        resume_checkpoint: bool,
        auto_recover_running: bool,
        completed_runs_loader: Optional[CompletedRunsLoader] = None,
        on_active_job_interrupted: Optional[ActiveJobInterruptedCallback] = None,
    ) -> None:
        self._node = node
        self._logger = node.get_logger()
        self._algorithm_name = algorithm_name
        self._experiment_identifier = experiment_identifier
        self._worker_id = worker_id
        self._heartbeat_interval = max(heartbeat_interval_sec, 1.0)
        self._heartbeat_timeout = max(
            heartbeat_timeout_sec, self._heartbeat_interval * 1.5
        )
        self._resume_checkpoint = resume_checkpoint
        self._auto_recover_running = auto_recover_running
        self._completed_runs_loader = completed_runs_loader
        self._on_active_job_interrupted = on_active_job_interrupted

        self._asyncio_loop: Optional[asyncio.AbstractEventLoop] = None
        self._heartbeat_task: Optional[asyncio.Task[None]] = None
        self._heartbeat_lost = False

        self._store: Optional[ExperimentCheckpointStore] = None
        self._local_job_queue: Deque[Tuple[str, int]] = deque()
        self._active_handle: Optional[ExperimentJobHandle] = None

        if checkpoint_dsn:
            self._store = ExperimentCheckpointStore(
                checkpoint_dsn,
                connect_timeout_sec=checkpoint_connect_timeout_sec,
            )
            self._logger.info(
                "Checkpoint database DSN configured for experiment checkpoints"
            )
            if self._auto_recover_running:
                try:
                    recovered = self._store.recover_stale_jobs(
                        worker_hint=self._worker_id
                    )
                    if recovered:
                        self._logger.warning(
                            f"Recovered {recovered} previously running job(s)"
                        )
                except Exception as exc:  # noqa: BLE001
                    self._logger.error(
                        f"Unable to recover running jobs from checkpoint store: {exc}"
                    )
        else:
            self._logger.info(
                "Checkpoint store disabled; experiments will run sequentially without database persistence"
            )

    def set_asyncio_loop(self, loop: Optional[asyncio.AbstractEventLoop]) -> None:
        self._asyncio_loop = loop

    @property
    def uses_checkpoint_store(self) -> bool:
        return self._store is not None

    @property
    def checkpoint_store(self) -> Optional[ExperimentCheckpointStore]:
        return self._store

    @property
    def heartbeat_interval(self) -> float:
        return self._heartbeat_interval

    @property
    def heartbeat_lost(self) -> bool:
        return self._heartbeat_lost

    def configure_jobs(
        self,
        *,
        experiment_tags: Iterable[str],
        repetitions: int,
    ) -> None:
        tags = list(experiment_tags)
        if self._store is not None:
            descriptors: List[ExperimentJobDescriptor] = [
                ExperimentJobDescriptor(
                    algorithm=self._algorithm_name,
                    experiment_identifier=self._experiment_identifier,
                    experiment_tag=tag,
                    run_id=run_id,
                )
                for tag in tags
                for run_id in range(1, repetitions + 1)
            ]
            self._store.register_jobs(descriptors)
            if not self._resume_checkpoint:
                try:
                    reset_count = self._store.reset_jobs(
                        algorithm=self._algorithm_name,
                        experiment_identifier=self._experiment_identifier,
                    )
                    if reset_count:
                        self._logger.info(
                            f"Reset {reset_count} job(s) to PENDING for a fresh run"
                        )
                except Exception as exc:  # noqa: BLE001
                    self._logger.error(
                        f"Unable to reset jobs in checkpoint store: {exc}"
                    )
        else:
            local_jobs: List[Tuple[str, int]] = [
                (tag, run_id) for tag in tags for run_id in range(1, repetitions + 1)
            ]
            if self._resume_checkpoint:
                skipped = self._prune_completed_local_jobs(local_jobs)
                if skipped:
                    self._logger.info(
                        f"Skipping {skipped} completed run(s) based on existing outcomes"
                    )
            self._local_job_queue = deque(local_jobs)

    def resubmit_failed_jobs(self) -> None:
        if self._store is None:
            return
        requeue_failed = getattr(self._store, "requeue_failed_jobs", None)
        if not callable(requeue_failed):
            return
        try:
            recovered_failed = requeue_failed(
                algorithm=self._algorithm_name,
                experiment_identifier=self._experiment_identifier,
            )
            if recovered_failed:
                self._logger.warning(
                    f"Requeued {recovered_failed} FAILED job(s) for retry"
                )
        except Exception as exc:  # noqa: BLE001
            self._logger.error(f"Unable to requeue FAILED jobs before dispatch: {exc}")

    def prepare_for_dispatch(self) -> None:
        if self._store is None:
            return
        try:
            requeued = self._store.requeue_stale_jobs(
                algorithm=self._algorithm_name,
                experiment_identifier=self._experiment_identifier,
                heartbeat_timeout_sec=self._heartbeat_timeout,
            )
            if requeued:
                self._logger.warning(
                    f"Requeued {requeued} stale RUNNING job(s) lacking heartbeats"
                )
        except Exception as exc:  # noqa: BLE001
            self._logger.error(f"Failed to refresh stale jobs: {exc}")

    async def on_watchdog_tick(self) -> None:
        self.prepare_for_dispatch()

    def claim_next_job(self) -> Optional[ExperimentJobHandle]:
        if self._store is None:
            if not self._local_job_queue:
                return None
            experiment_tag, run_id = self._local_job_queue.popleft()
            return ExperimentJobHandle(
                job_id=None,
                experiment_tag=experiment_tag,
                run_id=run_id,
            )
        job = self._store.claim_next_job(
            algorithm=self._algorithm_name,
            experiment_identifier=self._experiment_identifier,
            worker_id=self._worker_id,
        )
        if job is None:
            return None
        return ExperimentJobHandle(
            job_id=job.id,
            experiment_tag=job.experiment_tag,
            run_id=job.run_id,
        )

    def start_job(self, handle: ExperimentJobHandle) -> None:
        self._heartbeat_lost = False
        self._active_handle = handle
        if handle.job_id is not None and self._store is not None:
            self._logger.info(
                f"Worker {self._worker_id} starting {handle.experiment_tag} run {handle.run_id}"
            )
            self._start_job_heartbeat(handle.job_id)
        else:
            self._logger.info(f"Starting {handle.experiment_tag} run {handle.run_id}")

    def complete_job_success(
        self,
        handle: ExperimentJobHandle,
        result_text: str,
    ) -> None:
        if self._store is None or handle.job_id is None:
            return
        if self._heartbeat_lost:
            self._logger.warning(
                f"Lost heartbeat for job {handle.job_id}; skipping result storage so another worker can retry"
            )
            return
        try:
            self._store.mark_job_done(handle.job_id, result_text)
        except Exception as exc:  # noqa: BLE001
            self._logger.error(f"Failed to mark job {handle.job_id} as done: {exc}")

    def complete_job_failure(
        self,
        handle: ExperimentJobHandle,
        error_message: str,
    ) -> None:
        if self._store is None or handle.job_id is None:
            return
        if self._heartbeat_lost:
            self._logger.warning("Skipping failure mark because job ownership was lost")
            return
        try:
            self._store.mark_job_failed(handle.job_id, error_message)
        except Exception as exc:  # noqa: BLE001
            self._logger.error(f"Failed to mark job {handle.job_id} as failed: {exc}")

    def handle_lost_heartbeat(self, handle: ExperimentJobHandle) -> None:
        if self._store is None:
            self._local_job_queue.appendleft((handle.experiment_tag, handle.run_id))

    async def handle_watchdog_missing_nodes(self, missing: Set[str]) -> None:
        reason = ", ".join(sorted(missing))
        handle = self._active_handle
        if self._store is not None and handle and handle.job_id is not None:
            try:
                requeued = self._store.force_requeue_job(
                    handle.job_id,
                    reason=f"Dependent modules unavailable: {reason}",
                    expected_worker_id=self._worker_id,
                )
                if requeued:
                    self._logger.warning(
                        f"Requeued job {handle.job_id} because required modules disappeared"
                    )
                else:
                    self._logger.warning(
                        f"Job {handle.job_id} could not be requeued; it may have already been released"
                    )
            except Exception as exc:  # noqa: BLE001
                self._logger.error(
                    f"Failed to requeue job {handle.job_id} after module loss: {exc}"
                )
        elif self._store is None and handle is not None:
            self._logger.warning(
                f"Local job {handle.experiment_tag} run {handle.run_id} interrupted because required modules disappeared"
            )
        else:
            self._logger.warning(
                "Module loss detected but no active job was held for requeue"
            )
        self._heartbeat_lost = True
        await self._stop_job_heartbeat()
        if handle is not None:
            self.handle_lost_heartbeat(handle)
        if self._on_active_job_interrupted is not None and handle is not None:
            try:
                await self._on_active_job_interrupted(handle, reason)
            except Exception as exc:  # noqa: BLE001
                self._logger.error(f"Watchdog recovery callback failed: {exc}")

    async def finalize_job(self, handle: ExperimentJobHandle) -> None:
        await self._stop_job_heartbeat()
        if self._active_handle == handle:
            self._active_handle = None

    async def shutdown(self) -> None:
        await self._stop_job_heartbeat()

    def handle_unsuccessful_job(self, handle: ExperimentJobHandle) -> None:
        if self._store is None:
            self._local_job_queue.appendleft((handle.experiment_tag, handle.run_id))

    def cleanup_on_exit(self) -> None:
        handle = self._active_handle
        if handle is None or handle.job_id is None or self._store is None:
            return
        try:
            recovery_message = f"Worker {self._worker_id} terminated before finishing job {handle.job_id}"
            self._store.mark_job_failed(handle.job_id, recovery_message)
        except Exception:  # noqa: BLE001
            # Avoid raising during interpreter shutdown; logging may be unavailable.
            pass
        finally:
            self._active_handle = None

    def _prune_completed_local_jobs(self, jobs: List[Tuple[str, int]]) -> int:
        completed = self._load_completed_runs_from_outcomes()
        if not completed:
            return 0
        before = len(jobs)
        jobs[:] = [job for job in jobs if job not in completed]
        return before - len(jobs)

    def _load_completed_runs_from_outcomes(self) -> Set[Tuple[str, int]]:
        if self._completed_runs_loader is None:
            return set()
        try:
            return set(self._completed_runs_loader())
        except Exception as exc:  # noqa: BLE001
            self._logger.warning(
                f"Unable to read outcomes file for resume logic: {exc}"
            )
            return set()

    def _start_job_heartbeat(self, job_id: int) -> None:
        if self._store is None:
            self._heartbeat_lost = False
            return
        if self._heartbeat_task is not None and not self._heartbeat_task.done():
            self._heartbeat_task.cancel()
        loop = self._resolve_loop()
        self._heartbeat_lost = False
        self._heartbeat_task = loop.create_task(self._heartbeat_loop(job_id))

    async def _stop_job_heartbeat(self) -> None:
        if self._heartbeat_task is None:
            return
        task = self._heartbeat_task
        self._heartbeat_task = None
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:  # pragma: no cover - expected flow
            pass

    def _resolve_loop(self) -> asyncio.AbstractEventLoop:
        if self._asyncio_loop is not None:
            return self._asyncio_loop
        return asyncio.get_running_loop()

    async def _heartbeat_loop(self, job_id: int) -> None:
        if self._store is None:
            return
        try:
            while True:
                await asyncio.sleep(self._heartbeat_interval)
                updated = self._store.refresh_job_heartbeat(
                    job_id,
                    self._worker_id,
                )
                if not updated:
                    self._heartbeat_lost = True
                    self._logger.warning(
                        f"Heartbeat rejected for job {job_id}; job will be requeued"
                    )
                    return
        except asyncio.CancelledError:  # pragma: no cover - cooperative cancel
            raise
        except Exception as exc:  # noqa: BLE001
            self._logger.error(f"Failed to refresh heartbeat for job {job_id}: {exc}")
