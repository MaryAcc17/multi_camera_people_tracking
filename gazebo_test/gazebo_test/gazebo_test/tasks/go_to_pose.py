from rclpy.node import Node
import rclpy
import rclpy.duration

from gazebo_test.utils.gazebo_env_handler import GazeboEnvironmentHandler
from gazebo_test.utils.evaluation_handler import ExperimentEvaluator, ExperimentResult
from gazebo_test.utils.hunav_handler import HunavEvaluatorHandler
from gazebo_test.utils.bag_recorder import BagRecorder
from gazebo_test.utils.common_utils import (
    parse_entity_state_yaml,
    get_posestamped_from_entity,
)
from gazebo_msgs.msg import EntityState
from geometry_msgs.msg import PoseStamped
from typing import Dict, Iterator, List, Optional, Set, Tuple
from pathlib import Path
import pandas as pd
import os
import socket
import traceback
import uuid
import fcntl
import hashlib
import re
import asyncio
from urllib.parse import urlparse

import atexit
from contextlib import contextmanager

from gazebo_test.utils.navigation_handler import NavigationHandler
from gazebo_test.utils.watchdog_monitor import ExperimentWatchdog

from ament_index_python.packages import get_package_share_directory
from rclpy.parameter import Parameter

from gazebo_test.utils.checkpoint_store import (
    ExperimentJobCoordinator,
    ExperimentJobHandle,
)


class ExperimentManager(Node):
    def __init__(self):
        super().__init__("experiment_manager")

        self._asyncio_loop: Optional[asyncio.AbstractEventLoop] = None

        declared_algorithm = self.declare_parameter(
            "algorithm_name", Parameter.Type.STRING
        ).value
        self.algorithm_name = str(declared_algorithm or "").strip() or "unknown"

        yaml_path_param = self.declare_parameter(
            "goals_and_poses_file",
            Parameter.Type.STRING,
        ).value

        if not yaml_path_param:
            raise ValueError("Parameter 'goals_and_poses_file' must be provided")
        yaml_path = str(yaml_path_param)

        base_path_value = self.declare_parameter(
            "base_path",
            "results/gazebo_test",
        ).value
        self._base_path_root = Path(
            str(base_path_value) or "results/gazebo_test"
        ).expanduser()
        self._experiment_identifier = Path(yaml_path).stem

        timeout_duration = float(
            self.declare_parameter("timeout_duration", Parameter.Type.DOUBLE).value  # type: ignore
        )

        self.use_recorder = bool(
            self.declare_parameter("use_recorder", Parameter.Type.BOOL).value
        )
        self.use_evaluator = bool(
            self.declare_parameter("use_evaluator", Parameter.Type.BOOL).value
        )
        self.repetitions = int(self.declare_parameter("repetitions", Parameter.Type.INTEGER).value)  # type: ignore
        self.record_maps = bool(
            self.declare_parameter("record_maps", Parameter.Type.BOOL).value
        )
        self.wait_before_start = int(
            self.declare_parameter("wait_before_start", Parameter.Type.INTEGER).value  # type: ignore
        )
        checkpoint_dsn_param = self.declare_parameter(
            "checkpoint_dsn",
            "",
        ).value
        env_checkpoint_dsn = os.environ.get("GAZEBO_TEST_CHECKPOINT_DSN", "")
        checkpoint_dsn = str(checkpoint_dsn_param or "").strip()
        if checkpoint_dsn and checkpoint_dsn.lower() in {"null", "none", "~"}:
            # Treat YAML null-ish values as "not provided"
            checkpoint_dsn = ""

        if not checkpoint_dsn and env_checkpoint_dsn:
            checkpoint_dsn = env_checkpoint_dsn.strip()
        self.checkpoint_dsn = checkpoint_dsn

        if self.checkpoint_dsn:
            self._checkpoint_namespace = self._derive_checkpoint_namespace(
                self.checkpoint_dsn
            )
        else:
            self._checkpoint_namespace = "no_checkpoint"
        self.base_path = (
            self._base_path_root
            / self._checkpoint_namespace
            / self.algorithm_name
            / self._experiment_identifier
        ).expanduser()
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.get_logger().info(
            f"Results base path set to: {self.base_path} (checkpoint namespace: {self._checkpoint_namespace})"
        )

        self.experiment_outcomes_path = Path(
            f"{self.base_path.resolve()}/{self.algorithm_name}_outcomes.csv"
        )
        self._outcomes_lock_path = Path(f"{self.experiment_outcomes_path}.lock")

        checkpoint_connect_timeout_param = self.declare_parameter(
            "checkpoint_connect_timeout",
            5.0,
        ).value
        if checkpoint_connect_timeout_param is None:
            checkpoint_connect_timeout = 5.0
        else:
            checkpoint_connect_timeout = float(checkpoint_connect_timeout_param)

        self.resume_checkpoint = bool(
            self.declare_parameter("resume_checkpoint", Parameter.Type.BOOL).value
        )
        self.auto_recover_running = bool(
            self.declare_parameter(
                "auto_recover_running",
                Parameter.Type.BOOL,
            ).value
        )
        if not self.checkpoint_dsn and self.auto_recover_running:
            self.get_logger().warning(
                "Ignoring 'auto_recover_running' because checkpoint store is disabled"
            )
            self.auto_recover_running = False

        heartbeat_interval_param = self.declare_parameter(
            "job_heartbeat_interval_sec",
            5.0,
        ).value
        if heartbeat_interval_param is None:
            heartbeat_interval = 5.0
        else:
            heartbeat_interval = float(heartbeat_interval_param)
        self._job_heartbeat_interval = max(heartbeat_interval, 1.0)

        heartbeat_timeout_param = self.declare_parameter(
            "job_heartbeat_timeout_sec",
            30.0,
        ).value
        if heartbeat_timeout_param is None:
            heartbeat_timeout = max(self._job_heartbeat_interval * 3.0, 15.0)
        else:
            heartbeat_timeout = float(heartbeat_timeout_param)
        self._job_heartbeat_timeout = max(
            heartbeat_timeout, self._job_heartbeat_interval * 1.5
        )

        worker_id_param = self.declare_parameter(
            "worker_id",
            Parameter.Type.STRING,
        ).value
        if worker_id_param:
            self.worker_id = str(worker_id_param)
        else:
            hostname = socket.gethostname()
            pid = os.getpid()
            suffix = uuid.uuid4().hex[:8]
            self.worker_id = f"{hostname}-{pid}-{suffix}"
        self.get_logger().info(f"Worker id set to: {self.worker_id}")

        self._job_coordinator = ExperimentJobCoordinator(
            node=self,
            algorithm_name=self.algorithm_name,
            experiment_identifier=self._experiment_identifier,
            worker_id=self.worker_id,
            checkpoint_dsn=self.checkpoint_dsn or None,
            checkpoint_connect_timeout_sec=checkpoint_connect_timeout,
            heartbeat_interval_sec=self._job_heartbeat_interval,
            heartbeat_timeout_sec=self._job_heartbeat_timeout,
            resume_checkpoint=self.resume_checkpoint,
            auto_recover_running=self.auto_recover_running,
            completed_runs_loader=self._load_completed_runs_from_outcomes,
            on_active_job_interrupted=self._handle_active_job_interrupted,
        )
        self._current_job_handle: Optional[ExperimentJobHandle] = None

        required_nodes_param = self.declare_parameter(
            "watchdog_required_nodes",
            [
                "/gazebo",
                "/bt_navigator",
                "/controller_server",
                "/planner_server",
                "/hunav_evaluator_node" if self.use_evaluator else "",
                "/hunav_agent_manager",
                "/hunav_plugin",
            ],
        ).value
        if isinstance(required_nodes_param, (list, tuple)):
            required_nodes = [
                str(node).strip() for node in required_nodes_param if str(node).strip()
            ]
        elif isinstance(required_nodes_param, str):
            required_nodes = (
                [required_nodes_param.strip()] if required_nodes_param else []
            )
        else:
            required_nodes = []

        module_grace_param = self.declare_parameter(
            "watchdog_startup_grace_sec",
            20.0,
        ).value
        if module_grace_param is None:
            module_grace = 20.0
        else:
            module_grace = float(module_grace_param)
        raw_error_patterns = self.declare_parameter(
            "watchdog_error_patterns",
            ["/hunav_plugin|Service /compute_agents timeout"],
        ).value
        threshold_param = self.declare_parameter(
            "watchdog_error_threshold",
            3,
        ).value
        window_param = self.declare_parameter(
            "watchdog_error_window_sec",
            5.0,
        ).value
        min_level_param = self.declare_parameter(
            "watchdog_error_min_level",
            Parameter.Type.INTEGER,
        ).value
        error_threshold = int(threshold_param) if threshold_param is not None else 3
        error_window = float(window_param) if window_param is not None else 5.0
        self.watchdog = ExperimentWatchdog(
            node=self,
            required_nodes=required_nodes,
            startup_grace_sec=max(module_grace, 0.0),
            raw_error_patterns=raw_error_patterns,
            error_threshold=error_threshold,
            error_window_sec=error_window,
            error_min_level=min_level_param,
            on_missing_nodes=self._handle_watchdog_missing_nodes,
        )

        self._watchdog_task: Optional[asyncio.Task[None]] = None

        self.gazebo_env_handler = GazeboEnvironmentHandler(self)
        self.evaluation_handler = ExperimentEvaluator(
            self,
            timeout_duration=timeout_duration,
        )
        if self.use_recorder:
            self.bag_recorder = BagRecorder(
                self,
                algorithm=self.algorithm_name,  # type: ignore
                base_path=self.base_path,  # type: ignore
                record_maps=self.record_maps,
            )
        if self.use_evaluator:
            self.hunav_evaluator_handler = HunavEvaluatorHandler(
                self,
                algorithm=self.algorithm_name,  # type: ignore
                base_path=self.base_path,  # type: Path
            )

        self.navigator = NavigationHandler(
            node=self,
            navigation_result_callback=self.evaluation_handler.set_navigation_result_event,
        )

        self.initial_state_entities: Dict[str, EntityState] = {}
        self.goal_entities: Dict[str, EntityState] = {}

        entity_dictionary = parse_entity_state_yaml(Path(yaml_path))
        self.initial_state_entities = entity_dictionary["initial_state_entities"]
        self.goal_entities = entity_dictionary["goal_entities"]

        self.goal_box_xml = (
            Path(
                f"{get_package_share_directory('gazebo_sim')}/models/goal_box/model.sdf"
            )
            .open("r")
            .read()
        )
        self._job_coordinator.configure_jobs(
            experiment_tags=self.initial_state_entities.keys(),
            repetitions=self.repetitions,
        )

        atexit.register(self._job_coordinator.cleanup_on_exit)
        self.end = False
        self.get_logger().info("ExperimentManager initialized")
        # set log level to debug
        # self.get_logger().set_level(rclpy.logging.LoggingSeverity.DEBUG)

    async def initialize(self):
        # Placeholder for initialization logic
        self.get_logger().debug("Initializing experiment ...")
        if self._asyncio_loop is None:
            try:
                self._asyncio_loop = asyncio.get_running_loop()
            except RuntimeError:
                self._asyncio_loop = None
        # Set the asyncio loop for the job coordinator and watchdog
        self._job_coordinator.set_asyncio_loop(self._asyncio_loop)
        self.watchdog.set_asyncio_loop(self._asyncio_loop)
        # Initialize the experiment manager
        await self.gazebo_env_handler.wait_for_gazebo_ready()
        self.get_logger().debug("All entities are in the world")
        await self.navigator.initialize_navigation()
        await self.gazebo_env_handler.pause_gazebo()
        self.get_logger().debug("Navigation stack initialized")
        self.evaluation_handler.initialize()
        self.get_logger().debug("Experiment evaluator initialized")
        if self.use_evaluator:
            await self.hunav_evaluator_handler.wait_for_hunav_evaluator()
            self.get_logger().debug("Hunav evaluator handler initialized")

    async def run_experiments(self):
        """
        Run the experiments sequentially.
        This method will run the experiments for each experiment_tag defined in the YAML file.
        """
        self.get_logger().debug("Running experiments ...")
        self.watchdog.start_monitoring()
        if self._watchdog_task is None or self._watchdog_task.done():
            self._watchdog_task = asyncio.create_task(self._stale_job_watchdog())
        self._job_coordinator.resubmit_failed_jobs()
        try:
            while True:
                health_event = self.watchdog.health_event
                if not health_event.is_set():
                    if not self.watchdog.waiting_logged:
                        self.get_logger().warning(
                            "Waiting for required modules to come online before claiming next job"
                        )
                        self.watchdog.set_waiting_logged(True)
                    await health_event.wait()
                    self.watchdog.set_waiting_logged(False)
                # Check for available jobs and claim the next one
                self._job_coordinator.prepare_for_dispatch()
                job_handle = self._job_coordinator.claim_next_job()
                if job_handle is None:
                    break
                try:
                    _ = self._job_coordinator.start_job(job_handle)
                    self._current_job_handle = job_handle
                    success = await self._execute_job(job_handle)
                    if not success and not self._job_coordinator.heartbeat_lost:
                        self._job_coordinator.handle_unsuccessful_job(job_handle)
                except Exception:
                    if not self._job_coordinator.heartbeat_lost:
                        self._job_coordinator.handle_unsuccessful_job(job_handle)
                    raise
                finally:
                    await self._job_coordinator.finalize_job(job_handle)
                    self._current_job_handle = None
        finally:
            await self._stop_watchdog_task()
            await self.watchdog.stop()
            await self._job_coordinator.shutdown()
        await self.navigator.shutdown_navigation()
        self.get_logger().info("All experiments completed")
        # Save the experiment outcomes to a CSV file
        self.get_logger().info(
            f"Experiment outcomes saved to {self.experiment_outcomes_path}"
        )
        self.end = True

    async def _execute_job(self, handle: ExperimentJobHandle) -> bool:
        """Execute a single experiment job.
        Args:
            handle (ExperimentJobHandle): The job handle to execute.
        Returns:
            bool: True if the job was successful, False otherwise.
        """
        experiment_tag = handle.experiment_tag
        run_id = handle.run_id

        try:
            if not self.experiment_outcomes_path.exists():
                self.experiment_outcomes_path.parent.mkdir(parents=True, exist_ok=True)
            result = await self.run_experiment(experiment_tag, run_id)
            if self._job_coordinator.heartbeat_lost:
                if handle.job_id is not None:
                    self.get_logger().warning(
                        f"Lost heartbeat for job {handle.job_id}; skipping result storage so another worker can retry"
                    )
                else:
                    self.get_logger().warning(
                        f"Lost heartbeat while running {experiment_tag} run {run_id}; will retry locally"
                    )
                self._job_coordinator.handle_lost_heartbeat(handle)
                return False
            self.get_logger().info(
                f"Run {run_id} for {experiment_tag} completed with result: {result}"
            )
            experiment_outcome = pd.DataFrame(
                {
                    "experiment_tag": [experiment_tag],
                    "run_id": [run_id],
                    "result": [result],
                }
            )
            if self.use_evaluator:
                hunav_results = self.hunav_evaluator_handler.get_result_df()
                experiment_outcome = pd.merge(
                    experiment_outcome,
                    hunav_results,
                    on=["experiment_tag", "run_id"],
                    how="left",
                )
            with self._acquire_outcomes_lock():
                file_exists = self.experiment_outcomes_path.exists()
                experiment_outcome.to_csv(
                    self.experiment_outcomes_path,
                    mode="a",
                    header=not file_exists,
                    index=False,
                )
                # Atomic re-sort: write to a temp file first, then replace
                df = pd.read_csv(self.experiment_outcomes_path)
                df = df.sort_values(by=["experiment_tag", "run_id"])
                tmp_path = self.experiment_outcomes_path.with_suffix(".csv.tmp")
                df.to_csv(
                    tmp_path,
                    mode="w",
                    header=True,
                    index=False,
                )
                tmp_path.replace(self.experiment_outcomes_path)
            # Mark the job as complete
            _ = self._job_coordinator.complete_job_success(handle, str(result))
            return True
        except Exception as exc:  # noqa: BLE001
            error_message = f"Experiment {experiment_tag} run {run_id} failed: {exc}"
            self.get_logger().error(error_message)
            self.get_logger().debug(traceback.format_exc())
            if not self._job_coordinator.heartbeat_lost:
                _ = self._job_coordinator.complete_job_failure(handle, error_message)
            else:
                self.get_logger().warning(
                    "Skipping failure mark because job ownership was lost"
                )
            raise

    async def run_experiment(
        self, experiment_tag: str = "episode_1", run_id: int = 1
    ) -> ExperimentResult:
        """
        Run the experiment for a given experiment_tag.
        This method will reset the environment, start the experiment, and return the result.
        The experiment is considered successful if the robot reaches the goal
        without any collisions and within the timeout duration.
        The experiment is considered a failure if:
        - A collision is detected with an agent or the environment
        - The timeout duration is reached
        Args:
            experiment_tag (str): The experiment_tag to run.
            run_id (int): The run ID for the experiment.
        Returns:
            ExperimentResult: The result of the experiment.
        """
        # Placeholder for experiment running logic
        entities_to_reset = [
            self.initial_state_entities[experiment_tag],
        ]
        goal_pose = get_posestamped_from_entity(
            self.goal_entities[experiment_tag], "map"
        )
        await self.gazebo_env_handler.reset_environment_for_experiment(
            entities_to_reset,
            goal_entity=self.goal_entities[experiment_tag],
            goal_xml=self.goal_box_xml,
        )
        await self.navigator.reset_navigation()
        # wait for a few seconds before starting the experiment (to let the environment settle)
        if self.wait_before_start:
            self.get_logger().debug(
                f"Waiting for {self.wait_before_start} seconds before starting the experiment ..."
            )
            # Use a sim-time-aware timer via asyncio sleep tied to the ROS clock
            wait_end = self.get_clock().now() + rclpy.duration.Duration(seconds=self.wait_before_start)
            while self.get_clock().now() < wait_end:
                await asyncio.sleep(0.1)
            self.get_logger().debug("Wait completed")
        # start navigation
        # create task to wait for the robot to reach the goal
        self.navigator.start_navigation_task(
            goal_pose=goal_pose,
            navigation_result_callback=self.evaluation_handler.set_navigation_result_event,
        )
        self.get_logger().debug("Environment reset successfully")

        if self.use_recorder:
            self.get_logger().debug("Starting recording ...")
            self.bag_recorder.start_recording_and_set_goal(
                experiment_name=experiment_tag,
                run_id=str(run_id),
                goal=goal_pose,
            )
        if self.use_evaluator:
            self.get_logger().info("Starting hunav evaluator recording ...")
            await self.hunav_evaluator_handler.start_recording(
                goal=goal_pose,
                experiment_tag=experiment_tag,
                run_id=run_id,
            )

        experiment_result = await self.evaluation_handler.run_experiment()
        # cancel the navigation task if it is still running
        self.get_logger().debug("Experiment completed")
        if self.use_recorder:
            self.bag_recorder.set_result_and_stop_recording(
                result=str(experiment_result)
            )
            self.get_logger().debug("Recording stopped")
        if self.use_evaluator:
            await self.hunav_evaluator_handler.stop_recording()
            self.get_logger().info("Hunav evaluator recording stopped")
        # Log the experiment result
        self.get_logger().debug(
            f"Experiment {experiment_tag} run {run_id} completed with result: {experiment_result}"
        )
        await self.navigator.cancel_navigation()
        return experiment_result

    async def _stale_job_watchdog(self) -> None:
        try:
            while not self.end:
                await asyncio.sleep(self._job_coordinator.heartbeat_interval)
                await self._job_coordinator.on_watchdog_tick()
                try:
                    await self.watchdog.evaluate_module_health()
                except Exception as exc:  # noqa: BLE001
                    self.get_logger().error(
                        f"Watchdog failed module health check: {exc}"
                    )
        except asyncio.CancelledError:
            raise

    async def _handle_watchdog_missing_nodes(self, missing: Set[str]) -> None:
        await self._job_coordinator.handle_watchdog_missing_nodes(missing)

    async def _handle_active_job_interrupted(
        self, handle: ExperimentJobHandle, reason: str
    ) -> None:
        if self._current_job_handle is not None and handle != self._current_job_handle:
            return
        self.get_logger().warning(
            f"Job interruption detected for {handle.experiment_tag} run {handle.run_id}: {reason}"
        )
        try:
            await self.navigator.cancel_navigation()
        except Exception as exc:  # noqa: BLE001
            self.get_logger().warning(
                f"Failed to cancel navigation during watchdog recovery: {exc}"
            )
        try:
            if hasattr(self.evaluation_handler, "collision_event"):
                self.evaluation_handler.experiment_result = (
                    ExperimentResult.FAILURE_NAVIGATION
                )
                self.evaluation_handler.collision_event.set()
        except Exception as exc:  # noqa: BLE001
            self.get_logger().debug(
                f"Failed to notify evaluator after watchdog interruption: {exc}"
            )

    async def _stop_watchdog_task(self) -> None:
        if self._watchdog_task is None:
            return
        task = self._watchdog_task
        self._watchdog_task = None
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    @contextmanager
    def _acquire_outcomes_lock(self) -> Iterator[None]:
        self._outcomes_lock_path.parent.mkdir(parents=True, exist_ok=True)
        with self._outcomes_lock_path.open("a+") as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

    @staticmethod
    def _derive_checkpoint_namespace(dsn: str) -> str:
        digest = hashlib.sha256(dsn.encode("utf-8")).hexdigest()[:12]
        components: List[str] = []
        parsed = urlparse(dsn)
        if parsed.scheme:
            if parsed.hostname:
                components.append(parsed.hostname)
            if parsed.path and parsed.path not in {"", "/"}:
                components.append(parsed.path.strip("/"))
            if parsed.port:
                components.append(str(parsed.port))
        if not components:
            for key, value in re.findall(r"([A-Za-z_]+)=([^\s]+)", dsn):
                if key.lower() in {"dbname", "database", "host", "port"} and value:
                    components.append(value)
        sanitized = [
            re.sub(r"[^A-Za-z0-9_.-]+", "_", piece.strip())
            for piece in components
            if piece.strip()
        ]
        label = "_".join(filter(None, sanitized))
        if label:
            return f"{label}__{digest}"
        return f"checkpoint__{digest}"

    def _load_completed_runs_from_outcomes(self) -> Set[Tuple[str, int]]:
        if not self.experiment_outcomes_path.exists():
            return set()
        try:
            df = pd.read_csv(self.experiment_outcomes_path)
        except Exception as exc:  # noqa: BLE001
            self.get_logger().warning(
                f"Unable to read outcomes file for resume logic: {exc}"
            )
            return set()
        if "experiment_tag" not in df.columns or "run_id" not in df.columns:
            return set()
        completed: Set[Tuple[str, int]] = set()
        for _, row in df[["experiment_tag", "run_id"]].dropna().iterrows():
            try:
                completed.add((str(row["experiment_tag"]), int(row["run_id"])))
            except Exception:  # noqa: BLE001
                continue
        return completed
