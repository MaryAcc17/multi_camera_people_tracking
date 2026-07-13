# Gazebo Test

This repository contains utilities and scripts for testing Gazebo simulations in ROS2 in an automated way.
It is capable of running simulations, repeating them, and generating reports on the results.

The main goal is to provide a framework for testing Gazebo simulations in a consistent and repeatable manner.
The framework is designed to be extensible, allowing for the addition of new tests and features as needed.
The framework is designed to be used with ROS2 and Gazebo, but can be adapted for use with other simulation environments as well.

## Requirements

- ROS2 Humble
- Gazebo
- Python 3.10 or higher

## Installation

1. Clone the repository into your ROS2 workspace:

   ```bash
   cd ~/ros2_ws/src
   git clone https://github.com/PIC4SeR/gazebo_test
   ```

2. Install the required dependencies:

   ```bash
   cd ~/ros2_ws
   rosdep install --from-paths src --ignore-src -r -y
   ```

3. Build the workspace:

   ```bash
   cd ~/ros2_ws
   colcon build --symlink-install
   ```

4. Source the workspace:

   ```bash
    source ~/ros2_ws/install/setup.bash
    ```

5. install the required python packages

   ```bash
   pip install --user -r src/gazebo_test/requirements.txt
   ```

6. Install tmux if not already installed

   ```bash
   sudo apt install tmux
   ```

7. Resolve the issue with tf_transformations

   change the lines in /usr/lib/python3/dist-packages/transforms3d/quaternions.py
   search for _MAX_FLOAT and _FLOAT_EPS and change them to:

   ```python
   _MAX_FLOAT = np.maximum_sctype(np.float32)
   _FLOAT_EPS = np.finfo(np.float32).eps
   ```

   This resolves the issue with tf_transformations in ROS2 Humble. (save the file as root to be able to edit it)

8. Add the following line to your ~/.tmux.conf file to enable mouse support in tmux:

   ```bash
   set -g mouse on
   ```

   This allows you to scroll and select panes using the mouse.

9. Change the base_path in gazebo_test/gazebo_experiments/experiment_config.yaml to your workspace path

   ```yaml
   base_path: "<your_workspace_path>/results"
   ```

10. Verify that you have the correct nav2_params files in gazebo_test/gazebo_experiments/nav2_params
    You can use the ones provided in the repository or create your own.

## Packages

- ['gazebo_test'](gazebo_test/README.md): The main package containing the test framework and utilities.
- ['gazebo_sim'](gazebo_sim/README.md): A package containing Gazebo simulation models and worlds for testing.

## Usage

Check the [usage documentation](docs/usage.md) for detailed instructions on how to use the framework.

## TODO

- rviz panel for experiment setting


# Gazebo Social Navigation Experiments

End-to-end automation for running social navigation benchmarks in Gazebo with ROS 2. The stack coordinates multiple workers through a PostgreSQL-backed checkpoint store, records rosbag data alongside experiment metrics, and now includes watchdogs to recover stalled runs automatically.

## Overview

- **Experiment Manager (`gazebo_test/tasks/go_to_pose.py`)** orchestrates each run, writes CSV outcomes, and keeps bag recordings in a deterministic structure keyed by checkpoint, algorithm, and experiment identifier.
- **Checkpoint Store (`gazebo_test/utils/checkpoint_store.py`)** uses PostgreSQL to register jobs, claim work atomically, track progress, and requeue stalled jobs based on heartbeat timestamps.
- **Automation CLI (`scripts/full_experiment`)** spins up Gazebo, navigation, evaluators, and the experiment manager via tmux. It consults the checkpoint store before launching to skip sessions with no remaining work.
- **Bag Recorder (`gazebo_test/utils/bag_recorder.py`)** writes bags into the shared results directory, reusing `run_<id>` folders when an experiment is re-run.

## Prerequisites

- Docker with non-root access (for the provided dev container).
- Visual Studio Code with the **Dev Containers** extension.
- PostgreSQL 14+ reachable from the container, with a database user that can create tables.
- ROS 2 Humble (installed inside the container via the workspace setup scripts).

## Getting Started

1. **Open in Dev Container** – Reopen this repository in VS Code using `Dev Containers: Rebuild and Reopen in Container`.
2. **Create and configure the database** if you have not already:

	 ```bash
	 createdb hunav_experiments
	 psql hunav_experiments <<'SQL'
	 CREATE USER hunav WITH PASSWORD 'hunav';
	 GRANT ALL PRIVILEGES ON DATABASE hunav_experiments TO hunav;
	 SQL
	 ```

3. **Set the DSN** in your shell (or `.env` file) so every component sees the same checkpoint database:

	 ```bash
	 export GAZEBO_TEST_CHECKPOINT_DSN="postgresql://hunav:hunav@localhost:5432/hunav_experiments"
	 ```

4. **Install dependencies** once per workspace:

	 ```bash
	 colcon build --symlink-install
	 source install/setup.bash
	 ```

## Workflow

### 1. Register experiments

Prepare experiment YAML files (goals and poses) and navigation configs under `src/`. The experiment manager automatically registers all combinations of `experiment_tag` and `run_id` defined by the YAML file and the `repetitions` parameter.

### 2. Launch a full experiment session

Use the automation CLI, which manages Gazebo, navigation, optional evaluators, and the experiment manager:

```bash
./src/gazebo_test/scripts/full_experiment run \
	social_env_test_benchmark \
	--navigator MPPI \
	--bag-record \
	--checkpoint-dsn "$GAZEBO_TEST_CHECKPOINT_DSN"
```

The script checks the checkpoint store before launching. If all jobs for the chosen algorithm/experiment are completed, nothing is started.

### 3. Worker behaviour

- When `ExperimentManager` starts, it enforces the DSN requirement, derives a **checkpoint namespace** (hashed from the DSN), and builds the output path:

	```text
	results/<checkpoint_namespace>/<algorithm>/<experiment_identifier>/
	```

- Bag recordings and CSV outputs live under this directory so resumed runs reuse the same storage.
- Each worker writes job heartbeats every `job_heartbeat_interval_sec` seconds (default 5s).
- A watchdog requeues RUNNING jobs if their heartbeat is older than `job_heartbeat_timeout_sec` (default 30s). Requeued jobs go back to PENDING so other workers can claim them.
- If a heartbeat is lost during execution, the worker stops writing results to avoid duplicating output and lets the job be re-run elsewhere.

### 4. Resuming experiments

- To resume, relaunch `full_experiment` with the same DSN, algorithm, and experiment. Only PENDING jobs will run.
- You can pass `--resume-checkpoint` to keep DONE jobs untouched; without it, you may use the `ExperimentManager.reset_jobs()` method (exposed via CLI arguments) to reset all runs to PENDING.

## Key Parameters

These ROS parameters can be set via launch arguments or on the command line:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `checkpoint_dsn` | *(required)* | PostgreSQL DSN for job coordination.
| `job_heartbeat_interval_sec` | `5.0` | How often a worker updates its heartbeat.
| `job_heartbeat_timeout_sec` | `30.0` | How long before a RUNNING job is considered stale.
| `base_path` | `results/gazebo_test` | Root directory for results; the manager appends the checkpoint namespace, algorithm, and experiment name.
| `use_recorder` / `record_maps` | `false` | Enable bag recording and include map topics.
| `use_evaluator` | `false` | Enable HuNav evaluator integration.

Environment variable `GAZEBO_TEST_CHECKPOINT_DSN` provides a fallback for the DSN if the parameter is omitted.

## Directory Layout

```
results/
	<checkpoint_namespace>/
		<algorithm>/
			<experiment_identifier>/
				<algorithm>_outcomes.csv
				episode_<n>/
					run_<id>/
						metadata.yaml
						*.db3  (rosbag2 storage)
```

CSV files are append-only with flock-based locking to prevent concurrent writes. Bag directories are reused per `run_<id>`; existing runs are replaced after a warning.

## Watchdog & Recovery

- **Heartbeats:** After claiming a job, a worker emits heartbeats. If `refresh_job_heartbeat()` fails (because another worker reclaimed it), the current run aborts without writing results.
- **Requeue:** The watchdog (and the main loop) periodically calls `requeue_stale_jobs()`. Any RUNNING job with a heartbeat older than the timeout is returned to PENDING and a note is recorded.
- **Manual recovery:** You can still call `recover_stale_jobs()` to force RUNNING jobs to FAILED; the watchdog already handles this automatically for heartbeats.

## Known Tips

- When modifying Python code, remember to rebuild with `colcon build --packages-select gazebo_test --symlink-install` and re-source `install/setup.bash` before running.
- For multi-worker setups, ensure every process uses the same DSN and base path; the checkpoint namespace prevents data overlap between different databases.
- If you need to inspect checkpoints manually: `SELECT * FROM jobs ORDER BY algorithm, experiment_identifier, run_id;` in PostgreSQL.

## Troubleshooting

- **“Database directory already exists”** – Rebuild/relaunch after picking up the updated bag recorder; it now cleans existing run directories before recording.
- **No jobs run** – Confirm the DSN matches the checkpoint data. `full_experiment` will exit immediately when all jobs are DONE.
- **Jobs never finish** – Check heartbeat parameters. Too aggressive `job_heartbeat_timeout_sec` may requeue long runs prematurely; increase it to a higher multiple of the expected run duration.

## License

See the repository’s LICENSE file for details.