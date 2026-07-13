from pathlib import Path
from typing import List, Optional, Dict
import yaml
import pandas as pd
import os

from ament_index_python.packages import (
    get_package_share_directory,
    get_resources,
    get_resource,
)


def print_summary(log_path: Path):
    """
    Print the summary of the gazebo test.

    log_path: The path to the log file.
    """
    # Check if the log file exists
    if not os.path.exists(log_path):
        print(f"Log file {log_path} does not exist.")
        return

    # Read the csv file and print the summary in a table format
    df = pd.read_csv(log_path)
    # Print the summary of the gazebo test
    print("Summary of the gazebo test:")
    print("--------------------------------------------------")
    print(f"Log file: {log_path}")
    print("--------------------------------------------------")
    print(f"Number of total experiments: {len(df)}")
    print("--------------------------------------------------")
    # Print the columns of the dataframe
    print("--------------------------------------------------")
    print(df)
    print("--------------------------------------------------")


def parse_experiment_name(experiment_name: str) -> Dict:
    """
    Parse the experiment name and return the world, map, and goals_and_poses.

    experiment_name: The name of the experiment.
    """
    experiment = get_experiment_by_id(experiment_name)

    if experiment is None:
        raise ValueError(f"Experiment {experiment_name} does not exist.")

    # retrieve the package name for the experiment
    experiment_pkg_name = experiment["pkg_name"]

    world = experiment["world"]
    map = experiment["map"]
    goals_and_poses = experiment["goals_and_poses"]
    world_pkg_name = experiment["world_pkg_name"]
    camera_config_file = experiment.get("camera_config_file", None)

    world_pkg_path = os.path.join(
        get_package_share_directory(world_pkg_name),
        "worlds",
        world,
    )
    agents_configuration_file = experiment["agents_configuration_file"]
    agents_pkg_name = experiment["config_pkg_name"]
    agents_pkg_share = get_package_share_directory(agents_pkg_name)
    agents_pkg_path = os.path.join(
        agents_pkg_share,
        "config",
        agents_configuration_file,
    )

    # check if there is a map_pkg name in the file, if not use the experiment package name
    map_pkg_name = experiment.get("map_pkg_name", experiment_pkg_name)

    # check if the world, map, and goals_and_poses are absolute paths
    if not os.path.isabs(map):
        map = os.path.join(
            get_package_share_directory(map_pkg_name),
            map,
        )
    if not os.path.isabs(goals_and_poses):
        goals_and_poses = os.path.join(
            get_package_share_directory(experiment_pkg_name),
            goals_and_poses,
        )
    # Check if the world, map, and goals_and_poses exist
    if not os.path.exists(world_pkg_path):
        raise FileNotFoundError(f"World file {world_pkg_path} does not exist.")
    if not os.path.exists(map):
        raise FileNotFoundError(f"Map file {map} does not exist.")
    if not os.path.exists(goals_and_poses):
        raise FileNotFoundError(
            f"Goals and poses file {goals_and_poses} does not exist."
        )
    if not os.path.exists(agents_pkg_path):
        raise FileNotFoundError(
            f"Agents configuration file {agents_pkg_path} does not exist."
        )

    camera_config_path = None
    if camera_config_file:
        camera_config_path = camera_config_file
        if not os.path.isabs(camera_config_path):
            camera_config_path = os.path.join(
                get_package_share_directory(world_pkg_name),
                "config",
                camera_config_file,
            )
        if not os.path.exists(camera_config_path):
            raise FileNotFoundError(
                f"Camera configuration file {camera_config_path} does not exist."
            )
    return {
        "world": world,
        "map": map,
        "goals_and_poses": goals_and_poses,
        "agents_configuration_file": agents_configuration_file,
        "world_pkg_name": world_pkg_name,
        "agents_pkg_name": agents_pkg_name,
        "camera_config_file": camera_config_path,
    }


def get_experiment_files() -> Dict[str, str]:
    """
    Get the files associated with the experiment using the ament index

    """
    resources = {}
    for package_name in get_resources("gazebo_test_experiments"):
        try:
            resource = get_resource("gazebo_test_experiments", package_name)
            package_path = get_package_share_directory(package_name)
            for resources_items in resource[0].split("\n"):
                if not resources_items:
                    continue
                resource_name, resource_path = resources_items.split(";")
                i = 1
                while resource_name in resources.keys():
                    resource_name = f"{resource_name}_{i}"
                    i += 1

                resources[resource_name] = os.path.join(package_path, resource_path)
        except Exception as e:
            print(f"Error getting resource for package {package_name}: {e}")
            continue

    return resources


def get_experiment_ids() -> List[str]:
    """
    Get the IDs of the experiments.

    returns: A list of experiment IDs.
    """
    files = get_experiment_files().values()

    # Check if the files are valid
    if not files:
        raise ValueError("No valid experiment files found.")

    # Filter the files to only include YAML files
    yaml_files = [f for f in files if f.endswith(".yaml")]

    # Check if the files exists
    for f in yaml_files:
        if not os.path.exists(f):
            raise FileNotFoundError(f"File {f} does not exist.")

    # Get the names of the experiments from the YAML files
    # open the yaml files and extract the experiment names
    experiment_names = []
    for f in yaml_files:
        with open(f, "r") as file:
            try:
                data = yaml.safe_load(file)
                experiments = data.get("experiments", None)
                if experiments:
                    experiment_names.extend(experiments)
            except Exception as e:
                print(f"Error reading {f}: {e}")

    return experiment_names


def get_experiment_by_id(experiment_id: str) -> Optional[dict]:
    """
    Get the experiment details by ID.

    experiment_id: The ID of the experiment.
    returns: A dictionary containing the experiment details.
    """
    files = get_experiment_files().values()

    # Check if the files are valid
    if not files:
        raise ValueError("No valid experiment files found.")

    # Filter the files to only include YAML files
    yaml_files = [f for f in files if f.endswith(".yaml")]

    # Check if the files exists
    for f in yaml_files:
        if not os.path.exists(f):
            raise FileNotFoundError(f"File {f} does not exist.")

    # Search inside the files if there is the corrispondent key in the YAML
    for f in yaml_files:
        with open(f, "r") as file:
            try:
                data = yaml.safe_load(file)
                if experiment_id in data:
                    this_pkg_name = f.split("share/")[-1].split("/")[0]
                    data[experiment_id]["pkg_name"] = this_pkg_name
                    return data[experiment_id]
            except Exception as e:
                print(f"Error reading {f}: {e}")

    return None
