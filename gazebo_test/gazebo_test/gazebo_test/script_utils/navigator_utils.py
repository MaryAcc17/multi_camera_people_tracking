import os
from jinja2 import Environment, FileSystemLoader, StrictUndefined, Template
import yaml, pathlib
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from typing import List, Optional, Dict, Union

resources = ['controllers', 'planners', 'navigators', 'costmap_layers']

from ament_index_python.packages import get_package_share_directory
from ament_index_python.resources import get_resource, has_resource, get_resources

# Get all packages that have registered resources and add their shared directories to search path
def get_template_search_paths(resources: List[str]) -> List[str]:
    """Get all template search paths from packages with registered controllers
    Args:
        resources (List[str]): List of resource names to search for, e.g., ['gazebo_test_controllers', 'gazebo_test_planners']
    Returns:
        List[str]: List of unique package share directories containing the resources
    """
    search_paths = []  # Always include current directory first
    for resource in resources:
        for package_name in get_resources(f'gazebo_test_{resource}'):
            if has_resource(f'gazebo_test_{resource}', package_name):
                try:
                    package_share_path = get_package_share_directory(package_name)
                    if package_share_path not in search_paths:
                      search_paths.append(package_share_path)
                      logging.debug(f"Added search path for package '{package_name}': {package_share_path}")
                except Exception as e:
                    logging.warning(f"Could not get share directory for package '{package_name}': {e}")
    return search_paths

def parse_resource_tuple(t: tuple[str, str]) -> Dict[str, str]:
    """Parse resource tuple into a dictionary and adjust paths for current working directory.
    
    Args:
        t (tuple[str, str]): Tuple where the first element is a newline-separated string of
                           "key;value" pairs, and the second element is the package name.
    Returns:
        Dict[str, str]: Dictionary mapping keys to their corresponding values with adjusted paths.
    """
    d = {}
    for item in t[0].split('\n'):
        if item.strip() == "":
            continue
        key, value = item.split(';')
        d[key] = value
    return d

def get_registered_resources(resource_name: str, search_paths: List[str]) -> Dict[str, str]:
    """Get registered resources from ament resource."""
    registered_resources = {}
    for package_name in get_resources(resource_name):
        if has_resource(resource_name, package_name):
            registered_resources_ = get_resource(resource_name, package_name)
            registered_resources.update(parse_resource_tuple(registered_resources_))
    return registered_resources


def get_navigator_config(navigator: str, search_paths: List[str]) -> Dict:
    """Get navigator configuration YAML file from registered navigators.
    Args:
        navigator (str): Name of the navigator to load.
        search_paths (List[str]): List of search paths to locate the navigator config file.
    Returns:
        Dict: Parsed YAML configuration as a dictionary.
    """
    
    available_navigators = list_available_navigators(search_paths)
    if navigator not in available_navigators:
        raise ValueError(f"Navigator '{navigator}' not registered. Available: {available_navigators}")
    registered_navigators = get_registered_resources('gazebo_test_navigators', search_paths)
    for pkg_name, path in registered_navigators.items():
        navigator_configs_path = os.path.join(get_package_share_directory(pkg_name), path)
        # Verify the file exists
        if not os.path.exists(navigator_configs_path):
            raise FileNotFoundError(f"Navigator config file '{navigator_configs_path}' not found.")
        with open(navigator_configs_path, 'r') as f:
            configs = yaml.safe_load(f)
            if navigator in configs['navigators']:
                return configs[navigator]
    raise ValueError(f"Navigator '{navigator}' not registered. Available: {list_available_navigators(search_paths)}")

def list_available_navigators(search_paths: List[str]=get_template_search_paths(resources)) -> List[str]:
    """Get a List of all available navigator names from registered navigators."""
    registered_navigators = get_registered_resources('gazebo_test_navigators', search_paths)
    available_navigators = []
    for pkg_name, path in registered_navigators.items():
        navigator_configs_path = os.path.join(get_package_share_directory(pkg_name), path)
        if not os.path.exists(navigator_configs_path):
            continue
        with open(navigator_configs_path, 'r') as f:
            configs = yaml.safe_load(f)
            available_navigators.extend(configs.get('navigators', []))
    return available_navigators


def get_resource_template(resource_type: str, name: str, search_paths: List[str]) -> str:
    """Get the template path for a given resource type and name."""
    registered_resources = get_registered_resources(f'gazebo_test_{resource_type}_templates', search_paths)
    if name not in registered_resources:
        available = list(registered_resources.keys())
        raise ValueError(f"{resource_type.capitalize()} '{name}' not registered. Available: {available}")
    
    return registered_resources[name]  

def select_resource_template(resource_type: str, resource: Union[str, Dict], search_paths: List[str], env: Environment) -> str:
    """ Select and return the resource template.
    Args:
        resource_type (str): Type of the resource ('navigator', 'controller', 'planner', 'costmap_layer').
        resource (Union[str, Dict]): Resource name or configuration dictionary.
        search_paths (List[str]): List of search paths to locate the resource template.
        env (Environment): Jinja2 Environment to load the template.
    Returns:
        str: Rendered resource template as a string.
    Raises:
        ValueError: If the resource is not registered.
    """
    logging.debug(f"Selecting {resource_type} template for resource: {resource}")
    if isinstance(resource, str):
        logging.debug(f"{resource_type} specified as string: {resource}")
        resource_name = resource
        resource_config = {}
    elif isinstance(resource, dict) and 'name' in resource:
        logging.debug(f"{resource_type} specified as dict: {resource}")
        resource_name = resource['name']
        resource_config = resource
    else:
        logging.error(f"Invalid {resource_type} specification: {resource}")
        raise ValueError(f"Invalid {resource_type} specification: {resource}")

    # include the resource template
    resource_template_path = get_resource_template(resource_type, resource_name, search_paths)
    return env.get_template(resource_template_path).render({resource_type: resource_config})

def get_navigator_yaml(navigator: str) -> pathlib.Path:
    """ Get the YAML configuration for the navigator.
    Args:
        navigator (str): Name of the navigator to get the configuration for.
    Returns:
        pathlib.Path: Path to the generated YAML configuration file.
    Raises:
        ValueError: If the navigator is not registered.
    """
    search_paths = get_template_search_paths(resources)
    navigator_config = get_navigator_config(navigator, search_paths)
    
    env = Environment(
        loader=FileSystemLoader(search_paths),
        undefined=StrictUndefined,
        keep_trailing_newline=True,
        lstrip_blocks=True,
        trim_blocks=True,
    )

    env.globals['select_navigator'] = lambda nav: select_resource_template('navigator', nav, search_paths, env)
    env.globals['select_controller'] = lambda ctrl: select_resource_template('controller', ctrl, search_paths, env)
    env.globals['select_planner'] = lambda plnr: select_resource_template('planner', plnr, search_paths, env)
    env.globals['select_costmap_layer'] = lambda layer: select_resource_template('costmap_layer', layer, search_paths, env)

    # Load the main navigator template
    navigator_yaml = select_resource_template('navigator', navigator_config['nav_template'], search_paths, env)
    # Render the template with the navigator configuration

    import tempfile, os
    output_dir = pathlib.Path(tempfile.mkdtemp(prefix="gazebo_test_nav_"))
    output_path = output_dir / f"{navigator}_pid{os.getpid()}_config.yaml"
    with open(output_path, 'w') as f:
        f.write(navigator_yaml)
    if not output_path.exists():
        raise FileNotFoundError(f"Failed to create navigator YAML file at {output_path}")
    return output_path
        
