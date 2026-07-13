from launch_pal.arg_utils import LaunchArgumentsBase
from dataclasses import dataclass
from launch.substitutions import LaunchConfiguration
from typing import Optional, List, Dict
from launch import LaunchContext


@dataclass(frozen=True, kw_only=True)
class LaunchArgumentsBaseParam(LaunchArgumentsBase):

    def launch_configurations(self) -> List[LaunchConfiguration]:
        """Return a List of launch configurations for the launch file

        Returns:
            A List of launch configurations.
        """

        return [
            LaunchConfiguration(f"{name}") for name in self.__dataclass_fields__.keys()
        ]

    def launch_configurations_dict(self) -> Dict[str, LaunchConfiguration]:
        """Return a dictionary of launch configurations.

        Returns:
            A dictionary of launch configurations.
        """
        return {
            name: LaunchConfiguration(f"{name}")
            for name in self.__dataclass_fields__.keys()
        }

    def launch_configurations_dict_with_context(
        self,
        context: Optional[LaunchContext] = None,
    ) -> Dict[str, str]:
        """Return a dictionary of launch configurations with context."""
        if context is None:
            raise ValueError(
                "Context cannot be None, please use this function in a OpaqueFunction"
            )
        return {
            name: config.perform(context)
            for name, config in self.launch_configurations_dict().items()
        }

    def param_rewrites_dict(
        self,
        context: Optional[LaunchContext] = None,
    ) -> Dict[str, str]:
        """Return a dictionary of parameter rewrites."""
        if context is None:
            raise ValueError(
                "Context cannot be None, please use this function in a OpaqueFunction"
            )
        return {
            name: config
            for name, config in self.launch_configurations_dict_with_context(
                context=context
            ).items()
            if config
        }
