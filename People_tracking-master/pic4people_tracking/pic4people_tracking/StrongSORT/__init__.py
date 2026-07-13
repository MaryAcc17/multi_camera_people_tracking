from pic4people_tracking.StrongSORT.trackers.strongsort.strong_sort import StrongSORT

TRACKERS = ['strongsort']

__version__ = '10.0.52'

__all__ = ("__version__",
           "StrongSORT", "create_tracker", "get_tracker_config", "gsi")
