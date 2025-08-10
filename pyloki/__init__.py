from importlib import metadata

__version__ = metadata.version(__name__)

from .utils.misc import setup_root_logging

setup_root_logging("pyloki")
