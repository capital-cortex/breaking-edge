from importlib import metadata as _md

try:
    __version__ = _md.version("be")
except _md.PackageNotFoundError:
    __version__ = "0.0.0"
