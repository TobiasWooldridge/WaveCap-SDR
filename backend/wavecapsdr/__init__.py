__all__ = [
    "__version__",
    "create_app",
]

__version__ = "0.1.0"

# Lazy import to avoid requiring FastAPI for submodule usage
def create_app(*args, **kwargs):
    from .app import create_app as _create_app
    return _create_app(*args, **kwargs)

