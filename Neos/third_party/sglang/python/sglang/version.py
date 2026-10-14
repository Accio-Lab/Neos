try:
    from sglang._version import __version__, __version_tuple__
except ImportError:
    try:
        import importlib.metadata

        __version__ = importlib.metadata.version("sglang")
        __version_tuple__ = tuple(__version__.split("."))
    except Exception:
        try:
            import pathlib

            from setuptools_scm import get_version

            
            project_root = pathlib.Path(__file__).parent.parent.parent
            __version__ = get_version(
                root=str(project_root), fallback_version="0.0.0.dev0"
            )
            __version_tuple__ = tuple(__version__.split("."))
        except Exception:
            
            __version__ = "0.0.0.dev0"
            __version_tuple__ = (0, 0, 0, "dev0")
