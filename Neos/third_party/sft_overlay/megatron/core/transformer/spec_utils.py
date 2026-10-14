# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
import functools
import logging
import types
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Tuple, Union

logger = logging.getLogger(__name__)


@dataclass
class ModuleSpec:
    """This is a Module Specification dataclass.

    Specification defines the location of the module (to import dynamically)
    or the imported module itself. It also defines the params that need to be
    passed to initialize the module.

    Args:
        module (Union[Tuple, type]): A tuple describing the location of the
            module class e.g. `(module.location, ModuleClass)` or the imported
            module class itself e.g. `ModuleClass` (which is already imported
            using `from module.location import ModuleClass`).
        params (dict): A dictionary of params that need to be passed while init.

    """

    module: Union[Tuple, type]
    params: dict = field(default_factory=lambda: {})
    submodules: object = None
    metainfo: dict = field(default_factory=lambda: {})

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Builds an instance of the module from the spec.

        Args:
            *args: Positional arguments to be passed to the module init.
            **kwargs: Keyword arguments to be passed to the module init.
        """
        return build_module(self, *args, **kwargs)


def import_module(module_path: Tuple[str]):
    """Import a named object from a module in the context of this function.

    TODO: make this importer module more robust, at least make sure there
    are no side effects of using this as is
    """
    base_path, name = module_path
    try:
        module = __import__(base_path, globals(), locals(), [name])
    except ImportError as e:
        logger.error(f"couldn't import module due to {e}")
        return None
    return vars(module)[name]



def get_module(spec_or_module: Union[ModuleSpec, type], **additional_kwargs):
    """Returns or imports the provided module."""
    
    if isinstance(spec_or_module, (type, types.FunctionType)):
        return spec_or_module

    
    if isinstance(spec_or_module.module, (type, types.FunctionType)):
        return spec_or_module.module

    
    return import_module(spec_or_module.module)


def build_module(spec_or_module: Union[ModuleSpec, type], *args, **kwargs):
    """Builds an instance of the module from the spec.

    Args:
        spec_or_module: The module spec or module class to build.
        *args: Positional arguments to be passed to the module init.
        **kwargs: Keyword arguments to be passed to the module init.
    """
    
    
    
    
    if isinstance(spec_or_module, types.FunctionType):
        return spec_or_module

    
    
    
    if isinstance(spec_or_module, ModuleSpec) and isinstance(
        spec_or_module.module, types.FunctionType
    ):
        return spec_or_module.module

    
    
    if isinstance(spec_or_module, type):
        module = spec_or_module
    elif hasattr(spec_or_module, "module") and isinstance(spec_or_module.module, type):
        module = spec_or_module.module
    else:
        
        module = import_module(spec_or_module.module)

    
    if isinstance(module, types.FunctionType):
        return module

    
    

    
    
    if hasattr(spec_or_module, "submodules") and spec_or_module.submodules is not None:
        kwargs["submodules"] = spec_or_module.submodules

    try:
        return module(
            *args, **spec_or_module.params if hasattr(spec_or_module, "params") else {}, **kwargs
        )
    except Exception as e:
        
        import sys

        raise type(e)(f"{str(e)} when instantiating {module.__name__}").with_traceback(
            sys.exc_info()[2]
        )


def get_submodules(spec: Callable[..., Any]) -> object:
    """Gets the `submodules` field from the provided spec.

    Supports `partial` objects, as well as ModuleSpec or any other object with a `submodules` attr.
    Raises a ValueError if the provided spec is not supported.
    """
    if isinstance(spec, functools.partial):
        return spec.keywords['submodules']
    if hasattr(spec, "submodules"):
        return spec.submodules  # type: ignore
    raise ValueError(f"Could not find `submodules` in the provided spec: {spec!r}")
