"""Fixtures for E2E tests.

This package contains modular pytest fixtures split by responsibility:
- hooks.py: Pytest collection hooks and marker registration
- pool.py: Model pool fixtures (session-scoped worker management)
- setup_backend.py: Backend setup fixtures (class/function-scoped)
- markers.py: Helper utilities for marker extraction

Legacy modules (to be removed during e2e_response_api migration):
- ports.py: Use infra.get_open_port() instead
- router_manager.py: Use infra.Gateway instead
"""


from .hooks import (
    get_pool_requirements,
    is_parallel_execution,
    pytest_collection_finish,
    pytest_collection_modifyitems,
    pytest_configure,
    pytest_runtest_setup,
    validate_gpu_requirements,
)


from .markers import get_marker_kwargs, get_marker_value


from .pool import model_base_url, model_client, model_pool
from .setup_backend import backend_router, setup_backend

__all__ = [
    
    "pytest_collection_modifyitems",
    "pytest_collection_finish",
    "pytest_configure",
    "pytest_runtest_setup",
    "get_pool_requirements",
    "validate_gpu_requirements",
    "is_parallel_execution",
    
    "model_pool",
    "model_client",
    "model_base_url",
    
    "setup_backend",
    "backend_router",
    
    "get_marker_value",
    "get_marker_kwargs",
]
