"""Opt-in expert-selection router freeze matching the Qwen SFT parameter scope."""
from __future__ import annotations

import json
import re
import torch

_PATTERN = re.compile(r'(?:^|\.)mlp\.router\.')


def router_parameters(models):
    models = models if isinstance(models, (list, tuple)) else [models]
    found = {}
    for model in models:
        for name, parameter in model.named_parameters():
            if _PATTERN.search(name):
                found[id(parameter)] = (name, parameter)
    return list(found.values())


def assert_expert_routers_frozen(models):
    parameters = router_parameters(models)
    if not parameters:
        raise ValueError('Expert-router freeze requested but no .mlp.router. parameters found')
    bad = [n for n, p in parameters if p.requires_grad or p.grad is not None]
    if bad:
        raise RuntimeError('Expert-router freezing was lost: ' + ', '.join(bad[:5]))
    return {
        'router_tensors': len(parameters),
        'router_parameters_on_this_rank': sum(p.numel() for _, p in parameters),
        'trainable_router_parameters': 0,
        'router_gradients_present': False,
        'scope': 'expert_router_only',
    }


def freeze_expert_routers(model, **_provider_context):
    """Runs before DDP/optimizer. Do NOT freeze shared-expert or attention gates."""
    parameters = router_parameters(model)
    if not parameters:
        raise ValueError('Expert-router freeze requested but no .mlp.router. parameters found')
    for _, parameter in parameters:
        parameter.requires_grad_(False)
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else None
    print(json.dumps({'router_freeze_stage': 'before_ddp', 'rank': rank,
                      **assert_expert_routers_frozen(model)}), flush=True)
    return model


def expert_router_only_requested(engine_config):
    config = (engine_config.override_mcore_model_config or {}).get('moe_config', {})
    return bool(config.get('freeze_moe_router', False)) and config.get('router_freeze_scope') == 'expert_router_only'
