"""Vision-tower freezing that runs before DDP/optimizer parameter collection."""
from __future__ import annotations

import json
from typing import Any

import torch

_VISION_NAMES = frozenset({'vision_model', 'vision_tower', 'vision_encoder', 'visual'})


def configure_actor_vision_freeze(actor_config: Any) -> None:
    """Carry the public Actor flag into the Megatron engine configuration."""
    if actor_config.engine.strategy == 'megatron':
        actor_config.engine.freeze_vision_tower = bool(actor_config.freeze_vision_tower)


def vision_tower_parameters(model):
    parameters = {}
    for name, module in model.named_modules():
        if name.rsplit('.', 1)[-1] in _VISION_NAMES:
            for parameter in module.parameters():
                parameters[id(parameter)] = parameter
    return list(parameters.values())


def freeze_vision_tower(model, **_provider_context):
    """Freeze the complete tower, including an internal merger, before wrapping."""
    parameters = vision_tower_parameters(model)
    if not parameters:
        if getattr(model, 'pre_process', True) is False or getattr(model, 'add_encoder', True) is False:
            return model  
        raise ValueError('freeze_vision_tower requested, but no supported vision tower was found')
    for parameter in parameters:
        parameter.requires_grad_(False)
    print(json.dumps({'vision_freeze_stage': 'before_ddp',
                      'vision_parameters': sum(p.numel() for p in parameters),
                      'trainable_vision_parameters': 0,
                      'remaining_trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad)}),
          flush=True)
    return model


def assert_vision_tower_frozen(models):
    """Fail before optimizer creation if weight loading/other hooks undid freezing."""
    models = models if isinstance(models, (list, tuple)) else [models]
    parameters = {id(p): p for model in models for p in vision_tower_parameters(model)}
    trainable = [p for p in parameters.values() if p.requires_grad]
    if trainable:
        raise RuntimeError(f'Vision freezing was lost: {sum(p.numel() for p in trainable)} trainable parameters')
    return {'vision_parameters': sum(p.numel() for p in parameters.values()),
            'trainable_vision_parameters': 0}
