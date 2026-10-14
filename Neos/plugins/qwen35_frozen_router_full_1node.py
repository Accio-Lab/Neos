
import functools
import importlib.util
import json
import os
from pathlib import Path
import re
import sys

_base = Path(__file__).with_name('qwen35_neos_full_1node.py')
_spec = importlib.util.spec_from_file_location('neos_proven_35b_single_node', _base)
_module = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _module
_spec.loader.exec_module(_module)

import torch
import torch.distributed as dist
from swift.megatron.trainers.base import BaseMegatronTrainer

ROUTER_PATTERN = r'(?:^|\.)mlp\.router\.'


def inspect_routers(trainer, models, stage):
    parameters = [(name, p) for model in models for name, p in model.named_parameters()]
    routers = [(name, p) for name, p in parameters if re.search(ROUTER_PATTERN, name)]
    assert routers and all(not p.requires_grad for _, p in routers)
    assert all(p.grad is None for _, p in routers)
    assert not trainer.config.moe_router_enable_expert_bias
    report = {'stage': stage, 'rank': dist.get_rank(), 'router_tensors': len(routers),
              'router_parameters_on_this_rank': sum(p.numel() for _, p in routers),
              'router_trainable_parameters': 0, 'router_gradients_present': False,
              'router_names': [name for name, _ in routers],
              'other_trainable_parameters_on_this_rank': sum(p.numel() for name, p in parameters if p.requires_grad)}
    target = Path(os.environ['SFT_ROUTER_AUDIT_DIR'])
    target.mkdir(parents=True, exist_ok=True)
    path = target / f'rank{dist.get_rank()}.{stage}.json'
    temporary = path.with_suffix(f'.tmp.{os.getpid()}')
    temporary.write_text(json.dumps(report, indent=2) + '\n')
    temporary.replace(path)
    print('frozen_router_verified ' + json.dumps({k: v for k, v in report.items() if k != 'router_names'}), flush=True)
    return routers


if not getattr(BaseMegatronTrainer, '_neos_frozen_router_verified', False):
    _prepare = BaseMegatronTrainer._prepare_peft_model
    _setup = BaseMegatronTrainer.setup_model_training

    @functools.wraps(_prepare)
    def prepare_with_router_check(self, models):
        
        
        result = _prepare(self, models)
        inspect_routers(self, models, 'before_ddp_optimizer')
        return result

    @functools.wraps(_setup)
    def setup_with_router_check(self):
        result = _setup(self)
        routers = inspect_routers(self, self.unwrapped_models, 'before_training')
        
        
        
        sentinels = [(p, p.detach().reshape(-1)[:16].clone()) for _, p in routers]
        original_step = self.optimizer.step
        checked = False

        @functools.wraps(original_step)
        def step_with_router_check(*args, **kwargs):
            nonlocal checked
            value = original_step(*args, **kwargs)
            if not checked:
                assert all(torch.equal(p.detach().reshape(-1)[:16], before) for p, before in sentinels)
                inspect_routers(self, self.unwrapped_models, 'after_first_optimizer_step')
                sentinels.clear()
                checked = True
            return value

        self.optimizer.step = step_with_router_check
        return result

    BaseMegatronTrainer._prepare_peft_model = prepare_with_router_check
    BaseMegatronTrainer.setup_model_training = setup_with_router_check
    BaseMegatronTrainer._neos_frozen_router_verified = True
