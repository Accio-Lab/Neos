
import functools
import importlib.util
from pathlib import Path
import sys

base = Path(__file__).with_name('qwen35_neos_native_tools.py')
spec = importlib.util.spec_from_file_location('neos_qwen35_native_base', base)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)

from swift.megatron.trainers.base import BaseMegatronTrainer
from verl.utils.megatron.optimizer_memory import release_optimizer_step_cache, warmup_optimizer_grad_stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from neos_sft_linear_ce import install, distributed_kernel_selftest
install()

if not getattr(BaseMegatronTrainer, '_neos_qwen35_single_node', False):
    original_setup = BaseMegatronTrainer.setup_model_training

    @functools.wraps(original_setup)
    def setup_with_memory_guards(self):
        warmup_optimizer_grad_stats(self.optimizer)
        from megatron.core import parallel_state
        distributed_kernel_selftest(parallel_state.get_tensor_model_parallel_group())
        original_step = self.optimizer.step

        @functools.wraps(original_step)
        def step_with_headroom(*args, **kwargs):
            release_optimizer_step_cache()
            return original_step(*args, **kwargs)

        self.optimizer.step = step_with_headroom
        return original_setup(self)

    BaseMegatronTrainer.setup_model_training = setup_with_memory_guards
    BaseMegatronTrainer._neos_qwen35_single_node = True
