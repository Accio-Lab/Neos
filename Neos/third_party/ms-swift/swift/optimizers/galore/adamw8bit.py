
import torch
from bitsandbytes.optim.optimizer import Optimizer2State

from swift.utils import synchronize
from .galore_projector import GaLoreProjector


class AdamW8bit(Optimizer2State):

    def __init__(self,
                 params,
                 lr=1e-3,
                 betas=(0.9, 0.999),
                 eps=1e-8,
                 weight_decay=1e-2,
                 amsgrad=False,
                 optim_bits=32,
                 args=None,
                 min_8bit_size=4096,
                 percentile_clipping=100,
                 block_wise=True,
                 is_paged=False):
        super().__init__(
            'adam',
            params,
            lr,
            betas,
            eps,
            weight_decay,
            8,
            args,
            min_8bit_size,
            percentile_clipping,
            block_wise,
            is_paged=is_paged)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        if not self.initialized:
            self.check_overrides()
            self.to_gpu()  
            self.initialized = True

        
        for gindex, group in enumerate(self.param_groups):
            for pindex, p in enumerate(group['params']):
                if p.grad is None:
                    continue
                state = self.state[p]

                if 'step' not in state:
                    state['step'] = 0

                
                if 'rank' in group:
                    if 'projector' not in state:
                        state['projector'] = GaLoreProjector(
                            group['rank'],
                            update_proj_gap=group['update_proj_gap'],
                            scale=group['scale'],
                            proj_type=group['proj_type'])

                    if 'weight_decay' in group and group['weight_decay'] > 0:
                        
                        group['weight_decay_saved'] = group['weight_decay']
                        group['weight_decay'] = 0

                    grad = state['projector'].project(p.grad, state['step'])

                    
                    p.saved_data = p.data.clone()
                    p.data = grad.clone().to(p.data.dtype).to(p.data.device)
                    p.data.zero_()
                    p.grad = grad

                if 'state1' not in state:
                    self.init_state(group, p, gindex, pindex)

                self.prefetch_state(p)
                self.update_step(group, p, gindex, pindex)
                synchronize()

                
                if 'rank' in group:
                    p.data = p.saved_data.add_(state['projector'].project_back(p.data))

                    
                    if 'weight_decay_saved' in group:
                        p.data.add_(p.data, alpha=-group['lr'] * group['weight_decay_saved'])
                        group['weight_decay'] = group['weight_decay_saved']
                        del group['weight_decay_saved']

        if self.is_paged:
            
            
            synchronize()

        return loss


GaLoreAdamW8bit = AdamW8bit
