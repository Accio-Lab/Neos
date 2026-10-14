
import functools
import inspect
import json

import torch
import torch.distributed as dist

from verl.utils.kernel.linear_cross_entropy import linear_cross_entropy


def sft_token_loss(model, hidden_states, labels, lce=linear_cross_entropy):
    if model.config.sequence_parallel:
        from megatron.core.tensor_parallel import gather_from_sequence_parallel_region
        hidden_states = gather_from_sequence_parallel_region(hidden_states, group=model.pg_collection.tp)
    
    assert hidden_states.ndim == 3 and labels.ndim == 2
    assert tuple(hidden_states.shape[:2]) == (labels.shape[1], labels.shape[0])
    hidden = hidden_states.transpose(0, 1).contiguous()
    valid = labels != -100
    safe_labels = labels.masked_fill(~valid, 0).contiguous()
    weight = model.output_layer.weight
    if weight is None:
        weight = model.shared_embedding_or_output_weight()
    assert getattr(model.output_layer, 'bias', None) is None
    logprobs, _ = lce(hidden, weight, safe_labels, 1.0, 'none', model.pg_collection.tp)
    return -logprobs.reshape_as(labels) * valid


def install():
    from mcore_bridge.model.gpt_model import GPTModel
    if getattr(GPTModel, '_neos_sft_linear_ce', False): return
    original = GPTModel._postprocess
    signature = inspect.signature(original)

    @functools.wraps(original)
    def postprocess(self, *args, **kwargs):
        values = signature.bind(self, *args, **kwargs).arguments
        labels = values.get('labels')
        if not self.training or labels is None or not self.post_process:
            return original(self, *args, **kwargs)
        assert self.config.task_type == 'causal_lm'
        assert not self.mtp_process, 'This SFT profile does not train MTP'
        return sft_token_loss(self, values['hidden_states'], labels)

    GPTModel._postprocess = postprocess
    GPTModel._neos_sft_linear_ce = True


def distributed_kernel_selftest(tp_group):
    pass
    size, rank = dist.get_world_size(tp_group), dist.get_rank(tp_group)
    vocab_size = ((4096 + size - 1) // size) * size
    
    
    h = torch.linspace(-0.03, 0.04, 128*128, device='cuda', dtype=torch.float32).reshape(128,128).requires_grad_()
    w = torch.sin(torch.arange(vocab_size*128,device='cuda',dtype=torch.float32)*0.003).reshape(vocab_size,128)
    local_w = w.chunk(size)[rank].clone().requires_grad_()
    labels = (torch.arange(128,device='cuda')*37) % vocab_size
    lp, _ = linear_cross_entropy(h, local_w, labels, 1.0, 'none', tp_group)
    (-lp).sum().backward()
    actual_dh = h.grad.clone()
    dist.all_reduce(actual_dh,group=tp_group)
    ref_h = h.detach().clone().requires_grad_()
    ref_w = w.clone().requires_grad_()
    reference = torch.nn.functional.cross_entropy(ref_h@ref_w.T, labels, reduction='none')
    reference.sum().backward()
    torch.testing.assert_close(-lp.reshape_as(reference),reference,atol=2e-3,rtol=2e-3)
    torch.testing.assert_close(actual_dh,ref_h.grad,atol=2e-3,rtol=2e-3)
    torch.testing.assert_close(local_w.grad,ref_w.grad.chunk(size)[rank],atol=2e-3,rtol=2e-3)
    print('sft_linear_ce_selftest '+json.dumps({'rank':dist.get_rank(),'status':'PASS','tp':size}),flush=True)
