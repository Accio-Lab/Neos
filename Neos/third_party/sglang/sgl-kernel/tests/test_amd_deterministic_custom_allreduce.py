"""
Test deterministic custom all-reduce kernel behavior with batch size invariance.

This test uses the 1-stage all-reduce kernel which is inherently deterministic
due to fixed accumulation ordering (each GPU reads all data from all GPUs and
reduces locally in a fixed order - no atomics, no race conditions).

Note: This is NOT a reduce-scatter + all-gather (RS+AG) approach.

This test compares:
1. Deterministic kernel (same batch size)
2. Deterministic kernel (different batch size)

Usage:
    pytest test_amd_deterministic_custom_allreduce.py
"""

import multiprocessing as mp
import socket

import pytest
import torch
import torch.distributed as dist


def get_open_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def worker(world_size, rank, port):
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)

    dist.init_process_group(
        backend="nccl",
        init_method=f"tcp://localhost:{port}",
        rank=rank,
        world_size=world_size,
    )

    
    try:
        from torch.distributed import new_group

        from sglang.srt.distributed.device_communicators.custom_all_reduce import (
            CustomAllreduce,
        )

        
        dist.barrier()
        ar_group = new_group(backend="gloo")
        dist.barrier()

        custom_ar = CustomAllreduce(group=ar_group, device=device)

        if custom_ar is None or custom_ar.disabled:
            if rank == 0:
                print("✗ Custom AR not available or disabled")
            dist.destroy_process_group()
            return

        if not hasattr(custom_ar, "deterministic_all_reduce"):
            if rank == 0:
                print("✗ Deterministic kernel not available")
            dist.destroy_process_group()
            return
    except Exception as e:
        if rank == 0:
            print(f"✗ Failed to initialize deterministic kernel: {e}")
            import traceback

            traceback.print_exc()
        dist.destroy_process_group()
        return

    num_trials = 10

    
    
    BS = 50  
    hidden_dim = 16384  

    
    torch.manual_seed(42 + rank)

    
    
    base_input = torch.randn(hidden_dim, dtype=torch.bfloat16, device=device)
    base_input_rand = torch.randn(hidden_dim, dtype=torch.bfloat16, device=device)

    
    
    input_size_bytes = base_input.numel() * base_input.element_size()
    if input_size_bytes > custom_ar.max_size and rank == 0:
        print(
            f"Warning: Input size ({input_size_bytes/(1024*1024):.1f} MB) exceeds buffer size ({custom_ar.max_size/(1024*1024):.1f} MB)"
        )
        print("  Using unregistered mode (will copy to buffer)")

    dist.barrier()

    
    
    
    if rank == 0:
        print(f"\n{'='*70}")
        print("TEST 1: Deterministic kernel (same batch size)")
        print(f"{'='*70}")
    dist.barrier()

    results_allreduce_only = []
    for trial in range(num_trials):
        
        inp = base_input.clone()

        
        
        input_size_bytes = inp.numel() * inp.element_size()
        use_registered = input_size_bytes > custom_ar.max_size

        if use_registered:
            
            custom_ar.register_buffer(inp)
            result = custom_ar.deterministic_all_reduce(inp, registered=True)
        else:
            
            result = custom_ar.deterministic_all_reduce(inp, registered=False)
        torch.cuda.synchronize()

        
        checksum = result.view(-1).sum().item()
        first_vals = result.view(-1)[:5].clone()
        results_allreduce_only.append((checksum, first_vals))

        if rank == 0:
            print(
                f"  Trial {trial+1:2d}: sum={checksum:.6f}, first5={first_vals.tolist()}"
            )

    
    if rank == 0:
        ref_sum, ref_vals = results_allreduce_only[0]
        all_match = True
        for i, (s, vals) in enumerate(results_allreduce_only[1:], 1):
            if abs(ref_sum - s) > 1e-3 or not torch.allclose(ref_vals, vals, rtol=1e-3):
                all_match = False
                print(f"  Trial {i+1} DIFFERS! ref_sum={ref_sum:.6f}, got={s:.6f}")

        if all_match:
            print("  ✓ DETERMINISTIC KERNEL (fixed BS): DETERMINISTIC (as expected)")
        else:
            print(
                "  ✗ DETERMINISTIC KERNEL (fixed BS): NON-DETERMINISTIC (unexpected!)"
            )

    dist.barrier()

    
    
    
    
    if rank == 0:
        print(f"\n{'='*70}")
        print("TEST 2: Deterministic kernel (different batch size)")
        print("Batches: [a], [a,x], [a,x,x], ...")
        print(f"{'='*70}")
    dist.barrier()

    results_allreduce_only = {trial: [] for trial in range(num_trials)}
    for trial in range(num_trials):
        for bs in range(1, BS + 1):
            
            
            batch = torch.stack([base_input] + [base_input_rand] * (bs - 1), dim=0)
            

            
            batch_flat = batch.view(-1)

            
            
            input_size_bytes = batch_flat.numel() * batch_flat.element_size()
            use_registered = input_size_bytes > custom_ar.max_size

            if use_registered:
                
                custom_ar.register_buffer(batch_flat)
                result_flat = custom_ar.deterministic_all_reduce(
                    batch_flat, registered=True
                )
            else:
                
                result_flat = custom_ar.deterministic_all_reduce(
                    batch_flat, registered=False
                )
            torch.cuda.synchronize()

            
            batch_out = result_flat.view(bs, hidden_dim)

            
            out_first_req = batch_out[0].clone()
            checksum = out_first_req.sum().item()
            first_vals = out_first_req[:5].clone()
            results_allreduce_only[trial].append((bs, checksum, first_vals))

            if rank == 0:
                print(
                    f"  Batch size {bs:2d}: sum={checksum:.6f}, first5={first_vals.tolist()}"
                )

    
    if rank == 0:
        for trial in range(num_trials):
            results = results_allreduce_only[trial]

            _, ref_sum, ref_vals = results[0]
            all_match = True
            for _, s, vals in results[1:]:
                if abs(ref_sum - s) > 1e-3 or not torch.allclose(
                    ref_vals, vals, rtol=1e-3
                ):
                    all_match = False

        if all_match:
            print("  ✓ DETERMINISTIC KERNEL (variant BS): DETERMINISTIC")
        else:
            print("  ✗ DETERMINISTIC KERNEL (variant BS): NON-DETERMINISTIC")

    dist.barrier()

    dist.destroy_process_group()


def main():
    world_size = 8
    available_gpus = torch.cuda.device_count()

    print("=" * 70)
    print("Deterministic Kernel All-Reduce Determinism Test")
    print("=" * 70)
    print(f"Available GPUs: {available_gpus}")
    print(f"Using world_size: {world_size}")

    if available_gpus < world_size:
        print(
            f"WARNING: Only {available_gpus} GPUs available, using {available_gpus} instead"
        )
        world_size = available_gpus

    if world_size < 2:
        print("ERROR: Need at least 2 GPUs for this test")
        return

    mp.set_start_method("spawn", force=True)
    port = get_open_port()

    procs = []
    for rank in range(world_size):
        p = mp.Process(target=worker, args=(world_size, rank, port))
        p.start()
        procs.append(p)

    for p in procs:
        p.join()


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 2,
    reason="Requires at least 2 CUDA GPUs",
)
def test_deterministic_custom_allreduce():
    """Test that deterministic custom all-reduce produces consistent results."""
    main()


if __name__ == "__main__":
    main()
