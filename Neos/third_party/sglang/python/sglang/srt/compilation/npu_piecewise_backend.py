from contextlib import ExitStack
from typing import Any, Callable
from unittest.mock import patch

import torch
import torch.fx as fx

from sglang.srt.compilation.compilation_config import CompilationConfig
from sglang.srt.compilation.compilation_counter import compilation_counter
from sglang.srt.compilation.cuda_piecewise_backend import (
    CUDAPiecewiseBackend,
    weak_ref_tensors,
)


class NPUPiecewiseBackend(CUDAPiecewiseBackend):
    def __init__(
        self,
        graph: fx.GraphModule,
        compile_config: CompilationConfig,
        inductor_config: dict[str, Any],
        graph_pool: Any,
        piecewise_compile_index: int,
        total_piecewise_compiles: int,
        sym_shape_indices: list[int],
        compiled_graph_for_general_shape: Callable,
        sglang_backend,
    ):
        super().__init__(
            graph,
            compile_config,
            inductor_config,
            graph_pool,
            piecewise_compile_index,
            total_piecewise_compiles,
            sym_shape_indices,
            compiled_graph_for_general_shape,
            sglang_backend,
        )

    def __call__(self, *args):
        runtime_shape = args[self.sym_shape_indices[0]]
        if runtime_shape not in self.concrete_size_entries:
            
            return self.compiled_graph_for_general_shape(*args)

        entry = self.concrete_size_entries[runtime_shape]

        if entry.runnable is None:
            entry.runnable = self.compiled_graph_for_general_shape

        if entry.cudagraph is None:
            if entry.num_finished_warmup < 1:  
                entry.num_finished_warmup += 1
                return entry.runnable(*args)

            if self.compile_config.get_enable_debug_mode():
                input_addresses = [
                    x.data_ptr() for x in args if isinstance(x, torch.Tensor)
                ]
                entry.input_addresses = input_addresses
            npugraph = torch.npu.NPUGraph()

            with ExitStack() as stack:
                if not self.is_first_graph:
                    
                    
                    
                    
                    
                    
                    stack.enter_context(patch("gc.collect", lambda: None))
                    stack.enter_context(patch("torch.npu.empty_cache", lambda: None))

                
                with torch.npu.graph(npugraph, pool=self.graph_pool):
                    
                    output = entry.runnable(*args)
                    if self.is_last_graph:
                        
                        
                        
                        
                        
                        output = weak_ref_tensors(output)

            
            
            entry.output = weak_ref_tensors(output)
            entry.cudagraph = npugraph

            compilation_counter.num_cudagraph_captured += 1

            
            
            
            return output

        if self.compile_config.get_enable_debug_mode():
            
            new_input_addresses = [
                x.data_ptr() for x in args if isinstance(x, torch.Tensor)
            ]
            assert new_input_addresses == entry.input_addresses, (
                "Input addresses for cudagraphs are different during replay."
                f" Expected {entry.input_addresses}, got {new_input_addresses}"
            )
        entry.cudagraph.replay()
        return entry.output
