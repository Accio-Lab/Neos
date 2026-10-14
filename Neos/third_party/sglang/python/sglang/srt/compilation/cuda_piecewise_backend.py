

import dataclasses
import logging
from contextlib import ExitStack
from typing import Any, Callable, Optional
from unittest.mock import patch

import torch
import torch.fx as fx

from sglang.srt.compilation.compilation_config import CompilationConfig
from sglang.srt.compilation.compilation_counter import compilation_counter
from sglang.srt.compilation.piecewise_context_manager import (
    get_pcg_capture_stream,
    is_in_pcg_torch_compile,
)
from sglang.srt.compilation.weak_ref_tensor import weak_ref_tensors

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class ConcreteSizeEntry:
    runtime_shape: int
    need_to_compile: bool  
    use_cudagraph: bool  

    compiled: bool = False
    runnable: Callable = None  # type: ignore
    num_finished_warmup: int = 0
    cudagraph: Optional[torch.cuda.CUDAGraph] = None
    output: Optional[Any] = None

    
    
    input_addresses: Optional[list[int]] = None


class CUDAPiecewiseBackend:

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
        """
        The backend for piecewise compilation.
        It mainly handles the compilation and cudagraph capturing.

        We will compile `self.graph` once for the general shape,
        and then compile for different shapes specified in
        `compilation_config.compile_sizes`.

        Independently, we will capture cudagraph for different shapes.

        If a shape needs both compilation and cudagraph, we will
        compile it first, and then capture cudagraph.
        """
        self.graph = graph
        self.inductor_config = inductor_config
        self.graph_pool = graph_pool
        self.piecewise_compile_index = piecewise_compile_index
        self.total_piecewise_compiles = total_piecewise_compiles
        self.sglang_backend = sglang_backend

        self.is_first_graph = piecewise_compile_index == 0
        self.is_last_graph = piecewise_compile_index == total_piecewise_compiles - 1

        self.compile_sizes: set[int] = set([])
        self.compile_config = compile_config
        self.cudagraph_capture_sizes: set[int] = set(compile_config.get_capture_sizes())

        self.first_run_finished = False

        self.compiled_graph_for_general_shape = compiled_graph_for_general_shape  

        self.sym_shape_indices = sym_shape_indices

        
        
        self.concrete_size_entries: dict[int, ConcreteSizeEntry] = {}

        
        
        self.to_be_compiled_sizes: set[int] = self.compile_sizes.copy()
        for shape in self.compile_sizes.union(self.cudagraph_capture_sizes):
            self.concrete_size_entries[shape] = ConcreteSizeEntry(
                runtime_shape=shape,
                need_to_compile=shape in self.compile_sizes,
                use_cudagraph=shape in self.cudagraph_capture_sizes,
            )

    def check_for_ending_compilation(self):
        if self.is_last_graph and not self.to_be_compiled_sizes:
            
            
            self.sglang_backend.compiler_manager.save_to_file()

    def __call__(self, *args) -> Any:
        if not self.first_run_finished:
            self.first_run_finished = True
            self.check_for_ending_compilation()
            return self.compiled_graph_for_general_shape(*args)

        if len(self.sym_shape_indices) == 0:
            return self.compiled_graph_for_general_shape(*args)

        runtime_shape = args[self.sym_shape_indices[0]]
        if runtime_shape not in self.concrete_size_entries:
            
            return self.compiled_graph_for_general_shape(*args)

        entry = self.concrete_size_entries[runtime_shape]

        if entry.runnable is None:
            entry.runnable = self.compiled_graph_for_general_shape

        if entry.need_to_compile and not entry.compiled:
            entry.compiled = True
            self.to_be_compiled_sizes.remove(runtime_shape)
            
            entry.runnable = self.sglang_backend.compiler_manager.compile(
                self.graph,
                args,
                self.inductor_config,
                graph_index=self.piecewise_compile_index,
                num_graphs=self.total_piecewise_compiles,
                runtime_shape=runtime_shape,
            )

            
            if self.is_last_graph and not self.to_be_compiled_sizes:
                self.check_for_ending_compilation()

        if is_in_pcg_torch_compile():
            return entry.runnable(*args)

        if entry.cudagraph is None:
            if entry.num_finished_warmup < 1:  
                entry.num_finished_warmup += 1
                return entry.runnable(*args)

            if self.compile_config.get_enable_debug_mode():
                input_addresses = [
                    x.data_ptr() for x in args if isinstance(x, torch.Tensor)
                ]
                entry.input_addresses = input_addresses
            cudagraph = torch.cuda.CUDAGraph()

            with ExitStack() as stack:
                if not self.is_first_graph:
                    
                    
                    
                    
                    
                    
                    stack.enter_context(patch("gc.collect", lambda: None))
                    stack.enter_context(patch("torch.cuda.empty_cache", lambda: None))
                
                stream = get_pcg_capture_stream()
                assert (
                    stream is not None
                ), "PCG capture stream is not set, please check if runtime recompilation happened"
                with torch.cuda.graph(cudagraph, pool=self.graph_pool, stream=stream):
                    
                    output = entry.runnable(*args)
                    if self.is_last_graph:
                        
                        
                        
                        
                        
                        output = weak_ref_tensors(output)

            
            
            entry.output = weak_ref_tensors(output)
            entry.cudagraph = cudagraph

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
