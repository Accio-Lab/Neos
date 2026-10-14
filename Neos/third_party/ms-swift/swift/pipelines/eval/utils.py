"""
EvalScope integration utilities for ms-swift models.

This module provides a custom ModelAPI implementation that enables batch inference
for evaluation tasks using ms-swift's TransformersEngine. It implements an asynchronous
batch processing system to improve throughput when evaluating models.
"""

from concurrent.futures import Future
from dataclasses import dataclass
from evalscope.api.messages import ChatMessage as EvalChatMessage
from evalscope.api.model import GenerateConfig, ModelAPI, ModelOutput, ModelUsage
from evalscope.api.registry import register_model_api
from evalscope.api.tool import ToolChoice, ToolInfo
from evalscope.models.utils.openai import chat_choices_from_openai
from queue import Empty, Queue
from threading import Thread
from typing import Any, List, Optional, Tuple

from swift.infer_engine import InferRequest, RequestConfig, TransformersEngine


@dataclass
class BatchInferInput:
    """
    Container for batch inference input data.

    Holds all necessary information for a single inference request
    that will be processed as part of a batch.
    """
    ms_input: InferRequest  
    ms_config: RequestConfig  
    batch_size: int  
    engine: TransformersEngine  


@dataclass
class _QueueItem:
    """
    Internal queue item for batch processing.

    Pairs a batch input with its corresponding future for result delivery.
    """
    input: BatchInferInput
    future: Future[ModelOutput]  




batch_thread: Optional[Thread] = None  
batch_queue: Queue[_QueueItem] = Queue()  


@register_model_api('swift_custom')
class EvalModel(ModelAPI):
    """
    Custom ModelAPI implementation for ms-swift models with batch inference support.

    This class integrates ms-swift's TransformersEngine with EvalScope's evaluation framework,
    providing efficient batch processing for improved evaluation throughput.
    """

    def __init__(
            self,
            model_name: str,
            base_url: Optional[str] = None,
            api_key: Optional[str] = None,
            config: GenerateConfig = GenerateConfig(),
            **model_args: Any,
    ):
        """
        Initialize the EvalModel with ms-swift backend.

        Args:
            model_name: Name of the model for identification
            base_url: Not used in this implementation (for API compatibility)
            api_key: Not used in this implementation (for API compatibility)
            config: Generation configuration with batch settings
            **model_args: Additional arguments including 'model' and 'template'
        """
        super().__init__(
            model_name=model_name,
            base_url=base_url,
            api_key=api_key,
            config=config,
        )

        
        
        def collect_model_arg(name: str) -> Optional[Any]:
            value = model_args.get(name, None)
            if value is not None:
                model_args.pop(name)
            return value

        
        self.model = collect_model_arg('model')  
        self.template = collect_model_arg('template')  
        self.max_batch_size = collect_model_arg('max_batch_size')  

        
        self.engine = TransformersEngine(self.model, template=self.template, max_batch_size=self.max_batch_size)

    def generate(
        self,
        input: List[EvalChatMessage],
        tools: List[ToolInfo],
        tool_choice: ToolChoice,
        config: GenerateConfig,
    ) -> ModelOutput:
        """
        Generate model response using batch inference.

        This method queues the request for batch processing and waits for the result.
        The actual inference is performed asynchronously in a background thread.

        Args:
            input: List of chat messages forming the conversation
            tools: Available tools for function calling (if supported)
            tool_choice: Tool selection strategy
            config: Generation configuration

        Returns:
            ModelOutput containing the generated response
        """
        
        global batch_thread
        if batch_thread is None:
            batch_thread = Thread(target=_process_batches, daemon=True)
            batch_thread.start()

        
        ms_input = convert_request(input, tools)
        ms_config = convert_config(config)

        
        batch_input = BatchInferInput(
            ms_input=ms_input, ms_config=ms_config, batch_size=config.batch_size, engine=self.engine)

        
        future = Future[ModelOutput]()

        
        batch_queue.put(_QueueItem(input=batch_input, future=future))

        
        return future.result()


def _process_batches() -> None:
    """
    Background thread function that processes batched inference requests.

    This function runs continuously, collecting requests from the queue and
    processing them in batches for improved efficiency. It uses a timeout-based
    approach to balance between batch size and latency.
    """
    while True:
        
        inputs: List[Tuple[BatchInferInput, Future[ModelOutput]]] = []

        while True:
            try:
                
                item = batch_queue.get(timeout=2)
                inputs.append((item.input, item.future))

                
                if len(inputs) == item.input.batch_size:
                    break  

            except Empty:
                
                break

        
        if len(inputs) == 0:
            continue

        try:
            
            ms_inputs = [item[0].ms_input for item in inputs]
            ms_config = inputs[0][0].ms_config  
            engine = inputs[0][0].engine  

            
            completions = engine.infer(ms_inputs, ms_config, use_tqdm=False)

            
            for i, (batch_input, future) in enumerate(inputs):
                completion = completions[i]

                
                choices = chat_choices_from_openai(completion, tools=[])
                result = ModelOutput(
                    model=completion.model,
                    choices=choices,
                    usage=(ModelUsage(
                        input_tokens=completion.usage.prompt_tokens,
                        output_tokens=completion.usage.completion_tokens,
                        total_tokens=completion.usage.total_tokens,
                    ) if completion.usage else None),
                )

                
                future.set_result(result)

        except Exception as ex:
            
            for _, future in inputs:
                future.set_exception(ex)


def convert_config(config: GenerateConfig) -> RequestConfig:
    """
    Convert EvalScope GenerateConfig to ms-swift RequestConfig.

    Maps configuration parameters between the two frameworks, ensuring
    compatibility while maintaining the same generation behavior.

    Args:
        config: EvalScope generation configuration

    Returns:
        RequestConfig: ms-swift compatible configuration
    """
    return RequestConfig(
        max_tokens=config.max_tokens,
        temperature=config.temperature,
        top_k=config.top_k,
        top_p=config.top_p,
        presence_penalty=config.presence_penalty,
        frequency_penalty=config.frequency_penalty,
        seed=config.seed,
        stream=False,  
        logprobs=config.logprobs,
        top_logprobs=config.top_logprobs)


def convert_request(messages: List[EvalChatMessage], tools: List[ToolInfo]) -> InferRequest:
    """
    Convert EvalScope request format to ms-swift InferRequest format.

    Transforms the message and tool format from EvalScope's representation
    to the format expected by ms-swift's inference engine.

    Args:
        messages: List of chat messages in EvalScope format
        tools: List of available tools in EvalScope format

    Returns:
        InferRequest: ms-swift compatible request object
    """
    
    tools_list = []
    if len(tools) > 0:
        tools_list = [tool.model_dump(exclude_none=True) for tool in tools]

    
    ms_messages = []
    for message in messages:
        ms_messages.append(message.model_dump(exclude_none=True))

    return InferRequest(
        messages=ms_messages,
        tools=tools_list,
    )
