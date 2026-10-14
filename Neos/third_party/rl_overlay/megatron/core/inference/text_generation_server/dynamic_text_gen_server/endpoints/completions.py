# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

import asyncio
import logging
import time

from megatron.core.inference.sampling_params import SamplingParams

logger = logging.getLogger(__name__)


try:
    from flask import Blueprint, current_app, jsonify, request

    bp = Blueprint('completions_api', __name__)

    @bp.route('/completions', methods=['POST'])
    @bp.route('/v1/completions', methods=['POST'])
    async def completions():
        """Handles async POST requests for completions."""
        client = current_app.config['client']
        tokenizer = current_app.config['tokenizer']

        req = request.get_json()

        
        prompt_data = req.get("prompt")
        if not prompt_data:
            return "Missing 'prompt' field", 400

        try:
            if isinstance(prompt_data, str):
                prompts_as_tokens = [tokenizer.tokenize(prompt_data)]
                prompts_as_strings = [prompt_data]
            elif isinstance(prompt_data, list):
                if not prompt_data:
                    return "'prompt' list is empty", 400
                if all(isinstance(p, str) for p in prompt_data):
                    prompts_as_tokens = [tokenizer.tokenize(p) for p in prompt_data]
                    prompts_as_strings = prompt_data
                elif all(isinstance(p, int) for p in prompt_data):
                    prompts_as_tokens = [prompt_data]
                    prompts_as_strings = [tokenizer.detokenize(prompt_data)]
                elif all(
                    isinstance(p, list) and all(isinstance(t, int) for t in p) for p in prompt_data
                ):
                    prompts_as_tokens = prompt_data
                    prompts_as_strings = [tokenizer.detokenize(p) for p in prompt_data]
                else:
                    return (
                        (
                            "Invalid 'prompt' format. Must be str, list[str], "
                            "list[int], or list[list[int]]"
                        ),
                        400,
                    )
            else:
                return "Invalid 'prompt' type. Must be str or list", 400
        except Exception as e:
            return f"Error tokenizing prompt: {e}", 500

        
        try:
            temperature = float(req.get("temperature", 1.0))
            top_p = float(req.get("top_p", 1.0))
            top_k = int(req.get("top_k", 0))
            echo = bool(req.get("echo", False))

            if temperature == 0.0:
                top_k = 1
                top_p = 0.0

            
            logprobs_param = req.get("logprobs", None)

            if logprobs_param is not None:
                top_n_logprobs = int(logprobs_param)
                return_log_probs = True
            else:
                top_n_logprobs = 0
                return_log_probs = False

            
            
            skip_prompt_log_probs = not (echo and return_log_probs)

            sampling_params = SamplingParams(
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                return_log_probs=return_log_probs,
                top_n_logprobs=top_n_logprobs,
                skip_prompt_log_probs=skip_prompt_log_probs,
                num_tokens_to_generate=int(req.get("max_tokens", 16)),
            )
        except ValueError as e:
            return f"Invalid sampling parameter: {e}", 400

        
        tasks = []
        for prompt_tokens in prompts_as_tokens:
            per_req_params = SamplingParams(
                temperature=sampling_params.temperature,
                top_k=sampling_params.top_k,
                top_p=sampling_params.top_p,
                return_log_probs=sampling_params.return_log_probs,
                top_n_logprobs=sampling_params.top_n_logprobs,
                skip_prompt_log_probs=sampling_params.skip_prompt_log_probs,
                num_tokens_to_generate=sampling_params.num_tokens_to_generate,
            )
            tasks.append(client.add_request(prompt_tokens, per_req_params))

        start_time = time.perf_counter()
        try:
            batch_results = await asyncio.gather(*tasks)
        except Exception as e:
            return f"Error during inference: {e}", 500

        logger.info(
            f"Batch of {len(tasks)} requests processed in {time.perf_counter() - start_time:.2f}s"
        )

        
        choices = []

        request_idx = 0
        for record in batch_results:
            for result in record.requests:
                full_text = result.generated_text or ""
                text_output = (prompts_as_strings[request_idx] + full_text) if echo else full_text

                logprobs_data = None
                if sampling_params.return_log_probs:
                    
                    prompt_tokens_list = []
                    if result.prompt_tokens is not None:
                        if hasattr(result.prompt_tokens, 'tolist'):
                            prompt_tokens_list = result.prompt_tokens.tolist()
                        else:
                            prompt_tokens_list = list(result.prompt_tokens)

                    prompt_log_probs = getattr(result, 'prompt_log_probs', None) or []
                    prompt_top_n_logprobs = getattr(result, 'prompt_top_n_logprobs', None) or []

                    
                    generated_tokens_list = (
                        list(result.generated_tokens) if result.generated_tokens else []
                    )
                    generated_log_probs = getattr(result, 'generated_log_probs', None) or []
                    generated_top_n_logprobs = (
                        getattr(result, 'generated_top_n_logprobs', None) or []
                    )

                    if echo:
                        
                        
                        all_token_ids = prompt_tokens_list + generated_tokens_list
                        tokens = [tokenizer.detokenize([tok]) for tok in all_token_ids]

                        
                        
                        token_logprobs = [None] + list(prompt_log_probs) + list(generated_log_probs)

                        
                        
                        top_logprobs = None
                        if prompt_top_n_logprobs or generated_top_n_logprobs:
                            top_logprobs = (
                                [None]
                                + list(prompt_top_n_logprobs)
                                + list(generated_top_n_logprobs)
                            )

                        
                        text_offset = []
                        current_offset = 0
                        for tok_str in tokens:
                            text_offset.append(current_offset)
                            current_offset += len(tok_str)
                    else:
                        
                        tokens = [tokenizer.detokenize([tok]) for tok in generated_tokens_list]

                        
                        token_logprobs = [None] + list(generated_log_probs)

                        
                        top_logprobs = None
                        if generated_top_n_logprobs:
                            top_logprobs = [None] + list(generated_top_n_logprobs)

                        
                        text_offset = []
                        current_offset = 0
                        for tok_str in tokens:
                            text_offset.append(current_offset)
                            current_offset += len(tok_str)

                    logprobs_data = {
                        "token_logprobs": token_logprobs,
                        "tokens": tokens,
                        "text_offset": text_offset,
                        "top_logprobs": top_logprobs,
                    }

                choices.append(
                    {"index": request_idx, "text": text_output, "logprobs": logprobs_data}
                )
                request_idx += 1

        return jsonify({"choices": choices})

except ImportError as e:
    logger.warning(f"Could not import flask: {e}")
