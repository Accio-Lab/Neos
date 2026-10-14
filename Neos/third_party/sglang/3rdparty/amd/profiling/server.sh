#!/bin/bash

                                                
export SGLANG_TORCH_PROFILER_DIR=/sgl-workspace/sglang/profile/

                           
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

                                      
LOGFILE="sglang_server_log_$TIMESTAMP.json"

                                                            
loadTracer.sh python3 -m sglang.launch_server \
    --model-path /sgl-workspace/sglang/dummy_grok1 \
    --tokenizer-path Xenova/grok-1-tokenizer \
    --load-format dummy \
    --quantization fp8 \
    --tp 8 \
    --port 30000 \
    --disable-radix-cache 2>&1 | tee "$LOGFILE"
