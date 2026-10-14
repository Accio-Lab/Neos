#!/bin/bash

check_vram_clear() {
    local vram_threshold_percent=5                             
    local memory_threshold_mb=500                                   

    if command -v rocm-smi >/dev/null 2>&1; then
        echo "Checking ROCm GPU VRAM usage..."
                                                                 
        local high_usage=$(rocm-smi --showmemuse | grep -E "GPU Memory Allocated \(VRAM%\): ([6-9]|[1-9][0-9]|100)")
        if [ -n "$high_usage" ]; then
            echo "ERROR: VRAM usage exceeds threshold (${vram_threshold_percent}%) on some GPUs:"
            echo "$high_usage"
            rocm-smi --showmemuse
            return 1
        else
            echo "✓ VRAM usage is within acceptable limits on all GPUs"
            return 0
        fi
   fi
}

                                                             
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    set -e
    check_vram_clear
fi
