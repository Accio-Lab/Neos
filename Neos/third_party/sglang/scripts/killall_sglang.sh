#!/bin/bash

        
                                                                               
                                                                             
                                                                           
                                                                          

if [ "$1" = "rocm" ]; then
    echo "Running in ROCm mode"

                            
    pgrep -f 'sglang::|sglang\.launch_server|sglang\.bench|sglang\.data_parallel|sglang\.srt|sgl_diffusion::' | xargs -r kill -9

elif [ "$1" = "gpus" ] && [ -n "$2" ]; then
                                              
    echo "Killing all processes on GPUs: $2"

                             
    nvidia-smi

                                                                                              
    devices=$(echo "$2" | tr ',' '\n' | sed 's/^[[:space:]]*//;s/[[:space:]]*$//' | sed 's|^|/dev/nvidia|' | tr '\n' ' ')
    echo "Targeting devices: $devices"

                                                    
    [ -n "$devices" ] && lsof $devices 2>/dev/null | awk 'NR>1 {print $2}' | sort -u | xargs -r kill -9 2>/dev/null

                                    
    nvidia-smi

else
                             
    nvidia-smi

                            
    pgrep -f 'sglang::|sglang\.launch_server|sglang\.bench|sglang\.data_parallel|sglang\.srt|sgl_diffusion::' | xargs -r kill -9

                                                           
    if [ "$1" = "all" ]; then
                                    
        if command -v sudo >/dev/null 2>&1; then
            sudo apt-get update
            sudo apt-get install -y lsof
        else
            apt-get update
            apt-get install -y lsof
        fi
        kill -9 $(nvidia-smi | sed -n '/Processes:/,$p' | grep "   [0-9]" | awk '{print $5}') 2>/dev/null
        lsof /dev/nvidia* | awk '{print $2}' | xargs kill -9 2>/dev/null
    fi

                                    
    nvidia-smi
fi
