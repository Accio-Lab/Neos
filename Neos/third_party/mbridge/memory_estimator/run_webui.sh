#!/bin/bash

                                            
export MEGATRON_PATH="/PATH/TO/Megatron-LM"

                                                                
                                                   
ESTIMATOR_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

                                                           
export PYTHONPATH="${ESTIMATOR_DIR}:${MEGATRON_PATH}:${PYTHONPATH}"

                                                                           
cd "${ESTIMATOR_DIR}"
uvicorn webui.main:app --host 0.0.0.0 --port 8000 