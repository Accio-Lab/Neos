#!/bin/bash
                                                                                              
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Preparing CI runner..."
echo ""

                                                                           
python3 "${SCRIPT_DIR}/../utils/cleanup_hf_cache.py"
echo ""

                                                               
                                                                                 
python3 "${SCRIPT_DIR}/../utils/prevalidate_cached_models.py"
echo ""

echo "CI runner preparation complete!"
