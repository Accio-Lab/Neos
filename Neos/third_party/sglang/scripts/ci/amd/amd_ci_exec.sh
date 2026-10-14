#!/bin/bash
set -euo pipefail

                                                                              
HOSTNAME_VALUE=$(hostname)
GPU_FAMILY=""

                                                            
if [[ "${HOSTNAME_VALUE}" =~ ^linux-(mi[0-9]+[a-z]*)-gpu-[0-9]+ ]]; then
  GPU_FAMILY="${BASH_REMATCH[1]}"
  echo "Detected GPU family from hostname: ${GPU_FAMILY}"
else
  echo "Warning: could not parse GPU family from '${HOSTNAME_VALUE}'"
fi

WORKDIR="/sglang-checkout/test/srt"
declare -A ENV_MAP=(
  [SGLANG_IS_IN_CI_AMD]=1
  [SGLANG_IS_IN_CI]=1
  [SGLANG_USE_AITER]=1
)

                                            
if [[ "${GPU_FAMILY}" == "mi35x" ]]; then
  ENV_MAP[GPU_ARCHS]="gfx950"
fi

                                   
while [[ $# -gt 0 ]]; do
  case "$1" in
    -w|--workdir)
      WORKDIR="$2"
      shift 2
      ;;
    -e)
      IFS="=" read -r key val <<< "$2"
      ENV_MAP["$key"]="$val"
      shift 2
      ;;
    --)
      shift
      break
      ;;
    *)
      break
      ;;
  esac
done

                      
ENV_ARGS=()
for key in "${!ENV_MAP[@]}"; do
  ENV_ARGS+=("-e" "$key=${ENV_MAP[$key]}")
done

                                                                          
                                                                      
                                                                            
 
                                                  
if docker exec \
  -w "$WORKDIR" \
  "${ENV_ARGS[@]}" \
  ci_sglang "$@"; then
  exit 0
else
  FIRST_EXIT_CODE=$?
fi

echo "First attempt failed with exit code $FIRST_EXIT_CODE"

                                                                   
                                                                
                                             
                                                     
                                                                                
if [[ "$FIRST_EXIT_CODE" -eq 1 || "$FIRST_EXIT_CODE" -eq 137 || "$FIRST_EXIT_CODE" -eq 255 ]]; then
  echo "Exit code $FIRST_EXIT_CODE indicates test failure (not network issue), not retrying"
  exit $FIRST_EXIT_CODE
fi

echo "Retrying with HF_HUB_OFFLINE=1 (offline mode to use cached models)..."

                                                                 
docker exec \
  -w "$WORKDIR" \
  "${ENV_ARGS[@]}" \
  -e HF_HUB_OFFLINE=1 \
  ci_sglang "$@"
