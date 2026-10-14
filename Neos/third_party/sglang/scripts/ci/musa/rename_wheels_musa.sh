#!/usr/bin/env bash
set -euo pipefail

                                                          
        
                                                   
          
                                            

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: $0 <musa_suffix> [wheel_dir]" >&2
  exit 1
fi

MUSA_SUFFIX="$1"
WHEEL_DIR="${2:-dist}"

wheel_files=("$WHEEL_DIR"/*.whl)

if [[ ! -e "${wheel_files[0]}" ]]; then
  echo "No wheel files found in ${WHEEL_DIR}/, nothing to rename."
  exit 0
fi

for wheel in "${wheel_files[@]}"; do
                                           
  intermediate_wheel="${wheel/linux/manylinux2014}"

                                           
  if [[ $intermediate_wheel =~ -cp([0-9]+)- ]]; then
    cp_version="${BASH_REMATCH[1]}"
  else
    echo "Could not extract Python version from wheel name: $intermediate_wheel" >&2
    continue
  fi

                                                  
  new_wheel="${intermediate_wheel/-cp${cp_version}/+musa${MUSA_SUFFIX}-cp${cp_version}}"

  if [[ "$wheel" != "$new_wheel" ]]; then
    echo "Renaming $wheel -> $new_wheel"
    mv -- "$wheel" "$new_wheel"
  fi
done

echo "MUSA wheel renaming completed."
