#!/bin/bash

                       
                                             

                                       
                                                                  
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LIB_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)/lib"

                               
if [ ! -d "$LIB_DIR" ]; then
    echo "Error: Library directory not found at $LIB_DIR"
    echo "Please run 'make lib' first to build and export the library"
    exit 1
fi

                                                                 
PYTHON_LDFLAGS=$(python3-config --ldflags --embed 2>/dev/null || python3-config --ldflags 2>/dev/null || echo "")

                                               
export CGO_LDFLAGS="-L${LIB_DIR} -lsgl_model_gateway_go ${PYTHON_LDFLAGS} -ldl"

                                                          
if [[ "$OSTYPE" == "darwin"* ]]; then
    export DYLD_LIBRARY_PATH="${LIB_DIR}:${DYLD_LIBRARY_PATH}"
else
    export LD_LIBRARY_PATH="${LIB_DIR}:${LD_LIBRARY_PATH}"
fi

                                                                                              
                                                        
DEFAULT_TOKENIZER_PATH="${SGL_TOKENIZER_PATH:-../tokenizer}"
DEFAULT_ENDPOINT="${SGL_GRPC_ENDPOINT:-grpc://localhost:20000}"

TOKENIZER_PATH="${1:-${DEFAULT_TOKENIZER_PATH}}"
ENDPOINT="${2:-${DEFAULT_ENDPOINT}}"

echo "Running simple example..."
echo "Library path: ${LIB_DIR}"
echo "Tokenizer: $TOKENIZER_PATH"
echo "Endpoint: $ENDPOINT"
echo ""

cd "$(dirname "${BASH_SOURCE[0]}")"
SGL_TOKENIZER_PATH="$TOKENIZER_PATH" SGL_GRPC_ENDPOINT="$ENDPOINT" go run main.go
