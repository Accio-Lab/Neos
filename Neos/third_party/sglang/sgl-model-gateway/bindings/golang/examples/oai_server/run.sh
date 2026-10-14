#!/bin/bash

                                 
                                                                                    
 
          
                                                                  
                                                                         

                                       
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BINDINGS_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
LIB_DIR="${BINDINGS_DIR}/lib"

if [ ! -d "$LIB_DIR" ]; then
    echo "Error: Library directory not found at $LIB_DIR"
    echo "Please run 'make lib' first to build and export the library"
    exit 1
fi

                                                                 
PYTHON_LDFLAGS=$(python3-config --ldflags --embed 2>/dev/null || python3-config --ldflags 2>/dev/null || echo "")

                                               
                                                                                                   
                                                            
export CGO_LDFLAGS="-L${LIB_DIR} ${PYTHON_LDFLAGS}"

                                                          
if [[ "$OSTYPE" == "darwin"* ]]; then
    export DYLD_LIBRARY_PATH="${LIB_DIR}:${DYLD_LIBRARY_PATH}"
else
    export LD_LIBRARY_PATH="${LIB_DIR}:${LD_LIBRARY_PATH}"
fi

                 
ENABLE_PROFILE=false
PPROF_PORT="6060"
TOKENIZER_PATH=""
ENDPOINT=""
PORT=""

while [[ $# -gt 0 ]]; do
	case $1 in
		--profile)
			ENABLE_PROFILE=true
			shift
			;;
		--pprof-port)
			ENABLE_PROFILE=true
			PPROF_PORT="$2"
			shift 2
			;;
		*)
			if [[ -z "$TOKENIZER_PATH" ]]; then
				TOKENIZER_PATH="$1"
			elif [[ -z "$ENDPOINT" ]]; then
				ENDPOINT="$1"
			elif [[ -z "$PORT" ]]; then
				PORT="$1"
			fi
			shift
			;;
	esac
done

                       
DEFAULT_TOKENIZER_PATH="${SGL_TOKENIZER_PATH:-../tokenizer}"
DEFAULT_ENDPOINT="${SGL_GRPC_ENDPOINT:-grpc://localhost:20000}"
DEFAULT_PORT="${PORT:-8080}"

TOKENIZER_PATH="${TOKENIZER_PATH:-${DEFAULT_TOKENIZER_PATH}}"
ENDPOINT="${ENDPOINT:-${DEFAULT_ENDPOINT}}"
PORT="${PORT:-${DEFAULT_PORT}}"

echo "Running OpenAI-compatible server..."
echo "Library path: ${LIB_DIR}"
echo "Tokenizer: $TOKENIZER_PATH"
echo "Endpoint: $ENDPOINT"
echo "Port: $PORT"
echo "Client Mode: gRPC (default)"
echo "FFI Postprocessing: ENABLED (normal mode)"
echo "FFI Preprocessing: ENABLED (normal mode)"
if [[ "$ENABLE_PROFILE" == "true" ]]; then
	echo "Profiling: enabled (port: $PPROF_PORT)"
	echo "  pprof endpoint: http://localhost:$PPROF_PORT/debug/pprof/"
	export PPROF_ENABLED=true
	export PPROF_PORT="$PPROF_PORT"
else
	echo "Profiling: disabled"
fi
echo ""

                            
cd "$(dirname "${BASH_SOURCE[0]}")"

                                          
if [ ! -f "go.mod" ]; then
    echo "Error: go.mod not found in $(pwd)"
    exit 1
fi

                               
export GO111MODULE=on

                             
echo "Syncing Go module dependencies..."
go mod tidy

                                                                    
SGL_TOKENIZER_PATH="$TOKENIZER_PATH" SGL_GRPC_ENDPOINT="$ENDPOINT" PORT="$PORT" go run ./main.go
