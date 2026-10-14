#!/bin/bash
set -euxo pipefail

                            
if command -v sudo >/dev/null 2>&1; then
    sudo apt-get update
    sudo apt-get install -y libssl-dev pkg-config protobuf-compiler redis-server
else
    apt-get update
    apt-get install -y libssl-dev pkg-config protobuf-compiler redis-server
fi

                                                     
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain 1.90


                                                         
. "$HOME/.cargo/env"
source $HOME/.cargo/env

                     
rustc --version
cargo --version
protoc --version
