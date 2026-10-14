set -ex

USE_MEGATRON=${USE_MEGATRON:-1}

echo "1. install basic packages"
pip uninstall -y triton triton-ascend
                               
pip install torchvision==0.24.0
pip install torchaudio==2.9.0
pip install triton-ascend==3.2.1 --extra-index-url https://triton-ascend.osinfra.cn/pypi/simple/ --trusted-host triton-ascend.osinfra.cn
pip install "transformers==5.3.0" 
pip install setuptools-scm


echo "2. install vllm & vllm-ascend"
git clone --depth 1 --branch v0.18.0 https://github.com/vllm-project/vllm.git
cd vllm
VLLM_TARGET_DEVICE=empty pip install -v -e .
cd ..
git clone -b releases/v0.18.0 https://github.com/vllm-project/vllm-ascend.git
cd vllm-ascend
git submodule update --init --recursive
pip install -v -e . --no-build-isolation --extra-index-url https://triton-ascend.osinfra.cn/pypi/simple/ --trusted-host triton-ascend.osinfra.cn
cd ..


if [ $USE_MEGATRON -eq 1 ]; then
    echo "3. install Megatron & MindSpeed"
                                                  
    git clone https://gitcode.com/Ascend/MindSpeed.git
    cd MindSpeed && git checkout core_r0.16.0 && cd ..
    git clone --depth 1 --branch core_r0.16.0 https://github.com/NVIDIA/Megatron-LM.git

                             
    pip install -e Megatron-LM
    pip install -e MindSpeed

                
    pip install mbridge
fi

echo "4. install verl"
cd verl
pip install -r requirements-npu.txt --extra-index-url https://triton-ascend.osinfra.cn/pypi/simple/ --trusted-host triton-ascend.osinfra.cn
pip install -v -e .
                                             
cd recipe
git checkout main
cd ..

echo "5. May need to check other neccessary packages"
pip install transformers==5.3.0 xgrammar==0.1.33
echo "Successfully installed all packages"