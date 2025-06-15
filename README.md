# LIA: A Single-GPU LLM Inference Acceleration with Cooperative AMX-Enabled CPU-GPU Computation and CXL Offloading

LIA is a CPU-GPU collaborative computing framework that accelerates LLM inference on a single GPU, leveraging Intel's AMX technology and CXL.
# 1. System Requirement
Hardware requirements:
- CPU: >= 4th generation Intel Xeon Scalable Processor
- GPU: NVIDIA A100/H100 GPUs

# 2. Docker-based environment setup

## 2.1. (Recommended) Pull Docker Image

```
# Directly download the image from Docker.io hub
docker pull hyungyo/lia-amxgpu:latest
```

## 2.2. Build Docker Image with Compilation from Source

```
# Download the Git repository
git clone https://github.com/Hyungyo1/LIA_AMXGPU.git
# Update submodules
cd LIA_AMXGPU
git submodule sync
git submodule update --init --recursive
# Build an image with the provided Dockerfile
DOCKER_BUILDKIT=1 docker build -f examples/cpu/inference/python/llm/Dockerfile --build-arg COMPILE=ON -t lia-amxgpu:main .
```


# 3. Run Docker Image with GPU
Install CUDA container toolkit with the following command:
```
sudo apt-get install -y nvidia-container-toolkit
```
Then, run the docker image with the command below:
```
docker run --rm -it --gpus all --privileged -v {your_storage_dir}:/home/storage lia-amxgpu:main bash
```
Activate and update environment variables:
```
cd llm
source ./tools/env_activate.sh
```
# 4. How to Run
## 4.1 Creating Dummy Model Weights (Only for OPT-175B)
As OPT-175B model is not open-sourced, the following command can be used to generate dummy weights:
```
# create a directory on the mounted storage to store OPT-175B model
mkdir -p "/home/storage/opt-175b"
# generate dummy weights
python utils/opt_dummy_weights.py --model-"opt-175b" --save_dir="/home/storage/opt-175b/"
# copy the prepared tokenizer for OPT-175b to the directory
cp /home/ubuntu/llm/utils/tokenizer/* /home/storage/opt-175b/
```
## 4.2 Quick Example for Running OPT-30B Inference with LIA
```
OMP_NUM_THREADS=32 numactl -m 0 -C 0-31 python run.py --benchmark -m facebook/opt-30b --dtype bfloat16 --ipex --input-tokens 256 --max-new-tokens 32 --batch-size 64 --token-latency --num-iter 10 --num-warmup 2 --greedy --prefill-policy 0 --decoding-policy 1 --gpu-percentage 10 --num-minibatch 2 --pin-weight --enable-cxl
```
LIA-specific Parameters –

`--prefill-policy/decoding-policy`: {0, 1, 2}

0 –> (0,0,0,0,0,0) full GPU compute 

1 –> (1,1,1,1,1,1) full CPU compute 

2 –> (0,1,1,0,0,0) partial CPU offloading

(The vector values follow the notation described in the paper)

`--gpu-percentage`: The percentage of model parameters to load to the GPU memory

`--num-minibatch`: The number of minibatches which the batch would be split into during the prefill stage

`--pin-weight`: Pin the entire model parameters on CPU memory.

`--enable-cxl`: Offload the model parameters to the CXL memory.

## 4.2 Performance Profiling
We provide example scripts for opt-30b and opt-175b models to reproduce the results in an SPR-A100 system.
The script will collect data for LIA and IPEX for online and offline inference.
```
bash scripts/run_performance.sh
```

## 4.3 CXL-Offloading for Large-batch Inference
Run the following example bash script as follows:
```
bash scripts/cxl_offloading.sh
```
The example script is for opt-30b that reproduces the results in an SPR-A100 system.
