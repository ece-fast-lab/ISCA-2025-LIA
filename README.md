# LIA: A Single-GPU LLM Inference Acceleration with Cooperative AMX-Enabled CPU-GPU Computation and CXL Offloading

LIA is a CPU-GPU collaborative computing framework that accelerates LLM inference on a single GPU, leveraging Intel's AMX technology and CXL.
# 1 System Requirement
Hardware requirements:
- CPU: >= 4th generation Intel Xeon Scalable Processor
- GPU: NVIDIA A100/H100 GPUs

# 2 Docker-based environment setup

## 2.1 Build Docker Image with Compilation from Source

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

## 2.2 (Alternative) Pull Docker Image

```
# Directly download the image from Docker.io hub
docker pull hyungyo/lia-amxgpu:latest
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
As OPT-175B model is not open-sourced, the following command are used to generate dummy weights:
```
# create a directory on the mounted storage to store OPT-175B model
mkdir -p "/home/storage/opt-175b"
# generate dummy weights
python utils/opt_dummy_weights.py --model-"opt-175b" --save_dir="/home/storage/opt-175b/"
# copy the prepared tokenizer for OPT-175b to the directory
cp /home/ubuntu/llm/utils/tokenizer/* /home/storage/opt-175b/
```
## 4.2 Performance Profiling for Online Inference
Run the following bash script as follows:
```
bash online_opt-{size}.sh
```
We provide example scripts for opt-30b and opt-175b models to reproduce the results in an SPR-A100 system.

## 4.3 Performance Profiling for Offline Inference
Run the following bash script as follows:
```
bash offline_opt-{size}.sh
```
Similarly, example scripts for opt-30b and opt-175b models are provided.

## 4.4 CXL-Offloading for Large-batch inference
Run the following example bash script as follows:
```
bash cxl_opt-30b.sh
```
The example script is for opt-30b that reproduces the results in an SPR-A100 system.
