#!/bin/bash

if [ ! -d "/home/storage/data" ]; then
  mkdir -p "/home/storage/data"
fi

if [ ! -d "/home/storage/data/cxl" ]; then
  mkdir -p "/home/storage/data/cxl"
fi

## Without CXL-offloading

OMP_NUM_THREADS=40 numactl -m 0 -C 0-39 python ../run.py --benchmark -m facebook/opt-30b --dtype bfloat16 --ipex --input-tokens 32 --max-new-tokens 32 --batch-size 900 --token-latency --num-iter 2 --num-warmup 1 --greedy --prefill-policy 0 --decoding-policy 2 --num-minibatch 2 --gpu-percentage 0 --pin-weight &> "/home/storage/data/cxl/opt30b_32_32_b=900.log"

OMP_NUM_THREADS=40 numactl -m 0 -C 0-39 python ../run.py --benchmark -m facebook/opt-30b --dtype bfloat16 --ipex --input-tokens 32 --max-new-tokens 64 --batch-size 900 --token-latency --num-iter 2 --num-warmup 1 --greedy --prefill-policy 0 --decoding-policy 2 --num-minibatch 2 --gpu-percentage 0 --pin-weight &> "/home/storage/data/cxl/opt30b_32_64_b=900.log"

OMP_NUM_THREADS=40 numactl -m 0 -C 0-39 python ../run.py --benchmark -m facebook/opt-30b --dtype bfloat16 --ipex --input-tokens 32 --max-new-tokens 128 --batch-size 900 --token-latency --num-iter 2 --num-warmup 1 --greedy --prefill-policy 0 --decoding-policy 2 --num-minibatch 2 --gpu-percentage 0 --pin-weight &> "/home/storage/data/cxl/opt30b_32_128_b=900.log"

OMP_NUM_THREADS=40 numactl -m 0 -C 0-39 python ../run.py --benchmark -m facebook/opt-30b --dtype bfloat16 --ipex --input-tokens 32 --max-new-tokens 256 --batch-size 900 --token-latency --num-iter 2 --num-warmup 1 --greedy --prefill-policy 0 --decoding-policy 2 --num-minibatch 2 --gpu-percentage 0 --pin-weight &> "/home/storage/data/cxl/opt30b_32_256_b=900.log"

## CXL-offloading with same batch size

OMP_NUM_THREADS=40 numactl -m 0 -C 0-39 python ../run.py --benchmark -m facebook/opt-30b --dtype bfloat16 --ipex --input-tokens 32 --max-new-tokens 32 --batch-size 900 --token-latency --num-iter 2 --num-warmup 1 --greedy --prefill-policy 0 --decoding-policy 2 --num-minibatch 2 --gpu-percentage 0 --pin-weight --enable-cxl &> "/home/storage/data/cxl/opt30b_32_32_b=900_cxl.log"

OMP_NUM_THREADS=40 numactl -m 0 -C 0-39 python ../run.py --benchmark -m facebook/opt-30b --dtype bfloat16 --ipex --input-tokens 32 --max-new-tokens 64 --batch-size 900 --token-latency --num-iter 2 --num-warmup 1 --greedy --prefill-policy 0 --decoding-policy 2 --num-minibatch 2 --gpu-percentage 0 --pin-weight --enable-cxl &> "/home/storage/data/cxl/opt30b_32_64_b=900_cxl.log"

OMP_NUM_THREADS=40 numactl -m 0 -C 0-39 python ../run.py --benchmark -m facebook/opt-30b --dtype bfloat16 --ipex --input-tokens 32 --max-new-tokens 128 --batch-size 900 --token-latency --num-iter 2 --num-warmup 1 --greedy --prefill-policy 0 --decoding-policy 2 --num-minibatch 2 --gpu-percentage 0 --pin-weight --enable-cxl &> "/home/storage/data/cxl/opt30b_32_128_b=900_cxl.log"

OMP_NUM_THREADS=40 numactl -m 0 -C 0-39 python ../run.py --benchmark -m facebook/opt-30b --dtype bfloat16 --ipex --input-tokens 32 --max-new-tokens 256 --batch-size 900 --token-latency --num-iter 2 --num-warmup 1 --greedy --prefill-policy 0 --decoding-policy 2 --num-minibatch 2 --gpu-percentage 0 --pin-weight --enable-cxl &> "/home/storage/data/cxl/opt30b_32_256_b=900_cxl.log"

## CXL-offloading with larger batch size

OMP_NUM_THREADS=40 numactl -m 0 -C 0-39 python ../run.py --benchmark -m facebook/opt-30b --dtype bfloat16 --ipex --input-tokens 32 --max-new-tokens 32 --batch-size 1580 --token-latency --num-iter 2 --num-warmup 1 --greedy --prefill-policy 0 --decoding-policy 2 --num-minibatch 4 --gpu-percentage 0 --pin-weight --enable-cxl &> "/home/storage/data/cxl/opt30b_32_32_b=1580_cxl.log"

OMP_NUM_THREADS=40 numactl -m 0 -C 0-39 python ../run.py --benchmark -m facebook/opt-30b --dtype bfloat16 --ipex --input-tokens 32 --max-new-tokens 64 --batch-size 1350 --token-latency --num-iter 2 --num-warmup 1 --greedy --prefill-policy 0 --decoding-policy 2 --num-minibatch 3 --gpu-percentage 0 --pin-weight --enable-cxl &> "/home/storage/data/cxl/opt30b_32_64_b=1350_cxl.log"

OMP_NUM_THREADS=40 numactl -m 0 -C 0-39 python ../run.py --benchmark -m facebook/opt-30b --dtype bfloat16 --ipex --input-tokens 32 --max-new-tokens 128 --batch-size 1150 --token-latency --num-iter 2 --num-warmup 1 --greedy --prefill-policy 0 --decoding-policy 2 --num-minibatch 3 --gpu-percentage 0 --pin-weight --enable-cxl &> "/home/storage/data/cxl/opt30b_32_128_b=1150_cxl.log"

OMP_NUM_THREADS=40 numactl -m 0 -C 0-39 python ../run.py --benchmark -m facebook/opt-30b --dtype bfloat16 --ipex --input-tokens 32 --max-new-tokens 256 --batch-size 1050 --token-latency --num-iter 2 --num-warmup 1 --greedy --prefill-policy 0 --decoding-policy 2 --num-minibatch 3 --gpu-percentage 0 --pin-weight --enable-cxl &> "/home/storage/data/cxl/opt30b_32_256_b=1050_cxl.log"
