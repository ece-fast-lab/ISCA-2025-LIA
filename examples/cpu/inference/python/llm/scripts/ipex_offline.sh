#!/bin/bash

if [ ! -d "/home/storage/data" ]; then
  mkdir -p "/home/storage/data"
fi

if [ ! -d "/home/storage/data/ipex" ]; then
  mkdir -p "/home/storage/data/ipex"
fi

## Offline for OPT-30B

OMP_NUM_THREADS=40 numactl -m 0 -C 0-39 python ../run.py --benchmark -m facebook/opt-30b --dtype bfloat16 --ipex --input-tokens 32 --max-new-tokens 32 --batch-size 64 --token-latency --num-iter 2 --num-warmup 1 --greedy --prefill-policy 1 --decoding-policy 1 --num-minibatch 1 --gpu-percentage 0 &> "/home/storage/data/ipex/opt30b_32_32_b=64.log"

OMP_NUM_THREADS=40 numactl -m 0 -C 0-39 python ../run.py --benchmark -m facebook/opt-30b --dtype bfloat16 --ipex --input-tokens 2016 --max-new-tokens 32 --batch-size 64 --token-latency --num-iter 2 --num-warmup 1 --greedy --prefill-policy 1 --decoding-policy 1 --num-minibatch 1 --gpu-percentage 0 &> "/home/storage/data/ipex/opt30b_2016_32_b=64.log"

OMP_NUM_THREADS=40 numactl -m 0 -C 0-39 python ../run.py --benchmark -m facebook/opt-30b --dtype bfloat16 --ipex --input-tokens 32 --max-new-tokens 256 --batch-size 64 --token-latency --num-iter 2 --num-warmup 1 --greedy --prefill-policy 1 --decoding-policy 1 --num-minibatch 1 --gpu-percentage 0 &> "/home/storage/data/ipex/opt30b_32_256_b=64.log"

OMP_NUM_THREADS=40 numactl -m 0 -C 0-39 python ../run.py --benchmark -m facebook/opt-30b --dtype bfloat16 --ipex --input-tokens 1792 --max-new-tokens 256 --batch-size 64 --token-latency --num-iter 2 --num-warmup 1 --greedy --prefill-policy 1 --decoding-policy 1 --num-minibatch 1 --gpu-percentage 0 &> "/home/storage/data/ipex/opt30b_1792_256_b=64.log"

OMP_NUM_THREADS=40 numactl -m 0 -C 0-39 python ../run.py --benchmark -m facebook/opt-30b --dtype bfloat16 --ipex --input-tokens 32 --max-new-tokens 32 --batch-size 900 --token-latency --num-iter 2 --num-warmup 1 --greedy --prefill-policy 1 --decoding-policy 1 --num-minibatch 1 --gpu-percentage 0 &> "/home/storage/data/ipex/opt30b_32_32_b=900.log"

OMP_NUM_THREADS=40 numactl -m 0 -C 0-39 python ../run.py --benchmark -m facebook/opt-30b --dtype bfloat16 --ipex --input-tokens 32 --max-new-tokens 256 --batch-size 900 --token-latency --num-iter 2 --num-warmup 1 --greedy --prefill-policy 1 --decoding-policy 1 --num-minibatch 1 --gpu-percentage 0 &> "/home/storage/data/ipex/opt30b_32_256_b=900.log"

## Offline for OPT-175B

OMP_NUM_THREADS=40 numactl -m 0 -C 0-39 python ../run.py --benchmark -m /home/storage/opt-175b/ --dtype bfloat16 --ipex --input-tokens 32 --max-new-tokens 32 --batch-size 1 --token-latency --num-iter 2 --num-warmup 1 --greedy --prefill-policy 1 --decoding-policy 1 --num-minibatch 1 --gpu-percentage 0 &> "/home/storage/data/ipex/opt175b_32_32_b=64.log"

OMP_NUM_THREADS=40 numactl -m 0 -C 0-39 python ../run.py --benchmark -m /home/storage/opt-175b/ --dtype bfloat16 --ipex --input-tokens 32 --max-new-tokens 256 --batch-size 1 --token-latency --num-iter 2 --num-warmup 1 --greedy --prefill-policy 1 --decoding-policy 1 --num-minibatch 1 --gpu-percentage 0 &> "/home/storage/data/ipex/opt175b_32_256_b=64.log"


