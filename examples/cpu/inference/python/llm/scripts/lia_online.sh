#!/bin/bash

if [ ! -d "/home/storage/data" ]; then
  mkdir -p "/home/storage/data"
fi

if [ ! -d "/home/storage/data/lia" ]; then
  mkdir -p "/home/storage/data/lia"
fi

## Online for OPT-30B

OMP_NUM_THREADS=40 numactl -m 0 -C 0-39 python ../run.py --benchmark -m facebook/opt-30b --dtype bfloat16 --ipex --input-tokens 32 --max-new-tokens 32 --batch-size 1 --token-latency --num-iter 2 --num-warmup 1 --greedy --prefill-policy 1 --decoding-policy 1 --num-minibatch 1 --gpu-percentage 66 --pin-weight &> "/home/storage/data/lia/opt30b_32_32_b=1.log"

OMP_NUM_THREADS=40 numactl -m 0 -C 0-39 python ../run.py --benchmark -m facebook/opt-30b --dtype bfloat16 --ipex --input-tokens 256 --max-new-tokens 32 --batch-size 1 --token-latency --num-iter 2 --num-warmup 1 --greedy --prefill-policy 1 --decoding-policy 1 --num-minibatch 1 --gpu-percentage 64 --pin-weight &> "/home/storage/data/lia/opt30b_256_32_b=1.log"

OMP_NUM_THREADS=40 numactl -m 0 -C 0-39 python ../run.py --benchmark -m facebook/opt-30b --dtype bfloat16 --ipex --input-tokens 2016 --max-new-tokens 32 --batch-size 1 --token-latency --num-iter 2 --num-warmup 1 --greedy --prefill-policy 0 --decoding-policy 1 --num-minibatch 1 --gpu-percentage 58 --pin-weight &> "/home/storage/data/lia/opt30b_2016_32_b=1.log"

OMP_NUM_THREADS=40 numactl -m 0 -C 0-39 python ../run.py --benchmark -m facebook/opt-30b --dtype bfloat16 --ipex --input-tokens 32 --max-new-tokens 256 --batch-size 1 --token-latency --num-iter 2 --num-warmup 1 --greedy --prefill-policy 1 --decoding-policy 1 --num-minibatch 1 --gpu-percentage 64 --pin-weight &> "/home/storage/data/lia/opt30b_32_256_b=1.log"

OMP_NUM_THREADS=40 numactl -m 0 -C 0-39 python ../run.py --benchmark -m facebook/opt-30b --dtype bfloat16 --ipex --input-tokens 256 --max-new-tokens 256 --batch-size 1 --token-latency --num-iter 2 --num-warmup 1 --greedy --prefill-policy 1 --decoding-policy 1 --num-minibatch 1 --gpu-percentage 62 --pin-weight &> "/home/storage/data/lia/opt30b_256_256_b=1.log"

OMP_NUM_THREADS=40 numactl -m 0 -C 0-39 python ../run.py --benchmark -m facebook/opt-30b --dtype bfloat16 --ipex --input-tokens 1792 --max-new-tokens 256 --batch-size 1 --token-latency --num-iter 2 --num-warmup 1 --greedy --prefill-policy 0 --decoding-policy 1 --num-minibatch 1 --gpu-percentage 58 --pin-weight &> "/home/storage/data/lia/opt30b_1792_256_b=1.log"

## Online for OPT-175B

OMP_NUM_THREADS=40 numactl -m 0 -C 0-39 python ../run.py --benchmark -m /home/storage/opt-175b/ --dtype bfloat16 --ipex --input-tokens 32 --max-new-tokens 32 --batch-size 1 --token-latency --num-iter 2 --num-warmup 1 --greedy --prefill-policy 1 --decoding-policy 1 --num-minibatch 1 --gpu-percentage 12 &> "/home/storage/data/lia/opt175b_32_32_b=1.log"

OMP_NUM_THREADS=40 numactl -m 0 -C 0-39 python ../run.py --benchmark -m /home/storage/opt-175b/ --dtype bfloat16 --ipex --input-tokens 256 --max-new-tokens 32 --batch-size 1 --token-latency --num-iter 2 --num-warmup 1 --greedy --prefill-policy 1 --decoding-policy 1 --num-minibatch 1 --gpu-percentage 12 &> "/home/storage/data/lia/opt175b_256_32_b=1.log"

OMP_NUM_THREADS=40 numactl -m 0 -C 0-39 python ../run.py --benchmark -m /home/storage/opt-175b/ --dtype bfloat16 --ipex --input-tokens 2016 --max-new-tokens 32 --batch-size 1 --token-latency --num-iter 2 --num-warmup 1 --greedy --prefill-policy 0 --decoding-policy 1 --num-minibatch 1 --gpu-percentage 8 &> "/home/storage/data/lia/opt175b_2016_32_b=1.log"

OMP_NUM_THREADS=40 numactl -m 0 -C 0-39 python ../run.py --benchmark -m /home/storage/opt-175b/ --dtype bfloat16 --ipex --input-tokens 32 --max-new-tokens 256 --batch-size 1 --token-latency --num-iter 2 --num-warmup 1 --greedy --prefill-policy 1 --decoding-policy 1 --num-minibatch 1 --gpu-percentage 12 &> "/home/storage/data/lia/opt175b_32_256_b=1.log"

OMP_NUM_THREADS=40 numactl -m 0 -C 0-39 python ../run.py --benchmark -m /home/storage/opt-175b/ --dtype bfloat16 --ipex --input-tokens 256 --max-new-tokens 256 --batch-size 1 --token-latency --num-iter 2 --num-warmup 1 --greedy --prefill-policy 1 --decoding-policy 1 --num-minibatch 1 --gpu-percentage 10 &> "/home/storage/data/lia/opt175b_256_256_b=1.log"

OMP_NUM_THREADS=40 numactl -m 0 -C 0-39 python ../run.py --benchmark -m /home/storage/opt-175b/ --dtype bfloat16 --ipex --input-tokens 1792 --max-new-tokens 256 --batch-size 1 --token-latency --num-iter 2 --num-warmup 1 --greedy --prefill-policy 0 --decoding-policy 1 --num-minibatch 1 --gpu-percentage 9 &> "/home/storage/data/lia/opt175b_1792_256_b=1.log"

