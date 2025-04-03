from numa_alloc import numa_alloc_tensor, numa_free_tensor
import numpy as np
import time
import torch
from threading import Thread
from queue import Queue
import argparse

def benchmark(is_compute, is_transfer, from_cxl):
    number = 3
    repeat = 3
    warmup = 2

    # CPU-GPU data trasnfer
    if is_transfer:
        b0, s0, h0 = 2048, 2048, 1024
        dtype = torch.int8
        size = b0 * s0 * h0 * number / (1024**3)  # Size in GB
        if from_cxl:
            t_cpu = numa_alloc_tensor((b0, s0, h0), dtype)
            if t_cpu is None:
                print("Failed to allocate NUMA memory.")
                return
        else:
            t_cpu = torch.ones((b0, s0, h0), dtype=dtype, pin_memory=True, device='cpu')
        t_gpu = torch.ones((b0, s0, h0), dtype=dtype, device='cuda:0')
        
        def memcpy(queue):
            costs = []
            total_time = time.time()
            for _ in range(repeat):
                torch.cuda.synchronize()
                start_time = time.time()
                for _ in range(number):
                    t_gpu.copy_(t_cpu, non_blocking=True)
                torch.cuda.synchronize()
                end_time = time.time()
                costs.append(end_time - start_time)
            total_time = time.time() - total_time
            queue.put(costs)
            queue.put(total_time)

    # CPU GEMM
    if is_compute:
        # b1, s1, h1 = 4096, 4096, 4096
        b1, s1, h1 = 8192, 8192, 8192
        # b1, s1, h1 = 16384, 16384, 16384
        mat1 = torch.ones(b1, s1, dtype=torch.float32, device='cpu')
        mat2 = torch.ones(s1, h1, dtype=torch.float32, device='cpu')
        
        def compute(queue):
            costs = []
            total_time = time.time()
            for _ in range(repeat):
                start_time = time.time()
                for _ in range(number):
                    _ = torch.mm(mat1, mat2)
                end_time = time.time()
                costs.append(end_time - start_time)
            total_time = time.time() - total_time
            queue.put(costs)
            queue.put(total_time)

    # Warming up
    for _ in range(warmup):
        if is_transfer:
            q1 = Queue()
            memcpy_thread = Thread(target=memcpy, args=(q1,))
        if is_compute:
            q2 = Queue()
            compute_thread = Thread(target=compute, args=(q2,))

        if is_transfer:
            memcpy_thread.start()
        if is_compute:
            compute_thread.start()

        if is_transfer:
            memcpy_thread.join()
        if is_compute:
            compute_thread.join()

    # Start timing
    if is_transfer:
        transfer_queue = Queue()
        memcpy_thread = Thread(target=memcpy, args=(transfer_queue,))
    if is_compute:
        compute_queue = Queue()
        compute_thread = Thread(target=compute, args=(compute_queue,))

    if is_transfer:
        memcpy_thread.start()
    if is_compute:
        compute_thread.start()

    if is_transfer:
        memcpy_thread.join()
    if is_compute:
        compute_thread.join()

    if is_transfer:
        transfer_times = transfer_queue.get()
        total_time = transfer_queue.get()
        avg_transfer_time = np.mean(transfer_times)
        total_bw = size / avg_transfer_time
        print(f"[{total_time:.3f} s] Average Transfer Bandwidth: {total_bw:.3f} GB/s")
    if is_compute:
        compute_times = compute_queue.get()
        total_time = compute_queue.get()
        avg_compute_time = np.mean(compute_times)
        print(f"[{total_time:.3f} s] Average Compute Time: {avg_compute_time:.3f} seconds")

    # Free the NUMA allocated memory
    if is_transfer:
        if from_cxl:
            numa_free_tensor(t_cpu)
        del t_cpu
        del t_gpu
    if is_compute:
        del mat1, mat2

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", action='store_true')
    parser.add_argument("--cpu", action='store_true')
    parser.add_argument("--cxl", action='store_true')
    args = parser.parse_args()
    
    is_compute=False
    is_transfer=False
    from_cxl=False
    if args.cpu:
        is_compute = True
    if args.gpu:
        is_transfer = True
    if args.cxl:
        from_cxl = True

    benchmark(is_compute, is_transfer, from_cxl)
