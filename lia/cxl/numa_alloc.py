import ctypes
from ctypes import c_size_t, c_void_p, c_int, c_char
import torch
import numpy as np

# Load the shared library
# lib = ctypes.CDLL('./libnuma_alloc.so')
lib = ctypes.CDLL('/home/ubuntu/miniconda3/envs/py310/lib/python3.10/site-packages/transformers/models/opt/libnuma_alloc.so')

# Define the function prototypes

numa_alloc_node = lib.numa_alloc_node
numa_alloc_node.argtypes = [c_size_t, c_int]
numa_alloc_node.restype = c_void_p

numa_alloc_interleave = lib.numa_alloc_interleave
numa_alloc_interleave.argtypes = [c_size_t]
numa_alloc_interleave.restype = c_void_p

numa_free_node = lib.numa_free_node
numa_free_node.argtypes = [c_void_p, c_size_t]
numa_free_node.restype = None

check_memory_node = lib.check_memory_node
check_memory_node.argtypes = [c_void_p, c_size_t]
check_memory_node.restype = None

def numa_alloc_tensor(shape, dtype):
    element_size = torch.tensor([], dtype=dtype).element_size()
    total_size = np.prod(shape) * element_size

    ptr = numa_alloc_interleave(total_size)
    # ptr = numa_alloc_node(total_size, 2)
    if not ptr:
        print("Memory allocation failed")
        return None
    # else:
    #     check_memory_node(ptr, total_size)

    # Buffer creation from the raw pointer
    buffer = (c_char * total_size).from_address(ptr)
    
    # Create an untyped storage from the buffer
    storage = torch.UntypedStorage.from_buffer(buffer, byte_order='native', dtype=dtype)

    # Create a tensor from the untyped storage
    tensor = torch.tensor(storage, dtype=dtype).reshape(shape)

    # return tensor.pin_memory()
    return tensor

def numa_free_tensor(tensor):
    ptr = ctypes.c_void_p(tensor.data_ptr())
    size = tensor.nelement() * tensor.element_size()
    numa_free_node(ptr, c_size_t(size))