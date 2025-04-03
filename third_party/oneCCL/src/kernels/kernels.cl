#include "common.h"

__kernel void empty_kernel(int my_rank,
                           int comm_size,
                           ulong count,
                           const __global void* input_buffer,
                           __global void* output_buffer,
                           const __global void* peer_input_buffer,
                           __global void* peer_output_buffer) {
    return;
}

// DtypeName - e.g. float32, uint16
// Dtype - e.g. float, ushort
// OpName - e.g. sum, prod
// OpFunc - e.g. __sum_int, __prod_float (convention __<OpName>_<Dtype>)

#define PTR_ARGS(Dtype, name, b) __global Dtype* name##b

// PTR_ARGS#: 1-7 args number, max case is 16 ranks
#define PTR_ARGS1(Dtype, name) PTR_ARGS(Dtype, name, 1)
#define PTR_ARGS2(Dtype, name) PTR_ARGS1(Dtype, name), PTR_ARGS(Dtype, name, 2)
#define PTR_ARGS3(Dtype, name) PTR_ARGS2(Dtype, name), PTR_ARGS(Dtype, name, 3)
#define PTR_ARGS4(Dtype, name) PTR_ARGS3(Dtype, name), PTR_ARGS(Dtype, name, 4)
#define PTR_ARGS5(Dtype, name) PTR_ARGS4(Dtype, name), PTR_ARGS(Dtype, name, 5)
#define PTR_ARGS6(Dtype, name) PTR_ARGS5(Dtype, name), PTR_ARGS(Dtype, name, 6)
#define PTR_ARGS7(Dtype, name) PTR_ARGS6(Dtype, name), PTR_ARGS(Dtype, name, 7)

#define ALL_PTR_ARGS(Dtype, name, N) PTR_ARGS(Dtype, name, 0), PTR_ARGS##N(Dtype, name)

#define CONST_ARGS(Dtype, name, b) const Dtype name##b

// CONST_ARGS#: 1-7 args number, max case is 16 ranks
#define CONST_ARGS1(Dtype, name) CONST_ARGS(Dtype, name, 1)
#define CONST_ARGS2(Dtype, name) CONST_ARGS1(Dtype, name), CONST_ARGS(Dtype, name, 2)
#define CONST_ARGS3(Dtype, name) CONST_ARGS2(Dtype, name), CONST_ARGS(Dtype, name, 3)
#define CONST_ARGS4(Dtype, name) CONST_ARGS3(Dtype, name), CONST_ARGS(Dtype, name, 4)
#define CONST_ARGS5(Dtype, name) CONST_ARGS4(Dtype, name), CONST_ARGS(Dtype, name, 5)
#define CONST_ARGS6(Dtype, name) CONST_ARGS5(Dtype, name), CONST_ARGS(Dtype, name, 6)
#define CONST_ARGS7(Dtype, name) CONST_ARGS6(Dtype, name), CONST_ARGS(Dtype, name, 7)

#define ALLTOALLV_ARGS(Dtype, b) \
    __global Dtype *in_buf##b, __global Dtype *out_buf##b, unsigned long count##b,

// ALLTOALLV_ARGS#: 2-16 args number, max case is 16 ranks
#define ALLTOALLV_ARGS2(Dtype) ALLTOALLV_ARGS(Dtype, 0) ALLTOALLV_ARGS(Dtype, 1)
#define ALLTOALLV_ARGS4(Dtype) \
    ALLTOALLV_ARGS2(Dtype) ALLTOALLV_ARGS(Dtype, 2) ALLTOALLV_ARGS(Dtype, 3)
#define ALLTOALLV_ARGS6(Dtype) \
    ALLTOALLV_ARGS4(Dtype) ALLTOALLV_ARGS(Dtype, 4) ALLTOALLV_ARGS(Dtype, 5)
#define ALLTOALLV_ARGS8(Dtype) \
    ALLTOALLV_ARGS6(Dtype) ALLTOALLV_ARGS(Dtype, 6) ALLTOALLV_ARGS(Dtype, 7)
#define ALLTOALLV_ARGS10(Dtype) \
    ALLTOALLV_ARGS8(Dtype) ALLTOALLV_ARGS(Dtype, 8) ALLTOALLV_ARGS(Dtype, 9)
#define ALLTOALLV_ARGS12(Dtype) \
    ALLTOALLV_ARGS10(Dtype) ALLTOALLV_ARGS(Dtype, 10) ALLTOALLV_ARGS(Dtype, 11)
#define ALLTOALLV_ARGS14(Dtype) \
    ALLTOALLV_ARGS12(Dtype) ALLTOALLV_ARGS(Dtype, 12) ALLTOALLV_ARGS(Dtype, 13)
#define ALLTOALLV_ARGS16(Dtype) \
    ALLTOALLV_ARGS14(Dtype) ALLTOALLV_ARGS(Dtype, 14) ALLTOALLV_ARGS(Dtype, 15)

#define ALLTOALLV_COPY(b) \
    for (size_t idx = thread_id; idx < count##b; idx += work_group_size) { \
        out_buf##b[idx] = in_buf##b[idx]; \
    }

#define BUFFER_COPY(dst, src, b) \
    for (size_t idx = thread_id; idx < count##b; idx += work_group_size) { \
        dst##b[idx] = src##b[idx]; \
    }

// ALLTOALLV_COPY#: 2-16 args number, max case is 16 ranks
#define ALLTOALLV_COPY2  ALLTOALLV_COPY(0) ALLTOALLV_COPY(1)
#define ALLTOALLV_COPY4  ALLTOALLV_COPY2 ALLTOALLV_COPY(2) ALLTOALLV_COPY(3)
#define ALLTOALLV_COPY6  ALLTOALLV_COPY4 ALLTOALLV_COPY(4) ALLTOALLV_COPY(5)
#define ALLTOALLV_COPY8  ALLTOALLV_COPY6 ALLTOALLV_COPY(6) ALLTOALLV_COPY(7)
#define ALLTOALLV_COPY10 ALLTOALLV_COPY8 ALLTOALLV_COPY(8) ALLTOALLV_COPY(9)
#define ALLTOALLV_COPY12 ALLTOALLV_COPY10 ALLTOALLV_COPY(10) ALLTOALLV_COPY(11)
#define ALLTOALLV_COPY14 ALLTOALLV_COPY12 ALLTOALLV_COPY(12) ALLTOALLV_COPY(13)
#define ALLTOALLV_COPY16 ALLTOALLV_COPY14 ALLTOALLV_COPY(14) ALLTOALLV_COPY(15)

// BUFFER_COPY#: 1-7 args number, max case is 16 ranks
#define BUFFER_COPY1(dst, src) BUFFER_COPY(dst, src, 1)
#define BUFFER_COPY2(dst, src) BUFFER_COPY1(dst, src) BUFFER_COPY(dst, src, 2)
#define BUFFER_COPY3(dst, src) BUFFER_COPY2(dst, src) BUFFER_COPY(dst, src, 3)
#define BUFFER_COPY4(dst, src) BUFFER_COPY3(dst, src) BUFFER_COPY(dst, src, 4)
#define BUFFER_COPY5(dst, src) BUFFER_COPY4(dst, src) BUFFER_COPY(dst, src, 5)
#define BUFFER_COPY6(dst, src) BUFFER_COPY5(dst, src) BUFFER_COPY(dst, src, 6)
#define BUFFER_COPY7(dst, src) BUFFER_COPY6(dst, src) BUFFER_COPY(dst, src, 7)

#define DEFINE_ALLTOALLV_KERNEL(DtypeName, Dtype, OpName, OpFunc, N) \
    __kernel void alltoallv_kernel_##N##_##DtypeName##_##OpName( \
        ALLTOALLV_ARGS##N(Dtype) int comm_size) { \
        size_t work_group_size = get_global_size(0); \
        size_t thread_id = get_global_id(0); \
        ALLTOALLV_COPY##N \
    }

// reduction for local_reduce
#define REDUCTION(OpFunc, b) \
    xelink_tmp_buf##b[idx] = OpFunc(local_send_buf##b[idx], mdfi_buf##b[idx]);

// REDUCTION#: 1-7 args number, max case is 16 ranks
#define REDUCTION1(OpFunc) REDUCTION(OpFunc, 0)
#define REDUCTION2(OpFunc) REDUCTION1(OpFunc) REDUCTION(OpFunc, 1)
#define REDUCTION3(OpFunc) REDUCTION2(OpFunc) REDUCTION(OpFunc, 2)
#define REDUCTION4(OpFunc) REDUCTION3(OpFunc) REDUCTION(OpFunc, 3)
#define REDUCTION5(OpFunc) REDUCTION4(OpFunc) REDUCTION(OpFunc, 4)
#define REDUCTION6(OpFunc) REDUCTION5(OpFunc) REDUCTION(OpFunc, 5)
#define REDUCTION7(OpFunc) REDUCTION6(OpFunc) REDUCTION(OpFunc, 6)

// reduction for local_reduce
#define FIRST_REDUCE(OpFunc, b0, b1) \
    output_buf[idx] = OpFunc(xelink_tmp_buf##b0[idx], xelink_tmp_buf##b1[idx]);

#define REDUCE(OpFunc, b) output_buf[idx] = OpFunc(output_buf[idx], xelink_tmp_buf##b[idx]);

// REDUCE#: 1-7 args number, max case is 16 ranks
#define REDUCE1(OpFunc) FIRST_REDUCE(OpFunc, 0, 1)
#define REDUCE2(OpFunc) REDUCE1(OpFunc) REDUCE(OpFunc, 2)
#define REDUCE3(OpFunc) REDUCE2(OpFunc) REDUCE(OpFunc, 3)
#define REDUCE4(OpFunc) REDUCE3(OpFunc) REDUCE(OpFunc, 4)
#define REDUCE5(OpFunc) REDUCE4(OpFunc) REDUCE(OpFunc, 5)
#define REDUCE6(OpFunc) REDUCE5(OpFunc) REDUCE(OpFunc, 6)
#define REDUCE7(OpFunc) REDUCE6(OpFunc) REDUCE(OpFunc, 7)

#define DEFINE_REDUCE_READ_WRITE_KERNEL(DtypeName, Dtype, OpName, OpFunc, N) \
    __kernel void reduce_read_write_kernel_##N##_##DtypeName##_##OpName( \
        ALL_PTR_ARGS(Dtype, local_send_buf, N), \
        ALL_PTR_ARGS(Dtype, mdfi_buf, N), \
        ALL_PTR_ARGS(Dtype, xelink_tmp_buf, N), \
        ulong count, \
        ulong last_count) { \
        DEBUG_BLOCK(printf("in reduce_read_write_kernel count %ld\n", count)); \
        size_t work_group_size = get_global_size(0); \
        size_t thread_id = get_global_id(0); \
        for (size_t idx = thread_id; idx < count; idx += work_group_size) { \
            REDUCTION##N(OpFunc) \
        } \
        for (size_t idx = thread_id; idx < last_count; idx += work_group_size) { \
            REDUCTION(OpFunc, N) \
        } \
    }

#define DEFINE_LOCAL_REDUCE_KERNEL(DtypeName, Dtype, OpName, OpFunc, N) \
    __kernel void local_reduce_kernel_##N##_##DtypeName##_##OpName( \
        ulong count, ALL_PTR_ARGS(Dtype, xelink_tmp_buf, N), __global Dtype* output_buf) { \
        size_t work_group_size = get_global_size(0); \
        size_t thread_id = get_global_id(0); \
        DEBUG_BLOCK(printf("in local_reduce_kernel count %ld\n", count)); \
        for (size_t idx = thread_id; idx < count; idx += work_group_size) { \
            REDUCE##N(OpFunc) \
        } \
    }

#define DEFINE_ALLREDUCE_KERNEL(DtypeName, Dtype, OpName, OpFunc) \
    __kernel void allreduce_kernel_##DtypeName##_##OpName(int my_rank, \
                                                          int comm_size, \
                                                          ulong count, \
                                                          const __global Dtype* input_buffer, \
                                                          __global Dtype* output_buffer, \
                                                          const __global Dtype* peer_input_buffer, \
                                                          __global Dtype* peer_output_buffer) { \
        DEBUG_BLOCK(printf("rank: %d, comm size: %d, count: %zu\n", my_rank, comm_size, count)); \
        size_t work_group_size = get_global_size(0); \
        size_t thread_id = get_global_id(0); \
        for (size_t i = 0; thread_id + i < count; i += work_group_size) { \
            const size_t idx = thread_id + i; \
            Dtype ret = OpFunc(input_buffer[idx], peer_input_buffer[idx]); \
            output_buffer[idx] = ret; \
            peer_output_buffer[idx] = ret; \
        } \
    }

#define DEFINE_REDUCE_LOCAL_OUTOFPLACE_KERNEL(DtypeName, Dtype, OpName, OpFunc) \
    __kernel void reduce_local_outofplace_kernel_##DtypeName##_##OpName( \
        int my_rank, \
        int comm_size, \
        ulong count, \
        const __global Dtype* input_buffer_1, \
        const __global Dtype* input_buffer_2, \
        __global Dtype* output_buffer) { \
        DEBUG_BLOCK(printf("rank: %d, comm size: %d, count: %zu\n", my_rank, comm_size, count)); \
        size_t work_group_size = get_global_size(0); \
        size_t thread_id = get_global_id(0); \
        for (size_t i = 0; thread_id + i < count; i += work_group_size) { \
            const size_t idx = thread_id + i; \
            output_buffer[idx] = OpFunc(input_buffer_1[idx], input_buffer_2[idx]); \
        } \
    }

#define DEFINE_REDUCE_LOCAL_INPLACE_KERNEL(DtypeName, Dtype, OpName, OpFunc) \
    __kernel void reduce_local_inplace_kernel_##DtypeName##_##OpName( \
        ulong count, const __global Dtype* input_buffer, __global Dtype* inoutput_buffer) { \
        DEBUG_BLOCK(printf("in reduce_local_inplace_kernel\n")); \
        size_t work_group_size = get_global_size(0); \
        size_t thread_id = get_global_id(0); \
        for (size_t i = 0; thread_id + i < count; i += work_group_size) { \
            const size_t idx = thread_id + i; \
            inoutput_buffer[idx] = OpFunc(input_buffer[idx], inoutput_buffer[idx]); \
        } \
    }

#define DEFINE_REDUCE_SINGLE_LOCAL_INPLACE_KERNEL(DtypeName, Dtype, OpName, OpFunc) \
    __kernel void reduce_single_local_inplace_kernel_##DtypeName##_##OpName( \
        ulong count, \
        int peer_count, \
        const __global Dtype* input_buffer, \
        __global Dtype* inoutput_buffer) { \
        DEBUG_BLOCK(printf("in reduce_single_local_inplace_kernel\n")); \
        size_t work_group_size = get_global_size(0); \
        size_t thread_id = get_global_id(0); \
        for (size_t i = 0; thread_id + i < count; i += work_group_size) { \
            const size_t idx = thread_id + i; \
            Dtype ret = OpFunc(input_buffer[idx], inoutput_buffer[idx]); \
            for (int j = 1; j < peer_count; j++) { \
                ret = OpFunc(inoutput_buffer[j * count + idx], ret); \
            } \
            inoutput_buffer[idx] = ret; \
        } \
    }

// DEFINE_REDUCE_MONOLITHIC_<n>_KERNEL: 1-7 kernels, max case is 16 ranks
#define DEFINE_REDUCE_MONOLITHIC_1_KERNEL(DtypeName, Dtype, OpName, OpFunc) \
    __kernel void reduce_monolithic_kernel_1_##DtypeName##_##OpName( \
        ulong count, \
        const __global Dtype* input_buffer, \
        const __global Dtype* peer_buffer1, \
        __global Dtype* output_buffer) { \
        DEBUG_BLOCK(printf("in reduce_monolithic_kernel_1\n")); \
        const size_t work_group_size = get_global_size(0); \
        const size_t thread_id = get_global_id(0); \
        for (size_t idx = thread_id; idx < count; idx += work_group_size) { \
            Dtype sum = input_buffer[idx]; \
            sum = OpFunc(sum, peer_buffer1[idx]); \
            output_buffer[idx] = sum; \
        } \
    }

#define DEFINE_REDUCE_MONOLITHIC_2_KERNEL(DtypeName, Dtype, OpName, OpFunc) \
    __kernel void reduce_monolithic_kernel_2_##DtypeName##_##OpName( \
        ulong count, \
        const __global Dtype* input_buffer, \
        const __global Dtype* peer_buffer1, \
        const __global Dtype* peer_buffer2, \
        __global Dtype* output_buffer) { \
        DEBUG_BLOCK(printf("in reduce_monolithic_kernel_2\n")); \
        const size_t work_group_size = get_global_size(0); \
        const size_t thread_id = get_global_id(0); \
        for (size_t idx = thread_id; idx < count; idx += work_group_size) { \
            Dtype sum = input_buffer[idx]; \
            sum = OpFunc(sum, peer_buffer1[idx]); \
            sum = OpFunc(sum, peer_buffer2[idx]); \
            output_buffer[idx] = sum; \
        } \
    }

#define DEFINE_REDUCE_MONOLITHIC_3_KERNEL(DtypeName, Dtype, OpName, OpFunc) \
    __kernel void reduce_monolithic_kernel_3_##DtypeName##_##OpName( \
        ulong count, \
        const __global Dtype* input_buffer, \
        const __global Dtype* peer_buffer1, \
        const __global Dtype* peer_buffer2, \
        const __global Dtype* peer_buffer3, \
        __global Dtype* output_buffer) { \
        DEBUG_BLOCK(printf("in reduce_monolithic_kernel_3\n")); \
        const size_t work_group_size = get_global_size(0); \
        const size_t thread_id = get_global_id(0); \
        for (size_t idx = thread_id; idx < count; idx += work_group_size) { \
            Dtype sum = input_buffer[idx]; \
            sum = OpFunc(sum, peer_buffer1[idx]); \
            sum = OpFunc(sum, peer_buffer2[idx]); \
            sum = OpFunc(sum, peer_buffer3[idx]); \
            output_buffer[idx] = sum; \
        } \
    }

#define DEFINE_REDUCE_MONOLITHIC_4_KERNEL(DtypeName, Dtype, OpName, OpFunc) \
    __kernel void reduce_monolithic_kernel_4_##DtypeName##_##OpName( \
        ulong count, \
        const __global Dtype* input_buffer, \
        const __global Dtype* peer_buffer1, \
        const __global Dtype* peer_buffer2, \
        const __global Dtype* peer_buffer3, \
        const __global Dtype* peer_buffer4, \
        __global Dtype* output_buffer) { \
        DEBUG_BLOCK(printf("in reduce_monolithic_kernel_4\n")); \
        const size_t work_group_size = get_global_size(0); \
        const size_t thread_id = get_global_id(0); \
        for (size_t idx = thread_id; idx < count; idx += work_group_size) { \
            Dtype sum = input_buffer[idx]; \
            sum = OpFunc(sum, peer_buffer1[idx]); \
            sum = OpFunc(sum, peer_buffer2[idx]); \
            sum = OpFunc(sum, peer_buffer3[idx]); \
            sum = OpFunc(sum, peer_buffer4[idx]); \
            output_buffer[idx] = sum; \
        } \
    }

#define DEFINE_REDUCE_MONOLITHIC_5_KERNEL(DtypeName, Dtype, OpName, OpFunc) \
    __kernel void reduce_monolithic_kernel_5_##DtypeName##_##OpName( \
        ulong count, \
        const __global Dtype* input_buffer, \
        const __global Dtype* peer_buffer1, \
        const __global Dtype* peer_buffer2, \
        const __global Dtype* peer_buffer3, \
        const __global Dtype* peer_buffer4, \
        const __global Dtype* peer_buffer5, \
        __global Dtype* output_buffer) { \
        DEBUG_BLOCK(printf("in reduce_monolithic_kernel_5\n")); \
        const size_t work_group_size = get_global_size(0); \
        const size_t thread_id = get_global_id(0); \
        for (size_t idx = thread_id; idx < count; idx += work_group_size) { \
            Dtype sum = input_buffer[idx]; \
            sum = OpFunc(sum, peer_buffer1[idx]); \
            sum = OpFunc(sum, peer_buffer2[idx]); \
            sum = OpFunc(sum, peer_buffer3[idx]); \
            sum = OpFunc(sum, peer_buffer4[idx]); \
            sum = OpFunc(sum, peer_buffer5[idx]); \
            output_buffer[idx] = sum; \
        } \
    }

#define DEFINE_REDUCE_MONOLITHIC_6_KERNEL(DtypeName, Dtype, OpName, OpFunc) \
    __kernel void reduce_monolithic_kernel_6_##DtypeName##_##OpName( \
        ulong count, \
        const __global Dtype* input_buffer, \
        const __global Dtype* peer_buffer1, \
        const __global Dtype* peer_buffer2, \
        const __global Dtype* peer_buffer3, \
        const __global Dtype* peer_buffer4, \
        const __global Dtype* peer_buffer5, \
        const __global Dtype* peer_buffer6, \
        __global Dtype* output_buffer) { \
        DEBUG_BLOCK(printf("in reduce_monolithic_kernel_6\n")); \
        const size_t work_group_size = get_global_size(0); \
        const size_t thread_id = get_global_id(0); \
        for (size_t idx = thread_id; idx < count; idx += work_group_size) { \
            Dtype sum = input_buffer[idx]; \
            sum = OpFunc(sum, peer_buffer1[idx]); \
            sum = OpFunc(sum, peer_buffer2[idx]); \
            sum = OpFunc(sum, peer_buffer3[idx]); \
            sum = OpFunc(sum, peer_buffer4[idx]); \
            sum = OpFunc(sum, peer_buffer5[idx]); \
            sum = OpFunc(sum, peer_buffer6[idx]); \
            output_buffer[idx] = sum; \
        } \
    }

#define DEFINE_REDUCE_MONOLITHIC_7_KERNEL(DtypeName, Dtype, OpName, OpFunc) \
    __kernel void reduce_monolithic_kernel_7_##DtypeName##_##OpName( \
        ulong count, \
        const __global Dtype* input_buffer, \
        const __global Dtype* peer_buffer1, \
        const __global Dtype* peer_buffer2, \
        const __global Dtype* peer_buffer3, \
        const __global Dtype* peer_buffer4, \
        const __global Dtype* peer_buffer5, \
        const __global Dtype* peer_buffer6, \
        const __global Dtype* peer_buffer7, \
        __global Dtype* output_buffer) { \
        DEBUG_BLOCK(printf("in reduce_monolithic_kernel_7\n")); \
        const size_t work_group_size = get_global_size(0); \
        const size_t thread_id = get_global_id(0); \
        for (size_t idx = thread_id; idx < count; idx += work_group_size) { \
            Dtype sum = input_buffer[idx]; \
            sum = OpFunc(sum, peer_buffer1[idx]); \
            sum = OpFunc(sum, peer_buffer2[idx]); \
            sum = OpFunc(sum, peer_buffer3[idx]); \
            sum = OpFunc(sum, peer_buffer4[idx]); \
            sum = OpFunc(sum, peer_buffer5[idx]); \
            sum = OpFunc(sum, peer_buffer6[idx]); \
            sum = OpFunc(sum, peer_buffer7[idx]); \
            output_buffer[idx] = sum; \
        } \
    }

// DEFINE_WRITE_MONOLITHIC_<n>_KERNEL: 1-7 kernels, max case is 16 ranks
#define DEFINE_WRITE_MONOLITHIC_1_KERNEL(DtypeName, Dtype, OpName, OpFunc) \
    __kernel void write_monolithic_kernel_1_##DtypeName##_##OpName( \
        ulong count, const __global Dtype* input_buffer, __global Dtype* peer_buffer1) { \
        DEBUG_BLOCK(printf("in write_monolithic_kernel_1 count %d\n", count)); \
        const size_t work_group_size = get_global_size(0); \
        const size_t thread_id = get_global_id(0); \
        for (size_t idx = thread_id; idx < count; idx += work_group_size) { \
            const Dtype val = input_buffer[idx]; \
            peer_buffer1[idx] = val; \
        } \
    }

#define DEFINE_WRITE_MONOLITHIC_2_KERNEL(DtypeName, Dtype, OpName, OpFunc) \
    __kernel void write_monolithic_kernel_2_##DtypeName##_##OpName( \
        ulong count, \
        const __global Dtype* input_buffer, \
        __global Dtype* peer_buffer1, \
        __global Dtype* peer_buffer2) { \
        DEBUG_BLOCK(printf("in write_monolithic_kernel_2 count %d\n", count)); \
        const size_t work_group_size = get_global_size(0); \
        const size_t thread_id = get_global_id(0); \
        for (size_t idx = thread_id; idx < count; idx += work_group_size) { \
            const Dtype val = input_buffer[idx]; \
            peer_buffer1[idx] = val; \
            peer_buffer2[idx] = val; \
        } \
    }

#define DEFINE_WRITE_MONOLITHIC_3_KERNEL(DtypeName, Dtype, OpName, OpFunc) \
    __kernel void write_monolithic_kernel_3_##DtypeName##_##OpName( \
        ulong count, \
        const __global Dtype* input_buffer, \
        __global Dtype* peer_buffer1, \
        __global Dtype* peer_buffer2, \
        __global Dtype* peer_buffer3) { \
        DEBUG_BLOCK(printf("in write_monolithic_kernel_3 count %d\n", count)); \
        const size_t work_group_size = get_global_size(0); \
        const size_t thread_id = get_global_id(0); \
        for (size_t idx = thread_id; idx < count; idx += work_group_size) { \
            const Dtype val = input_buffer[idx]; \
            peer_buffer1[idx] = val; \
            peer_buffer2[idx] = val; \
            peer_buffer3[idx] = val; \
        } \
    }

#define DEFINE_WRITE_MONOLITHIC_4_KERNEL(DtypeName, Dtype, OpName, OpFunc) \
    __kernel void write_monolithic_kernel_4_##DtypeName##_##OpName( \
        ulong count, \
        const __global Dtype* input_buffer, \
        __global Dtype* peer_buffer1, \
        __global Dtype* peer_buffer2, \
        __global Dtype* peer_buffer3, \
        __global Dtype* peer_buffer4) { \
        DEBUG_BLOCK(printf("in write_monolithic_kernel_4 count %d\n", count)); \
        const size_t work_group_size = get_global_size(0); \
        const size_t thread_id = get_global_id(0); \
        for (size_t idx = thread_id; idx < count; idx += work_group_size) { \
            const Dtype val = input_buffer[idx]; \
            peer_buffer1[idx] = val; \
            peer_buffer2[idx] = val; \
            peer_buffer3[idx] = val; \
            peer_buffer4[idx] = val; \
        } \
    }

#define DEFINE_WRITE_MONOLITHIC_5_KERNEL(DtypeName, Dtype, OpName, OpFunc) \
    __kernel void write_monolithic_kernel_5_##DtypeName##_##OpName( \
        ulong count, \
        const __global Dtype* input_buffer, \
        __global Dtype* peer_buffer1, \
        __global Dtype* peer_buffer2, \
        __global Dtype* peer_buffer3, \
        __global Dtype* peer_buffer4, \
        __global Dtype* peer_buffer5) { \
        DEBUG_BLOCK(printf("in write_monolithic_kernel_5 count %d\n", count)); \
        const size_t work_group_size = get_global_size(0); \
        const size_t thread_id = get_global_id(0); \
        for (size_t idx = thread_id; idx < count; idx += work_group_size) { \
            const Dtype val = input_buffer[idx]; \
            peer_buffer1[idx] = val; \
            peer_buffer2[idx] = val; \
            peer_buffer3[idx] = val; \
            peer_buffer4[idx] = val; \
            peer_buffer5[idx] = val; \
        } \
    }

#define DEFINE_WRITE_MONOLITHIC_6_KERNEL(DtypeName, Dtype, OpName, OpFunc) \
    __kernel void write_monolithic_kernel_6_##DtypeName##_##OpName( \
        ulong count, \
        const __global Dtype* input_buffer, \
        __global Dtype* peer_buffer1, \
        __global Dtype* peer_buffer2, \
        __global Dtype* peer_buffer3, \
        __global Dtype* peer_buffer4, \
        __global Dtype* peer_buffer5, \
        __global Dtype* peer_buffer6) { \
        DEBUG_BLOCK(printf("in write_monolithic_kernel_6 count %d\n", count)); \
        const size_t work_group_size = get_global_size(0); \
        const size_t thread_id = get_global_id(0); \
        for (size_t idx = thread_id; idx < count; idx += work_group_size) { \
            const Dtype val = input_buffer[idx]; \
            peer_buffer1[idx] = val; \
            peer_buffer2[idx] = val; \
            peer_buffer3[idx] = val; \
            peer_buffer4[idx] = val; \
            peer_buffer5[idx] = val; \
            peer_buffer6[idx] = val; \
        } \
    }

#define DEFINE_WRITE_MONOLITHIC_7_KERNEL(DtypeName, Dtype, OpName, OpFunc) \
    __kernel void write_monolithic_kernel_7_##DtypeName##_##OpName( \
        ulong count, \
        const __global Dtype* input_buffer, \
        __global Dtype* peer_buffer1, \
        __global Dtype* peer_buffer2, \
        __global Dtype* peer_buffer3, \
        __global Dtype* peer_buffer4, \
        __global Dtype* peer_buffer5, \
        __global Dtype* peer_buffer6, \
        __global Dtype* peer_buffer7) { \
        DEBUG_BLOCK(printf("in write_monolithic_kernel_7 count %d\n", count)); \
        const size_t work_group_size = get_global_size(0); \
        const size_t thread_id = get_global_id(0); \
        for (size_t idx = thread_id; idx < count; idx += work_group_size) { \
            const Dtype val = input_buffer[idx]; \
            peer_buffer1[idx] = val; \
            peer_buffer2[idx] = val; \
            peer_buffer3[idx] = val; \
            peer_buffer4[idx] = val; \
            peer_buffer5[idx] = val; \
            peer_buffer6[idx] = val; \
            peer_buffer7[idx] = val; \
        } \
    }

// Monolithic kernel reads data from buffers in Xelink peers and then writes it to buffers in MDFI peer
#define DEFINE_READ_WRITE_MONOLITHIC_KERNEL(DtypeName, Dtype, OpName, OpFunc, N) \
    __kernel void read_write_monolithic_kernel_##N##_##DtypeName##_##OpName( \
        const int pipeline_count, \
        PTR_ARGS##N(Dtype, peer_buffer), \
        PTR_ARGS##N(Dtype, output_buffer), \
        PTR_ARGS##N(Dtype, peer_output_buffer), \
        CONST_ARGS##N(ulong, count)) { \
        DEBUG_BLOCK(printf("in read_write_monolithic_kernel_%d\n", N)); \
        size_t work_group_size = get_global_size(0); \
        size_t thread_id = get_global_id(0); \
        BUFFER_COPY##N(output_buffer, peer_buffer) if (pipeline_count > 1) { \
            BUFFER_COPY##N(peer_output_buffer, output_buffer) \
        } \
    }

// Define kernels for a specific reduction operation for all supported datatypes
#define DEFINE_KERNELS_WITH_OP(KernelName, OpName) \
    DEFINE_##KernelName##_KERNEL(int8, char, OpName, __##OpName##_##char) \
        DEFINE_##KernelName##_KERNEL(uint8, uchar, OpName, __##OpName##_##uchar) \
\
            DEFINE_##KernelName##_KERNEL(int16, short, OpName, __##OpName##_##short) \
                DEFINE_##KernelName##_KERNEL(uint16, ushort, OpName, __##OpName##_##ushort) \
\
                    DEFINE_##KernelName##_KERNEL(int32, int, OpName, __##OpName##_##int) \
                        DEFINE_##KernelName##_KERNEL(uint32, uint, OpName, __##OpName##_##uint) \
\
                            DEFINE_##KernelName##_KERNEL(int64, long, OpName, __##OpName##_##long) \
                                DEFINE_##KernelName##_KERNEL( \
                                    uint64, ulong, OpName, __##OpName##_##ulong) \
\
                                    DEFINE_##KernelName##_KERNEL( \
                                        float32, float, OpName, __##OpName##_##float) \
                                        DEFINE_##KernelName##_KERNEL( \
                                            float64, double, OpName, __##OpName##_##double)

//Define kernels for a specific reduction operation for all supported datatypes
#define DEFINE_KERNELS_WITH_OP_N(KernelName, OpName, N) \
    DEFINE_##KernelName##_KERNEL(int8, char, OpName, __##OpName##_##char, N) \
        DEFINE_##KernelName##_KERNEL(uint8, uchar, OpName, __##OpName##_##uchar, N) \
\
            DEFINE_##KernelName##_KERNEL(int16, short, OpName, __##OpName##_##short, N) \
                DEFINE_##KernelName##_KERNEL(uint16, ushort, OpName, __##OpName##_##ushort, N) \
\
                    DEFINE_##KernelName##_KERNEL(int32, int, OpName, __##OpName##_##int, N) \
                        DEFINE_##KernelName##_KERNEL(uint32, uint, OpName, __##OpName##_##uint, N) \
\
                            DEFINE_##KernelName##_KERNEL( \
                                int64, long, OpName, __##OpName##_##long, N) \
                                DEFINE_##KernelName##_KERNEL( \
                                    uint64, ulong, OpName, __##OpName##_##ulong, N) \
\
                                    DEFINE_##KernelName##_KERNEL( \
                                        float32, float, OpName, __##OpName##_##float, N) \
                                        DEFINE_##KernelName##_KERNEL( \
                                            float64, double, OpName, __##OpName##_##double, N)

#define DEFINE_KERNELS_WITH_LP_OP(KernelName, OpName) \
    DEFINE_##KernelName##_KERNEL(bfloat16, ushort, OpName, __bf16_##OpName##_##ushort) \
        DEFINE_##KernelName##_KERNEL(float16, half, OpName, __##OpName##_##half)

#define DEFINE_KERNELS_WITH_LP_OP_N(KernelName, OpName, N) \
    DEFINE_##KernelName##_KERNEL(bfloat16, ushort, OpName, __bf16_##OpName##_##ushort, N) \
        DEFINE_##KernelName##_KERNEL(float16, half, OpName, __##OpName##_##half, N)

#define DEFINE_OPS(T) \
    DEFINE_SUM_OP(T) \
    DEFINE_PROD_OP(T) \
    DEFINE_MIN_OP(T) \
    DEFINE_MAX_OP(T)

#define DEFINE_BF16OPS(T) \
    DEFINE_BF16SUM_OP(T) \
    DEFINE_BF16PROD_OP(T) \
    DEFINE_BF16MIN_OP(T) \
    DEFINE_BF16MAX_OP(T)

#define DEFINE_FP16OPS(T) \
    DEFINE_FP16SUM_OP(T) \
    DEFINE_FP16PROD_OP(T) \
    DEFINE_FP16MIN_OP(T) \
    DEFINE_FP16MAX_OP(T)

// Define reduction operation function for each supported datatype
DEFINE_OPS(char)
DEFINE_OPS(uchar)

DEFINE_OPS(short)
DEFINE_OPS(ushort)

DEFINE_OPS(int)
DEFINE_OPS(uint)

DEFINE_OPS(long)
DEFINE_OPS(ulong)

DEFINE_OPS(float)
DEFINE_OPS(double)

DEFINE_BF16OPS(ushort)
DEFINE_FP16OPS(half)

// Define the actual kernels
#define DEFINE_ALL_KERNELS(KernelName) \
    DEFINE_KERNELS_WITH_OP(KernelName, sum) \
    DEFINE_KERNELS_WITH_OP(KernelName, prod) \
    DEFINE_KERNELS_WITH_OP(KernelName, min) \
    DEFINE_KERNELS_WITH_OP(KernelName, max) \
\
    DEFINE_KERNELS_WITH_LP_OP(KernelName, sum) \
    DEFINE_KERNELS_WITH_LP_OP(KernelName, prod) \
    DEFINE_KERNELS_WITH_LP_OP(KernelName, min) \
    DEFINE_KERNELS_WITH_LP_OP(KernelName, max)

//Define the actual kernels with peer_count
#define DEFINE_ALL_KERNELS_N(KernelName, N) \
    DEFINE_KERNELS_WITH_OP_N(KernelName, custom, N) \
\
    DEFINE_KERNELS_WITH_LP_OP_N(KernelName, custom, N)

// Define the actual kernels for all peer_counts
#define DEFINE_ALL_KERNELS_PEERS(KernelName) \
    DEFINE_ALL_KERNELS_N(KernelName, 2) \
    DEFINE_ALL_KERNELS_N(KernelName, 4) \
    DEFINE_ALL_KERNELS_N(KernelName, 6) \
    DEFINE_ALL_KERNELS_N(KernelName, 8) \
    DEFINE_ALL_KERNELS_N(KernelName, 10) \
    DEFINE_ALL_KERNELS_N(KernelName, 12) \
    DEFINE_ALL_KERNELS_N(KernelName, 14) \
    DEFINE_ALL_KERNELS_N(KernelName, 16)

#define DEFINE_ALL_KERNELS_PEERS_PLANE(KernelName) \
    DEFINE_ALL_KERNELS_N(KernelName, 1) \
    DEFINE_ALL_KERNELS_N(KernelName, 2) \
    DEFINE_ALL_KERNELS_N(KernelName, 3) \
    DEFINE_ALL_KERNELS_N(KernelName, 4) \
    DEFINE_ALL_KERNELS_N(KernelName, 5) \
    DEFINE_ALL_KERNELS_N(KernelName, 6) \
    DEFINE_ALL_KERNELS_N(KernelName, 7)

#define DEFINE_ALL_KERNELS_OP_N(KernelName, N) \
    DEFINE_KERNELS_WITH_OP_N(KernelName, sum, N) \
    DEFINE_KERNELS_WITH_OP_N(KernelName, prod, N) \
    DEFINE_KERNELS_WITH_OP_N(KernelName, min, N) \
    DEFINE_KERNELS_WITH_OP_N(KernelName, max, N) \
\
    DEFINE_KERNELS_WITH_LP_OP_N(KernelName, sum, N) \
    DEFINE_KERNELS_WITH_LP_OP_N(KernelName, prod, N) \
    DEFINE_KERNELS_WITH_LP_OP_N(KernelName, min, N) \
    DEFINE_KERNELS_WITH_LP_OP_N(KernelName, max, N)

#define DEFINE_ALL_KERNELS_PEERS_PLANE_OP(KernelName) \
    DEFINE_ALL_KERNELS_OP_N(KernelName, 1) \
    DEFINE_ALL_KERNELS_OP_N(KernelName, 2) \
    DEFINE_ALL_KERNELS_OP_N(KernelName, 3) \
    DEFINE_ALL_KERNELS_OP_N(KernelName, 4) \
    DEFINE_ALL_KERNELS_OP_N(KernelName, 5) \
    DEFINE_ALL_KERNELS_OP_N(KernelName, 6) \
    DEFINE_ALL_KERNELS_OP_N(KernelName, 7)

DEFINE_ALL_KERNELS_PEERS(ALLTOALLV)
DEFINE_ALL_KERNELS_PEERS_PLANE(READ_WRITE_MONOLITHIC)
DEFINE_ALL_KERNELS_PEERS_PLANE_OP(REDUCE_READ_WRITE)
DEFINE_ALL_KERNELS_PEERS_PLANE_OP(LOCAL_REDUCE)

DEFINE_ALL_KERNELS(ALLREDUCE)
DEFINE_ALL_KERNELS(REDUCE_LOCAL_OUTOFPLACE)
DEFINE_ALL_KERNELS(REDUCE_LOCAL_INPLACE)
DEFINE_ALL_KERNELS(REDUCE_SINGLE_LOCAL_INPLACE)
DEFINE_ALL_KERNELS(REDUCE_MONOLITHIC_1)
DEFINE_ALL_KERNELS(REDUCE_MONOLITHIC_2)
DEFINE_ALL_KERNELS(REDUCE_MONOLITHIC_3)
DEFINE_ALL_KERNELS(REDUCE_MONOLITHIC_4)
DEFINE_ALL_KERNELS(REDUCE_MONOLITHIC_5)
DEFINE_ALL_KERNELS(REDUCE_MONOLITHIC_6)
DEFINE_ALL_KERNELS(REDUCE_MONOLITHIC_7)

DEFINE_KERNELS_WITH_OP(WRITE_MONOLITHIC_1, custom)
DEFINE_KERNELS_WITH_OP(WRITE_MONOLITHIC_2, custom)
DEFINE_KERNELS_WITH_OP(WRITE_MONOLITHIC_3, custom)
DEFINE_KERNELS_WITH_OP(WRITE_MONOLITHIC_4, custom)
DEFINE_KERNELS_WITH_OP(WRITE_MONOLITHIC_5, custom)
DEFINE_KERNELS_WITH_OP(WRITE_MONOLITHIC_6, custom)
DEFINE_KERNELS_WITH_OP(WRITE_MONOLITHIC_7, custom)

DEFINE_KERNELS_WITH_LP_OP(WRITE_MONOLITHIC_1, custom)
DEFINE_KERNELS_WITH_LP_OP(WRITE_MONOLITHIC_2, custom)
DEFINE_KERNELS_WITH_LP_OP(WRITE_MONOLITHIC_3, custom)
DEFINE_KERNELS_WITH_LP_OP(WRITE_MONOLITHIC_4, custom)
DEFINE_KERNELS_WITH_LP_OP(WRITE_MONOLITHIC_5, custom)
DEFINE_KERNELS_WITH_LP_OP(WRITE_MONOLITHIC_6, custom)
DEFINE_KERNELS_WITH_LP_OP(WRITE_MONOLITHIC_7, custom)
