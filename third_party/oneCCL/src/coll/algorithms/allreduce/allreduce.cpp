/*
 Copyright 2016-2020 Intel Corporation
 
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at
 
     http://www.apache.org/licenses/LICENSE-2.0
 
 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
*/

/*
 *
 *  (C) 2001 by Argonne National Laboratory.
 *      See COPYRIGHT in top-level directory.
 */

#include "coll/algorithms/algorithms.hpp"
#include "coll/algorithms/algorithm_utils.hpp"
#include "coll/coll_util.hpp"
#include "comm/comm.hpp"
#include "sched/entry/copy/copy_helper.hpp"
#include "sched/entry/factory/chunked_entry_factory.hpp"
#include "sched/entry/factory/entry_factory.hpp"

using namespace ccl::utils;

ccl::status ccl_coll_build_direct_allreduce(ccl_sched* sched,
                                            ccl_buffer send_buf,
                                            ccl_buffer recv_buf,
                                            size_t count,
                                            const ccl_datatype& dtype,
                                            ccl::reduction op,
                                            ccl_comm* comm) {
    LOG_DEBUG("build direct allreduce");

    // count is the same for all ranks
    // if one rank skips mpi collectives, all ranks skip
    // this means we can safely skip all operations with zero count
    if (count == 0) {
        return ccl::status::success;
    }

    entry_factory::create<allreduce_entry>(sched, send_buf, recv_buf, count, dtype, op, comm);
    return ccl::status::success;
}

ccl::status ccl_coll_build_rabenseifner_allreduce(ccl_sched* sched,
                                                  ccl_buffer send_buf,
                                                  ccl_buffer recv_buf,
                                                  size_t count,
                                                  const ccl_datatype& dtype,
                                                  ccl::reduction op,
                                                  ccl_comm* comm) {
    LOG_DEBUG("build Rabenseifner's allreduce");
    CCL_ASSERT(sched != nullptr, "empty sched");

    ccl::status status = ccl::status::success;
    int comm_size, rank, newrank, pof2, rem;
    int i, send_idx, recv_idx, last_idx, mask, newdst, dst, send_cnt, recv_cnt;
    int *cnts = NULL, *disps = NULL;
    size_t dtype_size = dtype.size();

    comm_size = comm->size();

    // count is the same for all ranks
    // if one rank skips mpi collectives, all ranks skip
    // this means we can safely skip all operations with zero count
    if (count == 0) {
        return ccl::status::success;
    }

    rank = comm->rank();
    ccl_buffer tmp_buf = sched->alloc_buffer({ count * dtype_size, send_buf });

    /* copy local data into recv_buf */

    if (send_buf != recv_buf) {
        entry_factory::create<copy_entry>(sched, send_buf, recv_buf, count, dtype);
        sched->add_barrier();
    }

    if (comm_size == 1)
        return status;

    /* get nearest power-of-two less than or equal to comm_size */
    pof2 = comm->pof2();

    rem = comm_size - pof2;

    /* In the non-power-of-two case, all even-numbered
     * processes of rank < 2*rem send their data to
     * (rank+1). These even-numbered processes no longer
     * participate in the algorithm until the very end. The
     * remaining processes form a nice power-of-two. */

    if (rank < 2 * rem) {
        if (rank % 2 == 0) { /* even */
            entry_factory::create<send_entry>(sched, recv_buf, count, dtype, rank + 1, comm);
            sched->add_barrier();

            /* temporarily set the rank to -1 so that this
             * process does not pariticipate in recursive
             * doubling */
            newrank = CCL_INVALID_PROC_IDX;
        }
        else { /* odd */
            entry_factory::create<recv_entry>(sched, tmp_buf, count, dtype, rank - 1, comm);
            sched->add_barrier();

            /* do the reduction on received data. since the
             * ordering is right, it doesn't matter whether
             * the operation is commutative or not. */
            entry_factory::create<reduce_local_entry>(
                sched, tmp_buf, count, recv_buf, nullptr, dtype, op);
            sched->add_barrier();

            /* change the rank */
            newrank = rank / 2;
        }
    }
    else /* rank >= 2*rem */
        newrank = rank - rem;

    if (newrank != CCL_INVALID_PROC_IDX) {
        /* for the reduce-scatter, calculate the count that
         * each process receives and the displacement within
         * the buffer */
        cnts = static_cast<int*>(CCL_MALLOC(pof2 * sizeof(int), "counts"));
        disps = static_cast<int*>(CCL_MALLOC(pof2 * sizeof(int), "displacements"));

        /* the cnts calculations assume this */
        CCL_ASSERT(count >= static_cast<size_t>(pof2), "count ", count, ", pof2 ", pof2);

        for (i = 0; i < (pof2 - 1); i++)
            cnts[i] = count / pof2;
        cnts[pof2 - 1] = count - (count / pof2) * (pof2 - 1);

        disps[0] = 0;
        for (i = 1; i < pof2; i++)
            disps[i] = disps[i - 1] + cnts[i - 1];

        mask = 0x1;
        send_idx = recv_idx = 0;
        last_idx = pof2;
        while (mask < pof2) {
            newdst = newrank ^ mask;
            /* find real rank of dest */
            dst = (newdst < rem) ? newdst * 2 + 1 : newdst + rem;

            send_cnt = recv_cnt = 0;
            if (newrank < newdst) {
                send_idx = recv_idx + pof2 / (mask * 2);
                for (i = send_idx; i < last_idx; i++)
                    send_cnt += cnts[i];
                for (i = recv_idx; i < send_idx; i++)
                    recv_cnt += cnts[i];
            }
            else {
                recv_idx = send_idx + pof2 / (mask * 2);
                for (i = send_idx; i < recv_idx; i++)
                    send_cnt += cnts[i];
                for (i = recv_idx; i < last_idx; i++)
                    recv_cnt += cnts[i];
            }

            int can_use_recv_reduce = 0;
            ccl_buffer buf1 = (recv_buf + disps[send_idx] * dtype_size);
            ccl_buffer buf2 = (recv_buf + disps[recv_idx] * dtype_size);
            if (buf1 != buf2 &&
                ((buf1 + send_cnt * dtype_size <= buf2) || (buf2 + recv_cnt * dtype_size <= buf1)))
                can_use_recv_reduce = 1;

            CCL_ASSERT(can_use_recv_reduce);

            if (can_use_recv_reduce) {
                entry_factory::create<recv_reduce_entry>(sched,
                                                         (recv_buf + disps[recv_idx] * dtype_size),
                                                         recv_cnt,
                                                         dtype,
                                                         op,
                                                         dst,
                                                         comm);
                entry_factory::create<send_entry>(
                    sched, (recv_buf + disps[send_idx] * dtype_size), send_cnt, dtype, dst, comm);
                sched->add_barrier();
            }

            else {
                /* Send data from recv_buf. Recv into tmp_buf */
                entry_factory::create<recv_entry>(
                    sched, (tmp_buf + disps[recv_idx] * dtype_size), recv_cnt, dtype, dst, comm);
                /* sendrecv, no barrier here */
                entry_factory::create<send_entry>(
                    sched, (recv_buf + disps[send_idx] * dtype_size), send_cnt, dtype, dst, comm);
                sched->add_barrier();

                /* tmp_buf contains data received in this step.
                 * recv_buf contains data accumulated so far */

                /* This algorithm is used only for predefined ops
                 * and predefined ops are always commutative. */
                entry_factory::create<reduce_local_entry>(sched,
                                                          (tmp_buf + disps[recv_idx] * dtype_size),
                                                          recv_cnt,
                                                          (recv_buf + disps[recv_idx] * dtype_size),
                                                          nullptr,
                                                          dtype,
                                                          op);
                sched->add_barrier();
            }

            /* update send_idx for next iteration */
            send_idx = recv_idx;
            mask <<= 1;

            /* update last_idx, but not in last iteration
             * because the value is needed in the allgather
             * step below. */
            if (mask < pof2)
                last_idx = recv_idx + pof2 / mask;
        }

        /* now do the allgather */

        mask >>= 1;
        while (mask > 0) {
            newdst = newrank ^ mask;
            /* find real rank of dest */
            dst = (newdst < rem) ? newdst * 2 + 1 : newdst + rem;

            send_cnt = recv_cnt = 0;
            if (newrank < newdst) {
                /* update last_idx except on first iteration */
                if (mask != pof2 / 2)
                    last_idx = last_idx + pof2 / (mask * 2);

                recv_idx = send_idx + pof2 / (mask * 2);
                for (i = send_idx; i < recv_idx; i++)
                    send_cnt += cnts[i];
                for (i = recv_idx; i < last_idx; i++)
                    recv_cnt += cnts[i];
            }
            else {
                recv_idx = send_idx - pof2 / (mask * 2);
                for (i = send_idx; i < last_idx; i++)
                    send_cnt += cnts[i];
                for (i = recv_idx; i < send_idx; i++)
                    recv_cnt += cnts[i];
            }

            entry_factory::create<recv_entry>(
                sched, (recv_buf + disps[recv_idx] * dtype_size), recv_cnt, dtype, dst, comm);
            /* sendrecv, no barrier here */
            entry_factory::create<send_entry>(
                sched, (recv_buf + disps[send_idx] * dtype_size), send_cnt, dtype, dst, comm);
            sched->add_barrier();

            if (newrank > newdst)
                send_idx = recv_idx;

            mask >>= 1;
        }
    }

    /* In the non-power-of-two case, all odd-numbered
     * processes of rank < 2*rem send the result to
     * (rank-1), the ranks who didn't participate above. */
    if (rank < 2 * rem) {
        if (rank % 2) { /* odd */
            entry_factory::create<send_entry>(sched, recv_buf, count, dtype, rank - 1, comm);
        }
        else { /* even */
            entry_factory::create<recv_entry>(sched, recv_buf, count, dtype, rank + 1, comm);
        }
    }

    CCL_FREE(cnts);
    CCL_FREE(disps);

    return status;
}

ccl::status ccl_coll_build_nreduce_allreduce(ccl_sched* sched,
                                             ccl_buffer send_buf,
                                             ccl_buffer recv_buf,
                                             size_t count,
                                             const ccl_datatype& dtype,
                                             ccl::reduction op,
                                             ccl_comm* comm) {
    LOG_DEBUG("build nreduce allreduce");

    ccl::status status = ccl::status::success;

    // count is the same for all ranks
    // if one rank skips mpi collectives, all ranks skip
    // this means we can safely skip all operations with zero count
    if (count == 0) {
        return status;
    }

    int comm_size = comm->size();
    int comm_rank = comm->rank();
    std::vector<size_t> elem_counts(comm_size);
    std::vector<size_t> elem_offsets(comm_size);
    size_t dtype_size = dtype.size();
    bool is_inplace = (send_buf == recv_buf);

    if (comm_size == 1) {
        if (!is_inplace) {
            entry_factory::create<copy_entry>(sched, send_buf, recv_buf, count, dtype);
        }
        return status;
    }

    int use_buffering = ccl::global_data::env().allreduce_nreduce_buffering;

    size_t segment_size = 2 * 1024 * 1024;
    if (ccl::global_data::env().allreduce_nreduce_segment_size != CCL_ENV_SIZET_NOT_SPECIFIED) {
        segment_size = ccl::global_data::env().allreduce_nreduce_segment_size;
    }

    std::vector<size_t> segment_sizes;
    ccl_get_segment_sizes(dtype_size, count, segment_size, segment_sizes);

    size_t tmp_buf_size = *segment_sizes.rbegin() * comm_size * dtype_size;
    ccl_buffer tmp_buf = sched->alloc_buffer({ tmp_buf_size, send_buf });

    size_t seg_offset = 0;

    for (size_t seg_idx = 0; seg_idx < segment_sizes.size(); seg_idx++) {
        size_t seg_size = segment_sizes[seg_idx];

        ccl_buffer seg_send_buf = send_buf + seg_offset;
        ccl_buffer seg_recv_buf = recv_buf + seg_offset;
        ccl_buffer seg_tmp_buf = tmp_buf;

        seg_offset += seg_size * dtype_size;

        // calculate counts and offsets for each rank
        size_t common_buffer_count = seg_size / comm_size;
        for (int idx = 0; idx < comm_size; idx++) {
            elem_counts[idx] = common_buffer_count;
            elem_offsets[idx] = idx * elem_counts[idx] * dtype_size;
        }
        elem_counts[comm_size - 1] += seg_size % comm_size;

        size_t elem_count = elem_counts[comm_rank];

        ccl_buffer reduce_buf;
        if (use_buffering) {
            reduce_buf = seg_tmp_buf + elem_count * comm_rank * dtype_size;
        }
        else {
            reduce_buf = seg_recv_buf + elem_offsets[comm_rank];
        }

        if (!is_inplace || use_buffering) {
            entry_factory::create<copy_entry>(sched,
                                              seg_send_buf + elem_offsets[comm_rank],
                                              reduce_buf,
                                              elem_counts[comm_rank],
                                              dtype);
            sched->add_barrier();
        }

        // reduce-scatter
        for (int idx = 1; idx < comm_size; idx++) {
            // send part of buffer to other rank
            int dst = (comm_rank - idx + comm_size) % comm_size;
            entry_factory::create<send_entry>(
                sched, seg_send_buf + elem_offsets[dst], elem_counts[dst], dtype, dst, comm);

            // recv part of buffer from other rank and perform reduce
            int src = (comm_rank + idx) % comm_size;
            entry_factory::create<recv_reduce_entry>(sched,
                                                     reduce_buf,
                                                     elem_count,
                                                     dtype,
                                                     op,
                                                     src,
                                                     comm,
                                                     seg_tmp_buf + elem_count * src * dtype_size);
        }

        sched->add_barrier();

        // allgatherv
        if (use_buffering) {
            copy_attr attr;
            attr.direction = copy_direction::h2h;
            attr.use_nontemporal = true;

            // copy own result from tmp to recv buffer
            entry_factory::create<copy_entry>(
                sched, reduce_buf, seg_recv_buf + elem_offsets[comm_rank], elem_count, dtype, attr);
            sched->add_barrier();

            for (int idx = 1; idx < comm_size; idx++) {
                int dst = (comm_rank + idx) % comm_size;
                int src = (comm_rank - idx + comm_size) % comm_size;

                // send own result to other ranks
                entry_factory::create<send_entry>(
                    sched, reduce_buf, elem_counts[comm_rank], dtype, dst, comm);

                // recv other's rank result into tmp buffer and copy to recv buffer
                entry_factory::create<recv_copy_entry>(sched,
                                                       seg_tmp_buf + elem_offsets[src],
                                                       seg_recv_buf + elem_offsets[src],
                                                       elem_counts[src] * dtype_size,
                                                       src,
                                                       comm,
                                                       attr);
            }
        }
        else {
            CCL_CALL(ccl_coll_build_naive_allgatherv(sched,
                                                     seg_recv_buf,
                                                     elem_counts[comm_rank],
                                                     seg_recv_buf,
                                                     elem_counts.data(),
                                                     dtype,
                                                     comm));
        }

        sched->add_barrier();
    }

    return status;
}

ccl::status ccl_coll_build_ring_allreduce(ccl_sched* sched,
                                          ccl_buffer send_buf,
                                          ccl_buffer recv_buf,
                                          size_t count,
                                          const ccl_datatype& dtype,
                                          ccl::reduction op,
                                          ccl_comm* comm) {
    int inplace = (send_buf == recv_buf) ? 1 : 0;
    LOG_DEBUG("build ring allreduce ", inplace ? "in-place" : "out-of-place");

    // count is the same for all ranks
    // if one rank skips mpi collectives, all ranks skip
    // this means we can safely skip all operations with zero count
    if (count == 0) {
        return ccl::status::success;
    }

    CCL_THROW_IF_NOT(sched && send_buf && recv_buf,
                     "incorrect values, sched ",
                     sched,
                     ", send ",
                     send_buf,
                     " recv ",
                     recv_buf);

    ccl::status status = ccl::status::success;
    ccl_coll_build_ring_reduce_scatter(sched, send_buf, recv_buf, count, dtype, op, comm);

    sched->add_barrier();

    int comm_size = comm->size();
    size_t main_block_count = count / comm_size;
    size_t last_block_count = main_block_count + count % comm_size;
    std::vector<size_t> recv_counts(comm_size, main_block_count);
    if (count % comm_size) {
        recv_counts[comm_size - 1] = last_block_count;
    }

    std::vector<ccl_sched*> part_scheds = { sched };
    ccl_coll_build_ring_allgatherv(nullptr,
                                   part_scheds,
                                   recv_buf,
                                   recv_counts[comm->rank()],
                                   recv_buf,
                                   recv_counts.data(),
                                   dtype,
                                   comm);

    sched->add_barrier();

    return status;
}

ccl::status ccl_coll_build_recursive_doubling_allreduce(ccl_sched* sched,
                                                        ccl_buffer send_buf,
                                                        ccl_buffer recv_buf,
                                                        size_t count,
                                                        const ccl_datatype& dtype,
                                                        ccl::reduction op,
                                                        ccl_comm* comm) {
    LOG_DEBUG("build recursive_doubling allreduce");

    ccl::status status = ccl::status::success;

    // count is the same for all ranks
    // if one rank skips mpi collectives, all ranks skip
    // this means we can safely skip all operations with zero count
    if (count == 0) {
        return status;
    }

    int pof2, rem, comm_size, rank;
    int newrank, mask, newdst, dst;

    comm_size = comm->size();
    rank = comm->rank();

    size_t dtype_size = dtype.size();

    ccl_buffer tmp_buf = sched->alloc_buffer({ count * dtype_size, send_buf });

    /* copy local data into recv_buf */
    if (send_buf != recv_buf) {
        entry_factory::create<copy_entry>(sched, send_buf, recv_buf, count, dtype);
        sched->add_barrier();
    }

    if (comm_size == 1)
        return status;

    /* get nearest power-of-two less than or equal to comm_size */
    pof2 = comm->pof2();
    rem = comm_size - pof2;

    /* In the non-power-of-two case, all even-numbered
     * processes of rank < 2*rem send their data to
     * (rank+1). These even-numbered processes no longer
     * participate in the algorithm until the very end. The
     * remaining processes form a nice power-of-two. */

    if (rank < 2 * rem) {
        if (rank % 2 == 0) { /* even */
            entry_factory::create<send_entry>(sched, recv_buf, count, dtype, rank + 1, comm);
            sched->add_barrier();

            /* temporarily set the rank to -1 so that this
             * process does not pariticipate in recursive
             * doubling */
            newrank = -1;
        }
        else { /* odd */
            entry_factory::create<recv_entry>(sched, tmp_buf, count, dtype, rank - 1, comm);
            sched->add_barrier();

            /* do the reduction on received data. since the
             * ordering is right, it doesn't matter whether
             * the operation is commutative or not. */

            entry_factory::create<reduce_local_entry>(
                sched, tmp_buf, count, recv_buf, nullptr, dtype, op);
            sched->add_barrier();

            /* change the rank */
            newrank = rank / 2;
        }
    }
    else /* rank >= 2*rem */
        newrank = rank - rem;

    if (newrank != -1) {
        mask = 0x1;
        while (mask < pof2) {
            newdst = newrank ^ mask;
            /* find real rank of dest */
            dst = (newdst < rem) ? newdst * 2 + 1 : newdst + rem;

            /* Send the most current data, which is in recv_buf. Recv
             * into tmp_buf */
            entry_factory::create<recv_entry>(sched, tmp_buf, count, dtype, dst, comm);
            /* sendrecv, no barrier here */
            entry_factory::create<send_entry>(sched, recv_buf, count, dtype, dst, comm);
            sched->add_barrier();

            /* tmp_buf contains data received in this step.
             * recv_buf contains data accumulated so far */
            entry_factory::create<reduce_local_entry>(
                sched, tmp_buf, count, recv_buf, nullptr, dtype, op);
            sched->add_barrier();

            mask <<= 1;
        }
    }

    /* In the non-power-of-two case, all odd-numbered
     * processes of rank < 2*rem send the result to
     * (rank-1), the ranks who didn't participate above. */
    if (rank < 2 * rem) {
        if (rank % 2) { /* odd */
            entry_factory::create<send_entry>(sched, recv_buf, count, dtype, rank - 1, comm);
        }
        else { /* even */
            entry_factory::create<recv_entry>(sched, recv_buf, count, dtype, rank + 1, comm);
        }
        sched->add_barrier();
    }

    return status;
}

static void ccl_allreduce_2d_add_allreduce_allgather(ccl_sched* sched,
                                                     ccl_buffer send_buf,
                                                     ccl_buffer recv_buf,
                                                     size_t count,
                                                     const ccl_datatype& dtype,
                                                     ccl::reduction op,
                                                     ccl_comm* comm,
                                                     ccl_comm* first_dim_comm,
                                                     ccl_comm* second_dim_comm,
                                                     size_t chunk_idx,
                                                     size_t chunk_count) {
    size_t dtype_size = dtype.size();
    size_t main_chunk_size = count / chunk_count;
    size_t last_chunk_size = main_chunk_size + count % chunk_count;
    size_t cnt = (chunk_idx == (chunk_count - 1)) ? last_chunk_size : main_chunk_size;
    ccl_buffer rbuf = recv_buf + chunk_idx * main_chunk_size * dtype_size;

    size_t main_block_count = cnt / first_dim_comm->size();
    size_t last_block_count = main_block_count + cnt % first_dim_comm->size();
    size_t ar_count = (first_dim_comm->rank() == (first_dim_comm->size() - 1)) ? last_block_count
                                                                               : main_block_count;

    if (ar_count) {
        // TODO: add second level selection to distinguish high and low level algorithms
        ccl_buffer ar_buf = rbuf + first_dim_comm->rank() * main_block_count * dtype_size;
        ccl_coll_build_nreduce_allreduce(
            sched, ar_buf, ar_buf, ar_count, dtype, op, second_dim_comm);
        sched->add_barrier();
    }

    std::vector<size_t> ag_recv_counts(first_dim_comm->size(), main_block_count);
    ag_recv_counts[first_dim_comm->size() - 1] = last_block_count;

    // TODO: skip direct algo since it may be started
    // with different order on different ranks
    sched->hint_algo.allgatherv = ccl_coll_allgatherv_ring;
    ccl_coll_build_allgatherv(
        sched, rbuf, ar_count, rbuf, ag_recv_counts.data(), dtype, first_dim_comm, false);
    sched->hint_algo.allgatherv = ccl_coll_allgatherv_undefined;
}

static void ccl_allreduce_2d_add_reduce_scatter_allreduce_allgather(ccl_sched* sched,
                                                                    ccl_buffer send_buf,
                                                                    ccl_buffer recv_buf,
                                                                    size_t count,
                                                                    const ccl_datatype& dtype,
                                                                    ccl::reduction op,
                                                                    ccl_comm* comm,
                                                                    ccl_comm* first_dim_comm,
                                                                    ccl_comm* second_dim_comm,
                                                                    size_t chunk_idx,
                                                                    size_t chunk_count) {
    size_t dtype_size = dtype.size();
    size_t main_chunk_size = count / chunk_count;
    size_t last_chunk_size = main_chunk_size + count % chunk_count;
    size_t cnt = (chunk_idx == (chunk_count - 1)) ? last_chunk_size : main_chunk_size;
    ccl_buffer sbuf = send_buf + chunk_idx * main_chunk_size * dtype_size;
    ccl_buffer rbuf = recv_buf + chunk_idx * main_chunk_size * dtype_size;

    ccl_coll_build_reduce_scatter(sched, sbuf, rbuf, cnt, dtype, op, first_dim_comm, false, true);
    sched->add_barrier();

    if (chunk_idx == (chunk_count - 1) || (chunk_count == 1)) {
        ccl_allreduce_2d_add_allreduce_allgather(sched,
                                                 send_buf,
                                                 recv_buf,
                                                 count,
                                                 dtype,
                                                 op,
                                                 comm,
                                                 first_dim_comm,
                                                 second_dim_comm,
                                                 chunk_idx,
                                                 chunk_count);
    }
    else {
        entry_factory::create<subsched_entry>(
            sched,
            chunk_idx,
            [send_buf,
             recv_buf,
             count,
             &dtype,
             op,
             comm,
             first_dim_comm,
             second_dim_comm,
             chunk_idx,
             chunk_count](ccl_sched* s) {
                ccl_allreduce_2d_add_allreduce_allgather(s,
                                                         send_buf,
                                                         recv_buf,
                                                         count,
                                                         dtype,
                                                         op,
                                                         comm,
                                                         first_dim_comm,
                                                         second_dim_comm,
                                                         chunk_idx,
                                                         chunk_count);
            },
            "AR_AG");

        entry_factory::create<subsched_entry>(
            sched,
            chunk_idx + 1,
            [send_buf,
             recv_buf,
             count,
             &dtype,
             op,
             comm,
             first_dim_comm,
             second_dim_comm,
             chunk_idx,
             chunk_count](ccl_sched* s) {
                ccl_allreduce_2d_add_reduce_scatter_allreduce_allgather(s,
                                                                        send_buf,
                                                                        recv_buf,
                                                                        count,
                                                                        dtype,
                                                                        op,
                                                                        comm,
                                                                        first_dim_comm,
                                                                        second_dim_comm,
                                                                        chunk_idx + 1,
                                                                        chunk_count);
            },
            "RS_AR_AG");
    }
}

ccl::status ccl_coll_build_2d_allreduce(ccl_sched* sched,
                                        ccl_buffer send_buf,
                                        ccl_buffer recv_buf,
                                        size_t count,
                                        const ccl_datatype& dtype,
                                        ccl::reduction op,
                                        ccl_comm* comm) {
    ccl::status status = ccl::status::success;

    // count is the same for all ranks
    // if one rank skips mpi collectives, all ranks skip
    // this means we can safely skip all operations with zero count
    if (count == 0) {
        return status;
    }

    size_t chunk_count = ccl::global_data::env().allreduce_2d_chunk_count;

    bool switch_dims = ccl::global_data::env().allreduce_2d_switch_dims;
    ccl_comm* first_dim_comm =
        (switch_dims) ? comm->get_r2r_comm().get() : comm->get_node_comm().get();
    ccl_comm* second_dim_comm =
        (switch_dims) ? comm->get_node_comm().get() : comm->get_r2r_comm().get();

    LOG_DEBUG("build 2d allreduce: chunk_count: ",
              chunk_count,
              ", switch_dims: ",
              switch_dims,
              ", comm: ",
              comm->to_string(),
              ", 1st dim comm: ",
              first_dim_comm->to_string(),
              ", 2nd dim comm: ",
              second_dim_comm->to_string());

    ccl_allreduce_2d_add_reduce_scatter_allreduce_allgather(sched,
                                                            send_buf,
                                                            recv_buf,
                                                            count,
                                                            dtype,
                                                            op,
                                                            comm,
                                                            first_dim_comm,
                                                            second_dim_comm,
                                                            0 /* chunk_idx */,
                                                            chunk_count);

    return status;
}

#if defined(CCL_ENABLE_SYCL) && defined(CCL_ENABLE_ZE)

ccl::status ccl_coll_build_topo_allreduce_fill(ccl_sched* sched,
                                               ccl_buffer send_buf,
                                               ccl_buffer recv_buf,
                                               size_t count,
                                               const ccl_datatype& dtype,
                                               ccl::reduction op,
                                               ccl_comm* comm) {
    LOG_DEBUG("build topo allreduce");

    // count is the same for all ranks
    // if one rank skips mpi collectives, all ranks skip
    // this means we can safely skip all operations with zero count
    if (count == 0) {
        return ccl::status::success;
    }

    ccl_comm* pair_comm = comm->get_pair_comm().get();
    ccl_comm* even_comm = comm->get_even_comm().get();
    ccl_comm* node_comm = comm->get_node_comm().get();
    ccl_comm* r2r_comm = comm->get_r2r_comm().get();

    int comm_size = comm->size();
    int even_comm_size = even_comm->size();
    int node_comm_size = node_comm->size();

    const ccl::topo_manager& topo_manager = comm->get_topo_manager();
    bool is_single_node = topo_manager.is_single_node;
    bool is_single_card = topo_manager.is_single_card;
    bool is_multi_card = (even_comm_size > 1);

    // use mdfi + xelink pipeline kernel for reduce_scatter phase
    bool use_reduce_scatter_pipeline =
        ccl::global_data::env().reduce_scatter_monolithic_pipeline_kernel &&
        even_comm->size() > 1 && pair_comm->size() > 1 && count >= (size_t)comm_size &&
        is_multi_card && dtype != ccl::datatype::int8;

    // allgatherv pipeline uses xelink read and mdfi write
    const bool use_allgatherv_pipeline =
        ccl::global_data::env().allgatherv_monolithic_pipeline_kernel && count >= (size_t)comm_size;

    size_t base_count = count;
    size_t pair_comm_offset = 0;
    size_t pair_comm_offset_bytes = 0;

    bool barrier_1s_handle_exchange = false;
    if (ccl::global_data::env().enable_ze_bidir_algo && (base_count / pair_comm->size()) > 0) {
        base_count = count / pair_comm->size();
        pair_comm_offset = base_count * pair_comm->rank();
        pair_comm_offset_bytes = pair_comm_offset * dtype.size();

        if (pair_comm->rank() == pair_comm->size() - 1)
            base_count += count % pair_comm->size();
    }
    else if (pair_comm->rank() != ccl::global_data::env().kernel_1s_lead) {
        barrier_1s_handle_exchange = true;
    }

    size_t main_block_count = base_count / even_comm_size;
    size_t block_count = main_block_count;
    if (even_comm->rank() == even_comm_size - 1) {
        block_count += base_count % even_comm_size;
    }

    // tmp buff for write mode reduce scatter
    // TODO: fix - write based reduce_scatter fails intermittently for int8. However, such
    // failure has been seem for the read based path as well.
    ccl_buffer tmp_write_buf;
    bool is_rs_write =
        !ccl::global_data::env().reduce_scatter_topo_read && !use_reduce_scatter_pipeline &&
        !ccl::global_data::env().reduce_scatter_monolithic_kernel && dtype != ccl::datatype::int8;

    // optimized allreduce_entry is for single node and uses allgatherv write protocol.
    // this entry combines reduce_scatter and allgather across even_comm
    // If write based copy is being used reduce_scatter, we should skip a2a_allreduce_entry
    // because that path uses read based copy instead
    bool use_a2a_allreduce_entry = is_single_node &&
                                   !ccl::global_data::env().allgatherv_topo_read && !is_rs_write &&
                                   !use_reduce_scatter_pipeline && !use_allgatherv_pipeline;

    if (is_rs_write) {
        // local rank does not store its data in the tmp buffer, so skip 1 block
        // under uneven division, the last block can have extra data, reflected in block_count
        size_t tmp_buf_bytes = dtype.size() * ((even_comm->size() - 1) * block_count);
        // workaround with dummy 1 byte allocation to still enable handle exchange when tmp bytes is 0
        if (tmp_buf_bytes == 0)
            tmp_buf_bytes = 1;

        ccl::alloc_param alloc_param(
            tmp_buf_bytes, ccl::buffer_type::ze, ccl::buffer_place::device);
        tmp_write_buf = sched->alloc_buffer(alloc_param);
    }

    std::vector<ze_handle_exchange_entry::mem_desc_t> in_buffers{
        { send_buf.get_ptr(), ccl::ze::ipc_mem_type::memory }, // 0
        { recv_buf.get_ptr(), ccl::ze::ipc_mem_type::memory }, // 1
    };
    size_t tmp_write_buf_idx = -1;
    if (is_rs_write) {
        in_buffers.push_back({ tmp_write_buf.get_ptr(), ccl::ze::ipc_mem_type::memory }); // 2
        tmp_write_buf_idx = 2;
    }

    size_t recv_buf_idx = 1;

    int skip_rank = ccl_comm::invalid_rank;
    if (ccl::global_data::env().enable_kernel_1s_ipc_wa && is_single_card &&
        !use_reduce_scatter_pipeline) {
        skip_rank = ccl::global_data::env().kernel_1s_lead;
    }

    std::vector<ccl_buffer> tmp_bufs;
    size_t tmp_buf_idx_start = -1;
    if (use_reduce_scatter_pipeline) {
        ze_utils::alloc_tmp_bufs(
            sched, comm, tmp_bufs, in_buffers, tmp_buf_idx_start, count, dtype);
    }

    size_t ipc_event_count{};
    // note: increase max_ipc_event_count for each new IPC operation added in the source code,
    // which often also involves ipc_event_count++
    size_t max_ipc_event_count{ 8 };
    ze_event_pool_handle_t ipc_event_pool{};
    if (ccl::global_data::env().enable_ze_barrier) {
        ipc_event_pool = sched->get_memory().ipc_event_pool_manager.create(max_ipc_event_count);
        in_buffers.push_back({ static_cast<void*>(ipc_event_pool), ccl::ze::ipc_mem_type::pool });
    }

    // This algorithm uses ze_events to link different ze commands (from within an
    // entry or between different entries). It also uses these ze_events to link with
    // host-side tasks (such as barriers, scaleout, etc.).
    //
    // We have implemented this algorithm (at least from the perspective of this file,
    // without digging deeper into the contents of 'ze_entries') by linking tasks in a
    // very linear way. For example, in practice, we mostly create task graphs (DAG)
    // that look very linear (taskA -> taskB -> taskC), without forks.
    //
    // We are using wait_events to keep track of which events the next task needs to
    // wait on. In most cases, we use clear_and_push_back as an easy way to say "all
    // events in wait_events will have been signaled if the new event is signaled,
    // therefore we only need to wait on the latter."
    std::vector<ze_event_handle_t> wait_events{};
    ze_event_handle_t out_event{};

    sched->try_enable_ze_single_list();

    ccl::add_handle_exchange(sched,
                             node_comm,
                             wait_events,
                             out_event,
                             in_buffers,
                             skip_rank,
                             ipc_event_pool,
                             ipc_event_count++);
    clear_and_push_back(wait_events, out_event);

    CCL_THROW_IF_NOT(comm_size % 2 == 0, "unexpected comm_size ", comm_size);
    CCL_THROW_IF_NOT(node_comm_size % 2 == 0, "unexpected node_comm_size ", node_comm_size);

    if (barrier_1s_handle_exchange) {
        ccl::add_comm_barrier(
            sched, pair_comm, wait_events, out_event, ipc_event_pool, ipc_event_count++);
        CCL_THROW_IF_NOT(ipc_event_count <= max_ipc_event_count,
                         "unexpected ipc_event_count ",
                         ipc_event_count,
                         ", expected max ",
                         max_ipc_event_count);
        return ccl::status::success;
    }

    const size_t even_comm_offset = main_block_count * even_comm->rank();
    const size_t even_comm_offset_bytes = even_comm_offset * dtype.size();
    ccl_buffer even_comm_recv_buf{};
    ccl_buffer pair_comm_recv_buf{};

    bool is_read_allgatherv = ccl::global_data::env().allgatherv_topo_read;
    if (use_reduce_scatter_pipeline) {
        LOG_DEBUG("topo/scale_up/intra: use ze_a2a_pipeline_read_write_entry");

        ze_a2a_pipeline_read_write_entry::attr attrs{ .use_continous_data = true,
                                                      .use_remote_target = true };
        auto reduce_entry =
            entry_factory::create<ze_a2a_pipeline_read_write_entry>(sched,
                                                                    comm,
                                                                    send_buf,
                                                                    tmp_bufs,
                                                                    tmp_buf_idx_start,
                                                                    count,
                                                                    dtype,
                                                                    op,
                                                                    wait_events,
                                                                    attrs);
        wait_events.clear();
        wait_events.push_back(reduce_entry->entry_event);

        ccl::add_comm_barrier(
            sched, node_comm, wait_events, out_event, ipc_event_pool, ipc_event_count++);
        clear_and_push_back(wait_events, out_event);

        LOG_DEBUG("topo/scale_up/intra: use ze_a2a_pipeline_reduce_entry");
        even_comm_recv_buf = recv_buf + pair_comm_offset_bytes + even_comm_offset_bytes;
        pair_comm_recv_buf = recv_buf + pair_comm_offset_bytes;
        auto scatter_entry = entry_factory::create<ze_a2a_pipeline_reduce_entry>(
            sched, comm, even_comm_recv_buf, tmp_bufs, count, dtype, op, wait_events);
        wait_events.clear();
        wait_events.push_back(scatter_entry->entry_event);

        ccl::add_comm_barrier(
            sched, node_comm, wait_events, out_event, ipc_event_pool, ipc_event_count++);
        clear_and_push_back(wait_events, out_event);
    }
    else {
        ccl_buffer pair_comm_send_buf = send_buf + pair_comm_offset_bytes;
        even_comm_recv_buf = recv_buf + pair_comm_offset_bytes + even_comm_offset_bytes;
        pair_comm_recv_buf = recv_buf + pair_comm_offset_bytes;

        LOG_DEBUG("rank: ",
                  pair_comm->rank(),
                  ", count: ",
                  base_count,
                  ", pair_comm_offset: ",
                  pair_comm_offset);
        if (is_single_card) {
            // workaround for hardware issue that MDFI write from kernel
            // with 8 bit and 16 bit data types are slow. Instead perform
            // MDFI read + reduce in kernel and MDFI write using memcpy
            if (dtype == ccl::datatype::int8 || dtype == ccl::datatype::float16 ||
                dtype == ccl::datatype::bfloat16) {
                auto entry = entry_factory::create<ze_onesided_reduce_entry>(sched,
                                                                             pair_comm_send_buf,
                                                                             pair_comm_recv_buf,
                                                                             base_count,
                                                                             dtype,
                                                                             op,
                                                                             pair_comm->rank(),
                                                                             pair_comm,
                                                                             wait_events,
                                                                             pair_comm_offset);
                wait_events.clear();
                wait_events.push_back(entry->entry_event);
                int peer_rank = (pair_comm->rank() + 1) % pair_comm->size();
                auto copy_entry =
                    entry_factory::create<ze_copy_entry>(sched,
                                                         recv_buf,
                                                         ccl_buffer(),
                                                         base_count,
                                                         dtype,
                                                         copy_attr(peer_rank,
                                                                   recv_buf_idx,
                                                                   copy_direction::t2t,
                                                                   false, /*pt2pt_op*/
                                                                   pair_comm,
                                                                   pair_comm_offset,
                                                                   pair_comm_offset),
                                                         wait_events);
                clear_and_push_back(wait_events, copy_entry->entry_event);
            }
            else {
                auto entry = entry_factory::create<ze_onesided_allreduce_entry>(sched,
                                                                                pair_comm_send_buf,
                                                                                pair_comm_recv_buf,
                                                                                base_count,
                                                                                dtype,
                                                                                op,
                                                                                pair_comm,
                                                                                wait_events,
                                                                                pair_comm_offset);
                clear_and_push_back(wait_events, entry->entry_event);
            }
        }
        else {
            LOG_DEBUG("topo/scale_up/intra: use ze_onesided_reduce");
            auto entry = entry_factory::create<ze_onesided_reduce_entry>(sched,
                                                                         pair_comm_send_buf,
                                                                         pair_comm_recv_buf,
                                                                         base_count,
                                                                         dtype,
                                                                         op,
                                                                         pair_comm->rank(),
                                                                         pair_comm,
                                                                         wait_events,
                                                                         pair_comm_offset);
            clear_and_push_back(wait_events, entry->entry_event);
        }
        sched->add_barrier();

        if (is_multi_card) {
            ccl::add_comm_barrier(
                sched, even_comm, wait_events, out_event, ipc_event_pool, ipc_event_count++);
            clear_and_push_back(wait_events, out_event);

            // cannot use ze_a2a_allreduce_entry with allgatherv_read
            // since comm_barrier is not available inside the entry
            if (use_a2a_allreduce_entry) {
                LOG_DEBUG("topo/scale_up/intra: use ze_a2a_allreduce_entry");
                auto entry = entry_factory::create<ze_a2a_allreduce_entry>(sched,
                                                                           pair_comm_recv_buf,
                                                                           pair_comm_recv_buf,
                                                                           base_count,
                                                                           dtype,
                                                                           op,
                                                                           even_comm,
                                                                           wait_events,
                                                                           recv_buf_idx,
                                                                           recv_buf_idx,
                                                                           pair_comm_offset);
                clear_and_push_back(wait_events, entry->entry_event);
                ccl::add_comm_barrier(
                    sched, even_comm, wait_events, out_event, ipc_event_pool, ipc_event_count++);
                clear_and_push_back(wait_events, out_event);
            }
            else {
                std::vector<size_t> block_counts(even_comm->size(), main_block_count);
                block_counts.back() += base_count % even_comm_size;
                if (!is_rs_write) {
                    LOG_DEBUG("topo/scale_up/inter: use ze_a2a_reduce_scatter_entry");
                    auto entry =
                        entry_factory::create<ze_a2a_reduce_scatter_entry>(sched,
                                                                           pair_comm_recv_buf,
                                                                           even_comm_recv_buf,
                                                                           block_counts.data(),
                                                                           dtype,
                                                                           op,
                                                                           even_comm,
                                                                           wait_events,
                                                                           recv_buf_idx,
                                                                           pair_comm_offset);
                    clear_and_push_back(wait_events, entry->entry_event);
                    ccl::add_comm_barrier(sched,
                                          even_comm,
                                          wait_events,
                                          out_event,
                                          ipc_event_pool,
                                          ipc_event_count++);
                    clear_and_push_back(wait_events, out_event);
                }
                else {
                    // copy using write
                    LOG_DEBUG("topo/scale_up/inter: use ze_a2a_reduce_scatter_write_copy_entry ");

                    reduce_scatter_args rs_args = { even_comm, block_counts, dtype, op };
                    reduce_scatter_bufs rs_bufs = { recv_buf,
                                                    even_comm_recv_buf,
                                                    tmp_write_buf,
                                                    tmp_write_buf_idx,
                                                    pair_comm_offset };

                    auto copy_entry = entry_factory::create<ze_a2a_reduce_scatter_write_copy_entry>(
                        sched, rs_args, rs_bufs, wait_events);

                    clear_and_push_back(wait_events, copy_entry->entry_event);
                    ccl::add_comm_barrier(sched,
                                          even_comm,
                                          wait_events,
                                          out_event,
                                          ipc_event_pool,
                                          ipc_event_count++);
                    clear_and_push_back(wait_events, out_event);

                    // local reduction
                    LOG_DEBUG("topo/scale_up/inter: use ze_a2a_reduce_scatter_write_kernel_entry");
                    auto kernel_entry =
                        entry_factory::create<ze_a2a_reduce_scatter_write_kernel_entry>(
                            sched, rs_args, rs_bufs, wait_events);

                    clear_and_push_back(wait_events, kernel_entry->entry_event);
                    ccl::add_comm_barrier(sched,
                                          even_comm,
                                          wait_events,
                                          out_event,
                                          ipc_event_pool,
                                          ipc_event_count++);
                    clear_and_push_back(wait_events, out_event);
                }
            }
        }
    }

    ccl_coll_param coll_param{ false };
    coll_param.ctype = ccl_coll_allreduce;
    coll_param.send_buf = even_comm_recv_buf;
    coll_param.recv_buf = even_comm_recv_buf;
    coll_param.count = block_count;
    coll_param.dtype = dtype;
    coll_param.reduction = op;
    coll_param.comm = r2r_comm;

    out_event = nullptr;
    ccl::add_scaleout(sched, coll_param, is_single_node, wait_events, out_event);
    if (out_event) {
        clear_and_push_back(wait_events, out_event);
    }

    if (is_multi_card && !use_a2a_allreduce_entry) {
        LOG_DEBUG("topo/scale_up/inter: use ze_a2a_allgatherv");
        // for multinode with xelink read, use a comm_barrier to make sure all
        // r2r scaleout within even_comm has finished so that remote reads are valid
        if (!is_single_node && (is_read_allgatherv || use_allgatherv_pipeline)) {
            ccl::add_comm_barrier(
                sched, even_comm, wait_events, out_event, ipc_event_pool, ipc_event_count++);
            clear_and_push_back(wait_events, out_event);
        }

        std::vector<size_t> recv_counts(even_comm_size, main_block_count);
        recv_counts.back() += base_count % even_comm_size;
        // allgatherv entry takes an array of recv bufs with one buffer for each rank
        // rather than a single large recv buf that needs to be divided
        std::vector<ccl_buffer> recv_bufs;
        for (int i = 0; i < even_comm_size; i++) {
            recv_bufs.push_back(pair_comm_recv_buf + i * main_block_count * dtype.size());
        }
        // allreduce topo has ipc handle exchange of start of the buffer whereas
        // allgatherv topo use separate handles for the partitions of each rank
        constexpr bool is_separate_block_handles = false;
        auto entry = entry_factory::create<ze_a2a_allgatherv_entry>(sched,
                                                                    recv_bufs[even_comm->rank()],
                                                                    block_count,
                                                                    recv_bufs,
                                                                    recv_counts,
                                                                    dtype,
                                                                    even_comm,
                                                                    wait_events,
                                                                    recv_buf_idx,
                                                                    pair_comm_offset,
                                                                    use_allgatherv_pipeline,
                                                                    pair_comm,
                                                                    is_separate_block_handles);
        clear_and_push_back(wait_events, entry->entry_event);

        // copy local data to pair_comm neighbor
        if (pair_comm->size() > 1 && use_allgatherv_pipeline) {
            const size_t peer_pair_rank = (pair_comm->rank() + 1) % pair_comm->size();
            copy_attr attr{};
            attr.peer_rank = peer_pair_rank;
            attr.peer_buf_idx = recv_buf_idx;
            attr.direction = copy_direction::t2t;
            attr.pt2pt_op = false;
            attr.map_comm = pair_comm;
            attr.in_buf_offset = pair_comm_offset + even_comm_offset;
            attr.out_buf_offset = pair_comm_offset + even_comm_offset;
            // copy own buffer to the pair_comm neighbor recv buffer
            auto entry_pair = entry_factory::create<ze_copy_entry>(
                sched, recv_buf, ccl_buffer(), block_count, dtype, attr, wait_events);
            clear_and_push_back(wait_events, entry_pair->entry_event);
            wait_events.push_back(entry_pair->entry_event);
        }

        ccl::add_comm_barrier(
            sched, even_comm, wait_events, out_event, ipc_event_pool, ipc_event_count++);
        clear_and_push_back(wait_events, out_event);
    }

    // copy is not needed since allgatherv pipeline already
    // performed the copy to pair_comm neighbor tile
    if (!is_single_card && pair_comm->size() > 1 && !use_allgatherv_pipeline) {
        LOG_DEBUG("topo/scale_up/intra: use ze_onesided_bcast");
        int peer_rank = (pair_comm->rank() + 1) % pair_comm->size();
        auto entry = entry_factory::create<ze_copy_entry>(sched,
                                                          recv_buf,
                                                          ccl_buffer(),
                                                          base_count,
                                                          dtype,
                                                          copy_attr(peer_rank,
                                                                    recv_buf_idx,
                                                                    copy_direction::t2t,
                                                                    false, /*pt2pt_op*/
                                                                    pair_comm,
                                                                    pair_comm_offset,
                                                                    pair_comm_offset),
                                                          wait_events);
        clear_and_push_back(wait_events, entry->entry_event);
        sched->add_barrier();
    }

    ccl::add_comm_barrier(
        sched, pair_comm, wait_events, out_event, ipc_event_pool, ipc_event_count++);

    CCL_THROW_IF_NOT(ipc_event_count <= max_ipc_event_count,
                     "unexpected ipc_event_count ",
                     ipc_event_count,
                     ", expected max ",
                     max_ipc_event_count);

    return ccl::status::success;
}

ccl::status ccl_coll_build_topo_allreduce(ccl_sched* sched,
                                          ccl_buffer send_buf,
                                          ccl_buffer recv_buf,
                                          size_t count,
                                          const ccl_datatype& dtype,
                                          ccl::reduction op,
                                          ccl_comm* comm) {
    // Note about cache lines and pipelining: The same cache line must contain
    // a single chunk only.
    //
    // If the same cache line contains two chunks (or more), and we parallelize
    // the instructions required for both chunks, a conflict (race condition)
    // may appear between the copy-out for the scaleout portion and the
    // reduce_scatter phase.
    //
    // The easiest way to avoid that race condition is to require that each
    // cache line contains a single entry. If that is not the case, we must not
    // parallelize the instructions for different chunks.

    size_t chunk_count = ccl::global_data::env().allreduce_pipe_chunk_count;
    bool is_pipe = chunk_count > 0 && ccl::global_data::env().enable_ze_single_list;

    // TODO: why does oneCCL have CACHELINE_SIZE *and* CCL_KERNEL_MEM_ALIGN?
    size_t memalign = ccl::global_data::env().kernel_mem_align;
    size_t buf_size_bytes = count * dtype.size();

    // First, determine if we need to fallback to non-pipelined algorightm.
    // Such a fallback may happen in cases such as (1) the user requests it,
    // (2) message fits into a cache line, or (3) the cache line size is not
    // divisible by the data type size.

    size_t number_of_cache_lines_per_chunk =
        !is_pipe ? 1 : std::max(memalign, buf_size_bytes / chunk_count) / memalign;
    size_t main_chunk_size_bytes = memalign * number_of_cache_lines_per_chunk;

    bool is_dtype_non_divisible = main_chunk_size_bytes % dtype.size();
    bool is_msg_smaller_than_cache_line = buf_size_bytes <= main_chunk_size_bytes;

    bool is_multiworker =
        ccl::global_data::env().ze_multi_workers && ccl::global_data::env().worker_count > 1;

    if (!is_pipe || is_dtype_non_divisible || is_msg_smaller_than_cache_line || is_multiworker) {
        // Fall back to topo algorithm without pipelining

        if (!is_pipe) {
            LOG_DEBUG("Pipelining code disabled");
        }
        else if (is_dtype_non_divisible) {
            LOG_INFO("Running without pipelining because datatype size (",
                     dtype.size(),
                     ") is not divisible by cache line size (",
                     memalign,
                     ")");
        }
        else if (is_msg_smaller_than_cache_line) {
            LOG_INFO("Running without pipelining because message size (",
                     buf_size_bytes,
                     ") is smaller than a cache line (",
                     memalign,
                     ") and main_chunk_size_bytes (",
                     main_chunk_size_bytes,
                     ")");
        }
        else if (is_multiworker) {
            LOG_INFO(
                "Running without pipelining because ze_multi_workers was requested with more than one worker");
        }
        else {
            CCL_THROW("Unexpected fallback to non-pipe code");
        }

        ccl_coll_build_topo_allreduce_fill(sched, send_buf, recv_buf, count, dtype, op, comm);

        entry_factory::create<ze_execute_cmdlists_on_init_entry>(sched);

        return ccl::status::success;
    }

    LOG_DEBUG("build pipe allreduce");

    size_t main_chunk_count = main_chunk_size_bytes / dtype.size();

    // Need to re-calculate chunk_count after main_chunk_size_bytes calculation
    // with cache alignment in mind.
    chunk_count = count / main_chunk_count;
    size_t last_chunk_count = main_chunk_count + (count % main_chunk_count);

    sched->try_enable_ze_single_list();
    auto sync_obj = std::make_shared<sync_object>(chunk_count);
    bool is_parallelizable_chunks = true;

    for (size_t chunk_idx = 0; chunk_idx < chunk_count; ++chunk_idx) {
        size_t chunk_offset = chunk_idx * main_chunk_count * dtype.size();
        ccl_buffer sbuf = send_buf + chunk_offset;
        ccl_buffer rbuf = recv_buf + chunk_offset;
        size_t this_chunk_count =
            (chunk_idx == (chunk_count - 1)) ? last_chunk_count : main_chunk_count;

        if (this_chunk_count || (count == 0 && chunk_idx == 0)) {
            entry_factory::create<subsched_entry>(
                sched,
                chunk_idx,
                [sched, sbuf, rbuf, this_chunk_count, dtype, op, comm, sync_obj](ccl_sched* s) {
                    s->inherit_ze_managers_from(sched);
                    s->set_init_ze_hook_sync_obj(sync_obj);
                    s->set_ze_commands_bypass_flag(false);

                    ccl_coll_build_topo_allreduce_fill(
                        s, sbuf, rbuf, this_chunk_count, dtype, op, comm);
                },
                ("ALLREDUCE_PIPE" + std::to_string(chunk_idx)).c_str());
        }
        if (chunk_idx > 0) {
            auto ptr = reinterpret_cast<uintptr_t>(rbuf.get_ptr());
            auto prev_chunk_last_cache_line = (ptr - 1) / memalign;
            auto this_chunk_first_cache_line = ptr / memalign;

            if (prev_chunk_last_cache_line == this_chunk_first_cache_line) {
                // WARNING: previous chunk has part of this chunk's first cache
                // line. Cannot use pipelining. However, since this is a
                // "local" decision (i.e., other ranks may decide differently),
                // we still need to apply chunking. However, we will run one
                // chunk at a time, without parallelizing them.
                // Another way to have implemented this would be to link the
                // last task of the prev chunk with the first of this chunk
                // with an event.
                is_parallelizable_chunks = false;
            }
        }
    }

    static bool is_chunk_memalign_warning_printed{};
    if (!is_parallelizable_chunks && !is_chunk_memalign_warning_printed) {
        is_chunk_memalign_warning_printed = true;
        LOG_WARN(
            "[allreduce pipelining]: For best performance, (i) chunk size should be a multiple of a cache line (",
            memalign,
            " bytes), and (ii) buffers in all ranks should be aligned to ",
            memalign);
    }

    if (!is_parallelizable_chunks) {
        ccl::global_data::get()
            .metrics_profiler->allreduce_pipe_nonparallel_calls_per_count[count]++;
    }
    else {
        ccl::global_data::get().metrics_profiler->allreduce_pipe_parallel_calls_per_count[count]++;
    }

    entry_factory::create<ze_execute_cmdlists_on_start_entry>(
        sched,
        sync_obj,
        is_parallelizable_chunks ? submit_ze_commands_in_subsched_entries : nullptr);

    return ccl::status::success;
}

#endif // CCL_ENABLE_SYCL && CCL_ENABLE_ZE
