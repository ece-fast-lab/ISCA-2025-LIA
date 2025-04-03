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
#pragma once

#include "cpu_coll.hpp"
#include "reduce_scatter_strategy.hpp"

struct reduce_scatter_strategy_impl {
    static constexpr const char* class_name() {
        return "reduce_scatter";
    }

    size_t get_send_multiplier() {
        return 1;
    }

    size_t get_recv_multiplier() {
        return 1;
    }

    static const ccl::reduce_scatter_attr& get_op_attr(const bench_exec_attr& bench_attr) {
        return bench_attr.get_attr<ccl::reduce_scatter_attr>();
    }

    template <class Dtype, class... Args>
    void start_internal(ccl::communicator& comm,
                        size_t send_count,
                        const Dtype send_buf,
                        Dtype recv_buf,
                        const bench_exec_attr& bench_attr,
                        req_list_t& reqs,
                        Args&&... args) {
        size_t recv_count = send_count / comm.size();

        if (recv_count == 0) {
            reqs.push_back(ccl::event());
            return;
        }

        reqs.push_back(ccl::reduce_scatter(send_buf,
                                           recv_buf,
                                           recv_count,
                                           get_ccl_dtype<Dtype>(),
                                           bench_attr.reduction,
                                           comm,
                                           std::forward<Args>(args)...));
    }
};
