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
#include "oneapi/ccl/types.hpp"
#include "oneapi/ccl/types_policy.hpp"
#include "oneapi/ccl/coll_attr_ids.hpp"
#include "oneapi/ccl/coll_attr_ids_traits.hpp"
#include "coll/attr/ccl_common_op_attrs.hpp"

#define INVALID_GROUP_ID (-1)

namespace ccl {

class ccl_pt2pt_attr_impl_t : public ccl_operation_attr_impl_t {
public:
    using base_t = ccl_operation_attr_impl_t;

    ccl_pt2pt_attr_impl_t(
        const typename detail::ccl_api_type_attr_traits<operation_attr_id,
                                                        operation_attr_id::version>::type& version);

    using group_id_traits_t =
        detail::ccl_api_type_attr_traits<pt2pt_attr_id, pt2pt_attr_id::group_id>;
    const typename group_id_traits_t::return_type& get_attribute_value(
        const group_id_traits_t& id) const;

    typename group_id_traits_t::return_type set_attribute_value(
        typename group_id_traits_t::type val,
        const group_id_traits_t& t);

    int group_id = INVALID_GROUP_ID;
};
} // namespace ccl
