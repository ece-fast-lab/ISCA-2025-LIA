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

// enums
typedef enum { VALIDATE_OFF, VALIDATE_LAST_ITER, VALIDATE_ALL_ITERS } validate_values_t;

std::map<validate_values_t, std::string> validate_values_names = {
    std::make_pair(VALIDATE_OFF, "off"),
    std::make_pair(VALIDATE_LAST_ITER, "last"),
    std::make_pair(VALIDATE_ALL_ITERS, "all")
};

// defines
#define COL_WIDTH     (18)
#define COL_PRECISION (2)

#define DEFAULT_ITERS        (16)
#define DEFAULT_WARMUP_ITERS (16)

#define DEFAULT_CACHE_OPS   (1)
#define DEFAULT_QUEUE       (0)
#define DEFAULT_WAIT        (1)
#define DEFAULT_WINDOW_SIZE (64)

#define DEFAULT_MIN_ELEM_COUNT (1)
#define DEFAULT_MAX_ELEM_COUNT (33554432) // till 128 MBytes
#define DEFAULT_VALIDATE       (VALIDATE_LAST_ITER)

#define INVALID_VALUE  (-1)
#define INVALID_RETURN (-1)

int set_validate_values(const std::string& option_value, validate_values_t& validate) {
    std::string option_name = "validate";

    std::set<std::string> supported_option_values{ validate_values_names[VALIDATE_OFF],
                                                   validate_values_names[VALIDATE_LAST_ITER],
                                                   validate_values_names[VALIDATE_ALL_ITERS] };

    if (check_supported_options(option_name, option_value, supported_option_values)) {
        return INVALID_RETURN;
    }

    if (option_value == validate_values_names[VALIDATE_OFF]) {
        validate = VALIDATE_OFF;
    }
    else if (option_value == validate_values_names[VALIDATE_LAST_ITER]) {
        validate = VALIDATE_LAST_ITER;
    }
    else if (option_value == validate_values_names[VALIDATE_ALL_ITERS]) {
        validate = VALIDATE_ALL_ITERS;
    }

    return 0;
}
