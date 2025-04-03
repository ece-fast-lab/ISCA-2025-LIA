# coding=utf-8
# Copyright 2022 The Fairseq Authors and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" PyTorch OPT model."""
from typing import List, Optional, Tuple, Union

import torch
import copy
import math
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss 
from transformers.models.opt.numa_alloc import numa_alloc_tensor, numa_free_tensor

from ...activations import ACT2FN
from ...modeling_attn_mask_utils import _prepare_4d_causal_attention_mask
from ...modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutputWithPast,
)
from ...modeling_utils import PreTrainedModel
from ...utils import (
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_flash_attn_2_available,
    is_flash_attn_greater_or_equal_2_10,
    logging,
    replace_return_docstrings,
)
from .configuration_opt import OPTConfig


if is_flash_attn_2_available():
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input  # noqa


logger = logging.get_logger(__name__)

_CHECKPOINT_FOR_DOC = "facebook/opt-350m"
_CONFIG_FOR_DOC = "OPTConfig"

# Base model docstring
_EXPECTED_OUTPUT_SHAPE = [1, 8, 1024]

# SequenceClassification docstring
_CHECKPOINT_FOR_SEQUENCE_CLASSIFICATION = "ArthurZ/opt-350m-dummy-sc"
_SEQ_CLASS_EXPECTED_LOSS = 1.71
_SEQ_CLASS_EXPECTED_OUTPUT = "'LABEL_0'"

OPT_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "facebook/opt-125m",
    "facebook/opt-350m",
    "facebook/opt-1.3b",
    "facebook/opt-2.7b",
    "facebook/opt-6.7b",
    "facebook/opt-13b",
    "facebook/opt-30b",
    # See all OPT models at https://huggingface.co/models?filter=opt
]


# Copied from transformers.models.llama.modeling_llama._get_unpad_data
def _get_unpad_data(attention_mask):
    seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0))
    return (
        indices,
        cu_seqlens,
        max_seqlen_in_batch,
    )

def create_buffer(layers, device):
    if device == 'cuda':
        buffers = [
            torch.empty(layers.self_attn_layer_norm.weight.size(), dtype=torch.bfloat16, device=device),
            torch.empty(layers.self_attn_layer_norm.bias.size(), dtype=torch.bfloat16, device=device),
            torch.empty(layers.self_attn.q_proj.weight.size(), dtype=torch.bfloat16, device=device),
            torch.empty(layers.self_attn.q_proj.bias.size(), dtype=torch.bfloat16, device=device),
            torch.empty(layers.self_attn.k_proj.weight.size(), dtype=torch.bfloat16, device=device),
            torch.empty(layers.self_attn.k_proj.bias.size(), dtype=torch.bfloat16, device=device),
            torch.empty(layers.self_attn.v_proj.weight.size(), dtype=torch.bfloat16, device=device),
            torch.empty(layers.self_attn.v_proj.bias.size(), dtype=torch.bfloat16, device=device),
        ]
        
        if hasattr(layers.self_attn, 'out_proj'):
            buffers.append(torch.empty(layers.self_attn.out_proj.weight.size(), dtype=torch.bfloat16, device=device))
            buffers.append(torch.empty(layers.self_attn.out_proj.original_bias.size(), dtype=torch.bfloat16, device=device))
        elif hasattr(layers, 'mha_linear_add'):
            buffers.append(torch.empty(layers.mha_linear_add.linear.weight.size(), dtype=torch.bfloat16, device=device))
            buffers.append(torch.empty(layers.mha_linear_add.linear.bias.size(), dtype=torch.bfloat16, device=device))
        else:
            raise AttributeError("Neither 'out_proj' nor 'mha_linear_add' found in layers.self_attn")

        buffers.extend([
            torch.empty(layers.final_layer_norm.weight.size(), dtype=torch.bfloat16, device=device),
            torch.empty(layers.final_layer_norm.bias.size(), dtype=torch.bfloat16, device=device),
            torch.empty(layers.linear_relu.linear.weight.size(), dtype=torch.bfloat16, device=device),
            torch.empty(layers.linear_relu.linear.bias.size(), dtype=torch.bfloat16, device=device),
        ])

        if hasattr(layers, 'fc2'):
            buffers.append(torch.empty(layers.fc2.weight.size(), dtype=torch.bfloat16, device=device))
            buffers.append(torch.empty(layers.fc2.original_bias.size(), dtype=torch.bfloat16, device=device))
        elif hasattr(layers, 'mlp_linear_add'):
            buffers.append(torch.empty(layers.mlp_linear_add.linear.weight.size(), dtype=torch.bfloat16, device=device))
            buffers.append(torch.empty(layers.mlp_linear_add.linear.bias.size(), dtype=torch.bfloat16, device=device))
        else:
            raise AttributeError("Neither 'fc2' nor 'mlp_linear_add' found in layers.self_attn")
    
    else:
        buffers = [
            torch.empty(layers.self_attn_layer_norm.weight.size(), dtype=torch.bfloat16, device=device).pin_memory(),
            torch.empty(layers.self_attn_layer_norm.bias.size(), dtype=torch.bfloat16, device=device).pin_memory(),
            torch.empty(layers.self_attn.q_proj.weight.size(), dtype=torch.bfloat16, device=device).pin_memory(),
            torch.empty(layers.self_attn.q_proj.bias.size(), dtype=torch.bfloat16, device=device).pin_memory(),
            torch.empty(layers.self_attn.k_proj.weight.size(), dtype=torch.bfloat16, device=device).pin_memory(),
            torch.empty(layers.self_attn.k_proj.bias.size(), dtype=torch.bfloat16, device=device).pin_memory(),
            torch.empty(layers.self_attn.v_proj.weight.size(), dtype=torch.bfloat16, device=device).pin_memory(),
            torch.empty(layers.self_attn.v_proj.bias.size(), dtype=torch.bfloat16, device=device).pin_memory(),
        ]
        
        if hasattr(layers.self_attn, 'out_proj'):
            buffers.append(torch.empty(layers.self_attn.out_proj.weight.size(), dtype=torch.bfloat16, device=device).pin_memory())
            buffers.append(torch.empty(layers.self_attn.out_proj.original_bias.size(), dtype=torch.bfloat16, device=device).pin_memory())
        elif hasattr(layers, 'mha_linear_add'):
            buffers.append(torch.empty(layers.mha_linear_add.linear.weight.size(), dtype=torch.bfloat16, device=device).pin_memory())
            buffers.append(torch.empty(layers.mha_linear_add.linear.bias.size(), dtype=torch.bfloat16, device=device).pin_memory())
        else:
            raise AttributeError("Neither 'out_proj' nor 'mha_linear_add' found in layers.self_attn")

        buffers.extend([
            torch.empty(layers.final_layer_norm.weight.size(), dtype=torch.bfloat16, device=device).pin_memory(),
            torch.empty(layers.final_layer_norm.bias.size(), dtype=torch.bfloat16, device=device).pin_memory(),
            torch.empty(layers.linear_relu.linear.weight.size(), dtype=torch.bfloat16, device=device).pin_memory(),
            torch.empty(layers.linear_relu.linear.bias.size(), dtype=torch.bfloat16, device=device).pin_memory(),
        ])

        if hasattr(layers, 'fc2'):
            buffers.append(torch.empty(layers.fc2.weight.size(), dtype=torch.bfloat16, device=device).pin_memory())
            buffers.append(torch.empty(layers.fc2.original_bias.size(), dtype=torch.bfloat16, device=device).pin_memory())
        elif hasattr(layers, 'mlp_linear_add'):
            buffers.append(torch.empty(layers.mlp_linear_add.linear.weight.size(), dtype=torch.bfloat16, device=device).pin_memory())
            buffers.append(torch.empty(layers.mlp_linear_add.linear.bias.size(), dtype=torch.bfloat16, device=device).pin_memory())
        else:
            raise AttributeError("Neither 'fc2' nor 'mlp_linear_add' found in layers.self_attn")

    return buffers

def pin_memory(layers_ref, enable_cxl=False):
    def realloc_to_numa(tensor):
        numa_tensor = numa_alloc_tensor(tensor.shape, tensor.dtype)
        if numa_tensor is not None:
            numa_tensor.copy_(tensor)
            del tensor
            return numa_tensor
        else:
            raise MemoryError("Fail to allocate CXL memory!")
    if enable_cxl and not getattr(layers_ref, 'is_cxl', None):
        layers_ref.is_cxl = True
        layers_ref.self_attn_layer_norm.weight = torch.nn.Parameter(realloc_to_numa(layers_ref.self_attn_layer_norm.weight))
        layers_ref.self_attn_layer_norm.bias = torch.nn.Parameter(realloc_to_numa(layers_ref.self_attn_layer_norm.bias))
        layers_ref.self_attn.q_proj.weight = torch.nn.Parameter(realloc_to_numa(layers_ref.self_attn.q_proj.weight))
        layers_ref.self_attn.q_proj.bias = torch.nn.Parameter(realloc_to_numa(layers_ref.self_attn.q_proj.bias))
        layers_ref.self_attn.k_proj.weight = torch.nn.Parameter(realloc_to_numa(layers_ref.self_attn.k_proj.weight))
        layers_ref.self_attn.k_proj.bias = torch.nn.Parameter(realloc_to_numa(layers_ref.self_attn.k_proj.bias))
        layers_ref.self_attn.v_proj.weight = torch.nn.Parameter(realloc_to_numa(layers_ref.self_attn.v_proj.weight))
        layers_ref.self_attn.v_proj.bias = torch.nn.Parameter(realloc_to_numa(layers_ref.self_attn.v_proj.bias))
        if hasattr(layers_ref.self_attn, 'out_proj'):
            layers_ref.self_attn.out_proj.weight = torch.nn.Parameter(realloc_to_numa(layers_ref.self_attn.out_proj.weight))
            layers_ref.self_attn.out_proj.original_bias = torch.nn.Parameter(realloc_to_numa(layers_ref.self_attn.out_proj.original_bias))
        else:
            layers_ref.mha_linear_add.linear.weight = torch.nn.Parameter(realloc_to_numa(layers_ref.mha_linear_add.linear.weight))
            layers_ref.mha_linear_add.linear.bias = torch.nn.Parameter(realloc_to_numa(layers_ref.mha_linear_add.linear.bias))
        layers_ref.final_layer_norm.weight = torch.nn.Parameter(realloc_to_numa(layers_ref.final_layer_norm.weight))
        layers_ref.final_layer_norm.bias = torch.nn.Parameter(realloc_to_numa(layers_ref.final_layer_norm.bias))
        layers_ref.linear_relu.linear.weight = torch.nn.Parameter(realloc_to_numa(layers_ref.linear_relu.linear.weight))
        layers_ref.linear_relu.linear.bias = torch.nn.Parameter(realloc_to_numa(layers_ref.linear_relu.linear.bias))
        if hasattr(layers_ref, 'fc2'):
            layers_ref.fc2.weight = torch.nn.Parameter(realloc_to_numa(layers_ref.fc2.weight))
            layers_ref.fc2.original_bias = torch.nn.Parameter(realloc_to_numa(layers_ref.fc2.original_bias))
        else:
            layers_ref.mlp_linear_add.linear.weight = torch.nn.Parameter(realloc_to_numa(layers_ref.mlp_linear_add.linear.weight))
            layers_ref.mlp_linear_add.linear.bias = torch.nn.Parameter(realloc_to_numa(layers_ref.mlp_linear_add.linear.bias))

    if not enable_cxl:
        layers_ref.self_attn_layer_norm.weight=nn.Parameter(layers_ref.self_attn_layer_norm.weight.pin_memory())
        layers_ref.self_attn_layer_norm.bias=nn.Parameter(layers_ref.self_attn_layer_norm.bias.pin_memory())
        layers_ref.self_attn.q_proj.weight=nn.Parameter(layers_ref.self_attn.q_proj.weight.pin_memory())
        layers_ref.self_attn.q_proj.bias=nn.Parameter(layers_ref.self_attn.q_proj.bias.pin_memory())
        layers_ref.self_attn.k_proj.weight=nn.Parameter(layers_ref.self_attn.k_proj.weight.pin_memory())
        layers_ref.self_attn.k_proj.bias=nn.Parameter(layers_ref.self_attn.k_proj.bias.pin_memory())
        layers_ref.self_attn.v_proj.weight=nn.Parameter(layers_ref.self_attn.v_proj.weight.pin_memory())
        layers_ref.self_attn.v_proj.bias=nn.Parameter(layers_ref.self_attn.v_proj.bias.pin_memory())
        if hasattr(layers_ref.self_attn, 'out_proj'):
            layers_ref.self_attn.out_proj.weight=nn.Parameter(layers_ref.self_attn.out_proj.weight.pin_memory())
            layers_ref.self_attn.out_proj.original_bias=nn.Parameter(layers_ref.self_attn.out_proj.original_bias.pin_memory())
        else:
            layers_ref.mha_linear_add.linear.weight=nn.Parameter(layers_ref.mha_linear_add.linear.weight.pin_memory())
            layers_ref.mha_linear_add.linear.bias=nn.Parameter(layers_ref.mha_linear_add.linear.bias.pin_memory())
        layers_ref.final_layer_norm.weight=nn.Parameter(layers_ref.final_layer_norm.weight.pin_memory())
        layers_ref.final_layer_norm.bias=nn.Parameter(layers_ref.final_layer_norm.bias.pin_memory())
        layers_ref.linear_relu.linear.weight=nn.Parameter(layers_ref.linear_relu.linear.weight.pin_memory())
        layers_ref.linear_relu.linear.bias=nn.Parameter(layers_ref.linear_relu.linear.bias.pin_memory())
        if hasattr(layers_ref, 'fc2'):
            layers_ref.fc2.weight=nn.Parameter(layers_ref.fc2.weight.data.pin_memory())
            layers_ref.fc2.original_bias=nn.Parameter(layers_ref.fc2.original_bias.data.pin_memory())
        else:
            layers_ref.mlp_linear_add.linear.weight=nn.Parameter(layers_ref.mlp_linear_add.linear.weight.pin_memory())
            layers_ref.mlp_linear_add.linear.bias=nn.Parameter(layers_ref.mlp_linear_add.linear.bias.pin_memory())

def move_gpu_layer(layers_ref):
    if not layers_ref.self_attn.k_proj.weight.is_cuda:
        if hasattr(layers_ref, 'fc2'):
            dim0, dim1, dim2, dim3, dim4 = layers_ref.self_attn.q_proj.weight.shape
            new_dim = int(math.sqrt((dim0 * dim1 * dim2 * dim3 * dim4)/2))
            layers_ref.self_attn_layer_norm.weight = nn.Parameter(layers_ref.self_attn_layer_norm.weight.to('cuda'))
            layers_ref.self_attn_layer_norm.bias = nn.Parameter(torch.Tensor(layers_ref.self_attn_layer_norm.bias).to('cuda'))
            layers_ref.self_attn.q_proj.weight = nn.Parameter(layers_ref.self_attn.q_proj.weight.to('cuda').permute([0, 3, 1, 2, 4]).contiguous().view(new_dim, 2 * new_dim))
            layers_ref.self_attn.q_proj.bias = nn.Parameter(torch.Tensor(layers_ref.self_attn.q_proj.bias).to('cuda'))
            layers_ref.self_attn.k_proj.weight = nn.Parameter(layers_ref.self_attn.k_proj.weight.to('cuda').permute([0, 3, 1, 2, 4]).contiguous().view(new_dim, 2 * new_dim))
            layers_ref.self_attn.k_proj.bias = nn.Parameter(torch.Tensor(layers_ref.self_attn.k_proj.bias).to('cuda'))
            layers_ref.self_attn.v_proj.weight = nn.Parameter(layers_ref.self_attn.v_proj.weight.to('cuda').permute([0, 3, 1, 2, 4]).contiguous().view(new_dim, 2 * new_dim))
            layers_ref.self_attn.v_proj.bias = nn.Parameter(torch.Tensor(layers_ref.self_attn.v_proj.bias).to('cuda'))
            layers_ref.self_attn.out_proj.weight = nn.Parameter(layers_ref.self_attn.out_proj.weight.to('cuda').permute([0, 3, 1, 2, 4]).contiguous().view(2 * new_dim, new_dim))
            layers_ref.self_attn.out_proj.original_bias = nn.Parameter(torch.Tensor(layers_ref.self_attn.out_proj.original_bias/2).to('cuda'))
            layers_ref.final_layer_norm.weight = nn.Parameter(layers_ref.final_layer_norm.weight.to('cuda'))
            layers_ref.final_layer_norm.bias = nn.Parameter(torch.Tensor(layers_ref.final_layer_norm.bias).to('cuda'))
            layers_ref.linear_relu.linear.weight = nn.Parameter(layers_ref.linear_relu.linear.weight.to('cuda').permute([0, 3, 1, 2, 4]).contiguous().view(4 * new_dim, 2 * new_dim))
            layers_ref.linear_relu.linear.bias = nn.Parameter(torch.Tensor(layers_ref.linear_relu.linear.bias).to('cuda'))
            layers_ref.fc2.weight = nn.Parameter(layers_ref.fc2.weight.to('cuda').permute([0, 3, 1, 2, 4]).contiguous().view(2 * new_dim, 4 * new_dim))
            layers_ref.fc2.original_bias = nn.Parameter(torch.Tensor(layers_ref.fc2.original_bias).to('cuda')/2)
        else:
            dim0, dim1, dim2, dim3, dim4 = layers_ref.self_attn.q_proj.weight.shape
            new_dim = int(math.sqrt(dim0 * dim1 * dim2 * dim3 * dim4))
            layers_ref.self_attn_layer_norm.weight = nn.Parameter(layers_ref.self_attn_layer_norm.weight.to('cuda'))
            layers_ref.self_attn_layer_norm.bias = nn.Parameter(torch.Tensor(layers_ref.self_attn_layer_norm.bias).to('cuda'))
            layers_ref.self_attn.q_proj.weight = nn.Parameter(layers_ref.self_attn.q_proj.weight.to('cuda').permute([0, 3, 1, 2, 4]).contiguous().view(new_dim, new_dim))
            layers_ref.self_attn.q_proj.bias = nn.Parameter(torch.Tensor(layers_ref.self_attn.q_proj.bias).to('cuda'))
            layers_ref.self_attn.k_proj.weight = nn.Parameter(layers_ref.self_attn.k_proj.weight.to('cuda').permute([0, 3, 1, 2, 4]).contiguous().view(new_dim, new_dim))
            layers_ref.self_attn.k_proj.bias = nn.Parameter(torch.Tensor(layers_ref.self_attn.k_proj.bias).to('cuda'))
            layers_ref.self_attn.v_proj.weight = nn.Parameter(layers_ref.self_attn.v_proj.weight.to('cuda').permute([0, 3, 1, 2, 4]).contiguous().view(new_dim, new_dim))
            layers_ref.self_attn.v_proj.bias = nn.Parameter(torch.Tensor(layers_ref.self_attn.v_proj.bias).to('cuda'))
            layers_ref.mha_linear_add.weight = nn.Parameter(layers_ref.mha_linear_add.linear.weight.to('cuda').permute([0, 3, 1, 2, 4]).contiguous().view(new_dim, new_dim))
            layers_ref.mha_linear_add.bias = nn.Parameter(torch.Tensor(layers_ref.mha_linear_add.linear.bias).to('cuda'))
            layers_ref.final_layer_norm.weight = nn.Parameter(layers_ref.final_layer_norm.weight.to('cuda'))
            layers_ref.final_layer_norm.bias = nn.Parameter(torch.Tensor(layers_ref.final_layer_norm.bias).to('cuda'))
            layers_ref.linear_relu.linear.weight = nn.Parameter(layers_ref.linear_relu.linear.weight.to('cuda').permute([0, 3, 1, 2, 4]).contiguous().view(4 * new_dim, new_dim))
            layers_ref.linear_relu.linear.bias = nn.Parameter(torch.Tensor(layers_ref.linear_relu.linear.bias).to('cuda'))
            layers_ref.mlp_linear_add.weight = nn.Parameter(layers_ref.mlp_linear_add.linear.weight.to('cuda').permute([0, 3, 1, 2, 4]).contiguous().view(new_dim, 4 * new_dim))
            layers_ref.mlp_linear_add.bias = nn.Parameter(torch.Tensor(layers_ref.mlp_linear_add.linear.bias).to('cuda'))

def load_layer(layers, layers_ref, i, overlap=True):
    if i == 0:
        layers[0].copy_(layers_ref.self_attn_layer_norm.weight, non_blocking=overlap)
        layers[1].copy_(layers_ref.self_attn_layer_norm.bias, non_blocking=overlap)
        layers[2].copy_(layers_ref.self_attn.q_proj.weight, non_blocking=overlap)
        layers[3].copy_(layers_ref.self_attn.q_proj.bias, non_blocking=overlap)
        layers[4].copy_(layers_ref.self_attn.k_proj.weight, non_blocking=overlap)
        layers[5].copy_(layers_ref.self_attn.k_proj.bias, non_blocking=overlap)
        layers[6].copy_(layers_ref.self_attn.v_proj.weight, non_blocking=overlap)
        layers[7].copy_(layers_ref.self_attn.v_proj.bias, non_blocking=overlap)
        layers[10].copy_(layers_ref.final_layer_norm.weight, non_blocking=overlap)
        layers[11].copy_(layers_ref.final_layer_norm.bias, non_blocking=overlap)
        layers[12].copy_(layers_ref.linear_relu.linear.weight, non_blocking=overlap)
        layers[13].copy_(layers_ref.linear_relu.linear.bias, non_blocking=overlap)
        if hasattr(layers_ref, 'fc2'):
            layers[8].copy_(layers_ref.self_attn.out_proj.weight, non_blocking=overlap)
            layers[9].copy_(layers_ref.self_attn.out_proj.original_bias, non_blocking=overlap)
            layers[14].copy_(layers_ref.fc2.weight, non_blocking=overlap)
            layers[15].copy_(layers_ref.fc2.original_bias, non_blocking=overlap)
        else:
            layers[8].copy_(layers_ref.mha_linear_add.linear.weight, non_blocking=overlap)
            layers[9].copy_(layers_ref.mha_linear_add.linear.bias, non_blocking=overlap)
            layers[14].copy_(layers_ref.mlp_linear_add.linear.weight, non_blocking=overlap)
            layers[15].copy_(layers_ref.mlp_linear_add.linear.bias, non_blocking=overlap)

def layer_copy(layers, layers_ref, i, overlap=True):
    if i == 0:
        layers[0].copy_(layers_ref[0], non_blocking=overlap)
        layers[1].copy_(layers_ref[1], non_blocking=overlap)
        layers[2].copy_(layers_ref[2], non_blocking=overlap)
        layers[3].copy_(layers_ref[3], non_blocking=overlap)
        layers[4].copy_(layers_ref[4], non_blocking=overlap)
        layers[5].copy_(layers_ref[5], non_blocking=overlap)
        layers[6].copy_(layers_ref[6], non_blocking=overlap)
        layers[7].copy_(layers_ref[7], non_blocking=overlap)
        layers[10].copy_(layers_ref[10], non_blocking=overlap)
        layers[11].copy_(layers_ref[11], non_blocking=overlap)
        layers[12].copy_(layers_ref[12], non_blocking=overlap)
        layers[13].copy_(layers_ref[13], non_blocking=overlap)
        if hasattr(layers_ref, 'fc2'):
            layers[8].copy_(layers_ref[8], non_blocking=overlap)
            layers[9].copy_(layers_ref[9], non_blocking=overlap)
            layers[14].copy_(layers_ref[14], non_blocking=overlap)
            layers[15].copy_(layers_ref[15], non_blocking=overlap)
        else:
            layers[8].copy_(layers_ref[8], non_blocking=overlap)
            layers[9].copy_(layers_ref[9], non_blocking=overlap)
            layers[14].copy_(layers_ref[14], non_blocking=overlap)
            layers[15].copy_(layers_ref[15], non_blocking=overlap)
    
def load_activation(activation, hidden_states, mini_bsz, i, overlap=True):
    if overlap:
        hidden_states[i * mini_bsz : (i + 1) * mini_bsz] = hidden_states[i * mini_bsz : (i + 1) * mini_bsz].contiguous().pin_memory()
    
    activation.copy_(hidden_states[i * mini_bsz : (i + 1) * mini_bsz])

def load_kv_cache(past_key_value, tgt_len, key_buff, value_buff, mini_bsz, i, overlap=True):
    if overlap:
        past_key_value[1][:tgt_len, i * mini_bsz : (i + 1) * mini_bsz] = past_key_value[1][:tgt_len, i * mini_bsz : (i + 1) * mini_bsz].contiguous().pin_memory()
        past_key_value[2][:tgt_len, i * mini_bsz : (i + 1) * mini_bsz] = past_key_value[2][:tgt_len, i * mini_bsz : (i + 1) * mini_bsz].contiguous().pin_memory()

    key_buff.copy_(past_key_value[1][:tgt_len, i * mini_bsz : (i + 1) * mini_bsz], non_blocking=overlap)
    value_buff.copy_(past_key_value[2][:tgt_len, i * mini_bsz : (i + 1) * mini_bsz], non_blocking=overlap)

def store_cache(past_key_value, tgt_len, key, value, mini_bsz, i, overlap=True, stream=None):
    if overlap:
        tensor1 = torch.empty(past_key_value[1][:tgt_len, i * mini_bsz : (i + 1) * mini_bsz].size(), dtype=torch.bfloat16).contiguous().pin_memory()
        tensor2 = torch.empty(past_key_value[2][:tgt_len, i * mini_bsz : (i + 1) * mini_bsz].size(), dtype=torch.bfloat16).contiguous().pin_memory()
        tensor1.copy_(key, non_blocking=overlap)
        tensor2.copy_(value, non_blocking=overlap)
        stream.synchronize()
        past_key_value[1][:tgt_len, i * mini_bsz : (i + 1) * mini_bsz].copy_(tensor1)
        past_key_value[2][:tgt_len, i * mini_bsz : (i + 1) * mini_bsz].copy_(tensor2)
    else:
        past_key_value[1][:tgt_len, i * mini_bsz : (i + 1) * mini_bsz].copy_(key, non_blocking=overlap)
        past_key_value[2][:tgt_len, i * mini_bsz : (i + 1) * mini_bsz].copy_(value, non_blocking=overlap)

def store_cache_decoding(past_key_value, cur_len, key, value, mini_bsz, i, overlap=True):
    past_key_value[1][cur_len: cur_len + 1, i * mini_bsz : (i + 1) * mini_bsz].copy_(key, non_blocking=overlap)
    past_key_value[2][cur_len: cur_len + 1, i * mini_bsz : (i + 1) * mini_bsz].copy_(value, non_blocking=overlap)

def store_hidden(hidden_states_buff, hidden_partial, mini_bsz, i, overlap=True):
    if overlap:
        hidden_states_buff[i * mini_bsz: (i + 1) * mini_bsz] = hidden_states_buff[i * mini_bsz: (i + 1) * mini_bsz].contiguous().pin_memory()
    
    hidden_states_buff[i * mini_bsz: (i + 1) * mini_bsz].copy_(hidden_partial, non_blocking=overlap)

class OPTLearnedPositionalEmbedding(nn.Embedding):
    """
    This module learns positional embeddings up to a fixed maximum size.
    """

    def __init__(self, num_embeddings: int, embedding_dim: int):
        # OPT is set up so that if padding_idx is specified then offset the embedding ids by 2
        # and adjust num_embeddings appropriately. Other models don't have this hack
        self.offset = 2
        super().__init__(num_embeddings + self.offset, embedding_dim)

    def forward(self, attention_mask: torch.LongTensor, past_key_values_length: int = 0):
        """`input_ids_shape` is expected to be [bsz x seqlen]."""
        attention_mask = attention_mask.long()

        # create positions depending on attention_mask
        positions = (torch.cumsum(attention_mask, dim=1).type_as(attention_mask) * attention_mask).long() - 1

        # cut positions if `past_key_values_length` is > 0
        positions = positions[:, past_key_values_length:]

        return super().forward(positions + self.offset)


class OPTAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
        self,
        config: OPTConfig,
        is_decoder: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.config = config

        def _handle_deprecated_argument(config_arg_name, config, fn_arg_name, kwargs):
            """
            If a the deprecated argument `fn_arg_name` is passed, raise a deprecation
            warning and return that value, otherwise take the equivalent config.config_arg_name
            """
            val = None
            if fn_arg_name in kwargs:
                logging.warning(
                    "Passing in {} to {self.__class__.__name__} is deprecated and won't be supported from v4.38."
                    " Please set it in the config instead"
                )
                val = kwargs.pop(fn_arg_name)
            else:
                val = getattr(config, config_arg_name)
            return val

        self.embed_dim = _handle_deprecated_argument("hidden_size", config, "embed_dim", kwargs)
        self.num_heads = _handle_deprecated_argument("num_attention_heads", config, "num_heads", kwargs)
        self.dropout = _handle_deprecated_argument("attention_dropout", config, "dropout", kwargs)
        self.enable_bias = _handle_deprecated_argument("enable_bias", config, "bias", kwargs)

        self.head_dim = self.embed_dim // self.num_heads
        self.is_causal = True

        if (self.head_dim * self.num_heads) != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}"
                f" and `num_heads`: {self.num_heads})."
            )
        self.scaling = self.head_dim**-0.5
        self.is_decoder = is_decoder

        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=self.enable_bias)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=self.enable_bias)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=self.enable_bias)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=self.enable_bias)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""

        # if key_value_states are provided this layer is used as a cross-attention layer
        # for the decoder
        is_cross_attention = key_value_states is not None

        bsz, tgt_len, _ = hidden_states.size()

        # get query proj
        query_states = self.q_proj(hidden_states) * self.scaling
        # get key, value proj
        if is_cross_attention and past_key_value is not None:
            # reuse k,v, cross_attentions
            key_states = past_key_value[0]
            value_states = past_key_value[1]
        elif is_cross_attention:
            # cross_attentions
            key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
            value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
        elif past_key_value is not None:
            # reuse k, v, self_attention
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)
        else:
            # self_attention
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

        if self.is_decoder:
            # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_states, value_states)

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        value_states = value_states.view(*proj_shape)

        src_len = key_states.size(1)
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))

        if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, tgt_len, src_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
            attn_weights = torch.max(
                attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min, device=attn_weights.device)
            )
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        # upcast to fp32 if the weights are in fp16. Please see https://github.com/huggingface/transformers/pull/17437
        if attn_weights.dtype == torch.float16:
            attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(torch.float16)
        else:
            attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        if layer_head_mask is not None:
            if layer_head_mask.size() != (self.num_heads,):
                raise ValueError(
                    f"Head mask for a single layer should be of size {(self.num_heads,)}, but is"
                    f" {layer_head_mask.size()}"
                )
            attn_weights = layer_head_mask.view(1, -1, 1, 1) * attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        if output_attentions:
            # this operation is a bit awkward, but it's required to
            # make sure that attn_weights keeps its gradient.
            # In order to do so, attn_weights have to be reshaped
            # twice and have to be reused in the following
            attn_weights_reshaped = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights_reshaped.view(bsz * self.num_heads, tgt_len, src_len)
        else:
            attn_weights_reshaped = None

        attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)

        attn_output = torch.bmm(attn_probs, value_states)

        if attn_output.size() != (bsz * self.num_heads, tgt_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2)

        # Use the `embed_dim` from the config (stored in the class) rather than `hidden_state` because `attn_output` can be
        # partitioned aross GPUs when using tensor-parallelism.
        attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)

        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights_reshaped, past_key_value


class OptFlashAttention2(OPTAttention):
    """
    OPT flash attention module. This module inherits from `OPTAttention` as the weights of the module stays untouched.
    The only required change would be on the forward pass where it needs to correctly call the public API of flash
    attention and deal with padding tokens in case the input contains any of them.
    """

    # Copied from transformers.models.llama.modeling_llama.LlamaFlashAttention2.__init__
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # TODO: Should be removed once Flash Attention for RoCm is bumped to 2.1.
        # flash_attn<2.1 generates top-left aligned causal mask, while what is needed here is bottom-right alignement, that was made default for flash_attn>=2.1. This attribute is used to handle this difference. Reference: https://github.com/Dao-AILab/flash-attention/releases/tag/v2.1.0.
        # Beware that with flash_attn<2.1, using q_seqlen != k_seqlen (except for the case q_seqlen == 1) produces a wrong mask (top-left).
        self._flash_attn_uses_top_left_mask = not is_flash_attn_greater_or_equal_2_10()

    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""

        # if key_value_states are provided this layer is used as a cross-attention layer
        # for the decoder
        is_cross_attention = key_value_states is not None

        bsz, _, _ = hidden_states.size()

        # get query proj
        query_states = self.q_proj(hidden_states)
        # get key, value proj
        if is_cross_attention and past_key_value is not None:
            # reuse k,v, cross_attentions
            key_states = past_key_value[0]
            value_states = past_key_value[1]
        elif is_cross_attention:
            # cross_attentions
            key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
            value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
        elif past_key_value is not None:
            # reuse k, v, self_attention
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)
        else:
            # self_attention
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

        if self.is_decoder:
            # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_states, value_states)

        query_length = query_states.shape[1]
        tgt_len = key_states.shape[-2]

        # Flash attention requires the input to have the shape
        # batch_size x seq_length x head_dim x hidden_dim
        query_states = query_states.view(bsz, query_length, self.num_heads, self.head_dim)
        key_states = key_states.transpose(1, 2).view(bsz, tgt_len, self.num_heads, self.head_dim)
        value_states = value_states.transpose(1, 2).view(bsz, tgt_len, self.num_heads, self.head_dim)

        attn_dropout = self.dropout if self.training else 0.0

        # In PEFT, usually we cast the layer norms in float32 for training stability reasons
        # therefore the input hidden states gets silently casted in float32. Hence, we need
        # cast them back in float16 just to be sure everything works as expected.
        input_dtype = query_states.dtype
        if input_dtype == torch.float32:
            if torch.is_autocast_enabled():
                target_dtype = torch.get_autocast_gpu_dtype()
            # Handle the case where the model is quantized
            elif hasattr(self.config, "_pre_quantization_dtype"):
                target_dtype = self.config._pre_quantization_dtype
            else:
                target_dtype = self.q_proj.weight.dtype

            logger.warning_once(
                f"The input hidden states seems to be silently casted in float32, this might be related to"
                f" the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in"
                f" {target_dtype}."
            )

            query_states = query_states.to(target_dtype)
            key_states = key_states.to(target_dtype)
            value_states = value_states.to(target_dtype)

        attn_output = self._flash_attention_forward(
            query_states, key_states, value_states, attention_mask, query_length, dropout=attn_dropout
        )

        attn_weights_reshaped = attn_output.reshape(bsz, query_length, self.num_heads * self.head_dim)
        attn_output = self.out_proj(attn_weights_reshaped)

        if not output_attentions:
            attn_weights_reshaped = None

        return attn_output, attn_weights_reshaped, past_key_value

    # Copied from transformers.models.llama.modeling_llama.LlamaFlashAttention2._flash_attention_forward
    def _flash_attention_forward(
        self, query_states, key_states, value_states, attention_mask, query_length, dropout=0.0, softmax_scale=None
    ):
        """
        Calls the forward method of Flash Attention - if the input hidden states contain at least one padding token
        first unpad the input, then computes the attention scores and pad the final attention scores.

        Args:
            query_states (`torch.Tensor`):
                Input query states to be passed to Flash Attention API
            key_states (`torch.Tensor`):
                Input key states to be passed to Flash Attention API
            value_states (`torch.Tensor`):
                Input value states to be passed to Flash Attention API
            attention_mask (`torch.Tensor`):
                The padding mask - corresponds to a tensor of size `(batch_size, seq_len)` where 0 stands for the
                position of padding tokens and 1 for the position of non-padding tokens.
            dropout (`int`, *optional*):
                Attention dropout
            softmax_scale (`float`, *optional*):
                The scaling of QK^T before applying softmax. Default to 1 / sqrt(head_dim)
        """
        if not self._flash_attn_uses_top_left_mask:
            causal = self.is_causal
        else:
            # TODO: Remove the `query_length != 1` check once Flash Attention for RoCm is bumped to 2.1. For details, please see the comment in LlamaFlashAttention2 __init__.
            causal = self.is_causal and query_length != 1

        # Contains at least one padding token in the sequence
        if attention_mask is not None:
            batch_size = query_states.shape[0]
            query_states, key_states, value_states, indices_q, cu_seq_lens, max_seq_lens = self._upad_input(
                query_states, key_states, value_states, attention_mask, query_length
            )

            cu_seqlens_q, cu_seqlens_k = cu_seq_lens
            max_seqlen_in_batch_q, max_seqlen_in_batch_k = max_seq_lens

            attn_output_unpad = flash_attn_varlen_func(
                query_states,
                key_states,
                value_states,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=max_seqlen_in_batch_q,
                max_seqlen_k=max_seqlen_in_batch_k,
                dropout_p=dropout,
                softmax_scale=softmax_scale,
                causal=causal,
            )

            attn_output = pad_input(attn_output_unpad, indices_q, batch_size, query_length)
        else:
            attn_output = flash_attn_func(
                query_states, key_states, value_states, dropout, softmax_scale=softmax_scale, causal=causal
            )

        return attn_output

    # Copied from transformers.models.llama.modeling_llama.LlamaFlashAttention2._upad_input
    def _upad_input(self, query_layer, key_layer, value_layer, attention_mask, query_length):
        indices_k, cu_seqlens_k, max_seqlen_in_batch_k = _get_unpad_data(attention_mask)
        batch_size, kv_seq_len, num_key_value_heads, head_dim = key_layer.shape

        key_layer = index_first_axis(
            key_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim), indices_k
        )
        value_layer = index_first_axis(
            value_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim), indices_k
        )
        if query_length == kv_seq_len:
            query_layer = index_first_axis(
                query_layer.reshape(batch_size * kv_seq_len, self.num_heads, head_dim), indices_k
            )
            cu_seqlens_q = cu_seqlens_k
            max_seqlen_in_batch_q = max_seqlen_in_batch_k
            indices_q = indices_k
        elif query_length == 1:
            max_seqlen_in_batch_q = 1
            cu_seqlens_q = torch.arange(
                batch_size + 1, dtype=torch.int32, device=query_layer.device
            )  # There is a memcpy here, that is very bad.
            indices_q = cu_seqlens_q[:-1]
            query_layer = query_layer.squeeze(1)
        else:
            # The -q_len: slice assumes left padding.
            attention_mask = attention_mask[:, -query_length:]
            query_layer, indices_q, cu_seqlens_q, max_seqlen_in_batch_q = unpad_input(query_layer, attention_mask)

        return (
            query_layer,
            key_layer,
            value_layer,
            indices_q,
            (cu_seqlens_q, cu_seqlens_k),
            (max_seqlen_in_batch_q, max_seqlen_in_batch_k),
        )


OPT_ATTENTION_CLASSES = {
    "eager": OPTAttention,
    "flash_attention_2": OptFlashAttention2,
}


class OPTDecoderLayer(nn.Module):
    def __init__(self, config: OPTConfig):
        super().__init__()
        self.embed_dim = config.hidden_size

        self.self_attn = OPT_ATTENTION_CLASSES[config._attn_implementation](config=config, is_decoder=True)

        self.do_layer_norm_before = config.do_layer_norm_before
        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.activation_function]

        self.self_attn_layer_norm = nn.LayerNorm(
            self.embed_dim, elementwise_affine=config.layer_norm_elementwise_affine
        )
        self.fc1 = nn.Linear(self.embed_dim, config.ffn_dim, bias=config.enable_bias)
        self.fc2 = nn.Linear(config.ffn_dim, self.embed_dim, bias=config.enable_bias)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim, elementwise_affine=config.layer_norm_elementwise_affine)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            layer_head_mask (`torch.FloatTensor`, *optional*): mask for attention heads in a given layer of size
                `(encoder_attention_heads,)`.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
        """

        residual = hidden_states

        # 125m, 1.7B, ..., 175B applies layer norm BEFORE attention
        if self.do_layer_norm_before:
            hidden_states = self.self_attn_layer_norm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            past_key_value=past_key_value,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            output_attentions=output_attentions,
        )
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states

        # 350m applies layer norm AFTER attention
        if not self.do_layer_norm_before:
            hidden_states = self.self_attn_layer_norm(hidden_states)

        # Fully Connected
        hidden_states_shape = hidden_states.shape
        hidden_states = hidden_states.reshape(-1, hidden_states.size(-1))
        residual = hidden_states

        # 125m, 1.7B, ..., 175B applies layer norm BEFORE attention
        if self.do_layer_norm_before:
            hidden_states = self.final_layer_norm(hidden_states)

        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)

        hidden_states = self.fc2(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

        hidden_states = (residual + hidden_states).view(hidden_states_shape)

        # 350m applies layer norm AFTER attention
        if not self.do_layer_norm_before:
            hidden_states = self.final_layer_norm(hidden_states)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


OPT_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`OPTConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""


@add_start_docstrings(
    "The bare OPT Model outputting raw hidden-states without any specific head on top.",
    OPT_START_DOCSTRING,
)
class OPTPreTrainedModel(PreTrainedModel):
    config_class = OPTConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["OPTDecoderLayer"]
    _supports_flash_attn_2 = True

    def _init_weights(self, module):
        std = self.config.init_std
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


OPT_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            If `past_key_values` is used, optionally only the last `decoder_input_ids` have to be input (see
            `past_key_values`).

            If you want to change padding behavior, you should read [`modeling_opt._prepare_decoder_attention_mask`]
            and modify to your needs. See diagram 1 in [the paper](https://arxiv.org/abs/1910.13461) for more
            information on the default strategy.
        head_mask (`torch.Tensor` of shape `(encoder_layers, encoder_attention_heads)`, *optional*):
            Mask to nullify selected heads of the attention modules in the encoder. Mask values selected in `[0, 1]`:

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.

        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and 2 additional tensors of shape
            `(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`.

            Contains pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
            blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.

            If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that
            don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of all
            `decoder_input_ids` of shape `(batch_size, sequence_length)`.
        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""


class OPTDecoder(OPTPreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`OPTDecoderLayer`]

    Args:
        config: OPTConfig
    """

    def __init__(self, config: OPTConfig):
        super().__init__(config)
        self.dropout = config.dropout
        self.layerdrop = config.layerdrop
        self.padding_idx = config.pad_token_id
        self.max_target_positions = config.max_position_embeddings
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.word_embed_proj_dim, self.padding_idx)
        self.embed_positions = OPTLearnedPositionalEmbedding(config.max_position_embeddings, config.hidden_size)

        if config.word_embed_proj_dim != config.hidden_size:
            self.project_out = nn.Linear(config.hidden_size, config.word_embed_proj_dim, bias=False)
        else:
            self.project_out = None

        if config.word_embed_proj_dim != config.hidden_size:
            self.project_in = nn.Linear(config.word_embed_proj_dim, config.hidden_size, bias=False)
        else:
            self.project_in = None

        # Note that the only purpose of `config._remove_final_layer_norm` is to keep backward compatibility
        # with checkpoints that have been fine-tuned before transformers v4.20.1
        # see https://github.com/facebookresearch/metaseq/pull/164
        if config.do_layer_norm_before and not config._remove_final_layer_norm:
            self.final_layer_norm = nn.LayerNorm(
                config.hidden_size, elementwise_affine=config.layer_norm_elementwise_affine
            )
        else:
            self.final_layer_norm = None

        self.layers = nn.ModuleList([OPTDecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self._use_flash_attention_2 = config._attn_implementation == "flash_attention_2"

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        prefill_policy: Optional[int] = None,
        decoding_policy: Optional[int] = None,
        no_overlap: Optional[bool] = None,
        pin_weight: Optional[bool] = None,
        gpu_percentage: Optional[int] = None,
        num_minibatch: Optional[int] = None,
        enable_cxl: Optional[bool] = None,
        max_new_tokens: Optional[int] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        r"""
        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you
                provide it.

                Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
                [`PreTrainedTokenizer.__call__`] for details.

                [What are input IDs?](../glossary#input-ids)
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                [What are attention masks?](../glossary#attention-mask)
            head_mask (`torch.Tensor` of shape `(num_hidden_layers, num_attention_heads)`, *optional*):
                Mask to nullify selected heads of the attention modules. Mask values selected in `[0, 1]`:

                - 1 indicates the head is **not masked**,
                - 0 indicates the head is **masked**.

            past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
                Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of
                shape `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and 2 additional tensors of

                Contains pre-computed hidden-states (key and values in the self-attention blocks and in the
                cross-attention blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.

                If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those
                that don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of
                all `decoder_input_ids` of shape `(batch_size, sequence_length)`.

            inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
                Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation.
                This is useful if you want more control over how to convert `input_ids` indices into associated vectors
                than the model's internal embedding lookup matrix.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        batch_size, seq_length = input_shape
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0
        # required mask seq length can be calculated via length of past
        mask_seq_length = past_key_values_length + seq_length

        # embed positions
        if self._use_flash_attention_2:
            # 2d mask is passed through the layers
            causal_attention_mask = attention_mask if (attention_mask is not None and 0 in attention_mask) else None
            attention_mask = (
                torch.ones(batch_size, mask_seq_length, device=inputs_embeds.device)
                if attention_mask is None
                else attention_mask
            )
        else:
            # 4d mask is passed through the layers
            if attention_mask is None:
                attention_mask = torch.ones(batch_size, mask_seq_length, device=inputs_embeds.device)
            elif attention_mask.shape[1] != mask_seq_length:
                raise ValueError(
                    f"The provided attention mask has length {attention_mask.shape[1]}, but its length should be "
                    f"{mask_seq_length} (sum of the lengths of current and past inputs)"
                )
            causal_attention_mask = _prepare_4d_causal_attention_mask(
                attention_mask, input_shape, inputs_embeds, past_key_values_length
            )

        pos_embeds = self.embed_positions(attention_mask, past_key_values_length)

        if self.project_in is not None:
            inputs_embeds = self.project_in(inputs_embeds)

        hidden_states = inputs_embeds + pos_embeds

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None

        # check if head_mask has a correct number of layers specified if desired
        for attn_mask, mask_name in zip([head_mask], ["head_mask"]):
            if attn_mask is not None:
                if attn_mask.size()[0] != (len(self.layers)):
                    raise ValueError(
                        f"The `{mask_name}` should be specified for {len(self.layers)} layers, but it is for"
                        f" {head_mask.size()[0]}."
                    )
        
        bsz, tgt_len, _ = hidden_states.size()

        # Policy 0: compute everything on GPU & store KV cache on CPU
        # Policy 1: compute everything on CPU
        # Policy 2: compute linear on GPU & attention on CPU (w AMX)
        # Policy 3: compute everything on GPU & store KV cache on CPU (for online)

        # IPEX baseline: prefill_policy = 1, decoding_policy = 1, gpu_percentage = 0
        overlap = not no_overlap

        prefill_policy_gpu = 3
        decoding_policy_gpu = 3

        mini_bsz = int(bsz/num_minibatch)

        activation = torch.empty_like(hidden_states, device='cuda')

        n_gpu_layers = int(len(self.layers) * gpu_percentage / 100)
        for i in range(n_gpu_layers):
            move_gpu_layer(self.layers[i])

        is_prefill = False
        if tgt_len != 1:
            is_prefill = True

        if prefill_policy != 1 or decoding_policy != 1: 
            activation_1 = torch.empty_like(hidden_states[:mini_bsz], device='cuda')
            if overlap and (prefill_policy == 0 or decoding_policy == 0):
                activation_2 = torch.empty_like(hidden_states[:mini_bsz], device='cuda')
            if gpu_percentage < 99:
                gpu_buff_1 = create_buffer(self.layers[-1], device='cuda')
                gpu_buff_2 = create_buffer(self.layers[-1], device='cuda')

            hidden_partial = None
            key_buff = None
            value_buff = None
            
            if tgt_len == 1 and decoding_policy == 0:
                key_buff_1 = torch.empty_like(past_key_values[0][1][:, :mini_bsz], device='cuda')
                key_buff_2 = torch.empty_like(past_key_values[0][1][:, :mini_bsz], device='cuda')
                value_buff_1 = torch.empty_like(past_key_values[0][2][:, :mini_bsz], device='cuda')
                value_buff_2 = torch.empty_like(past_key_values[0][2][:, :mini_bsz], device='cuda')

            load_weight_stream = torch.cuda.Stream()
            load_activation_stream = torch.cuda.Stream()
            compute_stream = torch.cuda.Stream()
            store_cache_stream = torch.cuda.Stream()
            store_hidden_stream = torch.cuda.Stream()

            if pin_weight and overlap:
                if self.layers[0].self_attn.q_proj.weight.is_pinned() == False:
                    for idx in range(len(self.layers)-n_gpu_layers):
                        pin_memory(self.layers[idx+n_gpu_layers], enable_cxl)
            
            if not pin_weight:
                cpu_buff = create_buffer(self.layers[-1], device='cpu')

        for idx, decoder_layer in enumerate(self.layers):
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.training:
                dropout_probability = torch.rand([])
                if dropout_probability < self.layerdrop:
                    continue

            past_key_value = past_key_values[idx] if past_key_values is not None else None
            # hidden_states_buff = torch.empty_like(hidden_states)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    causal_attention_mask,
                    head_mask[idx] if head_mask is not None else None,
                    None,
                    output_attentions,
                    use_cache,
                )
            else:
                if idx  < n_gpu_layers:
                    if idx == 0:
                        load_activation(activation, hidden_states, bsz, 0, overlap=False)
                    layer_outputs = decoder_layer(
                        activation,
                        attention_mask=causal_attention_mask,
                        layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                        past_key_value=past_key_value,
                        output_attentions=output_attentions,
                        use_cache=use_cache,
                        policy=prefill_policy_gpu if is_prefill else decoding_policy_gpu,
                        max_new_tokens=max_new_tokens,
                    )
                    activation = layer_outputs[0]
                    next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)
                else:
                    if idx == n_gpu_layers:
                        if overlap:
                            hidden_states = hidden_states.pin_memory()
                        if n_gpu_layers > 0:
                            hidden_states.copy_(activation)
                            del activation
                    if is_prefill and prefill_policy == 0:                        
                        past_key_value_decoder = (
                            torch.empty(
                                1,
                                tgt_len,
                                tgt_len,
                                1,
                                dtype=torch.long,
                            ).contiguous(),
                            torch.empty([(tgt_len+max_new_tokens), bsz, self.layers[0].self_attn.num_heads, self.layers[0].self_attn.head_dim], dtype=torch.bfloat16, device='cpu'),
                            torch.empty([(tgt_len+max_new_tokens), bsz, self.layers[0].self_attn.num_heads, self.layers[0].self_attn.head_dim], dtype=torch.bfloat16, device='cpu'),
                            torch.zeros([(tgt_len+max_new_tokens), past_key_value[3].size(1)], dtype=torch.long).contiguous(),
                        )

                        past_key_value = past_key_value_decoder
                        # Multi-batch processing (FlexGen)
                        for i in range(num_minibatch):
                            if overlap:
                                # Overlapping
                                if i == 0:
                                    with torch.cuda.stream(load_weight_stream):
                                        if idx == n_gpu_layers:
                                            if not pin_weight:
                                                load_layer(cpu_buff, self.layers[idx], i, overlap=True)
                                                load_weight_stream.synchronize()
                                                layer_copy(gpu_buff_2 if idx % 2 else gpu_buff_1, cpu_buff, i, overlap=overlap)
                                            else:
                                                load_layer(gpu_buff_2 if idx % 2 else gpu_buff_1, self.layers[idx], i, overlap=overlap)
                                    with torch.cuda.stream(load_activation_stream):
                                        load_activation(activation_1, hidden_states, mini_bsz, i, overlap=overlap)
                                    torch.cuda.synchronize()

                                if i > 0:
                                    with torch.cuda.stream(store_cache_stream):
                                        store_cache(past_key_value, tgt_len, key_buff_1 if i % 2 else key_buff_2, value_buff_1 if i % 2 else value_buff_2, mini_bsz, i-1, overlap=overlap, stream=store_cache_stream)
                                    with torch.cuda.stream(store_hidden_stream):
                                        store_hidden(hidden_states, hidden_partial_1 if i % 2 else hidden_partial_2, mini_bsz, i-1, overlap=overlap)
                                if idx < len(self.layers) - 1:
                                    with torch.cuda.stream(load_weight_stream):
                                        if not pin_weight:
                                                load_layer(cpu_buff, self.layers[idx+1], i, overlap=False)
                                                load_weight_stream.synchronize()
                                                layer_copy(gpu_buff_1 if idx % 2 else gpu_buff_2, cpu_buff, i, overlap=overlap)
                                        else:
                                            load_layer(gpu_buff_1 if idx % 2 else gpu_buff_2, self.layers[idx+1], i, overlap=overlap)
                                if i < num_minibatch - 1:
                                    with torch.cuda.stream(load_activation_stream):
                                        load_activation(activation_1 if i % 2 else activation_2, hidden_states, mini_bsz, i)

                                with torch.cuda.stream(compute_stream):
                                    layer_outputs = decoder_layer(
                                        activation_2 if i % 2 else activation_1,
                                        attention_mask=causal_attention_mask[:mini_bsz],
                                        layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                                        past_key_value=None,
                                        output_attentions=output_attentions,
                                        use_cache=use_cache,
                                        gpu_layer=gpu_buff_2 if idx % 2 else gpu_buff_1,
                                        policy = prefill_policy,
                                        max_new_tokens=max_new_tokens,
                                    )
                                    
                                    if i % 2 == 0:
                                        hidden_partial_1 = layer_outputs[0]
                                        key_buff_1 = layer_outputs[2]
                                        value_buff_1 = layer_outputs[3]
                                    else:
                                        hidden_partial_2 = layer_outputs[0]
                                        key_buff_2 = layer_outputs[2]
                                        value_buff_2 = layer_outputs[3]
                                
                                torch.cuda.synchronize()

                                if i == num_minibatch - 1:
                                    with torch.cuda.stream(store_cache_stream):
                                        store_cache(past_key_value, tgt_len, key_buff_2 if i % 2 else key_buff_1, value_buff_2 if i % 2 else value_buff_1, mini_bsz, i, overlap=overlap, stream=store_cache_stream)
                                    with torch.cuda.stream(store_hidden_stream):
                                        store_hidden(hidden_states, hidden_partial_2 if i % 2 else hidden_partial_1, mini_bsz, i, overlap=overlap)
                                    torch.cuda.synchronize()

                            else:
                                # Non-overlapping
                                load_activation(activation_1, hidden_states, mini_bsz, i, overlap=overlap)
                                load_layer(gpu_buff_1, self.layers[idx], i, overlap=overlap)

                                layer_outputs = decoder_layer(
                                    activation_1,
                                    attention_mask=causal_attention_mask[:mini_bsz],
                                    layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                                    past_key_value=None,
                                    output_attentions=output_attentions,
                                    use_cache=use_cache,
                                    gpu_layer=gpu_buff_1,
                                    policy=prefill_policy,
                                    max_new_tokens=max_new_tokens,
                                )
                                store_cache(past_key_value, tgt_len, layer_outputs[2], layer_outputs[3], mini_bsz, i, overlap=overlap)
                                hidden_states[i * mini_bsz: (i + 1) * mini_bsz].copy_(layer_outputs[0])

                    elif is_prefill and prefill_policy == 1:
                        layer_outputs = decoder_layer(
                                hidden_states,
                                attention_mask=causal_attention_mask,
                                layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                                past_key_value=past_key_value,
                                output_attentions=output_attentions,
                                use_cache=use_cache,
                                policy=prefill_policy,
                            )
                        hidden_states = layer_outputs[0]

                    elif not is_prefill and decoding_policy == 0:
                        # pinning memory
                        hidden_states = hidden_states.pin_memory()
                        past_key_value = (
                            past_key_value[0],
                            past_key_value[1].pin_memory(),
                            past_key_value[2].pin_memory(),
                            past_key_value[3],
                        )

                        cur_len = past_key_value[0].size(1)

                        # Multi-batch processing (FlexGen)
                        for i in range(num_minibatch):
                            if overlap:
                                past_key_value_decoder = (
                                    past_key_value[0],
                                    key_buff_2 if i % 2 else key_buff_1,
                                    value_buff_2 if i % 2 else value_buff_1,
                                    past_key_value[3],
                                )
                                # Overlapping
                                if i == 0:
                                    with torch.cuda.stream(load_weight_stream):
                                        if idx == n_gpu_layers:
                                            if not pin_weight:
                                                load_layer(cpu_buff, self.layers[idx], i, overlap=False)
                                                load_weight_stream.synchronize()
                                                layer_copy(gpu_buff_1, cpu_buff, i, overlap=overlap)
                                            else:
                                                load_layer(gpu_buff_1, self.layers[idx], i, overlap=overlap)
                                        load_activation(activation_1, hidden_states, mini_bsz, i, overlap=overlap)
                                        load_kv_cache(past_key_value, tgt_len, key_buff_1, value_buff_1, mini_bsz, i, overlap=overlap)
                                    torch.cuda.synchronize() 

                                with torch.cuda.stream(store_cache_stream):
                                    if i > 0:
                                        store_cache_decoding(past_key_value, cur_len, key_buff, value_buff, mini_bsz, i-1, overlap=overlap)
                                        store_hidden(hidden_states, hidden_partial, mini_bsz, i-1, overlap=overlap)

                                with torch.cuda.stream(load_weight_stream):
                                    if idx < len(self.layers) - 1:
                                        if not pin_weight:
                                                load_layer(cpu_buff, self.layers[idx+1], i, overlap=False)
                                                load_weight_stream.synchronize()
                                                layer_copy(gpu_buff_1 if idx % 2 else gpu_buff_2, cpu_buff, i, overlap=overlap)
                                        else:
                                            load_layer(gpu_buff_1 if idx % 2 else gpu_buff_2, self.layers[idx+1], i, overlap=overlap)
                                    if idx < num_minibatch - 1:
                                        load_activation(activation_2 if i % 2 else activation_1, hidden_states, mini_bsz, i, overlap=overlap)
                                        load_kv_cache(past_key_value, tgt_len, key_buff_2 if i % 2 else key_buff_1, value_buff_2 if i % 2 else value_buff_1, mini_bsz, i, overlap=overlap)

                                with torch.cuda.stream(compute_stream):
                                    layer_outputs = decoder_layer(
                                        activation_2 if i % 2 else activation_1,
                                        attention_mask=None,
                                        layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                                        past_key_value=past_key_value_decoder,
                                        output_attentions=output_attentions,
                                        use_cache=use_cache,
                                        gpu_layer=gpu_buff_2 if idx % 2 else gpu_buff_1,
                                        policy = decoding_policy,
                                    )
                                
                                torch.cuda.synchronize()
                                hidden_partial = layer_outputs[0]
                                key_buff = layer_outputs[2]
                                value_buff = layer_outputs[3]

                                if i == num_minibatch - 1:
                                    with torch.cuda.stream(store_cache_stream):
                                        store_cache_decoding(past_key_value, cur_len, key_buff, value_buff, mini_bsz, i, overlap=overlap)
                                        store_hidden(hidden_states, hidden_partial, mini_bsz, i, overlap=overlap)
                                    torch.cuda.synchronize()

                            else:
                                # Non-overlapping
                                load_activation(activation_1, hidden_states, mini_bsz, i, overlap=True)
                                load_layer(gpu_buff_1, self.layers[idx], i, overlap=True)
                                load_kv_cache(past_key_value, tgt_len, key_buff_1, value_buff_1, mini_bsz, i, overlap=True)
                                
                                past_key_value_decoder = (
                                    past_key_value[0],
                                    key_buff_1,
                                    value_buff_1,
                                    past_key_value[3],
                                )

                                layer_outputs = decoder_layer(
                                    activation_1,
                                    attention_mask=None,
                                    layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                                    past_key_value=past_key_value_decoder,
                                    output_attentions=output_attentions,
                                    use_cache=use_cache,
                                    gpu_layer=gpu_buff_1,
                                    policy=decoding_policy,
                                )
                                store_cache_decoding(past_key_value, cur_len, layer_outputs[2], layer_outputs[3], mini_bsz, i, overlap=True)
                                hidden_states[i * mini_bsz: (i + 1) * mini_bsz].copy_(layer_outputs[0], non_blocking=True)
                        
                        past_key_value = (
                            torch.empty(
                                1,
                                cur_len+1,
                                cur_len+1,
                                1,
                                dtype=torch.long,
                            ).contiguous(),
                            past_key_value[1],
                            past_key_value[2],
                            past_key_value[3],
                        )
                        
                    elif not is_prefill and decoding_policy in [2, 4]:
                        hidden_states = hidden_states.pin_memory()
                        activation = torch.empty_like(hidden_states, device='cuda')
                        load_activation(activation, hidden_states, bsz, 0, overlap=overlap)
                        if overlap:
                            if idx == n_gpu_layers:
                                with torch.cuda.stream(load_weight_stream):
                                    if not pin_weight:
                                        load_layer(cpu_buff, self.layers[idx], 0, overlap=overlap)
                                        load_weight_stream.synchronize()
                                        layer_copy(gpu_buff_2 if idx % 2 else gpu_buff_1, cpu_buff, 0, overlap=overlap)
                                    else:
                                        load_layer(gpu_buff_2 if idx % 2 else gpu_buff_1, self.layers[idx], 0, overlap=overlap)
                                torch.cuda.synchronize()

                            with torch.cuda.stream(load_weight_stream):
                                if idx < len(self.layers) - 1:
                                    if not pin_weight:
                                        load_layer(cpu_buff, self.layers[idx+1], 0, overlap=overlap)
                                        load_weight_stream.synchronize()
                                        layer_copy(gpu_buff_1 if idx % 2 else gpu_buff_2, cpu_buff, 0, overlap=overlap)
                                    else:
                                        load_layer(gpu_buff_1 if idx % 2 else gpu_buff_2, self.layers[idx+1], 0, overlap=overlap)

                            with torch.cuda.stream(compute_stream):
                                layer_outputs = decoder_layer(
                                    activation,
                                    attention_mask=causal_attention_mask,
                                    layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                                    past_key_value=past_key_value,
                                    output_attentions=output_attentions,
                                    use_cache=use_cache,
                                    gpu_layer=gpu_buff_2 if idx % 2 else gpu_buff_1,
                                    policy=decoding_policy,
                                )
                            torch.cuda.synchronize()

                        else:
                            load_layer(gpu_buff_1, self.layers[idx], 0, overlap=overlap)
                            layer_outputs = decoder_layer(
                                activation,
                                attention_mask=causal_attention_mask,
                                layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                                past_key_value=past_key_value,
                                output_attentions=output_attentions,
                                use_cache=use_cache,
                                gpu_layer=gpu_buff_1,
                                policy=decoding_policy,
                            )
                        
                        hidden_states.copy_(layer_outputs[0])

                    else:
                        layer_outputs = decoder_layer(
                            hidden_states,
                            attention_mask=causal_attention_mask,
                            layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                            past_key_value=past_key_value,
                            output_attentions=output_attentions,
                            use_cache=use_cache,
                            policy=decoding_policy,
                        )
                        hidden_states = layer_outputs[0]

                    if use_cache:
                        next_decoder_cache += (past_key_value,) if ((prefill_policy == 0 and is_prefill) or (decoding_policy == 0 and not is_prefill)) else (layer_outputs[2 if output_attentions else 1],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        if self.final_layer_norm is not None:
            hidden_states = self.final_layer_norm(hidden_states)

        if self.project_out is not None:
            hidden_states = self.project_out(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)
        
        if tgt_len == 1 and decoding_policy == 0:
            del key_buff_1, key_buff_2, value_buff_1, value_buff_2
        
        torch.cuda.empty_cache()

        next_cache = next_decoder_cache if use_cache else None
        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


@add_start_docstrings(
    "The bare OPT Model outputting raw hidden-states without any specific head on top.",
    OPT_START_DOCSTRING,
)
class OPTModel(OPTPreTrainedModel):
    def __init__(self, config: OPTConfig):
        super().__init__(config)
        self.decoder = OPTDecoder(config)
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.decoder.embed_tokens

    def set_input_embeddings(self, value):
        self.decoder.embed_tokens = value

    def get_decoder(self):
        return self.decoder

    @add_start_docstrings_to_model_forward(OPT_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=BaseModelOutputWithPast,
        config_class=_CONFIG_FOR_DOC,
        expected_output=_EXPECTED_OUTPUT_SHAPE,
    )
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, past_key_value, dec_hidden, dec_attn)
        decoder_outputs = self.decoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if not return_dict:
            return decoder_outputs

        return BaseModelOutputWithPast(
            last_hidden_state=decoder_outputs.last_hidden_state,
            past_key_values=decoder_outputs.past_key_values,
            hidden_states=decoder_outputs.hidden_states,
            attentions=decoder_outputs.attentions,
        )


class OPTForCausalLM(OPTPreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = OPTModel(config)

        # the lm_head weight is automatically tied to the embed tokens weight
        self.lm_head = nn.Linear(config.word_embed_proj_dim, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.decoder.embed_tokens

    def set_input_embeddings(self, value):
        self.model.decoder.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model.decoder = decoder

    def get_decoder(self):
        return self.model.decoder

    @replace_return_docstrings(output_type=CausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        prefill_policy: Optional[int] = None,
        decoding_policy: Optional[int] = None,
        no_overlap: Optional[bool] = None,
        pin_weight: Optional[bool] = None,
        gpu_percentage: Optional[int] = None,
        num_minibatch: Optional[int] = None,
        enable_cxl: Optional[bool] = None,
        max_new_tokens: Optional[int] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you
                provide it.

                Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
                [`PreTrainedTokenizer.__call__`] for details.

                [What are input IDs?](../glossary#input-ids)
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                [What are attention masks?](../glossary#attention-mask)
            head_mask (`torch.Tensor` of shape `(num_hidden_layers, num_attention_heads)`, *optional*):
                Mask to nullify selected heads of the attention modules. Mask values selected in `[0, 1]`:

                - 1 indicates the head is **not masked**,
                - 0 indicates the head is **masked**.

            past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
                Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of
                shape `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and 2 additional tensors of
                shape `(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`. The two additional
                tensors are only required when the model is used as a decoder in a Sequence to Sequence model.

                Contains pre-computed hidden-states (key and values in the self-attention blocks and in the
                cross-attention blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.

                If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those
                that don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of
                all `decoder_input_ids` of shape `(batch_size, sequence_length)`.
            inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
                Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation.
                This is useful if you want more control over how to convert `input_ids` indices into associated vectors
                than the model's internal embedding lookup matrix.
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, OPTForCausalLM

        >>> model = OPTForCausalLM.from_pretrained("facebook/opt-350m")
        >>> tokenizer = AutoTokenizer.from_pretrained("facebook/opt-350m")

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious. I'm just a little bit of a weirdo."
        ```"""

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model.decoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            prefill_policy=prefill_policy,
            decoding_policy=decoding_policy,
            no_overlap=no_overlap,
            pin_weight=pin_weight,
            gpu_percentage=gpu_percentage,
            num_minibatch=num_minibatch,
            enable_cxl=enable_cxl,
            max_new_tokens=max_new_tokens,
        )

        logits = self.lm_head(outputs[0]).contiguous()

        loss = None
        if labels is not None:
            # move labels to correct device to enable model parallelism
            labels = labels.to(logits.device)
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        if past_key_values is not None:
            past_length = past_key_values[0][0].shape[2]

            # Some generation methods already pass only the last input ID
            if input_ids.shape[1] > past_length:
                remove_prefix_length = past_length
            else:
                # Default to old behavior: keep only final ID
                remove_prefix_length = input_ids.shape[1] - 1

            input_ids = input_ids[:, remove_prefix_length:]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
            }
        )
        return model_inputs

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past),
            )
        return reordered_past


@add_start_docstrings(
    """
    The OPT Model transformer with a sequence classification head on top (linear layer).

    [`OPTForSequenceClassification`] uses the last token in order to do the classification, as other causal models
    (e.g. GPT-2) do.

    Since it does classification on the last token, it requires to know the position of the last token. If a
    `pad_token_id` is defined in the configuration, it finds the last token that is not a padding token in each row. If
    no `pad_token_id` is defined, it simply takes the last value in each row of the batch. Since it cannot guess the
    padding tokens when `inputs_embeds` are passed instead of `input_ids`, it does the same (take the last value in
    each row of the batch).
    """,
    OPT_START_DOCSTRING,
)
class OPTForSequenceClassification(OPTPreTrainedModel):
    def __init__(self, config: OPTConfig):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.model = OPTModel(config)
        self.score = nn.Linear(config.word_embed_proj_dim, self.num_labels, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    @add_start_docstrings_to_model_forward(OPT_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_SEQUENCE_CLASSIFICATION,
        output_type=SequenceClassifierOutputWithPast,
        config_class=_CONFIG_FOR_DOC,
        expected_output=_SEQ_CLASS_EXPECTED_OUTPUT,
        expected_loss=_SEQ_CLASS_EXPECTED_LOSS,
    )
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, SequenceClassifierOutputWithPast]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.model(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]
        logits = self.score(hidden_states)

        if input_ids is not None:
            batch_size, sequence_length = input_ids.shape[:2]
        else:
            batch_size, sequence_length = inputs_embeds.shape[:2]

        if self.config.pad_token_id is None:
            sequence_lengths = -1
        else:
            if input_ids is not None:
                # if no pad token found, use modulo instead of reverse indexing for ONNX compatibility
                sequence_lengths = torch.eq(input_ids, self.config.pad_token_id).int().argmax(-1) - 1
                sequence_lengths = sequence_lengths % input_ids.shape[-1]
                sequence_lengths = sequence_lengths.to(logits.device)
            else:
                sequence_lengths = -1
                logger.warning(
                    f"{self.__class__.__name__} will not detect padding tokens in `inputs_embeds`. Results may be "
                    "unexpected if using padding tokens in conjunction with `inputs_embeds.`"
                )

        pooled_logits = logits[torch.arange(batch_size, device=logits.device), sequence_lengths]

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(pooled_logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(pooled_logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(pooled_logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(pooled_logits, labels)
        if not return_dict:
            output = (pooled_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutputWithPast(
            loss=loss,
            logits=pooled_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )

    def get_input_embeddings(self):
        return self.model.decoder.embed_tokens

    def set_input_embeddings(self, value):
        self.model.decoder.embed_tokens = value


@add_start_docstrings(
    """
    The OPT Model transformer with a span classification head on top for extractive question-answering tasks like SQuAD
    (a linear layers on top of the hidden-states output to compute `span start logits` and `span end logits`).
    """,
    OPT_START_DOCSTRING,
)
class OPTForQuestionAnswering(OPTPreTrainedModel):
    def __init__(self, config: OPTConfig):
        super().__init__(config)
        self.model = OPTModel(config)
        self.qa_outputs = nn.Linear(config.word_embed_proj_dim, 2)

        # Initialize weights and apply final processing
        self.post_init()

    @add_start_docstrings_to_model_forward(OPT_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=QuestionAnsweringModelOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        start_positions: Optional[torch.LongTensor] = None,
        end_positions: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, QuestionAnsweringModelOutput]:
        r"""
        start_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the start of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.
        end_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the end of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, OPTForQuestionAnswering
        >>> import torch

        >>> torch.manual_seed(4)  # doctest: +IGNORE_RESULT
        >>> tokenizer = AutoTokenizer.from_pretrained("facebook/opt-350m")

        >>> # note: we are loading a OPTForQuestionAnswering from the hub here,
        >>> # so the head will be randomly initialized, hence the predictions will be random
        >>> model = OPTForQuestionAnswering.from_pretrained("facebook/opt-350m")

        >>> question, text = "Who was Jim Henson?", "Jim Henson was a nice puppet"

        >>> inputs = tokenizer(question, text, return_tensors="pt")
        >>> with torch.no_grad():
        ...     outputs = model(**inputs)

        >>> answer_start_index = outputs.start_logits.argmax()
        >>> answer_end_index = outputs.end_logits.argmax()

        >>> answer_offset = len(tokenizer(question)[0])

        >>> predict_answer_tokens = inputs.input_ids[
        ...     0, answer_offset + answer_start_index : answer_offset + answer_end_index + 1
        ... ]
        >>> predicted = tokenizer.decode(predict_answer_tokens)
        >>> predicted
        ' a nice puppet'
        ```"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.model(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]

        logits = self.qa_outputs(hidden_states)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()

        total_loss = None
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions = start_positions.clamp(0, ignored_index)
            end_positions = end_positions.clamp(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

        if not return_dict:
            output = (start_logits, end_logits) + transformer_outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output

        return QuestionAnsweringModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )

    def get_input_embeddings(self):
        return self.model.decoder.embed_tokens

    def set_input_embeddings(self, value):
        self.model.decoder.embed_tokens = value

