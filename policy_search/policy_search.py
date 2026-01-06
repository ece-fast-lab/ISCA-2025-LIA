"""
Policy Search
"""

import itertools
import argparse
from typing import Sequence

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, ScalarFormatter

CPU_CONFIG = {
    "SPR": {
        "cpu_mem_bw": 265e9,
        "cpu_gemm_th": 20e12,
    },
    "GNR": {
        "cpu_mem_bw": 490e9,
        "cpu_gemm_th": 41e12,
    },
}

GPU_CONFIG = {
    "A100": {
        "pcie_bw": 28e9,
        "gpu_mem_bw": 1555e9,
        "gpu_gemm_th": 230e12,
    },
    "H100": {
        "pcie_bw": 55e9,
        "gpu_mem_bw": 3350e9,
        "gpu_gemm_th": 451e12,
    },
}

MODEL_CONFIG = {
    "OPT-30B": 7168,
    "OPT-66B": 9216,
    "OPT-175B": 12288,
}

def layer_shapes_bytes(bsz: int, seq_len: int, cur_len: int, d_model: int) -> tuple[tuple[float, ...], tuple[float, ...], tuple[float, ...], float, float]:
    """
    Returns:
      weight_bytes: 6-tuple
      act_bytes:    6-tuple
      compute_ops:  6-tuple
      cache_bytes:  scalar
      act_ref_bytes:scalar (bsz*cur_len*d_model in bytes)
    """
    # element counts
    weight = (
        3 * d_model**2,                 # QKV proj
        bsz * seq_len * d_model,        # K cache (modeled as "weight")
        bsz * seq_len * d_model,        # V cache (modeled as "weight)
        d_model**2,                     # out proj
        4 * d_model**2,                 # FFN up
        4 * d_model**2,                 # FFN down
    )

    act_ref = bsz * cur_len * d_model
    act = (
        act_ref, act_ref, act_ref, act_ref, act_ref,
        4 * act_ref,
    )

    compute_ops = (
        bsz * cur_len * d_model * 3 * d_model,         # QKV GEMM
        bsz * cur_len * seq_len * d_model,             # attn score
        bsz * cur_len * seq_len * d_model,             # attn apply
        bsz * cur_len * d_model * d_model,             # out proj GEMM
        bsz * cur_len * 4 * d_model * d_model,         # FFN up GEMM
        bsz * cur_len * 4 * d_model * d_model,         # FFN down GEMM
    )

    # bytes (bf16 => 2 bytes/elem) – preserve original behavior
    weight_b = tuple(float(x * 2) for x in weight)
    act_b = tuple(float(x * 2) for x in act)
    compute_ops = tuple(float(x * 2) for x in compute_ops)
    cache_b = float((bsz * cur_len * d_model) * 2)
    act_ref_b = float(act_ref * 2)

    return weight_b, act_b, compute_ops, cache_b, act_ref_b


def compute_latency(
    ops: float,
    size_1: float,
    size_2: float,
    cpu_gemm_th: float,
    cpu_mem_bw: float,
    gpu_gemm_th: float,
    gpu_mem_bw: float,
    policy: int,
    *,
    is_kv_cache: bool = False,
) -> float:
    """
    Matches original `compute()` behavior.
    - policy == 1 => run on GPU
    - policy == 0 => run on CPU (AMX)
    """
    # KV cache path has extra memory-bound overhead model
    if is_kv_cache:
        if policy == 1:
            return ops / gpu_gemm_th + (size_1 + 3.5 * size_2) / gpu_mem_bw
        else:
            return ops / cpu_gemm_th + (size_1 + 3.5 * size_2) / cpu_mem_bw

    # Non-KV
    if policy == 1:
        return ops / gpu_gemm_th + (size_1 + size_2) / gpu_mem_bw
    return ops / cpu_gemm_th + (size_1 + size_2) / cpu_mem_bw


def xfer_time(size: float, bw: float, do_xfer: bool) -> float:
    return (size / bw) if do_xfer else 0.0


def ctog_w(size: float, pcie_bw: float, policy: int, *, is_kv_cache: bool, is_prefill: bool) -> float:
    # CPU->GPU weights: KV only on prefill, and only if policy==1 (same as original)
    if is_kv_cache:
        return xfer_time(size, pcie_bw, do_xfer=(is_prefill and policy == 1))
    return xfer_time(size, pcie_bw, do_xfer=(policy == 1))


def gtoc_w(size: float, pcie_bw: float, policy: int, prev_policy: int, *, is_kv_cache: bool, is_prefill: bool) -> float:
    # GPU->CPU weights: KV only on prefill when switching 1->0 (same as original)
    if is_kv_cache:
        return xfer_time(size, pcie_bw, do_xfer=(is_prefill and policy == 0 and prev_policy == 1))
    return 0.0


def ctog_a(size: float, pcie_bw: float, policy: int, prev_policy: int) -> float:
    return xfer_time(size, pcie_bw, do_xfer=(policy == 1 and prev_policy == 0))


def gtoc_a(size: float, pcie_bw: float, policy: int, prev_policy: int) -> float:
    return xfer_time(size, pcie_bw, do_xfer=(policy == 0 and prev_policy == 1))


def cache_transfer(size: float, pcie_bw: float, policy: int) -> float:
    return xfer_time(size, pcie_bw, do_xfer=(policy == 1))


def out_transfer(size: float, pcie_bw: float, policy: int) -> float:
    return xfer_time(size, pcie_bw, do_xfer=(policy == 1))


# ----------------------------
# Public API: per-layer latency
# ----------------------------

def layer_latency_gpu(
    gpu_mem_bw: float,
    gpu_gemm_th: float,
    bsz: int,
    seq_len: int,
    cur_len: int,
    d_model: int,
    is_prefill: bool,
) -> float:
    weight_b, act_b, ops, _, _ = layer_shapes_bytes(bsz, seq_len, cur_len, d_model)

    total = 0.0
    for i in range(6):
        is_kv = (i in (1, 2))
        total += compute_latency(
            ops[i], act_b[i], weight_b[i],
            cpu_gemm_th=0.0, cpu_mem_bw=0.0,
            gpu_gemm_th=gpu_gemm_th, gpu_mem_bw=gpu_mem_bw,
            policy=1,
            is_kv_cache=is_kv,
        )
    return total


def layer_latency(
    pcie_bw: float,
    cpu_mem_bw: float,
    gpu_mem_bw: float,
    cpu_gemm_th: float,
    gpu_gemm_th: float,
    bsz: int,
    seq_len: int,
    cur_len: int,
    d_model: int,
    vector: Sequence[int],
    is_prefill: bool,
) -> float:
    """
    Preserves original semantics:
      vector is 6-tuple of 0/1 per sub-op
    """
    weight_b, act_b, ops, _, act_ref_b = layer_shapes_bytes(bsz, seq_len, cur_len, d_model)

    def prev_pol(i: int) -> int:
        return 0 if i == 0 else vector[i - 1]

    def is_kv(i: int) -> bool:
        return i in (1, 2)

    total = 0.0
    for i in range(6):
        v = vector[i]
        vp = prev_pol(i)

        stage = 0.0

        # weight transfers
        if is_kv(i):
            stage += ctog_w(weight_b[i], pcie_bw, v, is_kv_cache=True, is_prefill=is_prefill)
            stage += gtoc_w(weight_b[i], pcie_bw, v, vector[0], is_kv_cache=True, is_prefill=is_prefill)
        else:
            stage += ctog_w(weight_b[i], pcie_bw, v, is_kv_cache=False, is_prefill=is_prefill)

        # activation transfers
        if i == 0:
            stage += ctog_a(act_b[i], pcie_bw, v, 0)
        else:
            stage += ctog_a(act_b[i], pcie_bw, v, vp)
            stage += gtoc_a(act_b[i], pcie_bw, v, vp)

        # compute
        stage += compute_latency(
            ops[i], act_b[i], weight_b[i],
            cpu_gemm_th=cpu_gemm_th, cpu_mem_bw=cpu_mem_bw,
            gpu_gemm_th=gpu_gemm_th, gpu_mem_bw=gpu_mem_bw,
            policy=v,
            is_kv_cache=is_kv(i),
        )

        total += stage

    # extra transfers (same as original)
    total += cache_transfer(weight_b[1], pcie_bw, vector[1]) + cache_transfer(weight_b[2], pcie_bw, vector[2])
    total += out_transfer(act_ref_b, pcie_bw, vector[5])

    if vector[0] ^ vector[3]:
        total += act_ref_b / pcie_bw
    if vector[3] ^ vector[5]:
        total += act_ref_b / pcie_bw

    return total

def collect_policy_points(
    *,
    is_prefill: bool,
    bsz_range,
    len_range,
    d_model,
    pcie_bw,
    cpu_mem_bw,
    gpu_mem_bw,
    cpu_gemm_th,
    gpu_gemm_th,
):
    policy_points = {
        (0, 0, 0, 0, 0, 0): ([], [], "Policy (0,0,0,0,0,0)", "blue"),
        (1, 1, 1, 1, 1, 1): ([], [], "Policy (1,1,1,1,1,1)", "green"),
        (1, 0, 0, 1, 1, 1): ([], [], "Policy (1,0,0,1,1,1)", "red"),
        (0, 0, 0, 1, 1, 1): ([], [], "Policy (0,0,0,1,1,1)", "yellow"),
        (0, 0, 0, 0, 1, 1): ([], [], "Policy (0,0,0,0,1,1)", "black"),
    }

    binary_vectors = list(itertools.product([0, 1], repeat=6))

    for b in bsz_range:
        for L in len_range:
            cur_len = L if is_prefill else 1

            best_lat = float("inf")
            best_pol = None

            for vec in binary_vectors:
                lat = layer_latency(
                    pcie_bw, cpu_mem_bw, gpu_mem_bw,
                    cpu_gemm_th, gpu_gemm_th,
                    b, L, cur_len, d_model,
                    vec, is_prefill
                )
                if lat < best_lat:
                    best_lat = lat
                    best_pol = vec

            if best_pol in policy_points:
                xs, ys, _, _ = policy_points[best_pol]
                xs.append(b)
                ys.append(L)

    return policy_points

def main(cpu_type: str, gpu_type: str, model_name: str):
    cpu_type = cpu_type.upper()
    gpu_type = gpu_type.upper()
    model_name = model_name.upper()

    if cpu_type not in CPU_CONFIG:
        raise ValueError(f"cpu_type must be one of {list(CPU_CONFIG)}")
    if gpu_type not in GPU_CONFIG:
        raise ValueError(f"gpu_type must be one of {list(GPU_CONFIG)}")
    if model_name not in MODEL_CONFIG:
        raise ValueError(f"model_name must be one of {list(MODEL_CONFIG)}")

    d_model = MODEL_CONFIG[model_name]

    # CPU params
    cpu_mem_bw = CPU_CONFIG[cpu_type]["cpu_mem_bw"]
    cpu_gemm_th = CPU_CONFIG[cpu_type]["cpu_gemm_th"]

    # GPU params
    pcie_bw = GPU_CONFIG[gpu_type]["pcie_bw"]
    gpu_mem_bw = GPU_CONFIG[gpu_type]["gpu_mem_bw"]
    gpu_gemm_th = GPU_CONFIG[gpu_type]["gpu_gemm_th"]

    bsz_range = np.power(2, np.arange(0, 13))
    len_range = np.power(2, np.arange(4, 13))

    plt.rc("font", family="Helvetica")
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharex=True, sharey=True)

    for ax, is_prefill, title in zip(
        axes,
        [True, False],
        ["Prefill", "Decode"],
    ):
        policy_points = collect_policy_points(
            is_prefill=is_prefill,
            bsz_range=bsz_range,
            len_range=len_range,
            d_model=d_model,
            pcie_bw=pcie_bw,
            cpu_mem_bw=cpu_mem_bw,
            gpu_mem_bw=gpu_mem_bw,
            cpu_gemm_th=cpu_gemm_th,
            gpu_gemm_th=gpu_gemm_th,
        )

        for _, (xs, ys, label, color) in policy_points.items():
            ax.scatter(xs, ys, color=color, marker="s", s=300, label=label)

        ax.set_xscale("log", base=2)
        ax.set_yscale("log", base=2)

        ax.xaxis.set_major_locator(LogLocator(base=4))
        ax.yaxis.set_major_locator(LogLocator(base=4))
        ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.yaxis.set_major_formatter(ScalarFormatter())

        ax.set_title(title, fontsize=22)
        ax.grid(True, which="both", linestyle="--", alpha=0.5)
        ax.tick_params(labelsize=16)

    axes[0].set_ylabel("Sequence Length (L)", fontsize=20)
    for ax in axes:
        ax.set_xlabel("Batch Size (B)", fontsize=20)

    axes[1].legend(fontsize=12, loc="upper right")
    fig.suptitle(f"{model_name} on {cpu_type} + {gpu_type}", fontsize=24)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Latency model")
    parser.add_argument("--cpu", choices=["SPR", "GNR"], default="SPR")
    parser.add_argument("--gpu", choices=["A100", "H100"], default="A100")
    parser.add_argument(
        "--model",
        choices=["OPT-30B", "OPT-66B", "OPT-175B"],
        default="OPT-30B",
        help="Model size"
    )

    args = parser.parse_args()
    main(cpu_type=args.cpu, gpu_type=args.gpu, model_name=args.model)
