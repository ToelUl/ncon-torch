# test_ncon_vs_einsum.py
# -*- coding: utf-8 -*-
import time
import math
import random
from typing import List, Tuple, Dict, Any

import torch

from ncon_torch import ncon_torch

# ===========================================================
# Utilities
# ===========================================================

LETTERS = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"


def make_label_dims(connects: List[List[int]],
                    dim_low: int = 32,
                    dim_high: int = 128,
                    rng: random.Random = random.Random(0)) -> Dict[int, int]:
    """Assign a dimension size to each (positive/negative) label appearing in `connects`.

    Positive labels share the same dimension across all occurrences.
    Negative labels are free indices; their sizes are assigned independently here.
    """
    all_labels = [lab for conn in connects for lab in conn]
    uniq = []
    seen = set()
    for x in all_labels:
        if x not in seen:
            uniq.append(x)
            seen.add(x)
    dims: Dict[int, int] = {}
    for lab in uniq:
        # Random dimension per unique label.
        dims[lab] = rng.randint(dim_low, dim_high)
    return dims


def build_tensors(connects: List[List[int]],
                  label_dims: Dict[int, int],
                  device: torch.device,
                  dtype: torch.dtype = torch.float32) -> List[torch.Tensor]:
    """Create random tensors consistent with `connects` and `label_dims`."""
    tensors: List[torch.Tensor] = []
    for conn in connects:
        shape = [label_dims[lab] for lab in conn]
        if len(shape) == 0:
            # Allow scalars (rare, but supported by outer-product logic).
            t = torch.randn((), device=device, dtype=dtype)
        else:
            t = torch.randn(*shape, device=device, dtype=dtype)
        tensors.append(t)
    return tensors


def ncon_to_einsum(connects: List[List[int]]) -> Tuple[str, Dict[int, str]]:
    """Convert integer labels (ncon_torch-style) to letter labels (einsum-style).

    - Positive labels (> 0): contracted/internal indices, typically appearing twice.
    - Negative labels (< 0): free indices, each appears once and define output order
      by descending order of (-1, -2, -3, ...).
    """
    # Ensure we have enough letters for distinct labels.
    if len({lab for conn in connects for lab in conn}) > len(LETTERS):
        raise RuntimeError("Too many distinct labels (>52) for einsum letters.")

    # Map labels to letters in first-appearance order.
    label2char: Dict[int, str] = {}
    next_idx = 0
    for conn in connects:
        for lab in conn:
            if lab not in label2char:
                label2char[lab] = LETTERS[next_idx]
                next_idx += 1

    # Output subscripts: negative labels sorted as -1, -2, -3, ...
    neg_all = sorted({lab for conn in connects for lab in conn if lab < 0}, reverse=True)
    out_sub = "".join(label2char[lab] for lab in neg_all)

    # Input subscripts for each tensor.
    in_subs = []
    for conn in connects:
        in_subs.append("".join(label2char[lab] for lab in conn))

    expr = ",".join(in_subs) + "->" + out_sub  # out_sub can be empty -> scalar result
    return expr, label2char


@torch.inference_mode()
def run_once_ncon(tensors: List[torch.Tensor], connects: List[List[int]]) -> torch.Tensor:
    out = ncon_torch(tensors, connects, check_network=True, make_contiguous_output=False)
    if not isinstance(out, torch.Tensor):
        # If ncon_torch returns a Python scalar, wrap it for comparison convenience.
        out = torch.tensor(out, device=tensors[0].device, dtype=tensors[0].dtype)
    return out


@torch.inference_mode()
def run_once_einsum(tensors: List[torch.Tensor], connects: List[List[int]]) -> torch.Tensor:
    expr, _ = ncon_to_einsum(connects)
    out = torch.einsum(expr, *tensors)
    return out


def benchmark(func, warmup: int, repeat: int, device: torch.device) -> float:
    """Return average runtime per call (milliseconds)."""
    if device.type == "cuda":
        torch.cuda.synchronize()
    # Warmup
    for _ in range(warmup):
        _ = func()
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(repeat):
        _ = func()
    if device.type == "cuda":
        torch.cuda.synchronize()
    t1 = time.perf_counter()
    return (t1 - t0) * 1000.0 / repeat


def choose_loops(connects: List[List[int]]) -> Tuple[int, int]:
    """Select (CPU_loops, GPU_loops) based on network complexity.

    Complexity is approximated by the number of distinct positive labels.
    Iteration counts are intentionally *increased* compared to the previous version.
    """
    pos_labels = sorted({lab for conn in connects for lab in conn if lab > 0})
    L = len(pos_labels)
    if L >= 6:
        return 120, 90      # previously ~30/20
    elif L >= 4:
        return 240, 160     # previously ~60/40
    elif L >= 2:
        return 350, 240     # previously ~100/60
    else:
        return 500, 350     # previously ~150/100


# ============================================================
# Test networks of varying complexity
# ============================================================

def build_test_suites() -> List[Dict[str, Any]]:
    """Define representative networks (from simple to complex)."""
    suites: List[Dict[str, Any]] = []

    # 1) Binary matmul (A_ij B_jk -> C_ik)
    suites.append({
        "name": "matmul_2",
        "connects": [[-1, 1], [1, -2]],
    })

    # 2) Three-tensor chain (A_ij B_jk C_kl -> D_il)
    suites.append({
        "name": "three_chain",
        "connects": [[-1, 1], [1, 2], [2, -2]],
    })

    # 3) Partial trace -> scalar (A_ii)
    suites.append({
        "name": "partial_trace_scalar",
        "connects": [[1, 1]],
    })

    # 4) Partial trace + chain: A_iij with B_jk -> out_k
    suites.append({
        "name": "trace_then_chain",
        "connects": [[1, 1, 2], [2, -1]],
    })

    # 5) Disconnected -> outer product: A_ab with B_c -> out_abc
    suites.append({
        "name": "outer_product",
        "connects": [[-1, -2], [-3]],
    })

    # 6) Four-tensor interlock (higher rank with branching)
    #   A(a,b,c), B(c,d,e), C(b,g,f), D(a,h,f) -> out(d,e,g,h)
    suites.append({
        "name": "four_tensor_interlock",
        "connects": [[1, 2, 3], [3, -1, -2], [2, -3, 4], [1, -4, 4]],
    })

    # 7) Five-matrix long chain: A_mk1 B_k1k2 C_k2k3 D_k3k4 E_k4n -> out_mn
    suites.append({
        "name": "long_chain_5",
        "connects": [[-1, 1], [1, 2], [2, 3], [3, 4], [4, -2]],
    })

    return suites


# ============================================================
# Runner
# ============================================================

def run_device(suites: List[Dict[str, Any]], device: torch.device) -> List[Dict[str, Any]]:
    rng = random.Random(0)
    results: List[Dict[str, Any]] = []

    for case in suites:
        name = case["name"]
        connects = case["connects"]

        # Assign dimensions per label (same distribution across cases).
        label_dims = make_label_dims(connects, dim_low=8, dim_high=24, rng=rng)
        tensors = build_tensors(connects, label_dims, device=device, dtype=torch.float32)

        # Correctness check first.
        out_ncon = run_once_ncon(tensors, connects)
        out_ein = run_once_einsum(tensors, connects)

        ok = torch.allclose(out_ncon, out_ein, rtol=1e-4, atol=5e-4)
        if not ok:
            max_abs = (out_ncon - out_ein).abs().max().item()
            denom = out_ein.abs().max().item() if out_ein.numel() > 0 else 1.0
            rel = max_abs / max(denom, 1e-12)
            print(f"[DEBUG] case={name} device={device} shape={tuple(out_ein.shape)} "
                  f"max_abs={max_abs:.3e} rel={rel:.3e}")
            results.append({
                "name": name,
                "device": str(device),
                "ok": False,
                "note": "Mismatch between ncon_torch and einsum results",
                "ncon_ms": float("nan"),
                "einsum_ms": float("nan"),
                "speedup_einsum_over_ncon": float("nan"),
            })
            continue

        # Choose iteration counts (now increased).
        loops_cpu, loops_gpu = choose_loops(connects)
        repeat = loops_gpu if device.type == "cuda" else loops_cpu
        warmup = max(3, repeat // 10)

        # Pre-bind callables to avoid recreation inside the timed loop.
        def _call_ncon():
            return ncon_torch(tensors, connects, check_network=False, make_contiguous_output=False)

        expr, _ = ncon_to_einsum(connects)

        def _call_einsum():
            return torch.einsum(expr, *tensors)

        ncon_ms = benchmark(_call_ncon, warmup=warmup, repeat=repeat, device=device)
        ein_ms = benchmark(_call_einsum, warmup=warmup, repeat=repeat, device=device)

        results.append({
            "name": name,
            "device": str(device),
            "ok": ok,
            "ncon_ms": ncon_ms,
            "einsum_ms": ein_ms,
            "speedup_einsum_over_ncon": (ncon_ms / ein_ms) if ein_ms > 0 else float("inf"),
            "loops": repeat,
            "warmup": warmup,
        })

    return results


def main():
    torch.manual_seed(0)

    print("=" * 72)
    print("ncon_torch vs torch.einsum — Performance & Correctness Benchmark")
    print("=" * 72)
    print(f"Torch version: {torch.__version__}")
    if torch.cuda.is_available():
        dev_name = torch.cuda.get_device_name(0)
        cc_major, cc_minor = torch.cuda.get_device_capability(0)
        print(f"CUDA available: True | Device: {dev_name} | CC: {cc_major}.{cc_minor}")
    else:
        print("CUDA available: False")
    print("-" * 72)

    suites = build_test_suites()

    # CPU tests
    cpu_device = torch.device("cpu")
    cpu_results = run_device(suites, cpu_device)

    # GPU tests (if available)
    gpu_results: List[Dict[str, Any]] = []
    if torch.cuda.is_available():
        gpu_device = torch.device("cuda")
        gpu_results = run_device(suites, gpu_device)

    # Pretty printer
    def print_block(title: str, rows: List[Dict[str, Any]]):
        print(f"\n{title}")

        # Adjusted widths to be large enough for the header text.
        # Using explicit alignment (< for left, > for right) for clarity.
        header = (f"{'Case':<28}"
                  f"{'Device':<8}"
                  f"{'OK':<4}"
                  f"{'ncon_torch (ms)':>18}"
                  f"{'einsum (ms)':>14}"
                  f"{'speedup(ein/ncon_torch)':>27}"
                  f"{'loops':>7}")

        # Make the divider line dynamically match the header length for robustness.
        line_len = len(header)
        print("-" * line_len)
        print(header)
        print("-" * line_len)

        for r in rows:
            name = r["name"]
            device = r["device"]
            ok = "✓" if r["ok"] else "✗"
            ncon_ms = r["ncon_ms"]
            ein_ms = r["einsum_ms"]
            spd = r["speedup_einsum_over_ncon"]
            loops = r.get("loops", 0)

            if math.isnan(ncon_ms):
                # Also adjust the 'n/a' strings to match the new widths.
                ncon_s = f"{'n/a':>18}"
                ein_s = f"{'n/a':>14}"
                spd_s = f"{'n/a':>27}"
            else:
                # Adjust the data formatting to match the new header widths.
                ncon_s = f"{ncon_ms:18.3f}"
                ein_s = f"{ein_ms:14.3f}"
                spd_s = f"{spd:27.3f}"

            # The final print statement combines the correctly sized components.
            # Note: name, device, and ok are formatted here, while the numeric
            # strings are pre-formatted.
            print(f"{name:<28}"
                  f"{device:<8}"
                  f"{ok:<4}"
                  f"{ncon_s}"
                  f"{ein_s}"
                  f"{spd_s}"
                  f"{loops:7d}")
        print("-" * line_len)

    print_block("CPU Results", cpu_results)
    if gpu_results:
        print_block("GPU Results", gpu_results)

    # Summary
    def summarize(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
        oks = sum(1 for r in rows if r["ok"])
        Ns = len(rows)
        denom = max(1, sum(1 for r in rows if r["ok"]))
        avg_speedup = sum(r["speedup_einsum_over_ncon"] for r in rows if r["ok"]) / denom
        return {"count": Ns, "ok": oks, "avg_speedup_ein_over_ncon": avg_speedup}

    print("\nSummary")
    print("-" * 72)
    cpu_sum = summarize(cpu_results)
    print(f"CPU    | Cases: {cpu_sum['count']:2d} | All-correct: {cpu_sum['ok']:2d} | "
          f"Avg speedup (einsum/ncon_torch): {cpu_sum['avg_speedup_ein_over_ncon']:.3f}")

    if gpu_results:
        gpu_sum = summarize(gpu_results)
        print(f"GPU    | Cases: {gpu_sum['count']:2d} | All-correct: {gpu_sum['ok']:2d} | "
              f"Avg speedup (einsum/ncon_torch): {gpu_sum['avg_speedup_ein_over_ncon']:.3f}")
    else:
        print("GPU    | Not available")

    print("-" * 72)
    print("Notes:")
    print("1) Label dimensions are sampled in [32, 128]; results are checked with rtol=1e-4, atol=5e-4.")
    print("2) Iteration counts are increased vs. the previous version to reduce timing noise.")
    print("3) speedup(einsum/ncon_torch) > 1 means einsum is faster; < 1 means ncon_torch is faster.")
    print("=" * 72)


if __name__ == "__main__":
    main()
