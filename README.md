# ncon-torch
A PyTorch implementation of `ncon` (Network CONtractor).


```
========================================================================
ncon_torch vs torch.einsum — Performance & Correctness Benchmark
========================================================================
Torch version: 2.7.1+cu128
CUDA available: True | Device: NVIDIA GeForce RTX 4080 | CC: 8.9
------------------------------------------------------------------------

CPU Results
----------------------------------------------------------------------------------------------------------
Case                        Device  OK     ncon_torch (ms)   einsum (ms)    speedup(ein/ncon_torch)  loops
----------------------------------------------------------------------------------------------------------
matmul_2                    cpu     ✓                0.013         0.013                      1.048    500
three_chain                 cpu     ✓                0.029         0.087                      0.329    350
partial_trace_scalar        cpu     ✓                0.006         0.003                      1.666    500
trace_then_chain            cpu     ✓                0.012         0.008                      1.535    350
outer_product               cpu     ✓                0.006         0.004                      1.337    500
four_tensor_interlock       cpu     ✓                0.109         0.244                      0.446    240
long_chain_5                cpu     ✓                0.050         0.216                      0.231    240
----------------------------------------------------------------------------------------------------------

GPU Results
----------------------------------------------------------------------------------------------------------
Case                        Device  OK     ncon_torch (ms)   einsum (ms)    speedup(ein/ncon_torch)  loops
----------------------------------------------------------------------------------------------------------
matmul_2                    cuda    ✓                0.019         0.014                      1.310    350
three_chain                 cuda    ✓                0.030         0.063                      0.467    240
partial_trace_scalar        cuda    ✓                0.010         0.007                      1.291    350
trace_then_chain            cuda    ✓                0.022         0.017                      1.289    240
outer_product               cuda    ✓                0.009         0.008                      1.210    350
four_tensor_interlock       cuda    ✓                0.056         0.155                      0.362    160
long_chain_5                cuda    ✓                0.056         0.219                      0.254    160
----------------------------------------------------------------------------------------------------------

Summary
------------------------------------------------------------------------
CPU    | Cases:  7 | All-correct:  7 | Avg speedup (einsum/ncon_torch): 0.942
GPU    | Cases:  7 | All-correct:  7 | Avg speedup (einsum/ncon_torch): 0.883
------------------------------------------------------------------------
Notes:
1) Label dimensions are sampled in [32, 128]; results are checked with rtol=1e-4, atol=5e-4.
2) Iteration counts are increased vs. the previous version to reduce timing noise.
3) speedup(einsum/ncon_torch) > 1 means einsum is faster; < 1 means ncon_torch is faster.
========================================================================
```