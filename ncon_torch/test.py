# tests/test_ncon.py
"""
Comprehensive pytest suite for a PyTorch ncon_numpy implementation.

Coverage highlights
-------------------
- Correctness: matrix multiply, multi-tensor chains, and complex patterns vs torch.einsum
- Output axis ordering: free (negative) labels appear as (-1, -2, -3, …)
- Partial trace on a single tensor (both vector and scalar results), including dim mismatch errors
- Outer products for disconnected components, including scalar × scalar
- Full contraction to scalar (no free indices)
- Custom contraction order (list) vs default equivalence; invalid orders raise
- Strict validation: rank/label mismatches, negative label uniqueness/contiguity, positive label counts and dim matches
- Dtype preservation and autograd gradients
- Contiguity flag for final output
- Robust edge cases: unit dims, scalar mixing, duplicated label >2 on same tensor
"""

import pytest
import torch

from .ncon_torch import ncon_torch


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(1234)
    yield


# ---------------------------
# Basic functionality
# ---------------------------
def test_matmul_basic():
    m, k, n = 5, 7, 3
    A = torch.randn(m, k)
    B = torch.randn(k, n)
    # A: [-1, 1], B: [1, -2]  =>  A @ B
    C = ncon_torch([A, B], [[-1, 1], [1, -2]])
    torch.testing.assert_close(C, A @ B)


def test_three_tensor_chain():
    # A(i,a), B(a,b), C(b,j) -> M(i,j)
    i, a, b, j = 4, 3, 2, 5
    A = torch.randn(i, a)
    B = torch.randn(a, b)
    C = torch.randn(b, j)
    out = ncon_torch([A, B, C], [[-1, 1], [1, 2], [2, -2]])
    ref = torch.einsum("ia,ab,bj->ij", A, B, C)
    torch.testing.assert_close(out, ref)


# ---------------------------
# Free-axis ordering (-1, -2, -3, …) and contiguity flag
# ---------------------------
def test_negative_label_ordering_permutation():
    X = torch.randn(2, 3, 4)  # labels: [-3, -1, -2] -> output axes should be (-1, -2, -3)
    Y = ncon_torch([X], [[-3, -1, -2]])
    # Expected permutation to (-1, -2, -3) corresponds to original indices (1, 2, 0)
    ref = X.permute(1, 2, 0)
    torch.testing.assert_close(Y, ref)


def test_make_contiguous_output_flag():
    X = torch.randn(2, 3, 4)
    Y1 = ncon_torch([X], [[-3, -1, -2]], make_contiguous_output=False)
    Y2 = ncon_torch([X], [[-3, -1, -2]], make_contiguous_output=True)
    assert Y2.is_contiguous()
    if not Y1.is_contiguous():
        assert not Y1.is_contiguous()


# ---------------------------
# Partial trace on a single tensor
# ---------------------------
def test_partial_trace_vector():
    d, k = 6, 5
    A = torch.randn(d, d, k)
    # labels [1,1,-1] -> trace over first two dims, keep the last
    out = ncon_torch([A], [[1, 1, -1]])
    ref = torch.einsum("iik->k", A)
    torch.testing.assert_close(out, ref)


def test_partial_trace_scalar():
    d = 7
    A = torch.randn(d, d)
    out = ncon_torch([A], [[1, 1]])
    # Should be scalar equal to trace(A)
    ref = torch.trace(A)
    assert out.ndim == 0
    torch.testing.assert_close(out, ref)


def test_partial_trace_dim_mismatch_raises():
    A = torch.randn(2, 3)
    # Fails in _check_inputs first
    with pytest.raises(ValueError, match=r"tensor dimension mismatch on index labelled 1"):
        _ = ncon_torch([A], [[1, 1]])


def test_partial_trace_label_appears_more_than_twice_on_same_tensor_raises():
    A = torch.randn(2, 2, 2)
    # Fails in _check_inputs first
    with pytest.raises(ValueError, match=r"more than two indices labelled 1"):
        _ = ncon_torch([A], [[1, 1, 1]])


# ---------------------------
# Outer products for disconnected components
# ---------------------------
def test_outer_product_vectors():
    a, b = 4, 3
    x = torch.randn(a)
    y = torch.randn(b)
    # x: [-1], y: [-2] -> outer product
    out = ncon_torch([x, y], [[-1], [-2]])
    ref = torch.outer(x, y)
    torch.testing.assert_close(out, ref)


def test_outer_product_scalars():
    # Two scalars -> still scalar (product)
    x = torch.randn(())
    y = torch.randn(())
    out = ncon_torch([x, y], [[], []])
    ref = (x * y).reshape(())
    assert out.ndim == 0
    torch.testing.assert_close(out, ref)


# ---------------------------
# Full contraction (no free indices) -> scalar
# ---------------------------
def test_full_contraction_scalar():
    i, j = 3, 4
    A = torch.randn(i, j)
    B = torch.randn(i, j)
    # labels: [1,2], [1,2] -> sum_ij A_ij * B_ij
    out = ncon_torch([A, B], [[1, 2], [1, 2]])
    ref = (A * B).sum().reshape(())
    assert out.ndim == 0
    torch.testing.assert_close(out, ref)


# ---------------------------
# Contraction order: custom order vs default; invalid orders
# ---------------------------
def test_custom_contraction_order_equivalence():
    i, a, b, j = 2, 3, 4, 5
    A = torch.randn(i, a)
    B = torch.randn(a, b)
    C = torch.randn(b, j)
    # Positive labels are {1,2}; default [1,2]; test reversed [2,1]
    out_default = ncon_torch([A, B, C], [[-1, 1], [1, 2], [2, -2]], con_order=None)
    out_reverse = ncon_torch([A, B, C], [[-1, 1], [1, 2], [2, -2]], con_order=[2, 1])
    torch.testing.assert_close(out_default, out_reverse)


def test_invalid_contraction_order_string_raises():
    A = torch.randn(2, 2)
    B = torch.randn(2, 2)
    # Passing a string order should raise when check_network=True
    with pytest.raises(ValueError, match="invalid contraction order"):
        _ = ncon_torch([A, B], [[-1, 1], [1, -2]], con_order="greedy", check_network=True)


def test_contraction_order_missing_label_raises():
    A = torch.randn(2, 3)
    B = torch.randn(3, 4)
    # Positive labels are {1}; providing [] should raise
    with pytest.raises(ValueError, match="invalid contraction order"):
        _ = ncon_torch([A, B], [[-1, 1], [1, -2]], con_order=[], check_network=True)


# ---------------------------
# Strict validation of labels/dims
# ---------------------------
def test_rank_label_mismatch_raises():
    A = torch.randn(2, 2)
    # Label list length != tensor rank
    with pytest.raises(ValueError, match="number of indices does not match number of labels"):
        _ = ncon_torch([A], [[-1, 1, 2]])


def test_negative_label_duplicates_raises():
    A = torch.randn(2, 2)
    with pytest.raises(ValueError, match="more than one index labelled -1"):
        _ = ncon_torch([A], [[-1, -1]])


def test_negative_label_gap_raises():
    # Have -1 and -3 but missing -2
    A = torch.randn(3)
    B = torch.randn(4)
    with pytest.raises(ValueError, match="no index labelled -2"):
        _ = ncon_torch([A, B], [[-1], [-3]])


def test_positive_label_only_once_raises():
    A = torch.randn(5)
    B = torch.randn(6)
    with pytest.raises(ValueError, match="only one index labelled 1"):
        _ = ncon_torch([A, B], [[1], [-1]])


def test_positive_label_more_than_twice_raises():
    a = torch.randn(3)
    b = torch.randn(3)
    c = torch.randn(3)
    with pytest.raises(ValueError, match="more than two indices labelled 1"):
        _ = ncon_torch([a, b, c], [[1], [1], [1]])


def test_positive_label_dim_mismatch_raises():
    A = torch.randn(2, 3)
    B = torch.randn(4, 5)
    # Connect A dim-0 (size 2) with B dim-0 (size 4) on label 1 -> mismatch
    with pytest.raises(ValueError, match="tensor dimension mismatch on index labelled 1"):
        _ = ncon_torch([A, B], [[1, -1], [1, -2]])


# ---------------------------
# Dtype preservation (CPU)
# ---------------------------
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_dtype_preservation(dtype):
    A = torch.randn(4, 6, dtype=dtype)
    B = torch.randn(6, 3, dtype=dtype)
    out = ncon_torch([A, B], [[-1, 1], [1, -2]])
    assert out.dtype == dtype


# ---------------------------
# Autograd: gradients flow through ncon_torch
# ---------------------------
def test_autograd_basic_matmul_sum():
    m, k, n = 4, 5, 6
    A = torch.randn(m, k, requires_grad=True)
    B = torch.randn(k, n, requires_grad=True)
    out = ncon_torch([A, B], [[-1, 1], [1, -2]]).sum()
    out.backward()

    A2 = A.detach().clone().requires_grad_(True)
    B2 = B.detach().clone().requires_grad_(True)
    ref = (A2 @ B2).sum()
    ref.backward()

    torch.testing.assert_close(A.grad, A2.grad)
    torch.testing.assert_close(B.grad, B2.grad)


def test_autograd_with_partial_trace():
    d, k = 4, 3
    A = torch.randn(d, d, k, requires_grad=True)
    y = ncon_torch([A], [[1, 1, -1]]).sum()
    y.backward()

    A2 = A.detach().clone().requires_grad_(True)
    ref = torch.einsum("iik->k", A2).sum()
    ref.backward()

    assert A.grad is not None
    torch.testing.assert_close(A.grad, A2.grad)


# ---------------------------
# Mixed: contraction + outer product
# ---------------------------
def test_mixed_contractions_and_outer_products():
    i, k, n = 3, 2, 4
    A = torch.randn(i, k)
    B = torch.randn(k, n)
    v = torch.randn(5)

    part = ncon_torch([A, B], [[-1, 1], [1, -2]])  # shape (i, n)
    out = ncon_torch([A, B, v], [[-1, 1], [1, -2], [-3]])
    ref = torch.einsum("in,m->inm", part, v)
    torch.testing.assert_close(out, ref)


# ---------------------------
# Edge cases
# ---------------------------
def test_scalar_network_mixed_with_tensor():
    s = torch.randn(())
    M = torch.randn(2, 3)
    out = ncon_torch([s, M], [[], [-1, -2]])
    ref = (s * M).contiguous()
    torch.testing.assert_close(out, ref)


def test_unit_dims_behavior():
    A = torch.randn(1, 4)
    B = torch.randn(4, 1)
    out = ncon_torch([A, B], [[-1, 1], [1, -2]])
    ref = A @ B
    torch.testing.assert_close(out, ref)


def test_complex_pattern_against_einsum():
    # T1: (p,q,r) -> [-1, 1, 2]
    # T2: (q,s)   -> [ 1, -2]
    # T3: (r,t,u) -> [ 2, -3, -4]
    p, q, r, s, t, u = 2, 3, 4, 5, 6, 7
    T1 = torch.randn(p, q, r)
    T2 = torch.randn(q, s)
    T3 = torch.randn(r, t, u)
    out = ncon_torch([T1, T2, T3], [[-1, 1, 2], [1, -2], [2, -3, -4]])
    ref = torch.einsum("pqr,qs,rtu->pstu", T1, T2, T3)
    torch.testing.assert_close(out, ref)
