from typing import List, Union, Tuple, Optional
import math
import torch

def ncon_torch(
        tensors: List[torch.Tensor],
        connects: List[Union[List[int], Tuple[int]]],
        con_order: Optional[Union[List[int], str]] = None,
        check_network: bool = True,
        make_contiguous_output: bool = False,
):
    """Contract a tensor network (PyTorch backend).

    This is a faithful PyTorch rewrite of the `ncon` (arXiv:1402.0939) algorithm:
    it contracts a list of tensors according to index labels, performing
    (N-1) binary contractions and any necessary partial traces. Positive
    labels denote *internal* (contracted) indices and negative labels denote
    *free* (open/output) indices. The final tensor is permuted so that free
    indices appear in descending order of their (negative) labels
    `-1, -2, -3, ...`.

    Args:
      tensors: List of tensors comprising the network. Each `tensors[i]`
        must be a `torch.Tensor` with rank equal to `len(connects[i])`.
      connects: List of index-label lists/tuples. `connects[i][j]` is the
        integer label attached to dimension `j` of `tensors[i]`.
        Positive labels (>0) appear exactly twice across the whole network
        (unless eliminated by a same-tensor partial trace). Negative labels
        (<0) appear at most once and determine the order of output axes.
      con_order: Optional contraction order for positive labels. If `None`,
        the order defaults to the ascending positive labels (i.e., `sorted(set)`).
        If a list is provided,it must contain each positive label exactly once.
        Supplying a string is reserved for higher-level "greedy"/"full" planners
        and is not interpreted here; with `check_network=True` this will raise an
        "invalid contraction order" error.
      check_network: If `True`, run consistency checks on the input network
        (label multiplicities, dimensionality matches, validity of
        `con_order`, etc.). Disable for best performance once inputs are
        known to be valid.
      make_contiguous_output: If True, call ``.contiguous()`` after final permutation.

    Returns:
      torch.Tensor | float: If there are free (negative) indices, returns a
        `torch.Tensor` whose axes are sorted by decreasing negative label
        (i.e., `-1, -2, -3, ...`). If there are no free indices, returns a
        Python scalar via `.reshape(())`.

    Raises:
      ValueError: If the network is malformed (e.g., rank/label mismatch,
        invalid contraction order, repeated positive labels on more than two
        axes, dimension mismatches, etc.).

    Notes:
      * A *partial trace* is performed whenever the same positive label
        appears twice on the **same** tensor; those axes are traced out
        before binary contractions.
      * Binary contractions are performed via `torch.tensordot`, with an
        index-ordering heuristic that follows the original implementation.
      * If multiple disconnected components remain after exhausting positive
        labels, they are combined by outer products (robust for scalar inputs).

    Example:
      >>> A = torch.randn(5, 7)
      >>> B = torch.randn(7, 3)
      >>> C = ncon_torch([A, B], [[-1, 1], [1, -2]])  # matrix multiply A @ B
      >>> C.shape
      torch.Size([5, 3])
    """
    # Copy references (avoid mutating user inputs) and normalize labels to int.
    T = [t for t in tensors]
    C = [list(map(int, conn)) for conn in connects]

    # Flatten all labels once for order derivation and validation.
    flat_connect = [lab for cl in C for lab in cl]

    # Derive contraction order for positive labels when not provided.
    if con_order is None:
        # Default: ascending order of unique positive labels.
        con_order_list = sorted(set(x for x in flat_connect if x > 0))
    else:
        # Only list of integers is supported here. A string (e.g., "greedy")
        # is not interpreted in this implementation.
        if isinstance(con_order, str):
            con_order_list = []
        else:
            con_order_list = list(map(int, con_order))

    # Optional sanity checks (shape/label multiplicity/order validity).
    if check_network:
        dims_list = [list(t.shape) for t in T]
        _check_inputs(C, flat_connect, dims_list, con_order_list)

    # ------------------------------------------------------------------
    # Phase 1: Partial traces on individual tensors
    # ------------------------------------------------------------------
    # If a label appears twice on the same tensor, trace those two axes out.
    # This can eliminate some positive labels from the global contraction order.
    for i in range(len(T)):
        t, labels, traced_pos_labels = _partial_trace_fast(T[i], C[i])
        T[i], C[i] = t, labels
        if traced_pos_labels and con_order_list:
            traced_set = set(traced_pos_labels)
            con_order_list = [x for x in con_order_list if x not in traced_set]

    # ------------------------------------------------------------------
    # Phase 2: Binary contractions following `con_order_list`
    # ------------------------------------------------------------------
    # Repeatedly pick the next positive label, locate the two tensors that
    # carry it, and contract them over all common positive labels.
    while len(con_order_list) > 0:
        lab = con_order_list[0]

        # Find the two tensors that contain this label.
        locs = [i for i, cl in enumerate(C) if lab in cl]
        if len(locs) != 2:
            raise ValueError(f"NCON error: invalid network around label {lab}")

        iA, iB = locs
        A, La = T[iA], C[iA]
        B, Lb = T[iB], C[iB]

        # All common positive labels shared between the two tensors.
        common = sorted(set(La).intersection(Lb))
        # Axes to contract on A and B for those labels.
        A_cont = [La.index(x) for x in common]
        B_cont = [Lb.index(x) for x in common]

        # Heuristic from original code: prefer the axis order of the smaller tensor.
        if A.numel() < B.numel():
            order = sorted(range(len(A_cont)), key=lambda k: A_cont[k])
        else:
            order = sorted(range(len(B_cont)), key=lambda k: B_cont[k])
        axes_a = [A_cont[k] for k in order]
        axes_b = [B_cont[k] for k in order]

        # Perform the contraction over the aligned axes.
        C_new = torch.tensordot(A, B, dims=(axes_a, axes_b))

        # Construct the new label list: remaining (non-contracted) axes of A then B.
        La_new = [La[i] for i in range(len(La)) if i not in set(axes_a)]
        Lb_new = [Lb[i] for i in range(len(Lb)) if i not in set(axes_b)]
        L_new = La_new + Lb_new

        # Append the result tensor and labels; remove the two inputs.
        T.append(C_new)
        C.append(L_new)
        for idx in sorted(locs, reverse=True):
            del T[idx]
            del C[idx]

        # Remove all just-contracted labels from the remaining order.
        common_set = set(common)
        con_order_list = [x for x in con_order_list if x not in common_set]

    # ------------------------------------------------------------------
    # Phase 3: Outer products between remaining disconnected components
    # ------------------------------------------------------------------
    # If more than one tensor remains (no positive labels left), combine them
    # via outer products. This handles scalars robustly (shape=()).
    while len(T) > 1:
        A, La = T[-2], C[-2]
        B, Lb = T[-1], C[-1]

        s1, s2 = A.shape, B.shape
        out_shape = tuple(s1) + tuple(s2)  # May be empty for scalars.
        v1 = A.reshape(-1)
        v2 = B.reshape(-1)
        out = torch.outer(v1, v2).reshape(out_shape)

        T[-2], C[-2] = out, La + Lb
        del T[-1]
        del C[-1]

    # ------------------------------------------------------------------
    # Phase 4: Final permutation of free indices (negative labels)
    # ------------------------------------------------------------------
    # Output axes are ordered by decreasing negative label: -1, -2, -3, ...
    final_labels = C[0]
    if len(final_labels) > 0:
        perm = sorted(range(len(final_labels)), key=lambda k: -final_labels[k])
        result = T[0].permute(perm)
        return result.contiguous() if make_contiguous_output else result
    else:
        # No free indices -> scalar contraction result.
        return T[0].reshape(())


# ========= helpers =========
def _partial_trace_fast(A: torch.Tensor, labels: List[int]):
    """Perform partial traces on a single tensor for duplicated labels.

    Any label that appears **twice on the same tensor** corresponds to a
    partial trace of those two axes. This function performs all such traces
    at once using a single `permute + reshape + diagonal.sum` sequence.

    Args:
      A: Input tensor.
      labels: List of integer labels for each axis of `A`.

    Returns:
      Tuple[torch.Tensor, List[int], List[int]]:
        * The partially traced tensor.
        * The updated label list (with traced axes removed).
        * The list of **positive** labels that were eliminated by the trace
          (to be removed from the global contraction order).

    Raises:
      ValueError: If any duplicated label appears other than exactly twice
        on the same tensor, or if traced pairs have mismatched dimensions.
    """
    # Map each label to the list of axis positions where it appears.
    pos_map = {}
    for i, lab in enumerate(labels):
        pos_map.setdefault(lab, []).append(i)

    # Labels that occur more than once (eligible for partial trace).
    dup_keys = [lab for lab, pos in pos_map.items() if len(pos) > 1]
    if not dup_keys:
        return A, labels, []

    # Each duplicated label must occur exactly twice on the same tensor.
    for lab in dup_keys:
        if len(pos_map[lab]) != 2:
            raise ValueError(
                f"NCON error: label {lab} appears {len(pos_map[lab])} times on a single tensor."
            )

    # Following original (Fortran-like) ordering:
    # collect first positions for all dup labels, then all second positions.
    first_positions = [pos_map[lab][0] for lab in dup_keys]
    second_positions = [pos_map[lab][1] for lab in dup_keys]
    cont_ind = first_positions + second_positions

    # Free axes are those not involved in any partial trace.
    free_ind = [i for i in range(len(labels)) if i not in cont_ind]

    # Bring free axes first, then the traced axes.
    perm = free_ind + cont_ind
    A_perm = A.permute(perm)

    # Dimension bookkeeping for reshape.
    free_dims = [A.shape[i] for i in free_ind]
    cont_dims_first = [A.shape[i] for i in first_positions]
    cont_dims_second = [A.shape[i] for i in second_positions]

    # Dimension safety check: traced pairs must have equal sizes.
    if any(d1 != d2 for d1, d2 in zip(cont_dims_first, cont_dims_second)):
        raise ValueError("NCON error: mismatched dims in partial trace pairs.")

    # Flatten to [free_dim, cont_dim, cont_dim], then sum over the diagonal.
    free_dim = int(math.prod(free_dims)) if free_dims else 1
    cont_dim = int(math.prod(cont_dims_first)) if cont_dims_first else 1
    A_rs = A_perm.reshape(free_dim, cont_dim, cont_dim)

    # Sum of diagonal along the last two dims yields traced result for each free row.
    B_vec = torch.diagonal(A_rs, dim1=1, dim2=2).sum(dim=1)
    B = B_vec.reshape(*free_dims) if free_dims else B_vec.reshape(())

    # New labels retain only free axes.
    new_labels = [labels[i] for i in free_ind]

    # Report which positive labels were eliminated (remove from global order).
    traced_pos_labels = [lab for lab in dup_keys if lab > 0]
    return B, new_labels, traced_pos_labels


def _check_inputs(
    connect_list: List[List[int]],
    flat_connect: List[int],
    dims_list: List[List[int]],
    con_order_list: List[int],
):
    """Validate network structure against `ncon_torch` invariants.

    It verifies:
      * rank/label list length matches for each tensor,
      * negative labels are unique and contiguous (-1, -2, ...),
      * `con_order_list` equals the set of all positive labels,
      * each positive label occurs exactly twice and with matching dimensions.

    Args:
      connect_list: Per-tensor label lists.
      flat_connect: Flattened list of all labels (same order as dims_list flattened).
      dims_list: Per-tensor dimension lists (shape sizes).
      con_order_list: Intended contraction order for positive labels.

    Returns:
      True if all checks pass.

    Raises:
      ValueError: If any structural inconsistency is found.
    """
    # Partition labels by sign for convenience.
    pos_ind = [x for x in flat_connect if x > 0]
    neg_ind = [x for x in flat_connect if x < 0]

    # Check number of label lists == number of tensors.
    if len(dims_list) != len(connect_list):
        raise ValueError(
            f"mismatch between {len(dims_list)} tensors given but {len(connect_list)} index sub lists given"
        )

    # Check rank == number of labels for each tensor.
    for i, dims in enumerate(dims_list):
        if len(dims) != len(connect_list[i]):
            raise ValueError(
                f"number of indices does not match number of labels on tensor {i}: "
                f"{len(dims)}-indices versus {len(connect_list[i])}-labels"
            )

    # Check contraction order equals the set of positive labels.
    if sorted(con_order_list) != sorted(set(pos_ind)):
        raise ValueError("NCON error: invalid contraction order")

    # Negative labels must be unique and contiguous (-1, -2, ..., in any subset).
    if neg_ind:
        needed = list(range(-1, -len(neg_ind) - 1, -1))
        for lab in needed:
            cnt = sum(1 for x in neg_ind if x == lab)
            if cnt == 0:
                raise ValueError(f"NCON error: no index labelled {lab}")
            if cnt > 1:
                raise ValueError(f"NCON error: more than one index labelled {lab}")

    # Flatten dims in the same order as `flat_connect`, then check positive pairs.
    flat_dims = [d for dl in dims_list for d in dl]
    all_labels = []
    for cl in connect_list:
        all_labels.extend(cl)

    # Collect dimensions associated with each positive label.
    from collections import defaultdict
    label_dims = defaultdict(list)
    for lab, dim in zip(all_labels, flat_dims):
        if lab > 0:
            label_dims[lab].append(dim)

    # Each positive label must occur exactly twice with equal dimensions.
    for lab, dims in label_dims.items():
        if len(dims) == 1:
            raise ValueError(f"NCON error: only one index labelled {lab}")
        if len(dims) > 2:
            raise ValueError(f"NCON error: more than two indices labelled {lab}")
        if dims[0] != dims[1]:
            raise ValueError(
                f"NCON error: tensor dimension mismatch on index labelled {lab}: "
                f"dim-{dims[0]} versus dim-{dims[1]}"
            )
    return True
