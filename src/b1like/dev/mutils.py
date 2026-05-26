import numpy as np


def target_ordering_index(idx_a, idx_b, n):
    """
    Return the pair index in the target ordering.

    Parameters
    ----------
    n
        Number of maps.
    """
    # Generate target ordering
    target = []
    # Diagonal elements
    for i in range(n):
        target.append((i, i))
    # Off-diagonal elements by increasing difference
    for diff in range(1, n):
        for i in range(n - diff):
            target.append((i, i + diff))

    return target.index((idx_a, idx_b))


def input_ordering_index(idx_a, idx_b, n):
    """
    Return the pair index in the input ordering.

    Parameters
    ----------
    n
        Number of maps.
    """
    # Generate original ordering
    original = []
    # Diagonal elements
    for i in range(n):
        original.append((i, i))
    # Off-diagonal elements sorted lexicographically
    for i in range(n):
        for j in range(i + 1, n):
            original.append((i, j))

    return original.index((idx_a, idx_b))


def get_rwfratio(rwf_raw, idx_a, idx_b, idx_x, nmaps):
    if idx_a == idx_b:
        return np.ones_like(rwf_raw[0])

    if idx_b == nmaps - 1 and idx_a != 0:  # only need to fix Dust/SyncxLT; assume LT always the last map
        print(idx_a, idx_b)
        rwfratio = (
            np.sqrt(rwf_raw[idx_a] * rwf_raw[idx_b]) / rwf_raw[idx_x][:, :6]
        )  # the x-spec has 9 (:6 to match the numerator)
    else:
        rwfratio = np.ones_like(rwf_raw[0])

    return rwfratio


def reorder_pairs(n):
    """
    Create mappings between two orderings of pairs.

    Original order: diagonal pairs first, then pairs with difference 1, 2, 3, etc.
    Target order: diagonal pairs first, then sorted lexicographically.

    Parameters
    ----------
    n
        The maximum value in pairs. For example, ``n=4`` gives pairs up to
        ``(3, 3)`` for zero-indexed pairs.

    Returns
    -------
    tuple
        Tuple of dictionaries: ``forward_mapping`` maps old indices to new
        indices, ``reverse_mapping`` maps new indices to old indices,
        ``original_pair_to_idx`` maps ``(i, j)`` to old indices, and
        ``target_pair_to_idx`` maps ``(i, j)`` to new indices.
    """
    # Generate target ordering
    target = []
    # Diagonal elements
    for i in range(n):
        target.append((i, i))
    # Off-diagonal elements by increasing difference
    for diff in range(1, n):
        for i in range(n - diff):
            target.append((i, i + diff))

    # Generate original ordering
    original = []
    # Diagonal elements
    for i in range(n):
        original.append((i, i))
    # Off-diagonal elements sorted lexicographically
    for i in range(n):
        for j in range(i + 1, n):
            original.append((i, j))

    # Create forward mapping: original index -> target index
    forward_mapping = {}
    for old_idx, pair in enumerate(original):
        new_idx = target.index(pair)
        forward_mapping[old_idx] = new_idx

    # Create reverse mapping: target index -> original index
    reverse_mapping = {v: k for k, v in forward_mapping.items()}

    # Create pair-to-index mappings
    original_pair_to_idx = {pair: idx for idx, pair in enumerate(original)}
    target_pair_to_idx = {pair: idx for idx, pair in enumerate(target)}

    # Create index-to-pari
    original_idx_to_pair = {idx: pair for idx, pair in enumerate(original)}
    target_idx_to_pair = {idx: pair for idx, pair in enumerate(target)}

    return (
        forward_mapping,
        reverse_mapping,
        original_pair_to_idx,
        target_pair_to_idx,
        original_idx_to_pair,
        target_idx_to_pair,
    )
