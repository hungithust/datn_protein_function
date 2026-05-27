"""DAG-consistent post-processing for GO predictions."""

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import depth_first_order


def propagate_scores_upward(probs: np.ndarray, dag_matrix: np.ndarray) -> np.ndarray:
    """
    Enforce True Path Rule: score(parent) >= score(child).

    Args:
        probs: (N, C) sigmoid output
        dag_matrix: (C, C) — A[i,j]=1 nếu j là parent của i

    Returns:
        (N, C) probs after upward propagation
    """
    probs_out = probs.copy().astype(np.float64)
    C = probs.shape[1]

    if dag_matrix.sum() == 0:
        return probs_out.astype(probs.dtype)

    # Topo sort: scipy không có pure topo, dùng csgraph trên parent→child graph
    # parent_to_child[j,i]=1 nếu j→i (j parent of i) == dag_matrix.T
    pc = csr_matrix(dag_matrix.T)
    # Tìm roots = nodes không có incoming edge trong parent→child = nodes không có parent
    has_parent = (dag_matrix.sum(axis=1) > 0)
    roots = np.where(~has_parent)[0]

    # DFS từ mỗi root, collect order, đảo ngược để có leaf→root
    visited = np.zeros(C, dtype=bool)
    order = []
    for r in roots:
        if visited[r]:
            continue
        nodes, _ = depth_first_order(pc, r, directed=True, return_predecessors=True)
        for n in nodes:
            if not visited[n]:
                visited[n] = True
                order.append(n)
    # Bổ sung node mồ côi (không reach từ root nào)
    for n in range(C):
        if not visited[n]:
            order.append(n)
    leaf_to_root = order[::-1]

    for i in leaf_to_root:
        parents = np.where(dag_matrix[i] == 1)[0]
        for j in parents:
            np.maximum(probs_out[:, j], probs_out[:, i], out=probs_out[:, j])

    return probs_out.astype(probs.dtype)
