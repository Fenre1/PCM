from typing import List, Tuple

import numpy as np



def _cosine_sim_matrix(features: np.ndarray) -> np.ndarray:
    """
    Compute the pairwise cosine similarity matrix for the given feature vectors,
    setting diagonal to zero.

    Args:
        features: Array of shape (n_samples, n_features).

    Returns:
        A (n_samples, n_samples) array of cosine similarities with zeros on the diagonal.
    """
    x = features.astype(np.float32, copy=False)
    x -= x.mean(axis=1, keepdims=True)
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    np.divide(x, norm, out=x, where=norm != 0)
    sim_matrix = x @ x.T
    np.fill_diagonal(sim_matrix, 0)
    return sim_matrix


def _pairwise_correlation(
    query: np.ndarray, reference: np.ndarray
) -> np.ndarray:
    """
    Compute pairwise Pearson correlation between query vectors and reference vectors.

    Args:
        query: Array of shape (k, d) or (d,), treated as at least 2D.
        reference: Array of shape (n, d) or (d,), treated as at least 2D.

    Returns:
        A flat array of length k * n containing correlation values.

    Raises:
        ValueError: If feature dimensions do not match.
    """
    q = np.atleast_2d(query).astype(np.float32, order="C")
    r = np.atleast_2d(reference).astype(np.float32, order="C")
    k, d = q.shape
    n, d2 = r.shape
    if d != d2:
        raise ValueError(f"Feature dimension mismatch: query has {d}, reference has {d2}")

    # unbiased standard deviations
    denom = d - 1
    q_mean = q.mean(axis=1, keepdims=True)
    r_mean = r.mean(axis=1, keepdims=True)
    q_var = (np.einsum("ij,ij->i", q, q)[:, None] - d * q_mean ** 2) / denom
    r_var = (np.einsum("ij,ij->i", r, r)[:, None] - d * r_mean ** 2) / denom
    q_std = np.sqrt(np.clip(q_var, 1e-12, None)).astype(np.float32)
    r_std = np.sqrt(np.clip(r_var, 1e-12, None)).astype(np.float32)

    cov = (q @ r.T - d * q_mean @ r_mean.T) / denom
    corr_matrix = cov / (q_std @ r_std.T)
    return corr_matrix.ravel()


def _remap_labels(assignments: np.ndarray, merges: List[List[int]]) -> np.ndarray:
    """
    Remap original labels according to merge groups.

    Args:
        assignments: 1D array of original cluster IDs for each sample.
        merges: List of groups, each group is a list of cluster IDs to merge.

    Returns:
        New assignments array with merged IDs.
    """
    mapping = {cid: min(group) for group in merges for cid in group}
    result = assignments.copy()
    for old_id, new_id in mapping.items():
        result[result == old_id] = new_id
    return result


def _reindex_labels(labels: np.ndarray) -> np.ndarray:
    """
    Reindex labels to be consecutive integers starting from zero.

    Args:
        labels: 1D array of integer labels.

    Returns:
        2D array of shape (n_samples, 1) with new indices.
    """
    flat = labels.astype(int)
    unique = np.unique(flat)
    index_map = {old: new for new, old in enumerate(unique)}
    reindexed = np.vectorize(index_map.get)(flat)
    return reindexed.reshape(-1, 1)


def _centroids_to_matrix(
    labels: np.ndarray, embeddings: np.ndarray
) -> Tuple[np.ndarray, List[int], List[np.ndarray]]:
    """
    Compute centroids for each label and return cosine similarity matrix.

    Args:
        labels: 1D array of cluster IDs.
        embeddings: 2D array of shape (n_samples, n_features).

    Returns:
        sim_matrix: Cosine similarity between centroids.
        counts: List of sample counts per cluster.
        centroids: List of centroid vectors.

    Raises:
        ValueError: If shapes mismatch or labels not 1D.
    """
    labels = labels.ravel()
    if embeddings.shape[0] != labels.size:
        raise ValueError("Number of embeddings must match number of labels.")

    unique_labels, inv = np.unique(labels, return_inverse=True)
    k = unique_labels.size
    d = embeddings.shape[1]

    counts = np.bincount(inv).tolist()
    sums = np.zeros((k, d), dtype=np.float32)
    np.add.at(sums, inv, embeddings.astype(np.float32))
    centroids = [sums[i] / counts[i] for i in range(k)]

    sim_matrix = _cosine_sim_matrix(np.vstack(centroids))
    return sim_matrix, counts, centroids


def _build_graph(sim_matrix: np.ndarray, threshold: float) -> np.ndarray:
    """
    Build adjacency matrix by thresholding similarity values.

    Args:
        sim_matrix: 2D similarity matrix.
        threshold: Similarity cutoff.

    Returns:
        Boolean adjacency matrix.
    """
    graph = sim_matrix > threshold
    return graph


def _jaccard(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute Jaccard similarity between two boolean vectors.
    """
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return inter / union if union > 0 else 0.0


def _merge_neighbors(
    graph: np.ndarray,
    sim_threshold: float,
    jaccard_threshold: float,
    counts: List[int],
    centroids: List[np.ndarray],
) -> List[List[int]]:
    """
    Iteratively merge clusters whose neighbors have high Jaccard similarity.

    Args:
        graph: Boolean adjacency matrix of clusters.
        sim_threshold: Cosine similarity threshold for new links.
        jaccard_threshold: Jaccard similarity threshold for merging.
        counts: Current sizes of each cluster.
        centroids: Current centroid vectors.

    Returns:
        List of merged cluster ID groups.
    """
    merged = [[i] for i in range(graph.shape[0])]
    idx = 1
    changed = True

    while changed and len(centroids) > 1:
        changed = False
        for i in range(idx - 1, graph.shape[0]):
            neighbors_i = graph[i].copy()
            neighbors_i[i] = True
            for j in range(i + 1, graph.shape[0]):
                neighbors_j = graph[j].copy()
                neighbors_j[j] = True
                if _jaccard(neighbors_i, neighbors_j) >= jaccard_threshold:
                    # merge j into i
                    merged[i].extend(merged[j])
                    merged.pop(j)

                    # update counts and centroid
                    total = counts[i] + counts.pop(j)
                    centroids[i] = (
                        centroids[i] * counts[i] + centroids.pop(j) * counts[j]
                    ) / total
                    counts[i] = total

                    # rebuild graph row i
                    new_links = _pairwise_correlation(
                        centroids[i], np.vstack(centroids)
                    ) > sim_threshold
                    graph[i, :] = new_links
                    graph[:, i] = new_links

                    # remove j-th row/col from graph
                    graph = np.delete(graph, j, axis=0)
                    graph = np.delete(graph, j, axis=1)

                    changed = True
                    break
            if changed:
                break
            idx += 1
    return merged


def pcm(
    labels: np.ndarray,
    embeddings: np.ndarray,
    sim_threshold: float,
    jaccard_threshold: float,
) -> np.ndarray:
    """
    Merge clusters in `labels` based on embedding similarities and neighborhood overlap.

    Args:
        labels: 1D array of initial cluster IDs.
        embeddings: 2D array of shape (n_samples, n_features).
        sim_threshold: Cosine similarity threshold to connect clusters.
        jaccard_threshold: Jaccard threshold to decide merges.

    Returns:
        1D array of new cluster IDs, reindexed consecutively.
    """
    sim_matrix, counts, centroids = _centroids_to_matrix(labels, embeddings)
    graph = _build_graph(sim_matrix, sim_threshold)
    merge_groups = _merge_neighbors(graph, sim_threshold, jaccard_threshold, counts, centroids)
    remapped = _remap_labels(labels, merge_groups)
    reindexed = _reindex_labels(remapped).squeeze()
    return reindexed


__all__ = ["pcm"]
