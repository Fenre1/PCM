import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from scipy.spatial.distance import pdist
from datasketch import MinHash
from joblib import Parallel, delayed

__all__ = [
    "pjss",
    "lds",
    "harmonic_mean",
]


def exact_jaccard(X_bool: np.ndarray) -> float:
    """
    Exact Jaccard via SciPy pdist (fast in C) for small clusters.
    """
    if X_bool.shape[0] < 2:
        return 1.0
    sims = 1.0 - pdist(X_bool, metric="jaccard")
    return float(np.mean(sims))


def sampled_jaccard(X_bool: np.ndarray, k: int = 5000) -> float:
    """
    Approximate Jaccard by uniform sampling k random pairs (vectorized).
    """
    n = X_bool.shape[0]
    if n < 2:
        return 1.0
    i = np.random.randint(0, n, size=k)
    j = np.random.randint(0, n, size=k)
    mask = i != j
    i, j = i[mask], j[mask]
    Xi, Xj = X_bool[i], X_bool[j]
    inter = np.logical_and(Xi, Xj).sum(axis=1)
    union = np.logical_or(Xi, Xj).sum(axis=1)
    valid = union > 0
    return float(np.mean(inter[valid] / union[valid])) if valid.any() else 0.0


def minhash_jaccard(label_lists: list, num_perm: int = 128) -> float:
    """
    MinHash-based approximate Jaccard for very large clusters.
    """
    n = len(label_lists)
    if n < 2:
        return 1.0
    mhs = []
    for labels in label_lists:
        mh = MinHash(num_perm=num_perm)
        for l in labels:
            mh.update(str(l).encode())
        mhs.append(mh)
    s, cnt = 0.0, 0
    for a in range(n):
        for b in range(a + 1, n):
            s += mhs[a].jaccard(mhs[b])
            cnt += 1
    return s / cnt if cnt else 1.0


def adaptive_cluster_score(
    label_lists,
    exact_thresh: int = 2000,
    sample_thresh: int = 4000,
    sample_k: int = 8000,
) -> float:
    """
    Dispatch to the best Jaccard routine based on cluster size.

    Accepts either:
      - label_lists: list of lists of integer labels, OR
      - a 2D boolean NumPy array
    """
    if not isinstance(label_lists, np.ndarray):
        mlb = MultiLabelBinarizer(sparse_output=False)
        X_bool = mlb.fit_transform(label_lists)
    else:
        X_bool = label_lists

    n = X_bool.shape[0]
    if n <= exact_thresh:
        return exact_jaccard(X_bool)
    elif n <= sample_thresh:
        return sampled_jaccard(X_bool, k=sample_k)
    else:
        if isinstance(label_lists, np.ndarray):
            label_lists = [list(np.where(row)[0]) for row in X_bool]
        return minhash_jaccard(label_lists)


def weighted_mean(cluster_scores: list, labels_pred: np.ndarray) -> float:
    """
    Weighted mean of cluster-level scores, weighted by cluster size.
    """
    total = labels_pred.shape[0]
    sc = 0.0
    if labels_pred.ndim == 1:
        for idx, score in enumerate(cluster_scores):
            count = int((labels_pred == idx).sum())
            sc += score * count / total
    else:
        for idx, score in enumerate(cluster_scores):
            count = int(labels_pred[:, idx].sum())
            sc += score * count / total
    return sc


def pjss(labels_pred: np.ndarray, labels_true, n_jobs: int = -1) -> float:
    """
    Parallel, adaptive PJSS for multi-label clustering.

    Args:
        labels_pred : 1D cluster IDs or 2D binary indicator array
        labels_true : list-of-lists or 2D binary array of true labels
        n_jobs      : cores for joblib (-1 = all cores)

    Returns:
        Scalar PJSS score (float)
    """
    labels_pred = np.asarray(labels_pred)
    if isinstance(labels_true, np.ndarray) and labels_true.ndim == 2:
        true_lists = [list(np.where(row > 0)[0]) for row in labels_true]
    else:
        true_lists = labels_true

    clusters = []
    if labels_pred.ndim == 1:
        for cid in np.unique(labels_pred):
            idx = np.where(labels_pred == cid)[0]
            clusters.append([true_lists[i] for i in idx])
    else:
        for cid in range(labels_pred.shape[1]):
            idx = np.where(labels_pred[:, cid] == 1)[0]
            clusters.append([true_lists[i] for i in idx])

    scores = Parallel(n_jobs=n_jobs, prefer="threads")(delayed(adaptive_cluster_score)(cluster) for cluster in clusters)
    return weighted_mean(scores, labels_pred)


def lds(labels_pred: np.ndarray, labels_true: np.ndarray) -> float:
    """
    Calculate Label Distribution Score (LDS) for multi-label clustering.

    Args:
        labels_pred : 1D array of cluster IDs per sample
        labels_true : 2D binary array of true labels (shape n_samples x n_classes)

    Returns:
        Scalar LDS score (mean over label distribution scores)
    """
    labels_pred = np.asarray(labels_pred).squeeze()
    labels_true = np.asarray(labels_true)

    n_classes = labels_true.shape[1]
    counts_per_class = np.sum(labels_true, axis=0)

    unique_clusters = np.unique(labels_pred)
    cluster_label_counts = np.zeros((unique_clusters.size, n_classes), dtype=int)
    for i, c in enumerate(unique_clusters):
        idx = np.where(labels_pred == c)[0]
        cluster_label_counts[i] = labels_true[idx].sum(axis=0)

    dist_scores = np.zeros(n_classes, dtype=float)
    for j in range(n_classes):
        total = counts_per_class[j]
        if total > 0:
            sorted_counts = np.sort(cluster_label_counts[:, j])[::-1]
            weights = sorted_counts / total
            ranks = np.arange(1, sorted_counts.size + 1)
            dist_scores[j] = np.sum(weights / ranks)

    return float(np.mean(dist_scores))


def harmonic_mean(pjss_score: float, lds_score: float) -> float:
    """
    Harmonic mean of PJSS and LDS to balance quality vs. efficiency.

    Args:
        pjss_score : PJSS scalar score
        lds_score  : LDS scalar score
    Returns:
        Harmonic mean (float)
    """
    if pjss_score + lds_score == 0:
        return 0.0
    return 2 * pjss_score * lds_score / (pjss_score + lds_score)
