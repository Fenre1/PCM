# PCM
PCM implements the Post‑Clustering Merging (PCM) algorithm for refining overclustered data, together with two novel evaluation metrics for multi‑label clustering and their harmonic mean.

## Features

- PCM algorithm (PCM.pcm):

  - Takes an existing clustering (labels + feature embeddings) and merges similar clusters based on centroid similarity and neighbor overlap.

  - Designed for multi‑label datasets where traditional clustering may over‑ or under‑cluster.

  - Quick post‑processing step that adds minimal overhead.

- Evaluation measures:

  - Supports multi-label data: Unlike most common clustering metrics that assume each item has only one true label, PJSS and LDS can evaluate clustering quality when each datapoint may have one or more labels.
  - Pairwise Jaccard Similarity Score (PJSS): average Jaccard index over all label-set pairs within a cluster, measuring cluster quality.

  - Label Distribution Score (LDS): rewards concentrating occurrences of each label in as few clusters as possible, measuring cluster efficiency.

  - Harmonic mean of PJSS and LDS to balance quality vs. efficiency.
 
## Quickstart

```python
import numpy as np
from pcm_merge import pcm

# labels: 1D array of initial cluster IDs
# embeddings: 2D feature matrix

new_labels = pcm(
    labels,
    embeddings,
    sim_threshold=0.6,      # cosine‐similarity link threshold
    jaccard_threshold=0.1,  # neighbor‐overlap threshold
)
print(new_labels)
```


```python
from PCM.metrics import pjss, lds, harmonic_mean

# Suppose `clusters` is a list of clusters (list of sample indices)
# and `label_sets` is a list of ground‑truth label sets per sample

pjss_score = pjss(clusters, label_sets)
lds_score  = lds(clusters, label_sets)
hmean      = harmonic_mean(pjss_score, lds_score)
print(f"PJSS={pjss_score:.3f}, LDS={lds_score:.3f}, harmonic={hmean:.3f}")
```

# Defaults & Recommendations

- Default thresholds (found via grid search on multiple image collections):

  - sim_threshold = 0.6

  - jaccard_threshold = 0.1

These defaults work well out of the box, but you can tune them to control the balance between merging aggressiveness (efficiency) and preserving cluster purity (quality).
