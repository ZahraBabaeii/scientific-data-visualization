"""Step 3 - UMAP runs for the stability figure (F04b).

Six runs of the same 211 systems: three seeds at the reference settings, plus other n_neighbors / min_dist values.
Every run is initialised from the first two principal components (PCA before UMAP).

Usage:  python scripts/umap_grid.py [OUTPUT_DIR]            (default: outputs/)
Output: umap_grid.npy
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import umap
from sklearn.decomposition import PCA

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]  # repository root
O = sys.argv[1] if len(sys.argv) > 1 else str(ROOT / "outputs")
N = ["Coal", "Gas", "Oil", "Nuclear", "Hydro", "Wind", "Solar", "Bioenergy", "Other renewables"]
x = pd.read_csv(f"{O}/xs2023_labeled.csv")
X = x[N].values
Z = PCA().fit_transform(X)
init = Z[:, :2] / Z[:, 0].std()
# (n_neighbors, min_dist, seed)
S = [(15, 0.1, 0), (15, 0.1, 1), (15, 0.1, 2), (10, 0.1, 0), (30, 0.1, 0), (15, 0.3, 0)]
embs = {}
for nn, md, seed in S:
    embs[(nn, md, seed)] = umap.UMAP(n_neighbors=nn, min_dist=md, init=init.copy(), random_state=seed).fit_transform(X)
np.save(f"{O}/umap_grid.npy", np.stack([embs[s] for s in S]))
print("ref matches stored:", np.abs(embs[(15, 0.1, 0)] - x[["U1", "U2"]].values).max())
print("pure-oil (>=95%) systems:", (x.Oil >= 95).sum(), "| exactly 100:", (x.Oil >= 99.99).sum())
