"""
Tier 2 -- vanilla GuardNet bolt-on: full graph over all entities in the test
condition, 2-layer GCN encoder trained with full-batch gradient descent
(GuardNet Eq. 7-9), plus a simplified Major-node-centered Downsampling (MCD,
GuardNet Sect. 4.2.2) applied to the *training mask* only, combined with
inverse-class-frequency loss reweighting (see SimpleGCN.train's docstring --
MCD's structural downsampling alone was not sufficient to prevent majority-
class collapse during development and needed this on top). MCES (subgraph
fusion around minority nodes) is not separately implemented -- at the graph
sizes used in this package (hundreds to low thousands of ADFA-LD entities)
every minority node's neighborhood is already included in the single global
graph, so MCES's main additional effect (pulling minority-node neighborhoods
into an otherwise-disjoint sample) is close to a no-op here. This is a
documented simplification, not an oversight -- see the study report's
"Risks and Open Questions" section.
"""
import numpy as np


class SimpleGCN:
    def __init__(self, in_dim, hid_dim, out_dim=2, lr=0.1, seed=0):
        rng = np.random.default_rng(seed)
        self.W1 = rng.normal(0, 0.1, (in_dim, hid_dim))
        self.W2 = rng.normal(0, 0.1, (hid_dim, out_dim))
        self.lr = lr

    def forward(self, A_norm, X):
        Z1 = A_norm @ X @ self.W1
        H1 = np.maximum(Z1, 0)
        Z2 = A_norm @ H1 @ self.W2
        return Z2, H1, Z1

    @staticmethod
    def softmax(Z):
        Z = Z - Z.max(axis=1, keepdims=True)
        e = np.exp(Z)
        return e / e.sum(axis=1, keepdims=True)

    def train(self, A_norm, X, y, train_mask, epochs=300, class_weight=True):
        """class_weight=True additionally reweights the per-node loss term
        by inverse class frequency within the training mask, on top of
        GuardNet's structural MCD downsampling -- MCD alone (a 3:1 ratio
        after downsampling here) was not sufficient to prevent majority-
        class collapse on this synthetic graph's feature separability, so
        this combines resampling (MCD) with reweighting, a standard and
        literature-consistent pairing (see GuardNet's own Related Work,
        Sect. 2.1, on reweighting vs. resampling as complementary strategies)."""
        n = X.shape[0]
        out_dim = self.W2.shape[1]
        Y = np.zeros((n, out_dim))
        Y[np.arange(n), y] = 1
        mask = train_mask.astype(float)
        if class_weight:
            w = np.ones(n)
            for c in range(out_dim):
                idx = train_mask & (y == c)
                cnt = idx.sum()
                if cnt > 0:
                    w[idx] = 1.0 / cnt
            w = w * mask
            w = w / (w.sum() + 1e-9)
        else:
            w = mask / max(mask.sum(), 1.0)
        for _ in range(epochs):
            Z2, H1, Z1 = self.forward(A_norm, X)
            P = self.softmax(Z2)
            dZ2 = (P - Y) * w[:, None]
            gW2 = H1.T @ (A_norm.T @ dZ2)
            dH1 = (A_norm.T @ dZ2) @ self.W2.T
            dZ1 = dH1 * (Z1 > 0)
            gW1 = X.T @ (A_norm.T @ dZ1)
            self.W2 -= self.lr * gW2
            self.W1 -= self.lr * gW1

    def predict_proba(self, A_norm, X):
        Z2, _, _ = self.forward(A_norm, X)
        return self.softmax(Z2)


def mcd_training_mask(y, minority_label=1, gamma=0.5, seed=0):
    """GuardNet's Major-node-centered Downsampling: drop a `gamma` fraction
    of majority-class nodes from the *training mask* (they remain in the
    graph -- still usable as neighbors -- just not directly optimized
    against), so the loss isn't dominated by the majority class."""
    r = np.random.default_rng(seed)
    mask = np.ones(len(y), dtype=bool)
    major_idx = np.where(y != minority_label)[0]
    drop_n = int(len(major_idx) * gamma)
    drop_idx = r.choice(major_idx, size=drop_n, replace=False)
    mask[drop_idx] = False
    return mask


def run_tier2(full_feats, labels, tau, gamma=0.5, hid_dim=16, epochs=300,
              test_frac=0.3, seed=0, eval_idx=None):
    """labels: 0=normal, 1=attack. Returns predictions + probabilities for
    the held-out test split, transductively (test nodes are in the graph,
    just excluded from the loss). If eval_idx is given, that exact partition
    is used as the held-out set instead of drawing a fresh one -- needed so
    Tier 1/2/3 are all scored on the identical entities in run_ablation_real.py."""
    from graph_utils import build_adjacency, normalize_adjacency
    n = len(labels)
    if eval_idx is not None:
        test_idx = set(eval_idx.tolist())
    else:
        r = np.random.default_rng(seed)
        idx = r.permutation(n)
        n_test = int(n * test_frac)
        test_idx = set(idx[:n_test].tolist())
    base_train_mask = np.array([i not in test_idx for i in range(n)])

    A = build_adjacency(full_feats, tau=tau)
    A_norm = normalize_adjacency(A)

    mcd_mask = mcd_training_mask(labels, minority_label=1, gamma=gamma, seed=seed)
    train_mask = base_train_mask & mcd_mask

    Xn = (full_feats - full_feats.mean(0)) / (full_feats.std(0) + 1e-6)
    model = SimpleGCN(in_dim=Xn.shape[1], hid_dim=hid_dim, out_dim=2, seed=seed)
    model.train(A_norm, Xn, labels, train_mask, epochs=epochs)
    P = model.predict_proba(A_norm, Xn)
    preds = P.argmax(axis=1)
    test_idx_arr = np.array(sorted(test_idx))
    return preds[test_idx_arr], P[test_idx_arr, 1], test_idx_arr, model, A_norm, Xn
