import numpy as np
from scipy.linalg import eigh
from sklearn.svm import SVC
from sklearn.model_selection import LeaveOneOut, ParameterGrid
from sklearn.metrics import accuracy_score, roc_auc_score
import matplotlib.pyplot as plt
import time
from typing import List, Tuple, Dict, Optional
import warnings

warnings.filterwarnings('ignore')

# Try to use POT for faster Sliced Wasserstein
try:
    import ot

    POT_AVAILABLE = True
except ImportError:
    POT_AVAILABLE = False
    print("POT not installed. Using custom sliced Wasserstein implementation.")


# ================================================================================================
# 1. SYNTHETIC BRAIN NETWORK GENERATION (Disease‑specific patterns)
# ================================================================================================

def generate_disease_specific_networks(
        disease: str,
        n_subjects: int = 50,
        n_nodes: int = 90,
        noise: float = 0.05,
        seed: int = 42
) -> Tuple[List[np.ndarray], np.ndarray]:
    """
    Generate synthetic functional connectivity matrices for a specific brain disease vs controls.

    Each disease introduces a distinct alteration in the modular connectivity pattern:
    - ADHD: weakens fronto‑parietal connections (modules 2 and 5).
    - ASD: increases connectivity within default mode network (module 3) and weakens inter‑module.
    - EMCI: disrupts temporal and occipital modules (modules 4 and 6).

    Parameters
    ----------
    disease : str
        One of {'ADHD', 'ASD', 'EMCI'}.
    n_subjects : int
        Total number of subjects (half patients, half controls).
    n_nodes : int
        Number of brain regions (default 90, AAL atlas).
    noise : float
        Standard deviation of Gaussian noise added to each connection.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    graphs : List[np.ndarray]
        List of adjacency matrices (n_nodes × n_nodes), symmetric, non‑negative.
    labels : np.ndarray
        Binary labels: 0 = healthy control (NC), 1 = patient.
    """
    rng = np.random.RandomState(seed)
    n_patients = n_subjects // 2
    n_controls = n_subjects - n_patients
    labels = np.array([0] * n_controls + [1] * n_patients)

    # Base modular structure (6 modules of ~15 nodes each)
    module_size = 15
    n_modules = 6
    template = np.zeros((n_nodes, n_nodes))
    for m in range(n_modules):
        start = m * module_size
        end = start + module_size
        template[start:end, start:end] = 0.7  # Strong within‑module connections
    # Add sparse between‑module connections
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            if template[i, j] == 0:
                template[i, j] = 0.15 * rng.rand()
    template = (template + template.T) / 2
    np.fill_diagonal(template, 0)

    graphs = []
    for i in range(n_subjects):
        adj = template.copy()
        # Add subject‑specific noise
        noise_mat = rng.randn(n_nodes, n_nodes) * noise
        adj += noise_mat

        if labels[i] == 1:  # Patient
            # Apply disease‑specific perturbation
            if disease == 'ADHD':
                # Weaken fronto‑parietal connections (modules 1 and 4, 0‑indexed)
                mod1 = slice(0, 15)
                mod4 = slice(45, 60)
                adj[mod1, mod4] *= 0.5
                adj[mod4, mod1] *= 0.5
            elif disease == 'ASD':
                # Increase connectivity within DMN (module 2) and reduce cross‑module
                mod2 = slice(15, 30)
                adj[mod2, mod2] *= 1.4
                # Reduce connections between DMN and other modules
                adj[mod2, :] *= 0.8
                adj[:, mod2] *= 0.8
                np.fill_diagonal(adj, 0)
            elif disease == 'EMCI':
                # Disrupt temporal (mod 3) and occipital (mod 5) modules
                mod3 = slice(30, 45)
                mod5 = slice(60, 75)
                adj[mod3, mod5] *= 0.4
                adj[mod5, mod3] *= 0.4
                adj[mod3, mod3] *= 0.7
                adj[mod5, mod5] *= 0.7
            else:
                raise ValueError(f"Unknown disease: {disease}")

        # Enforce symmetry, non‑negativity, zero diagonal
        adj = (adj + adj.T) / 2
        adj = np.clip(adj, 0, None)
        np.fill_diagonal(adj, 0)
        graphs.append(adj)

    return graphs, labels


# ================================================================================================
# 2. CORE ALGORITHM: SLICED WASSERSTEIN GRAPH KERNEL
# ================================================================================================

class SlicedWassersteinGraphKernel:
    """
    Positive Definite Sliced Wasserstein Graph Kernel.

    Implements:
        1. Laplacian spectral embedding Φ(G) ∈ R^{n×d} (Feature Projection).
        2. Graph Sliced Wasserstein distance D_GSW (Eq. 2 & 4).
        3. SWG kernel K_SWG(G1,G2) = exp(-λ · D_GSW(G1,G2)) (Eq. 5).

    Attributes
    ----------
    embedding_dim : int
        Dimension d of Laplacian embedding (number of eigenvectors).
    n_slices : int
        Number of random projection directions L.
    lambda_ : float
        Kernel bandwidth parameter λ.
    random_state : int
        Seed for reproducible random slices.
    """

    def __init__(
            self,
            embedding_dim: int = 10,
            n_slices: int = 50,
            lambda_: float = 1.0,
            random_state: int = 42
    ):
        self.embedding_dim = embedding_dim
        self.n_slices = n_slices
        self.lambda_ = lambda_
        self.random_state = random_state
        self._embeddings = None  # Cache embeddings for all graphs
        self._slice_directions = None  # Cached projection directions

    def _laplacian_embedding(self, adj: np.ndarray) -> np.ndarray:
        """
        Compute Laplacian spectral embedding (Eq. after Eq.3 in paper).

        L = D - A  →  eigenvectors corresponding to the d smallest non‑zero eigenvalues.
        """
        n = adj.shape[0]
        degree = np.sum(adj, axis=1)
        D = np.diag(degree)
        L = D - adj
        try:
            _, eigvecs = eigh(L, subset_by_index=[1, self.embedding_dim])
        except ValueError:
            # Fallback if graph is degenerate
            _, eigvecs = eigh(L)
            eigvecs = eigvecs[:, 1:self.embedding_dim + 1]
        return eigvecs

    def _generate_slice_directions(self, d: int) -> np.ndarray:
        """Generate L random unit vectors in R^d."""
        rng = np.random.RandomState(self.random_state)
        theta = rng.randn(self.n_slices, d)
        theta = theta / np.linalg.norm(theta, axis=1, keepdims=True)
        return theta

    def _gsw_distance(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """
        Compute Graph Sliced Wasserstein distance (Eq. 4 approximation).

        GSW₂(r,c) ≈ sqrt( 1/L Σ_l Σ_n |⟨θ_l, x_{i[n]}⟩ - ⟨θ_l, y_{j[n]}⟩|² )
        """
        n1, d = emb1.shape
        n2, _ = emb2.shape
        assert n1 == n2, "Both graphs must have the same number of nodes."
        assert d == self.embedding_dim

        if self._slice_directions is None:
            self._slice_directions = self._generate_slice_directions(d)

        if POT_AVAILABLE:
            # Use POT's efficient implementation (faster)
            a = np.ones(n1) / n1
            b = np.ones(n2) / n2
            M = ot.dist(emb1, emb2, metric='euclidean')
            # Sliced Wasserstein via random projections
            dist = ot.sliced_wasserstein_distance(emb1, emb2, a, b, self.n_slices, seed=self.random_state)
            return dist
        else:
            # Manual implementation
            total = 0.0
            for l in range(self.n_slices):
                theta = self._slice_directions[l]
                proj1 = emb1 @ theta
                proj2 = emb2 @ theta
                proj1_sorted = np.sort(proj1)
                proj2_sorted = np.sort(proj2)
                total += np.sum((proj1_sorted - proj2_sorted) ** 2)
            return np.sqrt(total / self.n_slices)

    def fit_transform(self, graphs: List[np.ndarray]) -> np.ndarray:
        """
        Compute the SWG kernel matrix for a list of graphs.

        Returns
        -------
        K : np.ndarray (n_graphs × n_graphs)
            Positive definite kernel matrix.
        """
        n_graphs = len(graphs)
        # Step 1: Laplacian embeddings
        self._embeddings = [self._laplacian_embedding(g) for g in graphs]

        # Reset slice directions (will be generated on first call)
        self._slice_directions = None

        # Step 2: Pairwise GSW distances and kernel values
        K = np.zeros((n_graphs, n_graphs))
        for i in range(n_graphs):
            for j in range(i, n_graphs):
                dist = self._gsw_distance(self._embeddings[i], self._embeddings[j])
                kval = np.exp(-self.lambda_ * dist)
                K[i, j] = kval
                K[j, i] = kval
        return K


# ================================================================================================
# 3. CLASSIFICATION PIPELINE WITH LEAVE‑ONE‑OUT CV AND HYPERPARAMETER TUNING (FIXED AUC)
# ================================================================================================

def evaluate_swg_for_disease(
        disease_name: str,
        graphs: List[np.ndarray],
        labels: np.ndarray,
        param_grid: Dict,
        random_state: int = 42
) -> Tuple[float, float, float, Dict]:
    """
    Perform Leave‑One‑Out cross‑validation with grid search for λ and C.

    Returns
    -------
    best_acc : float
        Best mean accuracy over LOO folds.
    best_auc : float
        Best AUC computed on aggregated LOO predictions.
    runtime : float
        Time taken for kernel computation and grid search (seconds).
    best_params : dict
        Best hyperparameters found.
    """
    loo = LeaveOneOut()
    best_score = -np.inf
    best_params = None
    best_acc = 0.0
    best_auc = 0.0

    start_time = time.time()

    # Pre‑compute embeddings once for all graphs
    base_kernel = SlicedWassersteinGraphKernel(
        embedding_dim=param_grid['embedding_dim'][0],
        n_slices=param_grid['n_slices'][0],
        random_state=random_state
    )
    base_kernel._embeddings = [base_kernel._laplacian_embedding(g) for g in graphs]
    base_kernel._slice_directions = None  # Will be generated once

    n_graphs = len(graphs)

    # Grid search
    for lambda_ in param_grid['lambda']:
        for C in param_grid['C']:
            kernel_instance = SlicedWassersteinGraphKernel(
                embedding_dim=base_kernel.embedding_dim,
                n_slices=base_kernel.n_slices,
                lambda_=lambda_,
                random_state=random_state
            )
            # Use precomputed embeddings
            kernel_instance._embeddings = base_kernel._embeddings
            kernel_instance._slice_directions = base_kernel._generate_slice_directions(base_kernel.embedding_dim)

            # Compute full kernel matrix
            K = np.zeros((n_graphs, n_graphs))
            for i in range(n_graphs):
                for j in range(i, n_graphs):
                    dist = kernel_instance._gsw_distance(kernel_instance._embeddings[i],
                                                         kernel_instance._embeddings[j])
                    K[i, j] = K[j, i] = np.exp(-lambda_ * dist)

            svm = SVC(kernel='precomputed', C=C, probability=True, random_state=random_state, max_iter=10000)

            y_true_all = []
            y_prob_all = []
            accs = []

            for train_idx, test_idx in loo.split(K):
                K_train = K[train_idx][:, train_idx]
                K_test = K[test_idx][:, train_idx]
                y_train, y_test = labels[train_idx], labels[test_idx]

                svm.fit(K_train, y_train)
                y_pred = svm.predict(K_test)
                y_prob = svm.predict_proba(K_test)[:, 1]

                accs.append(accuracy_score(y_test, y_pred))
                y_true_all.append(y_test[0])
                y_prob_all.append(y_prob[0])

            mean_acc = np.mean(accs)
            mean_auc = roc_auc_score(y_true_all, y_prob_all)

            if mean_acc > best_score:
                best_score = mean_acc
                best_acc = mean_acc
                best_auc = mean_auc
                best_params = {'lambda': lambda_, 'C': C}

    runtime = time.time() - start_time
    return best_acc, best_auc, runtime, best_params


def compute_distance_matrix(
        graphs: List[np.ndarray],
        embedding_dim: int = 10,
        n_slices: int = 50,
        random_state: int = 42
) -> np.ndarray:
    """Compute pairwise GSW distance matrix for visualisation."""
    kernel = SlicedWassersteinGraphKernel(
        embedding_dim=embedding_dim,
        n_slices=n_slices,
        lambda_=1.0,
        random_state=random_state
    )
    kernel._embeddings = [kernel._laplacian_embedding(g) for g in graphs]
    kernel._slice_directions = kernel._generate_slice_directions(embedding_dim)

    n = len(graphs)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            dist = kernel._gsw_distance(kernel._embeddings[i], kernel._embeddings[j])
            D[i, j] = D[j, i] = dist
    return D


def plot_heatmap_for_disease(disease: str, D: np.ndarray, labels: np.ndarray):
    """Generate distance heatmap similar to Figure 2 in the paper."""
    # Select 10 NC and 10 patients for clear visualisation
    nc_idx = np.where(labels == 0)[0][:10]
    pat_idx = np.where(labels == 1)[0][:10]
    idx = np.concatenate([nc_idx, pat_idx])
    D_sub = D[np.ix_(idx, idx)]

    plt.figure(figsize=(7, 6))
    plt.imshow(D_sub, cmap='hot', interpolation='nearest')
    plt.colorbar(label='GSW Distance')
    plt.title(f'{disease}: Distance Matrix (10 NC vs 10 Patients)')
    plt.axhline(y=9.5, color='cyan', linestyle='--', linewidth=2)
    plt.axvline(x=9.5, color='cyan', linestyle='--', linewidth=2)
    plt.text(4.5, -1.5, 'Healthy', ha='center', fontsize=12, color='cyan')
    plt.text(14.5, -1.5, 'Patient', ha='center', fontsize=12, color='cyan')
    plt.tight_layout()
    plt.savefig(f'heatmap_{disease}.png', dpi=150)
    plt.show()


# ================================================================================================
# 4. MAIN EXPERIMENT (Disease‑by‑Disease Evaluation)
# ================================================================================================

def main():
    print("=" * 90)
    print("POSITIVE DEFINITE SLICED WASSERSTEIN GRAPH KERNEL FOR BRAIN DISEASE DIAGNOSIS")
    print("Implementation based on Ma et al., MICCAI 2023")
    print("=" * 90)

    diseases = ['ADHD', 'ASD', 'EMCI']
    n_subjects_per_dataset = 100  # 50 NC + 50 patients per disease
    n_nodes = 90

    # Hyperparameter grid (as in paper: λ ∈ {1e-2, 1e-1, ..., 1e2}, C ∈ {1e-3, ..., 1e3})
    param_grid = {
        'embedding_dim': [10],
        'n_slices': [50],
        'lambda': [0.01, 0.1, 1.0, 10.0, 100.0],
        'C': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
    }

    results = []

    for disease in diseases:
        print(f"\n{'=' * 40} {disease} vs NC {'=' * 40}")
        # Generate disease‑specific synthetic data
        graphs, labels = generate_disease_specific_networks(
            disease=disease,
            n_subjects=n_subjects_per_dataset,
            n_nodes=n_nodes,
            noise=0.05,
            seed=42
        )
        print(f"Generated {len(graphs)} graphs ({np.sum(labels == 0)} NC, {np.sum(labels == 1)} {disease})")

        # Evaluate SWG kernel with grid search and LOO CV
        best_acc, best_auc, runtime, best_params = evaluate_swg_for_disease(
            disease_name=disease,
            graphs=graphs,
            labels=labels,
            param_grid=param_grid,
            random_state=42
        )

        print(f"Best λ = {best_params['lambda']}, C = {best_params['C']}")
        print(f"Accuracy = {best_acc:.4f} ({best_acc * 100:.2f}%)")
        print(f"AUC      = {best_auc:.4f}")
        print(f"Runtime  = {runtime:.2f} seconds")

        # Store for final table
        results.append({
            'Disease': disease,
            'Accuracy': best_acc,
            'AUC': best_auc,
            'Runtime (s)': runtime,
            'Best λ': best_params['lambda'],
            'Best C': best_params['C']
        })

        # Compute distance matrix for visualisation (using best λ for consistency)
        D = compute_distance_matrix(graphs, embedding_dim=10, n_slices=50, random_state=42)
        plot_heatmap_for_disease(disease, D, labels)

    # --------------------------------------------------------------------------------------------
    # Final Summary Table (reproduces Table 1 and Table 2 of the paper)
    # --------------------------------------------------------------------------------------------
    print("\n" + "=" * 90)
    print("SUMMARY TABLE (cf. Table 1 and Table 2 in the paper)")
    print("=" * 90)
    print(f"{'Disease':<10} {'Accuracy (%)':<15} {'AUC':<10} {'Runtime (s)':<12} {'Best λ':<10} {'Best C':<10}")
    print("-" * 70)
    for res in results:
        print(f"{res['Disease']:<10} {res['Accuracy'] * 100:<15.2f} {res['AUC']:<10.4f} "
              f"{res['Runtime (s)']:<12.2f} {res['Best λ']:<10.2f} {res['Best C']:<10.2f}")

    print("\nComparison with reported paper results :")
    print("Paper Table 1: ADHD: 78.83%, ASD: 90.54%, EMCI: 85.44%")
    print("Paper Table 2: SWG runtime: 8–83 seconds (depending on dataset size).")


if __name__ == "__main__":
    main()   
