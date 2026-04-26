"""
RiemannInfer: Improving Transformer Inference through Riemannian Geometry

Implementation of the RiemannInfer framework from:
Mao et al., "RiemannInfer: improving transformer inference through Riemannian geometry",
Scientific Reports 16, 6636 (2026). https://doi.org/10.1038/s41598-026-37328-x

The algorithm works in three stages:
1. Topological Dimensionality Reduction - UMAP on hidden states
2. Riemannian Manifold Construction - Metric tensor from attention weights
3. Reasoning Path Planning - Minimum work path via Dijkstra

Usage:
    from nanochat.riemann_infer import RiemannInfer
    ri = RiemannInfer(model, tokenizer)
    best_tokens, scores = ri.infer(prompt_tokens, num_candidates=5)
"""

import heapq
import numpy as np
import torch
import torch.nn.functional as F
from typing import List, Tuple, Optional, Dict


# ---------------------------------------------------------------------------
# Stage 1: Topological Dimensionality Reduction (UMAP)
# ---------------------------------------------------------------------------

def _try_import_umap():
    """Lazy import of UMAP to avoid hard dependency."""
    try:
        import umap
        return umap
    except ImportError:
        return None


def reduce_hidden_states_umap(
    hidden_states: List[np.ndarray],
    target_dim: int = 16,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    metric: str = "euclidean",
) -> List[np.ndarray]:
    """
    Apply UMAP dimensionality reduction to hidden states from each layer.
    Preserves topological structure per Theorem 1 (Manifold Hypothesis).

    Args:
        hidden_states: List of arrays, each (N, d) where N = B*T tokens, d = model dim
        target_dim: Target dimensionality for UMAP reduction
        n_neighbors: UMAP neighborhood size (controls local vs global structure)
        min_dist: UMAP minimum distance parameter
        metric: Distance metric for UMAP

    Returns:
        List of reduced arrays, each (N, target_dim)
    """
    umap_mod = _try_import_umap()
    if umap_mod is None:
        # Fallback: use PCA if UMAP is not installed
        return _reduce_hidden_states_pca(hidden_states, target_dim)

    reduced = []
    for h in hidden_states:
        if h.shape[1] <= target_dim:
            reduced.append(h)
            continue
        reducer = umap_mod.UMAP(
            n_components=target_dim,
            n_neighbors=min(n_neighbors, h.shape[0] - 1),
            min_dist=min_dist,
            metric=metric,
            random_state=42,
        )
        reduced.append(reducer.fit_transform(h))
    return reduced


def _reduce_hidden_states_pca(
    hidden_states: List[np.ndarray],
    target_dim: int,
) -> List[np.ndarray]:
    """PCA fallback when UMAP is not available."""
    reduced = []
    for h in hidden_states:
        if h.shape[1] <= target_dim:
            reduced.append(h)
            continue
        # Center the data
        mean = h.mean(axis=0, keepdims=True)
        h_centered = h - mean
        # SVD-based PCA
        U, S, Vt = np.linalg.svd(h_centered, full_matrices=False)
        reduced.append(U[:, :target_dim] * S[:target_dim])
    return reduced


# ---------------------------------------------------------------------------
# Stage 2: Riemannian Manifold Construction
# ---------------------------------------------------------------------------

def compute_metric_tensor(attn_weights: np.ndarray, epsilon: float = 1e-6) -> np.ndarray:
    """
    Construct Riemannian metric tensor g^A from attention weights (Theorem 4).

    g^A_ij = delta_ij + (a_ij + a_ji) / 2

    The metric captures semantic distances: high attention weight -> small geodesic distance.

    Args:
        attn_weights: Attention weight matrix (T, T), averaged across heads and batch
        epsilon: Small constant for numerical stability (positive-definiteness)

    Returns:
        Metric tensor g of shape (T, T), symmetric positive-definite
    """
    T = attn_weights.shape[0]
    # Symmetrize attention weights
    a_sym = (attn_weights + attn_weights.T) / 2.0
    # Construct metric: g_ij = delta_ij + a_sym_ij
    g = np.eye(T) + a_sym
    # Ensure positive definiteness with regularization
    g += epsilon * np.eye(T)
    return g


def compute_geodesic_distance(
    h_i: np.ndarray,
    h_j: np.ndarray,
    g: np.ndarray,
    i: int,
    j: int,
) -> float:
    """
    Approximate geodesic distance between hidden states h_i and h_j
    using the attention-induced metric (Theorem 5 & 7).

    d_g(h_i, h_j) ≈ sqrt( (h_i - h_j)^T * G_local * (h_i - h_j) )

    where G_local is derived from the metric tensor entries at positions i, j.

    Args:
        h_i, h_j: Hidden state vectors at positions i and j
        g: Full metric tensor (T, T)
        i, j: Token position indices

    Returns:
        Approximate geodesic distance (scalar)
    """
    diff = h_i - h_j
    # The metric tensor entry g[i,j] modulates the Euclidean distance
    # Higher g[i,j] -> shorter geodesic (inverse relationship per Theorem 5)
    metric_scale = g[i, j]
    euclidean_dist = np.sqrt(np.dot(diff, diff) + 1e-10)
    # Geodesic distance is inversely related to attention-induced metric
    # d_g ≈ ||h_i - h_j|| / sqrt(g_ij) (from the approximate formula in Theorem 7)
    geodesic = euclidean_dist / np.sqrt(max(metric_scale, 1e-10))
    return geodesic


def compute_scalar_curvature(
    g: np.ndarray,
    attn_weights: np.ndarray,
    position: int,
    epsilon: float = 1e-8,
) -> float:
    """
    Compute scalar curvature R(h) at a given position using the attention
    entropy relationship (Theorem 9).

    R(h) ∝ -C * H(a)^α

    where H(a) is the entropy of the attention distribution at position h.
    Low entropy (concentrated attention) -> high curvature.

    Args:
        g: Metric tensor (T, T)
        attn_weights: Attention weights (T, T)
        position: Token position to compute curvature at
        epsilon: Numerical stability constant

    Returns:
        Scalar curvature at the given position
    """
    # Get attention distribution for this position
    a = attn_weights[position]
    # Clip to avoid log(0)
    a_clipped = np.clip(a, epsilon, 1.0)
    # Normalize to ensure it sums to 1
    a_clipped = a_clipped / (a_clipped.sum() + epsilon)
    # Shannon entropy
    entropy = -np.sum(a_clipped * np.log(a_clipped + epsilon))
    # Curvature is inversely related to entropy (Theorem 9)
    # High entropy (uniform attention) -> low curvature (flat space)
    # Low entropy (concentrated attention) -> high curvature (curved space)
    max_entropy = np.log(len(a) + epsilon)
    if max_entropy < epsilon:
        return 0.0
    normalized_entropy = entropy / max_entropy
    # R ∝ 1 / (normalized_entropy + epsilon) - 1 (so R=0 at max entropy)
    curvature = (1.0 / (normalized_entropy + 0.1)) - (1.0 / 1.1)
    return curvature


def compute_curvature_gradient(
    curvatures: np.ndarray,
) -> np.ndarray:
    """
    Compute the discrete gradient of curvature along token positions.
    dR/dx ≈ R[i+1] - R[i]

    Args:
        curvatures: Array of scalar curvatures at each position (T,)

    Returns:
        Curvature gradient array (T,), zero-padded at boundaries
    """
    T = len(curvatures)
    grad = np.zeros(T)
    if T > 1:
        grad[:-1] = np.diff(curvatures)
        grad[-1] = grad[-2] if T > 1 else 0.0
    return grad


# ---------------------------------------------------------------------------
# Stage 3: Reasoning Path Planning
# ---------------------------------------------------------------------------

def compute_work(
    geodesic_dist: float,
    curvature_grad: float,
    alpha: float = 1.0,
) -> float:
    """
    Compute the reasoning work between two adjacent points on the manifold.

    W = geodesic_dist * (1 + alpha * |dR/dx|)

    The work increases with:
    - Geodesic distance (longer paths need more work)
    - Curvature gradient (steeper curvature changes need more force)

    Args:
        geodesic_dist: Geodesic distance between the two points
        curvature_grad: Rate of curvature change along the path
        alpha: Weighting constant for curvature contribution

    Returns:
        Work value (scalar)
    """
    return geodesic_dist * (1.0 + alpha * abs(curvature_grad))


def build_manifold_graph(
    reduced_states: np.ndarray,
    metric_tensor: np.ndarray,
    curvatures: np.ndarray,
    curvature_grads: np.ndarray,
    k_neighbors: int = 10,
    alpha: float = 1.0,
) -> Dict[int, List[Tuple[float, int]]]:
    """
    Build a discretized weighted graph on the Riemannian manifold.

    Each token position is a node. Edges connect each node to its k-nearest
    neighbors (in the reduced space), weighted by the reasoning work.

    Args:
        reduced_states: UMAP-reduced hidden states (T, d_reduced)
        metric_tensor: Attention-induced metric tensor (T, T)
        curvatures: Scalar curvature at each position (T,)
        curvature_grads: Curvature gradient at each position (T,)
        k_neighbors: Number of nearest neighbors for graph connectivity
        alpha: Curvature weight in work formula

    Returns:
        Adjacency list: node -> [(work, neighbor), ...]
    """
    T = reduced_states.shape[0]
    k = min(k_neighbors, T - 1)

    # Compute pairwise distances in reduced space
    dists = np.zeros((T, T))
    for i in range(T):
        for j in range(i + 1, T):
            d = compute_geodesic_distance(
                reduced_states[i], reduced_states[j],
                metric_tensor, i, j,
            )
            dists[i, j] = d
            dists[j, i] = d

    # Build k-NN graph
    graph = {i: [] for i in range(T)}
    for i in range(T):
        # Find k nearest neighbors
        neighbor_dists = [(dists[i, j], j) for j in range(T) if j != i]
        neighbor_dists.sort()
        for d, j in neighbor_dists[:k]:
            # Average the curvature gradients at both endpoints
            avg_curv_grad = (curvature_grads[i] + curvature_grads[j]) / 2.0
            w = compute_work(d, avg_curv_grad, alpha)
            graph[i].append((w, j))

    return graph


def dijkstra_min_work_path(
    graph: Dict[int, List[Tuple[float, int]]],
    start: int,
    end: int,
) -> Tuple[List[int], float]:
    """
    Find the minimum-work reasoning path using Dijkstra's algorithm.

    Args:
        graph: Adjacency list from build_manifold_graph
        start: Start node index
        end: End node index

    Returns:
        (path, total_work): Ordered list of node indices and total work
    """
    T = len(graph)
    dist = {i: float('inf') for i in range(T)}
    prev = {i: None for i in range(T)}
    dist[start] = 0.0

    # Priority queue: (distance, node)
    pq = [(0.0, start)]

    while pq:
        d, u = heapq.heappop(pq)
        if d > dist[u]:
            continue
        if u == end:
            break
        for w, v in graph[u]:
            new_dist = dist[u] + w
            if new_dist < dist[v]:
                dist[v] = new_dist
                prev[v] = u
                heapq.heappush(pq, (new_dist, v))

    # Reconstruct path
    path = []
    node = end
    while node is not None:
        path.append(node)
        node = prev[node]
    path.reverse()

    if path[0] != start:
        # No path found
        return list(range(start, end + 1)), float('inf')

    return path, dist[end]


def compute_total_path_work(
    reduced_states: np.ndarray,
    metric_tensor: np.ndarray,
    curvatures: np.ndarray,
    curvature_grads: np.ndarray,
    alpha: float = 1.0,
) -> float:
    """
    Compute the total reasoning work along the sequential path
    (position 0 -> 1 -> 2 -> ... -> T-1).

    This is a simpler alternative to Dijkstra that scores the natural
    sequential reasoning path.

    Args:
        reduced_states: UMAP-reduced hidden states (T, d_reduced)
        metric_tensor: Attention-induced metric tensor (T, T)
        curvatures: Scalar curvature at each position (T,)
        curvature_grads: Curvature gradient at each position (T,)
        alpha: Curvature weight

    Returns:
        Total work along the sequential path
    """
    T = reduced_states.shape[0]
    total = 0.0
    for i in range(T - 1):
        d = compute_geodesic_distance(
            reduced_states[i], reduced_states[i + 1],
            metric_tensor, i, i + 1,
        )
        avg_grad = (curvature_grads[i] + curvature_grads[i + 1]) / 2.0
        total += compute_work(d, avg_grad, alpha)
    return total


# ---------------------------------------------------------------------------
# Main RiemannInfer Class
# ---------------------------------------------------------------------------

class RiemannInfer:
    """
    RiemannInfer: Riemannian geometry-based reasoning path optimizer.

    Wraps a nanochat GPT model and applies the three-stage pipeline:
    1. Generate N candidate continuations
    2. For each, extract hidden states + attention weights via forward_with_intermediates
    3. Build Riemannian manifold, compute reasoning work for each candidate
    4. Return the candidate with minimum total reasoning work

    Args:
        model: A nanochat GPT model (in eval mode)
        tokenizer: The nanochat tokenizer
        umap_dim: Target dimensionality for UMAP reduction (default 16)
        umap_n_neighbors: UMAP neighborhood parameter (default 15)
        alpha: Curvature weighting constant in work formula (default 1.0)
        k_neighbors: k-NN graph connectivity for Dijkstra (default 10)
        analysis_layer: Which layer's hidden states to analyze.
                        'last' = final layer, 'all' = concatenate all layers,
                        int = specific layer index (default 'last')
        use_dijkstra: If True, find optimal path via Dijkstra.
                      If False, score the sequential path (faster). (default False)
        verbose: Print debug information (default False)
    """

    def __init__(
        self,
        model,
        tokenizer,
        umap_dim: int = 16,
        umap_n_neighbors: int = 15,
        alpha: float = 1.0,
        k_neighbors: int = 10,
        analysis_layer: str = "last",
        use_dijkstra: bool = False,
        verbose: bool = False,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.umap_dim = umap_dim
        self.umap_n_neighbors = umap_n_neighbors
        self.alpha = alpha
        self.k_neighbors = k_neighbors
        self.analysis_layer = analysis_layer
        self.use_dijkstra = use_dijkstra
        self.verbose = verbose

    def _extract_features(
        self, token_ids: torch.Tensor,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run forward_with_intermediates and extract the hidden states and
        averaged attention weights for manifold analysis.

        Args:
            token_ids: (1, T) tensor of token ids

        Returns:
            hidden_states: (T, d) numpy array of hidden states
            attn_weights: (T, T) numpy array of averaged attention weights
        """
        logits, all_hidden, all_attn = self.model.forward_with_intermediates(token_ids)

        # Select which layer's hidden states to analyze
        if self.analysis_layer == "last":
            h = all_hidden[-1][0].cpu().numpy()  # (T, d)
        elif self.analysis_layer == "all":
            # Concatenate all layers
            h = np.concatenate([hs[0].cpu().numpy() for hs in all_hidden], axis=1)
        elif isinstance(self.analysis_layer, int):
            h = all_hidden[self.analysis_layer][0].cpu().numpy()
        else:
            h = all_hidden[-1][0].cpu().numpy()

        # Average attention weights across all layers and heads
        # Each entry is (B, H, T, T); average over B=0, then over H, then over layers
        attn_avg = np.mean(
            [a[0].mean(dim=0).cpu().numpy() for a in all_attn],
            axis=0,
        )  # (T, T)

        return h, attn_avg

    def score_sequence(
        self, token_ids: torch.Tensor,
    ) -> Dict[str, float]:
        """
        Score a token sequence using the RiemannInfer pipeline.

        Returns a dict with:
        - total_work: Total reasoning work along the path
        - mean_curvature: Average scalar curvature
        - path_work: Work along the optimal (or sequential) path
        """
        # Stage 1: Extract and reduce
        h, attn = self._extract_features(token_ids)
        T = h.shape[0]

        if T < 3:
            return {"total_work": 0.0, "mean_curvature": 0.0, "path_work": 0.0}

        # UMAP reduction
        reduced = reduce_hidden_states_umap(
            [h],
            target_dim=min(self.umap_dim, T - 1, h.shape[1]),
            n_neighbors=min(self.umap_n_neighbors, T - 1),
        )[0]

        # Stage 2: Build manifold
        g = compute_metric_tensor(attn)
        curvatures = np.array([
            compute_scalar_curvature(g, attn, i) for i in range(T)
        ])
        curv_grads = compute_curvature_gradient(curvatures)

        # Stage 3: Compute work
        if self.use_dijkstra:
            graph = build_manifold_graph(
                reduced, g, curvatures, curv_grads,
                k_neighbors=self.k_neighbors,
                alpha=self.alpha,
            )
            path, path_work = dijkstra_min_work_path(graph, 0, T - 1)
        else:
            path_work = compute_total_path_work(
                reduced, g, curvatures, curv_grads, alpha=self.alpha,
            )

        return {
            "total_work": path_work,
            "mean_curvature": float(curvatures.mean()),
            "path_work": path_work,
        }

    @torch.inference_mode()
    def infer(
        self,
        prompt_tokens: List[int],
        num_candidates: int = 5,
        max_tokens: int = 256,
        temperature: float = 0.8,
        top_k: int = 50,
    ) -> Tuple[List[int], Dict[str, float]]:
        """
        Generate multiple candidate continuations and select the one
        with the minimum reasoning work on the Riemannian manifold.

        Args:
            prompt_tokens: List of token ids for the prompt
            num_candidates: Number of candidate sequences to generate and score
            max_tokens: Maximum tokens to generate per candidate
            temperature: Sampling temperature
            top_k: Top-k sampling parameter

        Returns:
            best_tokens: The full token sequence (prompt + best continuation)
            best_score: Dict of scoring metrics for the best candidate
        """
        device = self.model.get_device()
        prompt_len = len(prompt_tokens)

        # Import Engine for efficient batched generation
        from nanochat.engine import Engine
        engine = Engine(self.model, self.tokenizer)

        # Generate candidate continuations
        candidates = []
        for seed in range(num_candidates):
            result_tokens = list(prompt_tokens)
            for token_column, _ in engine.generate(
                prompt_tokens,
                num_samples=1,
                max_tokens=max_tokens,
                temperature=temperature,
                top_k=top_k,
                seed=seed * 1337 + 42,
            ):
                result_tokens.append(token_column[0])
                # Stop at end tokens
                assistant_end = self.tokenizer.encode_special("<|assistant_end|>")
                bos = self.tokenizer.get_bos_token_id()
                if token_column[0] in (assistant_end, bos):
                    break
            candidates.append(result_tokens)

        if self.verbose:
            print(f"Generated {len(candidates)} candidates, lengths: {[len(c) for c in candidates]}")

        # Score each candidate
        best_work = float('inf')
        best_idx = 0
        all_scores = []

        for i, tokens in enumerate(candidates):
            # Truncate to model's max sequence length
            max_seq = self.model.config.sequence_len
            tokens_truncated = tokens[:max_seq]

            ids = torch.tensor([tokens_truncated], dtype=torch.long, device=device)
            score = self.score_sequence(ids)
            all_scores.append(score)

            if self.verbose:
                print(f"  Candidate {i}: work={score['total_work']:.4f}, "
                      f"curvature={score['mean_curvature']:.4f}, "
                      f"len={len(tokens_truncated)}")

            if score["total_work"] < best_work:
                best_work = score["total_work"]
                best_idx = i

        best_tokens = candidates[best_idx]
        best_score = all_scores[best_idx]

        if self.verbose:
            print(f"  Selected candidate {best_idx} with work={best_work:.4f}")

        return best_tokens, best_score
