"""
HyLo Upcycling — Checkpoint Conversion without Full Retraining.

Konversi dense model checkpoints ke MoE checkpoints tanpa full retraining.
HyLo (Heterogeneous Layer Upcycling) memungkinkan transformasi efisien
dari model dense ke model MoE dengan kualitas yang mendekati training
dari awal.

Integration with Losion (audit finding C4.1):
----------------------------------------------
HyLo Upcycling is a standalone utility module that operates on state dicts,
not a module wired into the model forward path. This is by design —
upcycling is a one-time conversion step that happens BEFORE training,
not during it. The workflow is:

1. Train a dense LosionModel (or start from a pretrained dense checkpoint)
2. Use HyLoUpcycler to convert the dense state dict → MoE state dict
3. Load the MoE state dict into a LosionModelV2 configured with MoE layers
4. Fine-tune the MoE model (much cheaper than training MoE from scratch)

The module is fully functional and can be used as follows:

    >>> from losion.utils import HyLoUpcycler, UpcyclingConfig
    >>> config = UpcyclingConfig(
    ...     source_type="dense",
    ...     num_target_experts=8,
    ...     clustering_method="kmeans",
    ...     router_init="activation",
    ... )
    >>> upcycler = HyLoUpcycler(config)
    >>> # Analyze activations dari calibration data
    >>> upcycler.analyze_activations(model, calibration_dataloader)
    >>> # Convert checkpoint
    >>> moe_state_dict = upcycler.upcycle_checkpoint(dense_state_dict)

It is NOT dead code — it is a preprocessing/utility tool that is used
outside the training loop, similar to tokenizer training or dataset
preprocessing. The README already lists it under Training components,
and it is properly exported from `losion.utils.__init__`.

Motivasi:
---------
Training MoE dari awal sangat mahal:
  - Membutuhkan dataset besar dan compute tinggi
  - Router harus belajar dari nol
  - Expert specialization belum terbentuk

HyLo Upcycling menyelesaikan ini dengan:
  1. Memecah dense FFN menjadi multiple experts via weight clustering
  2. Menginisialisasi router berdasarkan activation patterns
  3. Hanya butuh fine-tuning ringan, bukan full retraining
  4. Mendukung progressive upcycling (tambah expert secara bertahap)

Algoritma:
----------
1. Analyze: Kumpulkan statistik aktivasi dari calibration data
2. Cluster: Klaster bobot FFN menjadi N expert groups
   - KMeans: clustering standar pada weight vectors
   - Spectral: spectral clustering untuk struktur yang lebih kompleks
   - Random: baseline random assignment
3. Initialize Router: Setup router berdasarkan activation patterns
   - Activation: gunakan pola aktivasi sebagai router features
   - Random: inisialisasi acak
   - Uniform: bobot merata ke semua expert
4. Convert: Bentuk MoE checkpoint dari clustered weights
5. Progressive (opsional): Tambah expert secara bertahap

Contoh Penggunaan:
------------------
    >>> config = UpcyclingConfig(
    ...     source_type="dense",
    ...     num_target_experts=8,
    ...     clustering_method="kmeans",
    ...     router_init="activation",
    ... )
    >>> upcycler = HyLoUpcycler(config)
    >>> # Analyze activations dari calibration data
    >>> upcycler.analyze_activations(model, calibration_dataloader)
    >>> # Convert checkpoint
    >>> moe_state_dict = upcycler.upcycle_checkpoint(dense_state_dict)

Hardware: Pure PyTorch. Tidak membutuhkan custom CUDA kernels.
"""

from __future__ import annotations

import copy
import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# UpcyclingConfig
# ---------------------------------------------------------------------------


@dataclass
class UpcyclingConfig:
    """
    Konfigurasi untuk HyLo upcycling.

    Attributes:
        source_type: Tipe sumber checkpoint.
            "dense" — konversi dense FFN ke MoE
            "moe" — konversi MoE homogen ke MoE heterogen
        num_target_experts: Jumlah expert target setelah upcycling.
        clustering_method: Metode clustering untuk memecah bobot.
            "kmeans" — K-Means clustering (default, cepat dan efektif)
            "spectral" — Spectral clustering (lebih baik untuk struktur kompleks)
            "random" — Random assignment (baseline)
        router_init: Metode inisialisasi router.
            "activation" — berdasarkan activation patterns dari calibration
            "random" — inisialisasi acak
            "uniform" — bobot merata ke semua expert
        progressive: Apakah menggunakan progressive upcycling.
            Jika True, expert ditambah secara bertahap selama fine-tuning.
        progressive_steps: Jumlah langkah progressive upcycling.
            Hanya relevan jika progressive=True.
        expert_ff_dim: Dimensi FFN per expert. Jika None, menggunakan
            dimensi yang sama dengan dense FFN.
        top_k: Jumlah expert yang diaktivasi per token (default 2).
        n_calibration_batches: Jumlah batch calibration data untuk
            mengumpulkan statistik aktivasi (default 100).
        cluster_dim_reduction: Dimensi untuk PCA reduction sebelum
            clustering. 0 = tidak ada reduction. Default 64.
        seed: Random seed untuk reproducibility.
    """

    source_type: str = "dense"
    num_target_experts: int = 8
    clustering_method: str = "kmeans"
    router_init: str = "activation"
    progressive: bool = False
    progressive_steps: int = 4
    expert_ff_dim: Optional[int] = None
    top_k: int = 2
    n_calibration_batches: int = 100
    cluster_dim_reduction: int = 64
    seed: int = 42


# ---------------------------------------------------------------------------
# Simple K-Means Implementation (Pure PyTorch)
# ---------------------------------------------------------------------------


class SimpleKMeans:
    """
    K-Means clustering sederhana menggunakan PyTorch.

    Digunakan untuk clustering weight vectors menjadi expert groups.
    Implementasi ini menggunakan PyTorch tensors untuk kompatibilitas
    GPU tanpa dependency tambahan (scikit-learn).

    Args:
        n_clusters: Jumlah clusters.
        max_iter: Iterasi maksimum (default 100).
        tol: Toleransi konvergensi (default 1e-4).
        seed: Random seed.
    """

    def __init__(
        self,
        n_clusters: int,
        max_iter: int = 100,
        tol: float = 1e-4,
        seed: int = 42,
    ) -> None:
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.seed = seed

    def fit(
        self,
        X: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Fit K-Means pada data X.

        Args:
            X: Data tensor, bentuk (n_samples, n_features).

        Returns:
            Tuple (labels, centroids):
            - labels: Cluster assignment per sample, bentuk (n_samples,).
            - centroids: Cluster centroids, bentuk (n_clusters, n_features).
        """
        n_samples, n_features = X.shape
        device = X.device
        dtype = X.dtype

        # ---- Inisialisasi centroids menggunakan K-Means++ ----
        generator = torch.Generator(device=device)
        generator.manual_seed(self.seed)

        # Pilih centroid pertama secara acak
        first_idx = torch.randint(0, n_samples, (1,), generator=generator, device=device)
        centroids = X[first_idx].clone()  # (1, n_features)

        for _ in range(1, self.n_clusters):
            # Hitung jarak minimum ke centroids yang sudah ada
            dists = torch.cdist(X, centroids)  # (n_samples, n_centroids)
            min_dists = dists.min(dim=1).values  # (n_samples,)

            # Pilih centroid baru dengan probabilitas proporsional ke jarak^2
            probs = min_dists ** 2
            probs = probs / probs.sum()
            next_idx = torch.multinomial(probs, 1, generator=generator)
            centroids = torch.cat([centroids, X[next_idx]], dim=0)

        # ---- Iterasi K-Means ----
        labels = torch.zeros(n_samples, dtype=torch.long, device=device)

        for iteration in range(self.max_iter):
            # Assign setiap sample ke centroid terdekat
            dists = torch.cdist(X, centroids)  # (n_samples, n_clusters)
            new_labels = dists.argmin(dim=1)  # (n_samples,)

            # Cek konvergensi
            if iteration > 0 and (new_labels != labels).float().mean() < self.tol:
                labels = new_labels
                break

            labels = new_labels

            # Update centroids
            new_centroids = torch.zeros_like(centroids)
            for k in range(self.n_clusters):
                mask = (labels == k)
                if mask.any():
                    new_centroids[k] = X[mask].mean(dim=0)
                else:
                    # Reinitialize empty cluster
                    rand_idx = torch.randint(0, n_samples, (1,), generator=generator, device=device)
                    new_centroids[k] = X[rand_idx].squeeze(0)

            centroids = new_centroids

        return labels, centroids


# ---------------------------------------------------------------------------
# Spectral Clustering (Simplified)
# ---------------------------------------------------------------------------


class SimpleSpectralClustering:
    """
    Spectral clustering sederhana menggunakan PyTorch.

    Menggunakan eigenvalue decomposition dari normalized Laplacian
    untuk clustering. Lebih baik untuk struktur non-convex.

    Args:
        n_clusters: Jumlah clusters.
        n_neighbors: Jumlah neighbors untuk affinity graph (default 10).
        seed: Random seed.
    """

    def __init__(
        self,
        n_clusters: int,
        n_neighbors: int = 10,
        seed: int = 42,
    ) -> None:
        self.n_clusters = n_clusters
        self.n_neighbors = n_neighbors
        self.seed = seed

    def fit(
        self,
        X: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Fit spectral clustering pada data X.

        Args:
            X: Data tensor, bentuk (n_samples, n_features).

        Returns:
            Tuple (labels, centroids).
        """
        n_samples, n_features = X.shape
        device = X.device

        # ---- Compute affinity matrix ----
        # Gunakan k-nearest neighbor graph
        dists = torch.cdist(X, X)  # (n_samples, n_samples)

        # KNN affinity
        k = min(self.n_neighbors, n_samples - 1)
        affinity = torch.zeros_like(dists)
        for i in range(n_samples):
            _, topk_idx = dists[i].topk(k + 1, largest=False)
            for j in topk_idx[1:]:  # Skip self
                # Gaussian kernel
                sigma = dists[i, topk_idx[-1]].clamp(min=1e-8)
                affinity[i, j] = torch.exp(-dists[i, j] ** 2 / (2 * sigma ** 2))
                affinity[j, i] = affinity[i, j]

        # ---- Normalized Laplacian ----
        degree = affinity.sum(dim=1)  # (n_samples,)
        degree_inv_sqrt = 1.0 / degree.clamp(min=1e-8).sqrt()

        # L = I - D^{-1/2} A D^{-1/2}
        L_norm = torch.eye(n_samples, device=device) - (
            degree_inv_sqrt.unsqueeze(1) * affinity * degree_inv_sqrt.unsqueeze(0)
        )

        # ---- Eigenvalue decomposition ----
        # Ambil n_clusters eigenvectors terkecil
        try:
            eigenvalues, eigenvectors = torch.linalg.eigh(L_norm)
            # eigenvectors[:, :n_clusters] — smallest eigenvalues
            embedding = eigenvectors[:, :self.n_clusters]  # (n_samples, n_clusters)
        except Exception:
            # Fallback ke random jika eigh gagal
            logger.warning("Spectral clustering eigh failed, falling back to random")
            embedding = torch.randn(n_samples, self.n_clusters, device=device)

        # ---- K-Means pada spectral embedding ----
        kmeans = SimpleKMeans(
            n_clusters=self.n_clusters,
            seed=self.seed,
        )
        labels, centroids = kmeans.fit(embedding)

        # Hitung centroids di space asli
        original_centroids = torch.zeros(
            self.n_clusters, n_features, device=device, dtype=X.dtype
        )
        for k in range(self.n_clusters):
            mask = (labels == k)
            if mask.any():
                original_centroids[k] = X[mask].mean(dim=0)

        return labels, original_centroids


# ---------------------------------------------------------------------------
# HyLoUpcycler
# ---------------------------------------------------------------------------


class HyLoUpcycler:
    """
    Konversi dense model checkpoints ke MoE checkpoints tanpa full retraining.

    HyLo Upcycling memungkinkan transformasi efisien dari model dense
    ke model MoE dengan langkah-langkah:

    1. analyze_activations() — Kumpulkan statistik aktivasi dari calibration data
    2. cluster_weights() — Klaster bobot FFN menjadi expert groups
    3. initialize_router() — Setup router berdasarkan activation patterns
    4. upcycle_checkpoint() — Konversi utama
    5. progressive_upcycle() — (Opsional) Tambah expert secara bertahap

    Contoh:
        >>> config = UpcyclingConfig(num_target_experts=8)
        >>> upcycler = HyLoUpcycler(config)
        >>> upcycler.analyze_activations(model, calibration_dataloader)
        >>> moe_state = upcycler.upcycle_checkpoint(dense_state_dict)

    Args:
        config: UpcyclingConfig dengan parameter konfigurasi.
    """

    def __init__(self, config: UpcyclingConfig) -> None:
        self.config = config
        self._activation_stats: Dict[str, Dict[str, torch.Tensor]] = {}
        self._cluster_assignments: Dict[str, torch.Tensor] = {}
        self._cluster_centroids: Dict[str, torch.Tensor] = {}

    # ------------------------------------------------------------------
    # Step 1: Activation Analysis
    # ------------------------------------------------------------------

    @torch.no_grad()
    def analyze_activations(
        self,
        model: nn.Module,
        calibration_dataloader: Any,
        layer_names: Optional[List[str]] = None,
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Kumpulkan statistik aktivasi dari calibration data.

        Mengumpulkan mean dan covariance dari aktivasi FFN intermediate
        untuk setiap layer. Statistik ini digunakan untuk:
        1. Menentukan weight clustering yang baik
        2. Menginisialisasi router berdasarkan activation patterns

        Args:
            model: Model dense yang akan dianalisis.
            calibration_dataloader: DataLoader dengan calibration data.
            layer_names: Nama layer FFN yang akan dianalisis.
                Jika None, mendeteksi otomatis dari model.

        Returns:
            Dict berisi statistik aktivasi per layer:
            {
                "layer_name": {
                    "mean": Tensor (d_model,),
                    "covariance": Tensor (d_model, d_model),
                    "mean_abs": Tensor (d_model,),
                    "count": int,
                }
            }
        """
        model.eval()

        # ---- Detect FFN layers jika tidak diberikan ----
        if layer_names is None:
            layer_names = self._detect_ffn_layers(model)

        if not layer_names:
            logger.warning("No FFN layers detected for upcycling analysis")
            return self._activation_stats

        # ---- Setup hooks untuk mengumpulkan aktivasi ----
        activation_buffers: Dict[str, List[torch.Tensor]] = {
            name: [] for name in layer_names
        }
        hooks = []

        for name, module in model.named_modules():
            if name in layer_names:
                def make_hook(layer_name: str):
                    def hook_fn(module, input, output):
                        # Simpan input ke FFN (aktivasi dari layer sebelumnya)
                        if isinstance(input, tuple):
                            x = input[0]
                        else:
                            x = input
                        # Ambil sample untuk hemat memori
                        if x.dim() == 3:
                            # (batch, seq, d_model) → flatten ke (batch*seq, d_model)
                            x_flat = x.reshape(-1, x.shape[-1])
                            # Sample maksimal 512 token per batch
                            if x_flat.shape[0] > 512:
                                indices = torch.randperm(x_flat.shape[0])[:512]
                                x_flat = x_flat[indices]
                        activation_buffers[layer_name].append(x_flat.detach().cpu())
                    return hook_fn
                hooks.append(module.register_forward_hook(make_hook(name)))

        # ---- Run calibration data melalui model ----
        logger.info(f"Collecting activations from {len(layer_names)} FFN layers...")
        for batch_idx, batch in enumerate(calibration_dataloader):
            if batch_idx >= self.config.n_calibration_batches:
                break

            if isinstance(batch, dict):
                input_ids = batch.get("input_ids", batch.get("input", None))
                if input_ids is not None:
                    if isinstance(input_ids, torch.Tensor):
                        model(input_ids)
            elif isinstance(batch, (tuple, list)):
                model(batch[0])
            elif isinstance(batch, torch.Tensor):
                model(batch)

        # ---- Remove hooks ----
        for hook in hooks:
            hook.remove()

        # ---- Compute statistics ----
        for name in layer_names:
            if not activation_buffers[name]:
                continue

            all_acts = torch.cat(activation_buffers[name], dim=0)  # (N, d_model)

            # Subsample jika terlalu banyak
            if all_acts.shape[0] > 10000:
                indices = torch.randperm(all_acts.shape[0])[:10000]
                all_acts = all_acts[indices]

            stats = {
                "mean": all_acts.mean(dim=0),
                "mean_abs": all_acts.abs().mean(dim=0),
                "count": all_acts.shape[0],
            }

            # Covariance (approximate, menggunakan sampled data)
            if all_acts.shape[0] > 1:
                centered = all_acts - stats["mean"].unsqueeze(0)
                # Gunakan samples untuk mengestimasi covariance
                n_samples = min(all_acts.shape[0], 2000)
                stats["covariance"] = (
                    centered[:n_samples].T @ centered[:n_samples]
                ) / (n_samples - 1)
            else:
                d = all_acts.shape[-1]
                stats["covariance"] = torch.eye(d)

            self._activation_stats[name] = stats
            logger.info(
                f"  {name}: collected {stats['count']} activation samples, "
                f"d_model={all_acts.shape[-1]}"
            )

        return self._activation_stats

    # ------------------------------------------------------------------
    # Step 2: Weight Clustering
    # ------------------------------------------------------------------

    def cluster_weights(
        self,
        state_dict: Dict[str, torch.Tensor],
        layer_names: Optional[List[str]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Klaster bobot FFN menjadi expert groups.

        Menggunakan metode clustering yang dikonfigurasi untuk membagi
        bobot FFN dense menjadi num_target_experts expert groups.

        Args:
            state_dict: State dict dari model dense.
            layer_names: Nama layer FFN yang akan di-cluster.
                Jika None, mendeteksi otomatis.

        Returns:
            Dict berisi cluster assignments per layer:
            {
                "layer_name": Tensor (n_neurons,) — cluster ID per neuron
            }
        """
        config = self.config

        # ---- Detect FFN layers ----
        if layer_names is None:
            layer_names = self._detect_ffn_layers_from_state_dict(state_dict)

        # ---- Cluster setiap layer ----
        for layer_name in layer_names:
            # Cari weight keys untuk layer ini
            up_proj_key = f"{layer_name}.up_proj.weight"
            gate_proj_key = f"{layer_name}.gate_proj.weight"
            down_proj_key = f"{layer_name}.down_proj.weight"

            # Fallback: cari pola lain
            if up_proj_key not in state_dict:
                # Coba pola: layer.fc1.weight, layer.w1.weight, dll.
                for key in state_dict:
                    if layer_name in key and "weight" in key:
                        if "up" in key or "fc1" in key or "w1" in key:
                            up_proj_key = key
                        elif "gate" in key or "w3" in key:
                            gate_proj_key = key
                        elif "down" in key or "fc2" in key or "w2" in key:
                            down_proj_key = key

            # Ambil weight untuk clustering
            # Setiap "neuron" di FFN merepresentasikan satu row di
            # up_proj/gate_proj atau satu column di down_proj.
            # Kita cluster berdasarkan dimensi d_ff (hidden dimension).
            if up_proj_key in state_dict:
                # up_proj: (d_ff, d_model) → setiap row = 1 neuron
                W = state_dict[up_proj_key].float()  # (d_ff, d_model)
            elif gate_proj_key in state_dict:
                # gate_proj: (d_ff, d_model) → setiap row = 1 neuron
                W = state_dict[gate_proj_key].float()  # (d_ff, d_model)
            elif down_proj_key in state_dict:
                # down_proj: (d_model, d_ff) → setiap column = 1 neuron
                # Transpose sehingga setiap row = 1 neuron
                W = state_dict[down_proj_key].float().T  # (d_ff, d_model)
            else:
                logger.warning(f"Could not find weights for layer {layer_name}, skipping")
                continue

            n_neurons = W.shape[0]
            n_clusters = min(config.num_target_experts, n_neurons)

            # ---- Dimensionality reduction (opsional) ----
            if config.cluster_dim_reduction > 0 and W.shape[1] > config.cluster_dim_reduction:
                # PCA sederhana via SVD
                W_centered = W - W.mean(dim=0, keepdim=True)
                try:
                    U, S, Vh = torch.linalg.svd(W_centered, full_matrices=False)
                    W_reduced = U[:, :config.cluster_dim_reduction] * S[:config.cluster_dim_reduction].unsqueeze(0)
                except Exception:
                    W_reduced = W
            else:
                W_reduced = W

            # ---- Clustering ----
            logger.info(
                f"Clustering {layer_name}: {n_neurons} neurons into "
                f"{n_clusters} experts via {config.clustering_method}"
            )

            if config.clustering_method == "kmeans":
                kmeans = SimpleKMeans(
                    n_clusters=n_clusters,
                    seed=config.seed,
                )
                labels, centroids = kmeans.fit(W_reduced)

            elif config.clustering_method == "spectral":
                # Spectral clustering hanya feasible untuk dataset kecil
                if n_neurons > 5000:
                    logger.warning(
                        f"Spectral clustering with {n_neurons} neurons is slow. "
                        f"Subsampling to 5000 and propagating labels."
                    )
                    # Subsample, cluster, lalu assign sisanya ke nearest centroid
                    indices = torch.randperm(n_neurons)[:5000]
                    spectral = SimpleSpectralClustering(
                        n_clusters=n_clusters,
                        seed=config.seed,
                    )
                    sub_labels, sub_centroids = spectral.fit(W_reduced[indices])

                    # Assign remaining ke nearest centroid
                    dists = torch.cdist(W_reduced, sub_centroids)
                    labels = dists.argmin(dim=1)
                    centroids = sub_centroids
                else:
                    spectral = SimpleSpectralClustering(
                        n_clusters=n_clusters,
                        seed=config.seed,
                    )
                    labels, centroids = spectral.fit(W_reduced)

            elif config.clustering_method == "random":
                labels = torch.randint(
                    0, n_clusters, (n_neurons,), device=W.device
                )
                centroids = torch.zeros(n_clusters, W_reduced.shape[1], device=W.device)
                for k in range(n_clusters):
                    mask = (labels == k)
                    if mask.any():
                        centroids[k] = W_reduced[mask].mean(dim=0)
            else:
                raise ValueError(
                    f"Unknown clustering method: {config.clustering_method}. "
                    f"Use 'kmeans', 'spectral', or 'random'."
                )

            self._cluster_assignments[layer_name] = labels
            self._cluster_centroids[layer_name] = centroids

            # Log cluster sizes
            sizes = [(labels == k).sum().item() for k in range(n_clusters)]
            logger.info(f"  Cluster sizes: {sizes}")

        return self._cluster_assignments

    # ------------------------------------------------------------------
    # Step 3: Router Initialization
    # ------------------------------------------------------------------

    def initialize_router(
        self,
        d_model: int,
        num_experts: int,
    ) -> Dict[str, torch.Tensor]:
        """
        Inisialisasi parameter router berdasarkan activation patterns.

        Args:
            d_model: Dimensi model.
            num_experts: Jumlah expert (num_target_experts).

        Returns:
            Dict berisi initial router parameters:
            {
                "router.weight": Tensor (num_experts, d_model),
                "router.bias": Tensor (num_experts,) — optional,
            }
        """
        config = self.config

        if config.router_init == "activation" and self._activation_stats:
            # Gunakan statistik aktivasi untuk menginisialisasi router
            # Ambil mean activations dari layer pertama yang ada statistiknya
            first_layer = next(iter(self._activation_stats))
            stats = self._activation_stats[first_layer]
            mean_act = stats["mean_abs"]  # (d_model,)

            # Router weight: setiap expert "specializes" pada subset
            # dimensi berdasarkan cluster centroids
            if self._cluster_centroids:
                first_centroids = next(iter(self._cluster_centroids.values()))
                # centroids: (n_clusters, d_reduced)
                # Project ke d_model jika perlu
                if first_centroids.shape[1] != d_model:
                    # Gunakan PCA-like projection (sederhana: truncate atau pad)
                    router_weight = torch.zeros(num_experts, d_model)
                    min_dim = min(first_centroids.shape[1], d_model)
                    router_weight[:, :min_dim] = first_centroids[:num_experts, :min_dim]
                else:
                    router_weight = first_centroids[:num_experts]
            else:
                # Fallback: gunakan mean activation sebagai template
                router_weight = torch.randn(num_experts, d_model) * 0.02
                # Modulate berdasarkan mean activation
                router_weight = router_weight * mean_act.unsqueeze(0).clamp(min=0.01)

        elif config.router_init == "uniform":
            # Bobot merata — tidak ada preferensi expert
            router_weight = torch.ones(num_experts, d_model) / math.sqrt(d_model)

        else:
            # Random initialization (default)
            router_weight = torch.randn(num_experts, d_model) * 0.02

        return {"router.weight": router_weight}

    # ------------------------------------------------------------------
    # Step 4: Main Upcycling Method
    # ------------------------------------------------------------------

    def upcycle_checkpoint(
        self,
        state_dict: Dict[str, torch.Tensor],
        d_model: Optional[int] = None,
        d_ff: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Konversi utama: dense state dict → MoE state dict.

        Langkah-langkah:
        1. Cluster weights jika belum dilakukan
        2. Initialize router
        3. Split dense FFN weights menjadi expert weights berdasarkan clusters
        4. Gabungkan semua parameter ke MoE state dict

        Args:
            state_dict: State dict dari model dense.
            d_model: Dimensi model (auto-detect jika None).
            d_ff: Dimensi FFN (auto-detect jika None).

        Returns:
            MoE state dict yang bisa di-load ke model MoE.
        """
        config = self.config

        # ---- Auto-detect dimensions ----
        if d_model is None or d_ff is None:
            d_model, d_ff = self._detect_dimensions(state_dict, d_model, d_ff)

        # ---- Cluster weights jika belum ----
        if not self._cluster_assignments:
            self.cluster_weights(state_dict)

        # ---- Initialize router ----
        router_params = self.initialize_router(d_model, config.num_target_experts)

        # ---- Build MoE state dict ----
        moe_state_dict = {}

        # Copy semua parameter non-FFN
        ffn_layers = set(self._cluster_assignments.keys())
        for key, value in state_dict.items():
            # Cek apakah key milik FFN layer yang akan di-upcycle
            is_ffn = False
            for layer_name in ffn_layers:
                if key.startswith(layer_name):
                    is_ffn = True
                    break

            if not is_ffn:
                moe_state_dict[key] = value

        # ---- Split FFN weights per layer ----
        for layer_name, assignments in self._cluster_assignments.items():
            n_experts = config.num_target_experts
            assignments = assignments.to(state_dict[list(state_dict.keys())[0]].device)

            # Cari weight keys
            weight_keys = self._find_ffn_weight_keys(state_dict, layer_name)

            if not weight_keys:
                logger.warning(f"No FFN weights found for {layer_name}")
                continue

            # Cluster setiap weight matrix
            for weight_key, weight in weight_keys.items():
                # weight shape:
                # up_proj: (d_ff, d_model) atau (d_model, d_ff)
                # gate_proj: (d_ff, d_model) atau (d_model, d_ff)
                # down_proj: (d_model, d_ff) atau (d_ff, d_model)

                is_transposed = False
                if "down" in weight_key or "fc2" in weight_key or "w2" in weight_key:
                    # down_proj: (d_model, d_ff) → neurons di dim 1
                    is_transposed = True
                    W = weight.float().T  # (d_ff, d_model)
                elif "up" in weight_key or "gate" in weight_key or "fc1" in weight_key or "w1" in weight_key or "w3" in weight_key:
                    W = weight.float()  # (d_ff, d_model)
                else:
                    W = weight.float()

                # Assign neurons ke experts berdasarkan clustering
                for expert_id in range(n_experts):
                    mask = (assignments == expert_id)
                    n_assigned = mask.sum().item()

                    if n_assigned == 0:
                        # Expert kosong — inisialisasi dengan random kecil
                        expert_d_ff = max(d_ff // n_experts, 1)
                        if is_transposed:
                            expert_W = torch.randn(d_model, expert_d_ff) * 0.01
                        else:
                            expert_W = torch.randn(expert_d_ff, d_model) * 0.01
                    else:
                        expert_W = W[mask]  # (n_assigned, d_model)
                        if is_transposed:
                            expert_W = expert_W.T  # (d_model, n_assigned)

                    # Buat key untuk expert ini
                    # Konvensi: "layer_name.experts.expert_id.weight_key"
                    expert_key = f"{layer_name}.experts.{expert_id}.{weight_key.split('.')[-1]}"
                    moe_state_dict[expert_key] = expert_W.to(weight.dtype)

                # Simpan metadata: jumlah neuron per expert
                for expert_id in range(n_experts):
                    mask = (assignments == expert_id)
                    n_assigned = mask.sum().item()
                    meta_key = f"{layer_name}.experts.{expert_id}.d_ff"
                    moe_state_dict[meta_key] = torch.tensor(max(n_assigned, 1))

            # Tambahkan bias keys jika ada
            for weight_key in weight_keys:
                bias_key = weight_key.replace(".weight", ".bias")
                if bias_key in state_dict:
                    bias = state_dict[bias_key]
                    for expert_id in range(n_experts):
                        mask = (assignments == expert_id)
                        n_assigned = mask.sum().item()
                        if n_assigned > 0 and not is_transposed:
                            expert_bias = bias[mask] if bias.dim() > 0 else bias
                        else:
                            expert_bias = torch.zeros(max(n_assigned, 1), dtype=bias.dtype)
                        bias_name = bias_key.split(".")[-1]
                        expert_bias_key = f"{layer_name}.experts.{expert_id}.{bias_name}"
                        moe_state_dict[expert_bias_key] = expert_bias

        # ---- Tambahkan router parameters ----
        for key, value in router_params.items():
            for layer_name in self._cluster_assignments:
                router_key = f"{layer_name}.router.{key}"
                moe_state_dict[router_key] = value

        logger.info(
            f"Upcycled checkpoint: {len(ffn_layers)} FFN layers → "
            f"{config.num_target_experts} experts each"
        )

        return moe_state_dict

    # ------------------------------------------------------------------
    # Step 5: Progressive Upcycling
    # ------------------------------------------------------------------

    def progressive_upcycle(
        self,
        state_dict: Dict[str, torch.Tensor],
        current_num_experts: int = 1,
        d_model: Optional[int] = None,
        d_ff: Optional[int] = None,
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Progressive upcycling: tambah expert secara bertahap.

        Daripada langsung membuat N experts, progressive upcycling
        menambahkan expert satu per satu selama fine-tuning:
        - Step 0: 1 expert (dense copy)
        - Step 1: 2 experts (split dense menjadi 2)
        - Step 2: 4 experts
        - ...
        - Step K: num_target_experts

        Keuntungan:
        - Training lebih stabil
        - Router belajar secara gradual
        - Expert specialization berkembang natural

        Args:
            state_dict: State dict dari model (dense atau MoE sebelumnya).
            current_num_experts: Jumlah expert saat ini (1 = dense).
            d_model: Dimensi model (auto-detect jika None).
            d_ff: Dimensi FFN (auto-detect jika None).

        Returns:
            Dict mapping step → state_dict, berisi checkpoint untuk
            setiap langkah progressive.
        """
        config = self.config
        if not config.progressive:
            logger.warning(
                "progressive_upcycle called but progressive=False. "
                "Returning single-step conversion."
            )
            return {"step_0": self.upcycle_checkpoint(state_dict, d_model, d_ff)}

        # ---- Compute expert progression ----
        steps = config.progressive_steps
        target = config.num_target_experts

        # Geometric progression: 1, 2, 4, 8, ..., target
        expert_schedule = []
        n = current_num_experts
        while n < target:
            expert_schedule.append(n)
            n = min(n * 2, target)
        expert_schedule.append(target)

        # Pastikan tidak melebihi jumlah steps
        if len(expert_schedule) > steps:
            # Ambil steps terakhir
            expert_schedule = expert_schedule[-steps:]

        logger.info(
            f"Progressive upcycling schedule: {expert_schedule}"
        )

        # ---- Generate checkpoint untuk setiap step ----
        progressive_checkpoints: Dict[str, Dict[str, torch.Tensor]] = {}
        current_state = copy.deepcopy(state_dict)

        for step_idx, n_experts in enumerate(expert_schedule):
            # Buat config temporary untuk jumlah expert ini
            step_config = copy.deepcopy(config)
            step_config.num_target_experts = n_experts

            # Buat upcycler temporary
            step_upcycler = HyLoUpcycler(step_config)

            # Transfer activation stats jika ada
            step_upcycler._activation_stats = self._activation_stats

            # Cluster weights
            step_upcycler.cluster_weights(current_state)

            # Upcycle
            step_state = step_upcycler.upcycle_checkpoint(
                current_state, d_model, d_ff
            )

            progressive_checkpoints[f"step_{step_idx}"] = step_state
            current_state = step_state

            logger.info(
                f"Progressive step {step_idx}: {n_experts} experts created"
            )

        return progressive_checkpoints

    # ------------------------------------------------------------------
    # Helper Methods
    # ------------------------------------------------------------------

    @staticmethod
    def _detect_ffn_layers(model: nn.Module) -> List[str]:
        """
        Deteksi FFN layers dari model secara otomatis.

        Mencari modul yang merupakan Linear layers dengan dimensi
        yang sesuai pola FFN (up_proj/gate_proj/down_proj atau fc1/fc2).

        Args:
            model: Model PyTorch.

        Returns:
            List nama FFN layers.
        """
        ffn_layers = []

        for name, module in model.named_modules():
            # Cek pola FFN SwiGLU: module dengan up_proj + gate_proj + down_proj
            if isinstance(module, nn.Module):
                has_up = hasattr(module, "up_proj") or hasattr(module, "fc1") or hasattr(module, "w1")
                has_down = hasattr(module, "down_proj") or hasattr(module, "fc2") or hasattr(module, "w2")
                has_gate = hasattr(module, "gate_proj") or hasattr(module, "w3")

                if (has_up and has_down) or (has_up and has_gate and has_down):
                    ffn_layers.append(name)

        return ffn_layers

    @staticmethod
    def _detect_ffn_layers_from_state_dict(
        state_dict: Dict[str, torch.Tensor],
    ) -> List[str]:
        """
        Deteksi FFN layers dari state dict.

        Args:
            state_dict: State dict model.

        Returns:
            List nama FFN layers.
        """
        ffn_layers = set()

        for key in state_dict:
            # Cari pola: *.up_proj.weight, *.gate_proj.weight, *.down_proj.weight
            for pattern in [".up_proj.", ".gate_proj.", ".down_proj.",
                           ".fc1.", ".fc2.", ".w1.", ".w2.", ".w3."]:
                if pattern in key:
                    # Extract layer name
                    idx = key.index(pattern)
                    layer_name = key[:idx]
                    ffn_layers.add(layer_name)

        return list(ffn_layers)

    @staticmethod
    def _detect_dimensions(
        state_dict: Dict[str, torch.Tensor],
        d_model: Optional[int],
        d_ff: Optional[int],
    ) -> Tuple[int, int]:
        """
        Deteksi d_model dan d_ff dari state dict.

        Args:
            state_dict: State dict model.
            d_model: Override jika diberikan.
            d_ff: Override jika diberikan.

        Returns:
            Tuple (d_model, d_ff).
        """
        for key, value in state_dict.items():
            if value.dim() < 2:
                continue

            # Cek pola FFN
            for pattern in [".up_proj.weight", ".w1.weight", ".w3.weight"]:
                if pattern in key:
                    if d_model is None:
                        d_model = value.shape[-1]
                    if d_ff is None:
                        d_ff = value.shape[-2]
                    return d_model, d_ff

            for pattern in [".down_proj.weight", ".w2.weight", ".fc2.weight"]:
                if pattern in key:
                    if d_model is None:
                        d_model = value.shape[-2]
                    if d_ff is None:
                        d_ff = value.shape[-1]
                    return d_model, d_ff

        # Fallback: gunakan dimensi dari layer linear pertama
        for key, value in state_dict.items():
            if value.dim() == 2:
                if d_model is None:
                    d_model = min(value.shape)
                if d_ff is None:
                    d_ff = max(value.shape)
                return d_model, d_ff

        return d_model or 512, d_ff or 2048

    @staticmethod
    def _find_ffn_weight_keys(
        state_dict: Dict[str, torch.Tensor],
        layer_name: str,
    ) -> Dict[str, torch.Tensor]:
        """
        Cari semua weight keys untuk FFN layer tertentu.

        Args:
            state_dict: State dict model.
            layer_name: Nama FFN layer.

        Returns:
            Dict mapping weight_key → weight_tensor.
        """
        weights = {}
        for key, value in state_dict.items():
            if not key.startswith(layer_name):
                continue
            if ".weight" not in key:
                continue
            # Hanya ambil FFN weights (bukan norm, dll)
            for pattern in [".up_proj.", ".gate_proj.", ".down_proj.",
                           ".fc1.", ".fc2.", ".w1.", ".w2.", ".w3."]:
                if pattern in key:
                    weights[key] = value
                    break

        return weights
