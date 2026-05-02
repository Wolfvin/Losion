"""
Active Learning Loop — GNoME-inspired self-improving training cycle.

Diadaptasi dari GNoME (Graph Networks for Materials Exploration, DeepMind 2023):
menggunakan active learning loop di mana model dilatih, membuat prediksi,
prediksi diverifikasi, dan data yang terverifikasi ditambahkan ke dataset
untuk iterasi berikutnya.

Konsep GNoME:
1. Train model pada data yang tersedia
2. Gunakan model untuk memprediksi kandidat baru
3. Filter kandidat (DFT verification dalam GNoME)
4. Tambahkan kandidat terverifikasi ke dataset
5. Retrain model dengan dataset yang diperluas
6. Ulangi

Adaptasi untuk Losion:
1. Train model pada data yang tersedia
2. Model menghasilkan output pada unlabeled/partial data
3. Confidence-based filtering: pilih output yang model yakin benar
4. Consistency-based filtering: pilih output yang konsisten antar paths
5. Tambahkan high-confidence outputs ke training data
6. Retrain model
7. Ulangi

Keunggulan:
- Self-improving: model semakin baik seiring iterasi
- Data-efficient: memanfaatkan unlabeled data
- Scalable: bisa parallel dengan banyak verifiers

Hardware: Pure PyTorch, kompatibel dengan CUDA, ROCm, dan CPU.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ActiveLearningConfig:
    """Konfigurasi untuk Active Learning Loop.

    Attributes:
        confidence_threshold: Threshold confidence untuk menerima prediksi.
        consistency_threshold: Threshold konsistensi antar paths.
        max_new_samples: Maksimum sampel baru per iterasi.
        retrain_epochs: Epoch retraining per iterasi.
        num_iterations: Jumlah iterasi active learning.
        use_curriculum: Gunakan curriculum dalam retraining.
    """

    confidence_threshold: float = 0.9
    consistency_threshold: float = 0.8
    max_new_samples: int = 10000
    retrain_epochs: int = 1
    num_iterations: int = 5
    use_curriculum: bool = True


class ConfidenceFilter(nn.Module):
    """Filter berdasarkan confidence model.

    Memilih output yang model paling yakin benar.
    Ini adalah "verification" analog dengan DFT dalam GNoME.

    Args:
        d_model: Dimensi model.
        threshold: Confidence threshold.
    """

    def __init__(self, d_model: int, threshold: float = 0.9) -> None:
        super().__init__()
        self.threshold = threshold

        # Confidence estimator
        self.confidence_head = nn.Sequential(
            nn.Linear(d_model, d_model // 4, bias=False),
            nn.SiLU(),
            nn.Linear(d_model // 4, 1, bias=False),
            nn.Sigmoid(),  # [0, 1]
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Filter berdasarkan confidence.

        Args:
            hidden_states: [batch, seq, d_model]

        Returns:
            Tuple (mask, confidence_scores):
            - mask: [batch] — True jika melebihi threshold
            - confidence_scores: [batch] — confidence per sample
        """
        # Pool over sequence
        pooled = hidden_states.mean(dim=1)  # [batch, d_model]
        confidence = self.confidence_head(pooled).squeeze(-1)  # [batch]
        mask = confidence >= self.threshold

        return mask, confidence


class ConsistencyFilter(nn.Module):
    """Filter berdasarkan konsistensi antar forward passes.

    Dengan dropout aktif, model menjalankan beberapa forward passes
    dan memilih output yang konsisten di antara mereka. Ini adalah
    implementasi self-consistency (Wang et al., 2022).

    Args:
        d_model: Dimensi model.
        num_passes: Jumlah forward passes untuk konsistensi check.
        threshold: Konsistensi threshold.
    """

    def __init__(
        self,
        d_model: int,
        num_passes: int = 3,
        threshold: float = 0.8,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_passes = num_passes
        self.threshold = threshold

        # Similarity metric
        self.similarity = nn.CosineSimilarity(dim=-1)

    def forward(
        self,
        hidden_states_list: List[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Filter berdasarkan konsistensi.

        Args:
            hidden_states_list: List of [batch, seq, d_model] tensors
                                dari multiple forward passes

        Returns:
            Tuple (mask, consistency_scores):
            - mask: [batch] — True jika konsisten
            - consistency_scores: [batch]
        """
        if len(hidden_states_list) < 2:
            # Tidak cukup passes untuk konsistensi check
            batch_size = hidden_states_list[0].shape[0]
            return torch.ones(batch_size, dtype=torch.bool, device=hidden_states_list[0].device), \
                   torch.ones(batch_size, device=hidden_states_list[0].device)

        # Pool each pass
        pooled_list = [h.mean(dim=1) for h in hidden_states_list]  # List of [batch, d_model]

        # Compute pairwise similarities
        batch_size = pooled_list[0].shape[0]
        total_sim = torch.zeros(batch_size, device=pooled_list[0].device)
        count = 0

        for i in range(len(pooled_list)):
            for j in range(i + 1, len(pooled_list)):
                sim = self.similarity(pooled_list[i], pooled_list[j])  # [batch]
                total_sim += sim
                count += 1

        consistency = total_sim / count  # [batch]
        mask = consistency >= self.threshold

        return mask, consistency


class ActiveLearningLoop:
    """Active Learning Loop — GNoME-inspired self-improving training.

    Mengotomatiskan siklus train → predict → verify → augment → retrain.

    Penggunaan:
        loop = ActiveLearningLoop(model, config)
        for iteration in range(config.num_iterations):
            loop.train_iteration(train_data)
            new_data = loop.predict_and_filter(unlabeled_data)
            loop.augment_training_data(new_data)

    Args:
        model: Losion model instance.
        config: Konfigurasi active learning.
    """

    def __init__(
        self,
        model: nn.Module,
        config: Optional[ActiveLearningConfig] = None,
    ) -> None:
        self.model = model
        self.config = config or ActiveLearningConfig()

        # Filters
        d_model = getattr(model, "d_model", 2048)
        self.confidence_filter = ConfidenceFilter(
            d_model, self.config.confidence_threshold
        )
        self.consistency_filter = ConsistencyFilter(
            d_model, threshold=self.config.consistency_threshold,
        )

        # Data buffer
        self.augmented_data: List[Dict[str, Any]] = []
        self.iteration_stats: List[Dict[str, Any]] = []

    def predict_and_filter(
        self,
        inputs: torch.Tensor,
    ) -> List[Dict[str, Any]]:
        """Predict pada unlabeled data dan filter berdasarkan confidence.

        Args:
            inputs: Unlabeled input data [batch, seq, d_model]

        Returns:
            List of accepted samples
        """
        self.model.eval()

        with torch.no_grad():
            # Forward pass
            outputs = self.model(inputs)

            # Confidence filtering
            conf_mask, conf_scores = self.confidence_filter(outputs)

            # Apply mask
            accepted_indices = conf_mask.nonzero(as_tuple=True)[0]

            # Collect accepted samples
            accepted = []
            for idx in accepted_indices:
                if len(accepted) >= self.config.max_new_samples:
                    break
                accepted.append({
                    "input": inputs[idx].clone(),
                    "output": outputs[idx].clone(),
                    "confidence": conf_scores[idx].item(),
                })

        return accepted

    def augment_training_data(
        self,
        new_samples: List[Dict[str, Any]],
    ) -> int:
        """Tambahkan sampel baru ke buffer training data.

        Args:
            new_samples: List sampel dari predict_and_filter

        Returns:
            Jumlah sampel yang ditambahkan
        """
        self.augmented_data.extend(new_samples)
        return len(new_samples)

    def get_iteration_summary(self) -> Dict[str, Any]:
        """Ringkasan iterasi terakhir.

        Returns:
            Dictionary statistik
        """
        if not self.iteration_stats:
            return {"status": "not_started"}

        return self.iteration_stats[-1]
