"""
EngramMemory — O(1) factual retrieval via hash table.

Diadaptasi dari konsep DeepSeek-V4: fakta disimpan dalam hash table
di DRAM (bukan VRAM), dengan retrieval waktu konstan dan overhead
VRAM < 3%.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class EngramEntry:
    """Satu entri dalam Engram Memory."""

    subject_hash: int  # Hash dari subject
    embedding: torch.Tensor  # Fact embedding [embedding_dim] — disimpan di CPU
    metadata: Optional[Dict[str, str]]  # Metadata opsional


class EngramMemory(nn.Module):
    """
    Engram Memory — O(1) factual retrieval via hash table (dari DeepSeek-V4).

    Konsep: Fakta disimpan dalam hash table di DRAM (bukan VRAM).
    Retrieval dilakukan dalam waktu konstan O(1) dengan overhead
    VRAM < 3%.

    Implementasi:
    - Hash table: key=subject_hash, value=fact_embedding
    - Query: hash the query subject -> lookup -> return fact embedding
    - Stored in CPU DRAM, moved to GPU only during retrieval
    - Supports dynamic insertion of new facts without retraining

    Args:
        d_model: Model dimension
        num_buckets: Number of hash buckets (default 1_000_000)
        embedding_dim: Dimension of fact embeddings (default 256)
    """

    def __init__(
        self,
        d_model: int,
        num_buckets: int = 1_000_000,
        embedding_dim: int = 256,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_buckets = num_buckets
        self.embedding_dim = embedding_dim

        # Hash table disimpan di CPU (DRAM) untuk efisiensi memori
        # Embedding table: [num_buckets, embedding_dim] — seluruhnya di CPU
        self.register_buffer(
            "embedding_table",
            torch.zeros(num_buckets, embedding_dim),
            persistent=True,
        )

        # Occupation mask: menandai bucket yang sudah terisi
        self.register_buffer(
            "occupation_mask",
            torch.zeros(num_buckets, dtype=torch.bool),
            persistent=True,
        )

        # Proyeksi dari embedding_dim ke d_model (di GPU)
        self.output_proj = nn.Linear(embedding_dim, d_model, bias=False)

        # Query encoder: meng-encode query menjadi subject representation
        self.query_encoder = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, embedding_dim),
        )

        # Counter untuk tracking penggunaan
        self._insert_count: int = 0

        # Mapping dari subject_hash ke bucket index (untuk collision handling)
        self._hash_to_bucket: Dict[int, int] = {}

    @staticmethod
    def _compute_hash(subject: str) -> int:
        """
        Hitung hash deterministik dari subject string.

        Menggunakan SHA-256 untuk distribusi yang merata.

        Args:
            subject: String subject yang akan di-hash

        Returns:
            Integer hash value
        """
        digest = hashlib.sha256(subject.encode("utf-8")).hexdigest()
        return int(digest, 16)

    def _hash_to_index(self, hash_value: int) -> int:
        """
        Konversi hash value ke bucket index.

        Menggunakan modulo untuk mapping ke range bucket.
        Collision handling: linear probing.

        Args:
            hash_value: Hash value dari subject

        Returns:
            Bucket index
        """
        return hash_value % self.num_buckets

    def _find_bucket(
        self, hash_value: int, check_occupied: bool = True
    ) -> Optional[int]:
        """
        Cari bucket untuk hash value tertentu.

        Linear probing untuk collision handling.

        Args:
            hash_value: Hash value
            check_occupied: Jika True, cari bucket yang sudah terisi
                           Jika False, cari bucket kosong

        Returns:
            Bucket index, atau None jika tidak ditemukan
        """
        start_idx = self._hash_to_index(hash_value)
        max_probes = min(100, self.num_buckets)  # Batasi probing

        for probe in range(max_probes):
            idx = (start_idx + probe) % self.num_buckets
            if check_occupied:
                # Mencari bucket yang berisi hash ini
                if self.occupation_mask[idx]:
                    # Verifikasi hash cocok (via hash_to_bucket mapping)
                    if self._hash_to_bucket.get(idx) == hash_value:
                        return idx
            else:
                # Mencari bucket kosong
                if not self.occupation_mask[idx]:
                    return idx

        return None

    def insert(
        self,
        subject: str,
        embedding: Optional[torch.Tensor] = None,
        metadata: Optional[Dict[str, str]] = None,
    ) -> bool:
        """
        Insert fakta baru ke Engram Memory.

        Jika embedding tidak diberikan, akan dibuat embedding acak
        yang nantinya bisa di-refine selama training.

        Args:
            subject: String subject (e.g., "Paris", "Einstein")
            embedding: Fact embedding [embedding_dim], opsional
            metadata: Metadata tambahan, opsional

        Returns:
            True jika berhasil di-insert, False jika table penuh
        """
        hash_value = self._compute_hash(subject)

        # Cek apakah subject sudah ada
        existing_bucket = self._find_bucket(hash_value, check_occupied=True)
        if existing_bucket is not None:
            # Update embedding yang sudah ada
            if embedding is not None:
                self.embedding_table[existing_bucket] = embedding.cpu()
            return True

        # Cari bucket kosong
        empty_bucket = self._find_bucket(hash_value, check_occupied=False)
        if empty_bucket is None:
            return False  # Table penuh

        # Set embedding
        if embedding is not None:
            self.embedding_table[empty_bucket] = embedding.cpu()
        else:
            # Inisialisasi embedding dengan distribusi normal
            self.embedding_table[empty_bucket] = torch.randn(
                self.embedding_dim
            ) * 0.02

        # Mark sebagai terisi
        self.occupation_mask[empty_bucket] = True
        self._hash_to_bucket[empty_bucket] = hash_value
        self._insert_count += 1

        return True

    def insert_batch(
        self,
        subjects: List[str],
        embeddings: Optional[torch.Tensor] = None,
    ) -> List[bool]:
        """
        Insert batch fakta ke Engram Memory.

        Args:
            subjects: List subject strings
            embeddings: Tensor [batch, embedding_dim], opsional

        Returns:
            List boolean status per subject
        """
        results: List[bool] = []
        for i, subject in enumerate(subjects):
            emb = embeddings[i] if embeddings is not None else None
            results.append(self.insert(subject, emb))
        return results

    def retrieve(
        self, query: torch.Tensor, subject_strings: Optional[List[str]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieve fakta dari Engram Memory.

        Dua mode operasi:
        1. Hash-based: Jika subject_strings diberikan, gunakan hash lookup (O(1))
        2. Similarity-based: Jika tidak, gunakan query encoding + similarity search

        Args:
            query: Query tensor [batch, d_model]
            subject_strings: Optional list of subject strings untuk hash-based lookup

        Returns:
            Tuple:
                - fact_embeddings: [batch, d_model] — retrieved fact embeddings
                - retrieval_scores: [batch] — confidence scores (1.0 untuk hash hit)
        """
        batch_size = query.shape[0]
        device = query.device
        dtype = query.dtype

        # Encode query ke embedding space
        query_encoded = self.query_encoder(query)  # [batch, embedding_dim]

        if subject_strings is not None:
            # === Hash-based O(1) retrieval ===
            fact_embeddings = torch.zeros(
                batch_size, self.embedding_dim, device="cpu"
            )
            retrieval_scores = torch.zeros(batch_size, device=device)

            for i, subject in enumerate(subject_strings):
                hash_value = self._compute_hash(subject)
                bucket_idx = self._find_bucket(hash_value, check_occupied=True)

                if bucket_idx is not None:
                    # Hit ditemukan — O(1) retrieval
                    fact_embeddings[i] = self.embedding_table[bucket_idx]
                    retrieval_scores[i] = 1.0
                else:
                    # Tidak ditemukan — gunakan similarity sebagai fallback
                    # Ini tetap cepat karena embedding_table bisa di-subset
                    if self.occupation_mask.any():
                        occupied_indices = self.occupation_mask.nonzero(
                            as_tuple=True
                        )[0]
                        occupied_embs = self.embedding_table[occupied_indices]
                        # Similarity ke semua occupied embeddings
                        sim = F.cosine_similarity(
                            query_encoded[i].cpu().unsqueeze(0),
                            occupied_embs,
                            dim=-1,
                        )
                        best_idx = sim.argmax()
                        fact_embeddings[i] = occupied_embs[best_idx]
                        retrieval_scores[i] = (
                            sim[best_idx].item() * 0.5
                        )  # Skor lebih rendah untuk fallback
                    else:
                        retrieval_scores[i] = 0.0

            # Pindahkan ke GPU dan proyeksikan ke d_model
            fact_embeddings = fact_embeddings.to(device=device, dtype=dtype)
            output = self.output_proj(fact_embeddings)
        else:
            # === Similarity-based retrieval ===
            # Pindahkan seluruh embedding table ke GPU sementara
            # Ini adalah operasi yang mahal, hanya dilakukan jika diperlukan
            if self.occupation_mask.any():
                occupied_indices = self.occupation_mask.nonzero(as_tuple=True)[0]
                # Pindahkan hanya occupied embeddings ke GPU
                occupied_embs = self.embedding_table[occupied_indices].to(
                    device=device, dtype=dtype
                )

                # Hitung similarity: [batch, num_occupied]
                similarity = F.cosine_similarity(
                    query_encoded.unsqueeze(1),
                    occupied_embs.unsqueeze(0),
                    dim=-1,
                )

                # Top-1 retrieval
                best_scores, best_indices = similarity.max(dim=-1)

                # Gather embeddings
                fact_embeddings = occupied_embs[best_indices]  # [batch, emb_dim]
                output = self.output_proj(fact_embeddings)
                retrieval_scores = best_scores
            else:
                # Engram kosong — return zeros
                output = torch.zeros(
                    batch_size, self.d_model, device=device, dtype=dtype
                )
                retrieval_scores = torch.zeros(batch_size, device=device)

        return output, retrieval_scores

    def retrieve_single(self, subject: str) -> Optional[torch.Tensor]:
        """
        Retrieve single fakta berdasarkan subject string.

        Args:
            subject: Subject string untuk lookup

        Returns:
            Fact embedding [embedding_dim] di CPU, atau None jika tidak ditemukan
        """
        hash_value = self._compute_hash(subject)
        bucket_idx = self._find_bucket(hash_value, check_occupied=True)

        if bucket_idx is not None:
            return self.embedding_table[bucket_idx].clone()
        return None

    def contains(self, subject: str) -> bool:
        """
        Cek apakah subject ada dalam Engram Memory.

        Args:
            subject: Subject string

        Returns:
            True jika subject ditemukan
        """
        hash_value = self._compute_hash(subject)
        return self._find_bucket(hash_value, check_occupied=True) is not None

    def clear(self) -> None:
        """Hapus seluruh isi Engram Memory."""
        self.embedding_table.zero_()
        self.occupation_mask.zero_()
        self._hash_to_bucket.clear()
        self._insert_count = 0

    def get_occupation_ratio(self) -> float:
        """
        Hitung rasio okupasi Engram Memory.

        Returns:
            Rasio bucket yang terisi (0.0 - 1.0)
        """
        return self.occupation_mask.float().mean().item()

    def get_stats(self) -> Dict[str, object]:
        """
        Dapatkan statistik Engram Memory.

        Returns:
            Dictionary berisi statistik penggunaan
        """
        return {
            "total_buckets": self.num_buckets,
            "occupied_buckets": self.occupation_mask.sum().item(),
            "occupation_ratio": self.get_occupation_ratio(),
            "insert_count": self._insert_count,
            "embedding_dim": self.embedding_dim,
            "d_model": self.d_model,
            # Estimasi penggunaan memori
            "cpu_memory_mb": (
                self.num_buckets * self.embedding_dim * 4 / (1024 * 1024)
            ),
            "vram_overhead_mb": (
                self.embedding_dim * self.d_model * 4 / (1024 * 1024)
            ),
        }

    def make_embeddings_trainable(self) -> None:
        """
        Jadikan embedding table trainable untuk fine-tuning.

        Secara default, embedding table bukan parameter nn.Module
        karena disimpan sebagai buffer di CPU. Panggil method ini
        untuk mengizinkan gradient-based refinement.
        """
        self.embedding_table.requires_grad_(True)

    def freeze_embeddings(self) -> None:
        """
        Freeze embedding table (default behavior).
        """
        self.embedding_table.requires_grad_(False)

    def save_to_file(self, path: str) -> None:
        """
        Simpan Engram Memory ke file.

        Hanya menyimpan occupied entries untuk efisiensi.

        Args:
            path: Path file tujuan
        """
        occupied_indices = self.occupation_mask.nonzero(as_tuple=True)[0]
        occupied_embs = self.embedding_table[occupied_indices]

        # Reconstruct hash values (sebagai Python list karena hash SHA-256
        # bisa melebihi range torch.long)
        hash_values = [
            self._hash_to_bucket.get(idx.item(), 0) for idx in occupied_indices
        ]

        save_dict = {
            "occupied_indices": occupied_indices,
            "occupied_embeddings": occupied_embs,
            "hash_values": hash_values,
            "num_buckets": self.num_buckets,
            "embedding_dim": self.embedding_dim,
            "d_model": self.d_model,
        }
        torch.save(save_dict, path)

    def load_from_file(self, path: str) -> None:
        """
        Muat Engram Memory dari file.

        Args:
            path: Path file sumber
        """
        save_dict = torch.load(path, map_location="cpu", weights_only=False)

        # Verifikasi kompatibilitas
        assert save_dict["num_buckets"] == self.num_buckets, (
            f"Incompatible num_buckets: {save_dict['num_buckets']} vs {self.num_buckets}"
        )
        assert save_dict["embedding_dim"] == self.embedding_dim, (
            f"Incompatible embedding_dim: {save_dict['embedding_dim']} vs {self.embedding_dim}"
        )

        # Clear existing data
        self.clear()

        # Restore entries
        occupied_indices = save_dict["occupied_indices"]
        occupied_embeddings = save_dict["occupied_embeddings"]
        hash_values = save_dict["hash_values"]

        for i, idx in enumerate(occupied_indices):
            idx_val = idx.item() if isinstance(idx, torch.Tensor) else idx
            self.embedding_table[idx_val] = occupied_embeddings[i]
            self.occupation_mask[idx_val] = True
            # hash_values adalah Python list (bisa berisi int besar)
            hv = hash_values[i]
            self._hash_to_bucket[idx_val] = hv.item() if isinstance(hv, torch.Tensor) else hv
            self._insert_count += 1
