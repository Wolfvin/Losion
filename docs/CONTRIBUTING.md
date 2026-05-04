# Contributing to Losion

> Terima kasih atas minat Anda untuk berkontribusi pada Losion! Dokumen ini
> menjelaskan proses kontribusi, standar kode, dan ekspektasi untuk kontributor.

> **Agent Context**
> ```
> Kontribusi utama: Jalur components (SSM/Attention/Retrieval),
>   Training techniques, Documentation, Testing
> Setup: pip install -e ".[dev]" + pre-commit install
> Test: pytest tests/ -v (semua), pytest tests/test_advanced.py (fitur lanjutan)
> Lint: flake8, mypy, black, isort
> PR format: Conventional Commits (feat/fix/docs/refactor/test/chore)
> Dokumentasi: Bahasa Indonesia + istilah teknis Inggris
> Agent Context box: wajib di setiap file kode dan dokumentasi baru
> Kode etik: CODE_OF_CONDUCT.md
> Keamanan: SECURITY.md
> ```

---

## Daftar Isi

1. [Kode Etik](#kode-etik)
2. [Cara Berkontribusi](#cara-berkontribusi)
3. [Setup Development](#setup-development)
4. [Panduan Gaya Kode](#panduan-gaya-kode)
5. [Menambah Komponen Pathway Baru](#menambah-komponen-pathway-baru)
6. [Menambah Teknik Training Baru](#menambah-teknik-training-baru)
7. [Persyaratan Testing](#persyaratan-testing)
8. [Standar Dokumentasi](#standar-dokumentasi)
9. [Template Issue](#template-issue)
10. [Proses Release](#proses-release)

---

## Kode Etik

### Prinsip Kami

Losion berkomitmen untuk menyediakan lingkungan yang ramah, inklusif,
dan bebas diskriminasi untuk semua kontributor. Kami mengharapkan
setiap orang yang berpartisipasi dalam proyek ini untuk mematuhi
standar berikut:

1. **Hormat**: Perlakukan semua kontributor dengan hormat. Hargai
   perbedaan pandangan dan pengalaman.
2. **Konstruktif**: Berikan feedback yang membangun. Kritik harus
   diarahkan pada kode, bukan pada orang.
3. **Inklusif**: Gunakan bahasa yang inklusif dan menghargai keberagaman.
4. **Kolaboratif**: Bantu kontributor lain, terutama yang baru.
5. **Profesional**: Jaga komunikasi profesional di semua channel.

### Perilaku Tidak Diterima

- Harassment atau diskriminasi dalam bentuk apapun
- Komentar ofensif terkait gender, ras, agama, orientasi, atau disabilitas
- Intimidasi atau bullying
- Spam atau self-promotion yang tidak relevan
- Doxxing atau pelanggaran privasi

### Pelaporan

Jika Anda mengalami atau menyaksikan pelanggaran kode etik, silakan
hubungi maintainers melalui email: losion-conduct@example.com

Kode etik lengkap tersedia di [CODE_OF_CONDUCT.md](../CODE_OF_CONDUCT.md).

---

## Cara Berkontribusi

### Alur Pull Request

```
1. Fork repository
2. Buat branch baru (git checkout -b fitur-baru)
3. Buat perubahan
4. Tulis/update test
5. Jalankan test suite (pytest tests/)
6. Jalankan test advanced (pytest tests/test_advanced.py)
7. Jalankan linter (flake8, mypy)
8. Commit dengan pesan yang deskriptif
9. Push ke fork Anda
10. Buat Pull Request
11. Tunggu review dari maintainers
```

### Jenis Kontribusi

| Jenis | Contoh | Label Issue |
|-------|--------|-------------|
| Bug fix | Memperbaiki error di forward pass | `bug` |
| Fitur baru | Menambahkan lapisan baru, optimizer baru | `enhancement` |
| Dokumentasi | Memperbaiki docs, menambahkan contoh | `documentation` |
| Performa | Mengoptimalkan komputasi, mengurangi memori | `performance` |
| Testing | Menambahkan test coverage | `testing` |
| Refactoring | Memperbaiki struktur kode tanpa mengubah fungsionalitas | `refactor` |

### Ukuran Pull Request

- **Kecil** (< 200 baris): Bug fix, typo, dokumentasi kecil
  - Review: 1-2 hari
- **Sedang** (200-500 baris): Fitur baru, refactoring
  - Review: 3-5 hari
- **Besar** (> 500 baris): Perubahan arsitektur, fitur mayor
  - Review: 1-2 minggu
  - **Wajib diskusi terlebih dahulu** melalui issue atau discussion

### Commit Messages

Gunakan format [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<scope>): <description>

[optional body]

[optional footer]
```

Tipe yang valid:
- `feat`: Fitur baru
- `fix`: Bug fix
- `docs`: Perubahan dokumentasi
- `style`: Formatting, whitespace (tidak mengubah kode)
- `refactor`: Refactoring kode
- `test`: Menambahkan atau memperbaiki test
- `chore`: Build, CI, dependencies

Contoh:
```
feat(ssm): add support for variable chunk sizes

Implement dynamic chunk size selection based on sequence length.
Shorter sequences use smaller chunks for better efficiency.

Closes #42
```

### Review Process

1. **Automated checks**: CI harus pass (lint, test, type check)
2. **Code review**: Minimal 1 approval dari maintainer
3. **Discussion**: Untuk perubahan besar, diskusi di PR atau issue
4. **Merge**: Maintainer akan merge setelah approval

---

## Setup Development

### Instalasi

```bash
# Clone fork Anda
git clone https://github.com/YOUR_USERNAME/Losion.git
cd Losion

# Buat virtual environment
python -m venv venv
source venv/bin/activate

# Install dengan dependencies development
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

### Menjalankan Test

```bash
# Semua test
pytest tests/ -v

# Test spesifik
pytest tests/test_ssm.py -v
pytest tests/test_router.py -v

# Test fitur lanjutan (Reasoning Engine, Elastic Inference, dll.)
pytest tests/test_advanced.py -v

# Test dengan coverage
pytest tests/ --cov=losion --cov-report=html

# Test hanya SSM (tanpa GPU)
pytest tests/test_ssm.py -v -k "not gpu"
```

### Linting dan Type Checking

```bash
# Flake8 (linting)
flake8 losion/ tests/

# mypy (type checking)
mypy losion/

# black (formatting)
black losion/ tests/

# isort (import sorting)
isort losion/ tests/
```

### Pre-commit Hooks

Pre-commit hooks menjalankan pengecekan otomatis sebelum setiap commit:

```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/psf/black
    rev: 24.1.0
    hooks:
      - id: black
  - repo: https://github.com/pycqa/flake8
    rev: 7.0.0
    hooks:
      - id: flake8
  - repo: https://github.com/pycqa/isort
    rev: 5.13.0
    hooks:
      - id: isort
```

---

## Panduan Gaya Kode

### Prinsip Umum

1. **Konsistensi**: Ikuti gaya kode yang sudah ada di codebase
2. **Keterbacaan**: Kode dibaca lebih sering daripada ditulis
3. **Simplicity**: Pilih solusi paling sederhana yang berfungsi
4. **Type Safety**: Gunakan type hints di semua fungsi publik

### Python Style

- **Versi**: Target Python 3.10+ (gunakan `from __future__ import annotations`)
- **Formatter**: Black (line length: 88)
- **Import order**: isort (stdlib → third-party → local)
- **Naming**:
  - `PascalCase` untuk class: `SSMTerpaduLayer`, `AdaptiveRouter`
  - `snake_case` untuk fungsi/variabel: `forward_pass`, `hidden_states`
  - `UPPER_CASE` untuk konstanta: `MAX_SEQ_LEN`, `DEFAULT_RATIO`
  - `_leading_underscore` untuk internal: `_init_weights`, `_compute_loss`

### Type Hints

```python
# WAJIB untuk semua fungsi publik
def forward(
    self,
    x: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, AttentionState]:
    ...

# WAJIB untuk dataclass fields
@dataclass
class SSMConfig:
    d_model: int = 2048
    d_state: int = 16
    use_wkv: bool = False
```

### Docstrings

Gunakan format Google-style docstrings:

```python
def compute_routing_weights(
    self,
    x: torch.Tensor,
    prev_weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Hitung bobot routing untuk setiap token.

    Menggunakan bias-based routing tanpa auxiliary loss.
    Bobot dinormalisasi dengan softmax.

    Args:
        x: Input tensor [batch, seq_len, d_model].
        prev_weights: Bobot routing dari layer sebelumnya
                     (untuk Evoformer feedback). Bentuk: [batch, seq_len, 3].

    Returns:
        Routing weights [batch, seq_len, num_pathways].
        Sum dim=-1 = 1.0 (softmax normalized).

    Raises:
        ValueError: Jika input tidak 3D.

    Example:
        >>> router = AdaptiveRouter(d_model=256)
        >>> x = torch.randn(2, 10, 256)
        >>> weights = router.compute_routing_weights(x)
        >>> weights.shape
        torch.Size([2, 10, 3])
    """
```

### Komentar Bahasa

- **Docstrings dan komentar**: Bahasa Indonesia dengan istilah teknis Inggris
- **Nama variabel/fungsi/class**: Bahasa Inggris (standar internasional)
- **Commit messages**: Bahasa Inggris (standar open-source)
- **Issue/PR description**: Bahasa Inggris atau Indonesia (bebas)

Contoh:
```python
# Komentar: Indonesia + istilah teknis Inggris
# Hitung routing weights menggunakan bias-based method
# tanpa auxiliary loss (aux-loss-free routing)
routing_weights = self.bias_router(x)

# Nama variabel: Inggris
adjusted_weights = self._adjust_for_thinking(routing_weights, assessment)
```

---

## Menambah Komponen Pathway Baru

Losion menggunakan arsitektur Tri-Jalur Router, dan setiap jalur dapat
diperluas dengan komponen baru. Bagian ini menjelaskan cara menambahkan
komponen baru ke Jalur 1 (SSM), Jalur 2 (Attention), atau Jalur 3 (Retrieval).

### Struktur Direktori

```
lossion/core/
├── ssm/                   # Jalur 1: State Space Model
│   ├── mamba2.py          # Mamba-2 SSD
│   ├── mamba3.py          # Mamba-3 SSD (v0.6)
│   ├── rwkv7.py           # RWKV-7 WKV
│   ├── delta_net.py       # Gated DeltaNet
│   ├── liquid_ssm.py      # Liquid SSM (v0.4)
│   ├── post_decay.py      # PoST Decay Spectra (v0.5)
│   ├── fg2_gdn.py         # FG2-GDN (v0.5)
│   ├── routing_mamba.py   # Routing Mamba (v0.6)
│   ├── structured_sparse.py # Structured Sparse SSM (v0.8)
│   └── ssm_layer.py       # Interleaving + dispatch
├── attention/             # Jalur 2: Attention + Compression
│   ├── mla.py             # Multi-head Latent Attention
│   ├── irope.py           # Interleaved RoPE/NoPE
│   ├── kda_mla.py         # KDA+MLA Hybrid (v0.5)
│   ├── gated_attention.py # Gated Attention (v0.6)
│   ├── moba.py            # MoBA (v0.6)
│   ├── lightning_attention.py # Lightning Attention (v0.4)
│   ├── shared_attention.py # Shared Attention (v0.4)
│   ├── attn_res.py        # Attention Residuals (v0.9)
│   ├── child_3w.py        # Child-3W QKV-level MoE (v0.9)
│   ├── context_extension.py # YaRN, NTK, SSM extension (v0.7)
│   ├── interleaving.py    # Adaptive local/global
│   ├── pairformer.py      # AlphaFold3-style triangular
│   └── attention_layer.py # Layer abstraction
├── retrieval/             # Jalur 3: Specialized Retrieval
│   ├── moe.py             # Mixture of Experts
│   ├── engram.py          # Engram Memory
│   ├── expert_choice.py   # Expert Choice routing
│   ├── aux_free_moe.py    # Aux-Loss-Free MoE (v0.5)
│   ├── infinite_moe.py    # ∞-MoE Infinite MoE (v0.8)
│   ├── mohge.py           # MoHGE Grouped Experts (v0.8)
│   ├── smore.py           # S'MoRE (v0.6)
│   ├── cross_jalur_routing.py # Cross-Jalur Routing (v0.8)
│   └── retrieval_layer.py # Gated Fusion layer
├── feedback/              # Evoformer Feedback (v0.9)
│   └── evoformer.py       # 5-level AlphaFold feedback
├── memory/                # Dual Memory System (v0.9)
│   └── dual_memory.py     # Working + Long-term memory
├── output/                # Output Heads
│   ├── flow_matching.py   # Flow Matching
│   ├── diffusion_refinement.py # Diffusion Refinement
│   ├── speculative_decoder.py  # MTP Speculative (v0.4)
│   ├── mirror_speculative.py   # Mirror Speculative (v0.5)
│   ├── leap_mtp.py        # L-MTP Leap MTP (v0.8)
│   └── anchored_decoder.py # Anchored Diffusion Decoder (v0.9)
├── reasoning/             # Reasoning Modules
│   ├── mcts.py            # Monte Carlo Tree Search
│   ├── neuro_symbolic.py  # Neuro-symbolic integration
│   ├── parallel_thinking.py # Parallel Thinking
│   └── path_lock_expert.py # Path-Lock Expert (v0.5)
└── router/                # Adaptive Router
    ├── bias_router.py     # Bias-based routing
    ├── thinking_toggle.py # Thinking Toggle
    └── router.py          # Router orchestration
```

### Langkah-Langkah Menambah Komponen SSM Baru (Jalur 1)

Contoh: Menambahkan komponen SSM baru bernama "RetNet".

**1. Buat file implementasi** (`lossion/core/ssm/retnet.py`):

```python
"""
RetNetLayer — RetNet implementation untuk Losion Framework.

Jalur 1 dari arsitektur Tri-Jalur Router. Mengimplementasikan
Retention mechanism sebagai alternatif SSM sub-layer.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple


class RetNetLayer(nn.Module):
    """RetNet sub-layer untuk SSM Terpadu.

    Mengimplementasikan retention mechanism dengan parallel training
    dan recurrent inference.

    Args:
        d_model: Dimensi model.
        n_heads: Jumlah retention heads.
        d_head: Dimensi per head.

    Example:
        >>> layer = RetNetLayer(d_model=256, n_heads=4, d_head=64)
        >>> x = torch.randn(2, 16, 256)
        >>> output, state = layer(x)
        >>> output.shape
        torch.Size([2, 16, 256])
    """

    def __init__(
        self,
        d_model: int = 2048,
        n_heads: int = 8,
        d_head: int = 64,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_head

        # ... implementation ...

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            x: Input tensor [batch, seq_len, d_model].
            state: Optional recurrent state.

        Returns:
            Tuple of (output [batch, seq_len, d_model], new_state).
        """
        # ... implementation ...
        return x, state
```

**2. Daftarkan di `SSMTerpaduLayer`** (`lossion/core/ssm/ssm_layer.py`):

```python
# Tambahkan import
from losion.core.ssm.retnet import RetNetLayer

# Di __init__, tambahkan ke registry
SSM_SUB_LAYER_REGISTRY = {
    "ssd": Mamba2SSD,
    "wkv": RWKV7WKV,
    "delta": GatedDeltaNet,
    "retnet": RetNetLayer,  # BARU
}
```

**3. Tambahkan ke konfigurasi** (`lossion/config.py`):

```python
@dataclass
class SSMConfig:
    # ... existing fields ...
    use_retnet: bool = False          # Aktifkan RetNet sub-layer
    retnet_n_heads: int = 8           # RetNet heads
    retnet_d_head: int = 64           # RetNet head dim
    # Update interleaving_ratios untuk include RetNet
    # Misalnya: [3, 1, 1, 1] = 3 SSD, 1 WKV, 1 Delta, 1 RetNet
```

**4. Tulis test** (`tests/test_ssm.py` atau `tests/test_advanced.py`):

```python
class TestRetNetLayer:
    """Test suite untuk RetNetLayer."""

    @pytest.fixture
    def layer(self):
        return RetNetLayer(d_model=64, n_heads=4, d_head=16)

    def test_forward_pass_shape(self, layer):
        batch, seq, d = 2, 32, 64
        x = torch.randn(batch, seq, d)
        output, state = layer(x)
        assert output.shape == (batch, seq, d)

    def test_forward_pass_no_nan(self, layer):
        x = torch.randn(2, 16, 64)
        output, _ = layer(x)
        assert not torch.isnan(output).any()
```

**5. Update dokumentasi** — Tambahkan Agent Context box dan referensi:

```python
# Di docstring module:
"""
RetNetLayer — RetNet implementation untuk Losion Framework.
...
"""

> **Agent Context**
> ```
> RetNetLayer: losion/core/ssm/retnet.py
> Config: SSMConfig.use_retnet, retnet_n_heads, retnet_d_head
> Registry: SSM_SUB_LAYER_REGISTRY["retnet"]
> Interleaving: tambahkan ke interleaving_ratios (misal [3,1,1,1])
> ```
```

### Menambah Komponen Attention Baru (Jalur 2)

Ikuti pola yang sama:

1. Buat file di `lossion/core/attention/` (misal `linear_attention.py`)
2. Daftarkan di `AttentionKompresiLayer`
3. Tambahkan ke `AttentionConfig`
4. Tulis test
5. Update dokumentasi dengan Agent Context box

### Menambah Komponen Retrieval Baru (Jalur 3)

1. Buat file di `lossion/core/retrieval/` (misal `dense_retrieval.py`)
2. Daftarkan di `RetrievalTerpaduLayer`
3. Tambahkan ke `RetrievalConfig`
4. Tulis test
5. Update dokumentasi dengan Agent Context box

### Checklist Komponen Pathway Baru

- [ ] Implementasi komponen di direktori yang sesuai (`ssm/`, `attention/`, `retrieval/`)
- [ ] Module docstring dengan Agent Context box
- [ ] Class docstring (Google-style) dengan Args, Returns, Example
- [ ] Type hints di semua fungsi publik
- [ ] Terdaftar di layer registry
- [ ] Konfigurasi ditambahkan ke `config.py` dataclass yang sesuai
- [ ] Test di `tests/test_ssm.py` atau `tests/test_advanced.py`
- [ ] Forward pass shape test
- [ ] No-NaN test
- [ ] Edge case test (empty sequence, batch size 1)
- [ ] Lulus semua test yang sudah ada
- [ ] Dokumentasi di ARCHITECTURE.md diupdate

> **Agent Context**
> ```
> Menambah pathway component:
>   1. Buat file di losion/core/{ssm,attention,retrieval}/
>   2. Daftarkan di layer registry
>   3. Tambahkan config di losion/config.py
>   4. Tulis test
>   5. Update docs
> Registry pattern: SSM_SUB_LAYER_REGISTRY, attention registry, retrieval registry
> Semua komponen baru WAJIB punya Agent Context box
> ```

---

## Menambah Teknik Training Baru

Losion mengintegrasikan berbagai teknik training canggih. Bagian ini
menjelaskan cara menambahkan teknik training baru ke framework.

### Struktur Direktori Training

```
lossion/training/
├── trainer.py               # Main training loop
├── grpo.py                  # GRPO reinforcement learning
├── curriculum.py            # 4-phase curriculum learning
├── advanced_rlhf.py         # Self-Play + Value Head + Self-Consistency
├── advanced_backprop.py     # Gradient Overlapping + Parallel Attn+FFN
│                            #   + Chinchilla + Per-Jalur LR + Soft Capping
│                            #   + Scheduled Sampling + Confidence Heads
├── advanced_memory_data.py  # KV Compression + Attention Sinks
│                            #   + Dynamic Expert Buffer + Modality-Aware Loss
│                            #   + Chinchilla Data Sizing + Sample-then-Filter
│                            #   + Template-Based Conditional Routing
├── active_learning.py       # GNoME-style active learning loop
└── evolutionary_search.py   # FunSearch-style evolutionary search
```

### Langkah-Langkah Menambah Teknik Training Baru

Contoh: Menambahkan teknik "Stochastic Weight Averaging (SWA)".

**1. Tentukan di mana teknik akan ditempatkan**

Teknik training yang berkaitan dengan:
- **RLHF/RL** → `advanced_rlhf.py`
- **Backprop/optimisasi** → `advanced_backprop.py`
- **Memori/data** → `advanced_memory_data.py`
- **Scheduling/curriculum** → `curriculum.py`
- **Baru dan unik** → Buat file baru (misal `swa.py`)

**2. Implementasi teknik** (`lossion/training/swa.py`):

```python
"""
Stochastic Weight Averaging (SWA) untuk Losion Framework.

Mengimplementasikan SWA yang mengaverage model weights dari
beberapa checkpoint terakhir untuk menghasilkan model yang
lebih generalizable.
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any
from copy import deepcopy


class StochasticWeightAverager:
    """Stochastic Weight Averaging untuk model Losion.

    Mengaverage weights dari n checkpoint terakhir untuk
    menghasilkan model yang lebih robust.

    Args:
        model: Model Losion yang akan di-average.
        avg_start_step: Langkah mulai averaging (default: 75% total steps).
        avg_frequency: Frekuensi snapshot (default: setiap 100 steps).
        n_snapshots: Jumlah maksimum snapshot (default: 10).

    Example:
        >>> averager = StochasticWeightAverager(model, avg_start_step=75000)
        >>> # Dalam training loop:
        >>> averager.maybe_snapshot(step=current_step)
        >>> # Di akhir training:
        >>> averaged_model = averager.average()
    """

    def __init__(
        self,
        model: nn.Module,
        avg_start_step: int = 75000,
        avg_frequency: int = 100,
        n_snapshots: int = 10,
    ) -> None:
        self.model = model
        self.avg_start_step = avg_start_step
        self.avg_frequency = avg_frequency
        self.n_snapshots = n_snapshots
        self.snapshots: list[Dict[str, torch.Tensor]] = []

    def maybe_snapshot(self, step: int) -> bool:
        """Ambil snapshot jika step sudah memenuhi kriteria.

        Args:
            step: Training step saat ini.

        Returns:
            True jika snapshot diambil, False jika tidak.
        """
        if step < self.avg_start_step:
            return False
        if (step - self.avg_start_step) % self.avg_frequency != 0:
            return False

        snapshot = deepcopy(self.model.state_dict())
        self.snapshots.append(snapshot)

        if len(self.snapshots) > self.n_snapshots:
            self.snapshots.pop(0)

        return True

    def average(self) -> nn.Module:
        """Average semua snapshot yang tersimpan.

        Returns:
            Model baru dengan averaged weights.
        """
        if not self.snapshots:
            return self.model

        averaged_state = {}
        keys = self.snapshots[0].keys()

        for key in keys:
            averaged_state[key] = torch.stack(
                [s[key] for s in self.snapshots], dim=0
            ).mean(dim=0)

        averaged_model = deepcopy(self.model)
        averaged_model.load_state_dict(averaged_state)
        return averaged_model
```

**3. Tambahkan ke konfigurasi** (`lossion/config.py`):

```python
@dataclass
class TrainingConfig:
    # ... existing fields ...
    use_swa: bool = False                  # Aktifkan SWA
    swa_start_step: int = 75000            # Langkah mulai SWA
    swa_frequency: int = 100               # Frekuensi snapshot
    swa_n_snapshots: int = 10              # Jumlah snapshot
```

**4. Integrasikan ke trainer** (`lossion/training/trainer.py`):

```python
# Di __init__:
if config.training.use_swa:
    from losion.training.swa import StochasticWeightAverager
    self.swa_averager = StochasticWeightAverager(
        model=self.model,
        avg_start_step=config.training.swa_start_step,
        avg_frequency=config.training.swa_frequency,
        n_snapshots=config.training.swa_n_snapshots,
    )

# Di training loop:
if self.config.training.use_swa:
    self.swa_averager.maybe_snapshot(step=self.global_step)

# Di akhir training:
if self.config.training.use_swa:
    averaged_model = self.swa_averager.average()
    self.save_checkpoint(averaged_model, "swa-averaged")
```

**5. Tulis test** (`tests/test_advanced.py`):

```python
class TestStochasticWeightAverager:
    """Test suite untuk StochasticWeightAverager."""

    def test_snapshot_logic(self):
        model = nn.Linear(10, 10)
        averager = StochasticWeightAverager(
            model, avg_start_step=10, avg_frequency=5, n_snapshots=3
        )
        assert not averager.maybe_snapshot(step=5)   # Sebelum start
        assert averager.maybe_snapshot(step=10)       # Tepat di start
        assert not averager.maybe_snapshot(step=12)   # Bukan frekuensi
        assert averager.maybe_snapshot(step=15)       # Frekuensi tepat

    def test_averaging(self):
        model = nn.Linear(10, 10)
        averager = StochasticWeightAverager(
            model, avg_start_step=0, avg_frequency=1, n_snapshots=3
        )
        for _ in range(3):
            averager.maybe_snapshot(step=0)

        averaged = averager.average()
        assert isinstance(averaged, nn.Linear)
```

**6. Update dokumentasi** — Tambahkan ke TRAINING.md dengan Agent Context box.

### Checklist Teknik Training Baru

- [ ] Implementasi di `lossion/training/`
- [ ] Module docstring dengan Agent Context box
- [ ] Class docstring (Google-style) dengan Args, Returns, Example
- [ ] Type hints di semua fungsi publik
- [ ] Konfigurasi ditambahkan ke `TrainingConfig` di `config.py`
- [ ] Integrasi ke `trainer.py` (opsional: bisa juga standalone)
- [ ] Test di `tests/test_advanced.py`
- [ ] Unit test untuk logika inti
- [ ] Integration test dengan model kecil
- [ ] Lulus semua test yang sudah ada
- [ ] Dokumentasi di TRAINING.md diupdate

> **Agent Context**
> ```
> Menambah training technique:
>   1. Buat file di losion/training/
>   2. Tambahkan config di TrainingConfig
>   3. Integrasikan ke trainer.py (jika diperlukan)
>   4. Tulis test di tests/test_advanced.py
>   5. Update TRAINING.md
> File placement: rlhf→advanced_rlhf.py, backprop→advanced_backprop.py,
>   memory→advanced_memory_data.py, new→new_file.py
> Semua teknik baru WAJIB punya Agent Context box
> ```

---

## Persyaratan Testing

### Standar Minimum

Setiap PR harus:
1. **Tidak menurunkan test coverage** yang sudah ada
2. **Menambahkan test** untuk fitur baru
3. **Memperbaiki test** yang rusak oleh perubahan
4. **Lulus semua test** yang sudah ada

### Jenis Test

| Jenis | Keterangan | Contoh |
|-------|-----------|--------|
| Unit test | Test satu komponen | `test_ssm_forward_pass` |
| Integration test | Test interaksi komponen | `test_model_with_router` |
| Smoke test | Test cepat apakah berjalan | `test_import_losion` |
| Regression test | Mencegah bug muncul kembali | `test_router_no_nan_weights` |
| Advanced feature test | Test fitur lanjutan | `test_mcts_search` |

### File Test

| File | Konten |
|------|--------|
| `tests/test_ssm.py` | SSM pathway tests (SSD, WKV, DeltaNet) |
| `tests/test_attention.py` | Attention pathway tests (MLA, iRoPE, Pairformer) |
| `tests/test_router.py` | Router tests (BiasRouter, ThinkingToggle) |
| `tests/test_model.py` | End-to-end model tests |
| `tests/test_advanced.py` | Advanced feature tests (MCTS, Parallel Thinking, Neuro-Symbolic, Matryoshka, GRPO, Advanced RLHF, Memory techniques, Training techniques) |

### Menulis Test yang Baik

```python
import pytest
import torch
from losion.core.ssm import SSMTerpaduLayer


class TestSSMTerpaduLayer:
    """Test suite untuk SSMTerpaduLayer."""

    @pytest.fixture
    def layer(self):
        """Buat layer untuk testing."""
        return SSMTerpaduLayer(
            d_model=64,
            d_state=16,
            d_conv=4,
            expand=2,
            chunk_size=16,
            n_heads=4,
            d_head=16,
        )

    def test_forward_pass_shape(self, layer):
        """Test bahwa output memiliki shape yang benar."""
        batch, seq, d = 2, 32, 64
        x = torch.randn(batch, seq, d)
        output, state = layer(x)
        assert output.shape == (batch, seq, d)

    def test_forward_pass_no_nan(self, layer):
        """Test bahwa output tidak mengandung NaN."""
        x = torch.randn(2, 16, 64)
        output, _ = layer(x)
        assert not torch.isnan(output).any()

    def test_empty_sequence(self, layer):
        """Test edge case: sekuens kosong."""
        x = torch.randn(2, 0, 64)
        output, state = layer(x)
        assert output.shape == (2, 0, 64)
```

### Menulis Test untuk Fitur Lanjutan

```python
# tests/test_advanced.py
import pytest
import torch
from losion.core.reasoning.mcts import MCTSReasoner, MCTSConfig


class TestMCTSReasoner:
    """Test suite untuk MCTS Reasoner."""

    def test_search_returns_result(self, small_model):
        """Test bahwa MCTS search mengembalikan hasil yang valid."""
        config = MCTSConfig(num_simulations=4, max_depth=3)
        reasoner = MCTSReasoner(small_model, config=config)

        prompt = torch.randint(0, 100, (1, 8))
        result = reasoner.search(prompt, max_new_tokens=16)

        assert result.best_output is not None
        assert result.value >= -1.0 and result.value <= 1.0
        assert result.num_simulations > 0
```

### Test Tanpa GPU

Semua test harus dapat dijalankan **tanpa GPU** (di CPU). Gunakan
tensor kecil dan parameter minimal:

```python
# BENAR: Test di CPU dengan parameter kecil
def test_ssm_layer():
    layer = SSMTerpaduLayer(d_model=32, d_state=8, n_heads=2)
    x = torch.randn(1, 8, 32)  # Batch kecil
    output, _ = layer(x)
    assert output.shape == (1, 8, 32)

# SALAH: Memerlukan GPU atau parameter besar
def test_ssm_layer():
    layer = SSMTerpaduLayer(d_model=2048, d_state=256)  # Terlalu besar!
    x = torch.randn(32, 2048, 2048).cuda()  # Memerlukan GPU!
```

---

## Standar Dokumentasi

### Dokumentasi Kode

Setiap **module**, **class**, dan **fungsi publik** harus memiliki docstring.

**Module docstring** (di atas file):
```python
"""
SSMTerpaduLayer — Layer SSM Terpadu untuk Losion Framework.

Jalur 1 dari arsitektur Tri-Jalur Router. Menggabungkan tiga inovasi
SSM dalam satu layer yang koheren:
1. Mamba-2 SSD — pemrosesan sekuensial paralel
2. RWKV-7 WKV — evolusi state dinamis
3. Gated DeltaNet — kemampuan in-context learning
"""
```

### Agent Context Box Convention

Setiap file kode dan dokumentasi baru **wajib** menyertakan Agent Context box
sebagai blockquote. Formatnya:

**Untuk file kode** (di bawah module docstring):

```python
"""
Module docstring...
"""

# > **Agent Context**
# > ```
# > ClassName: losion/core/path/file.py
# > Key config: SectionConfig.field_name (default: value)
# > Dependencies: losion.core.other.Thing
# > Used by: LosionModel, LosionTrainer
# > ```
```

**Untuk file dokumentasi** (di bawah judul):

```markdown
# Judul Dokumen

> Deskripsi singkat dokumen.

> **Agent Context**
> ```
> File: docs/DOCUMENT.md
> Related: ARCHITECTURE.md, TRAINING.md
> Key APIs: losion.module.ClassName
> Config: SectionConfig
> ```
```

**Tujuan Agent Context box**:
- Memberikan konteks cepat untuk AI agents yang membaca dokumentasi
- Menunjukkan lokasi implementasi, dependensi, dan penggunaan
- Memudahkan navigasi antar file dan modul
- Standar yang konsisten di seluruh proyek Losion

### Dokumentasi Proyek

Dokumentasi proyek ditulis dalam Markdown di folder `docs/`:

| File | Konten |
|------|--------|
| `ARCHITECTURE.md` | Arsitektur Tri-Jalur Router (detail teknis lengkap) |
| `TRAINING.md` | Panduan training (4-fase, GRPO, RLHF, hyperparameter) |
| `HARDWARE.md` | Dukungan hardware (CUDA, ROCm, FP8, memori) |
| `GETTING_STARTED.md` | Quick start guide (instalasi, training pertama, fitur lanjutan) |
| `CONTRIBUTING.md` | Panduan kontribusi (kode, testing, dokumentasi) |

Dokumentasi tambahan di root:

| File | Konten |
|------|--------|
| `CHANGELOG.md` | Riwayat perubahan per versi |
| `ROADMAP.md` | Rencana pengembangan ke depan |
| `SECURITY.md` | Panduan keamanan dan vulnerability reporting |
| `CODE_OF_CONDUCT.md` | Kode etik komunitas |

**Bahasa**: Bahasa Indonesia dengan istilah teknis Inggris.

**Standar**:
- Setiap dokumen harus memiliki daftar isi
- Setiap dokumen harus memiliki Agent Context box di bagian atas
- Contoh kode harus dapat dijalankan
- Diagram menggunakan ASCII art (bukan gambar)
- Link antar dokumen menggunakan relative path

---

## Template Issue

### Bug Report

```markdown
## Bug Description
Deskripsi singkat bug.

## Reproduction Steps
1. Langkah pertama
2. Langkah kedua
3. ...

## Expected Behavior
Apa yang seharusnya terjadi.

## Actual Behavior
Apa yang benar-benar terjadi.

## Environment
- OS: (e.g., Ubuntu 22.04)
- Python: (e.g., 3.11.5)
- PyTorch: (e.g., 2.4.0)
- GPU: (e.g., NVIDIA A100 80GB)
- Losion version: (e.g., 2.0.0)

## Additional Context
Log, screenshot, atau informasi tambahan.
```

### Feature Request

```markdown
## Feature Description
Deskripsi fitur yang diinginkan.

## Motivation
Mengapa fitur ini diperlukan? Masalah apa yang dipecahkan?

## Proposed Solution
Bagaimana fitur ini seharusnya bekerja?

## Alternatives Considered
Solusi alternatif yang telah dipertimbangkan.

## Additional Context
Referensi, paper, atau implementasi serupa.
```

---

## Proses Release

Losion mengikuti [Semantic Versioning](https://semver.org/) dan proses
release yang terstruktur. Riwayat perubahan dicatat di [CHANGELOG.md](../CHANGELOG.md).

### Versi

- **v2.0.0**: AuxFreeMoE MTP loss propagated to total loss — all model params now receive gradients (was: 32.2% of model params were dead weight with zero gradient)
- **v1.9.0**: Complete gradient flow — 10.0/10 score, vectorized attention
- Versi berikutnya mengikuti format `MAJOR.MINOR.PATCH`

### Alur Release

```
1. Update CHANGELOG.md dengan perubahan sejak release terakhir
2. Update version number di:
   - losion/__init__.py (__version__)
   - pyproject.toml / setup.py
3. Jalankan full test suite: pytest tests/ -v
4. Jalankan advanced tests: pytest tests/test_advanced.py -v
5. Jalankan linter: flake8, mypy
6. Buat tag: git tag v0.X.0
7. Push tag: git push origin v0.X.0
8. CI otomatis build dan publish ke PyPI
9. Buat GitHub Release dengan CHANGELOG entry
```

### CHANGELOG Format

Ikuti format [Keep a Changelog](https://keepachangelog.com/):

```markdown
## [0.2.0] - 2025-04-XX

### Added
- RetNet sub-layer untuk Jalur 1 SSM
- Stochastic Weight Averaging di training
- Dukungan FP8 untuk AMD MI300X

### Changed
- Perbaikan routing stability di Phase 3
- Update PyTorch minimum version ke 2.2+

### Fixed
- Bug: gradient overflow di fp8 training
- Bug: attention sink eviction error pada seq > 100K

### Deprecated
- SSMConfig.legacy_mode (akan dihapus di v0.3.0)

### Removed
- Tidak ada penghapusan di versi ini

### Security
- Update dependency untuk patch CVE-2025-XXXX
```

### Keamanan dalam Release

- Selalu jalankan `pip audit` sebelum release
- Review [SECURITY.md](../SECURITY.md) untuk prosedur vulnerability reporting
- Jangan include secrets atau credentials dalam release
- Verify semua dependencies tidak memiliki vulnerability yang diketahui

> **Agent Context**
> ```
> Release process: lihat CHANGELOG.md untuk riwayat
> Version: losion/__version__, pyproject.toml
> Semantic Versioning: MAJOR.MINOR.PATCH
> Security: SECURITY.md untuk vulnerability reporting
> CI: otomatis build + publish ke PyPI saat tag di-push
> ```

---

Terima kasih telah berkontribusi pada Losion! Setiap kontribusi —
baik kode, dokumentasi, bug report, atau saran — sangat berarti
untuk pengembangan proyek ini.

---

*Dokumen ini ditulis untuk Losion v2.0.0.*
