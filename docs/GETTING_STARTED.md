# Getting Started — Losion

> Panduan cepat untuk memulai menggunakan framework Losion, dari instalasi
> hingga training pertama Anda, termasuk Reasoning Engine dan Elastic Inference.

> **Agent Context**
> ```
> Entry points: LosionConfig, LosionForCausalLM, LosionTrainer
> Quick start config: configs/losion-1b.yaml
> Training script: scripts/train.py
> Reasoning Engine: losion/core/reasoning/ (MCTS, Parallel Thinking, Neuro-Symbolic)
> Elastic Inference: losion/core/elastic/matryoshka.py
> Verifikasi instalasi: import losion; losion.__version__
> Dokumentasi lengkap: docs/ARCHITECTURE.md, docs/TRAINING.md, docs/HARDWARE.md
> ```

---

## Daftar Isi

1. [Prerequisites](#prerequisites)
2. [Instalasi](#instalasi)
3. [Verifikasi Instalasi](#verifikasi-instalasi)
4. [Training Pertama](#training-pertama)
5. [Pre-Training Your First Model](#pre-training-your-first-model)
6. [Menggunakan Reasoning Engine](#menggunakan-reasoning-engine)
7. [Menggunakan Elastic Inference (Matryoshka)](#menggunakan-elastic-inference-matryoshka)
8. [Memahami Konfigurasi](#memahami-konfigurasi)
9. [Langkah Selanjutnya](#langkah-selanjutnya)
10. [FAQ](#faq)

---

## Prerequisites

### Kebutuhan Sistem

| Komponen | Minimum | Direkomendasikan |
|----------|---------|-----------------|
| Python | 3.10+ | 3.11+ |
| PyTorch | 2.1+ | 2.4+ |
| CUDA (NVIDIA) | 11.8+ | 12.1+ |
| ROCm (AMD) | 5.7+ | 6.0+ |
| RAM | 16 GB | 64 GB+ |
| Storage | 50 GB | 500 GB+ (untuk dataset) |

### GPU yang Didukung

**NVIDIA**: RTX 3090, RTX 4090, A100, H100, dan lainnya dengan CUDA support

**AMD**: RX 7900 XTX, MI210, MI250, MI300X, dan lainnya dengan ROCm support

**CPU**: Didukung untuk development/testing, tidak untuk training produksi

### Software Dependencies

```
torch >= 2.1.0
omegaconf >= 2.3.0
pytest >= 7.0
```

Dependencies tambahan akan diinstal otomatis oleh pip.

---

## Instalasi

### Metode 1: pip (Direkomendasikan)

```bash
# Clone repository
git clone https://github.com/losion/losion.git
cd losion

# Buat virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# atau: venv\Scripts\activate  # Windows

# Install Losion
pip install -e .
```

### Metode 2: Dengan CUDA (NVIDIA GPU)

```bash
# Install PyTorch dengan CUDA terlebih dahulu
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Kemudian install Losion
pip install -e .
```

### Metode 3: Dengan ROCm (AMD GPU)

```bash
# Install PyTorch dengan ROCm
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.0

# Install Losion
pip install -e .

# Verifikasi ROCm terdeteksi
python -c "import torch; print(f'ROCm: {torch.cuda.is_available()}')"
```

### Metode 4: Docker (Paling Mudah)

```bash
# NVIDIA
docker run --gpus all -it pytorch/pytorch:2.4.0-cuda12.4-cudnn9-devel
# Di dalam container:
git clone https://github.com/losion/losion.git && cd losion
pip install -e .

# AMD
docker run -it --device=/dev/kfd --device=/dev/dri \
    --group-add=video --ipc=host \
    rocm/pytorch:latest
# Di dalam container:
git clone https://github.com/losion/losion.git && cd losion
pip install -e .
```

### Install untuk Development

Jika Anda berencana berkontribusi ke Losion:

```bash
# Install dengan dependencies development
pip install -e ".[dev]"

# Ini menginstal:
# - pytest (testing)
# - black, isort (formatting)
# - mypy (type checking)
# - flake8 (linting)
```

---

## Verifikasi Instalasi

### Langkah 1: Cek Import

```python
python -c "import losion; print(f'Losion version: {losion.__version__}')"
# Output: Losion version: 0.1.0
```

### Langkah 2: Cek Hardware

```python
python -c "
from losion import get_device_info
info = get_device_info()
print('=== Losion Hardware Info ===')
print(f'CUDA available: {info[\"cuda_available\"]}')
print(f'ROCm available: {info[\"rocm_available\"]}')
print(f'Device count: {info[\"device_count\"]}')
print(f'Device name: {info[\"device_name\"]}')
print(f'Device memory: {info[\"device_memory_gb\"]} GB')
print(f'FP8 supported: {info.get(\"fp8_supported\", False)}')
"
```

Contoh output (NVIDIA A100):
```
=== Losion Hardware Info ===
CUDA available: True
ROCm available: False
Device count: 1
Device name: NVIDIA A100-SXM4-80GB
Device memory: 79.15 GB
FP8 supported: False
```

Contoh output (AMD MI300X):
```
=== Losion Hardware Info ===
CUDA available: True
ROCm available: True
Device count: 1
Device name: AMD Instinct MI300X
Device memory: 192.0 GB
FP8 supported: True
```

### Langkah 3: Jalankan Test Suite

```bash
# Jalankan semua test (tidak memerlukan GPU)
pytest tests/ -v

# Test spesifik
pytest tests/test_ssm.py -v
pytest tests/test_router.py -v

# Test fitur lanjutan (Reasoning Engine, Elastic Inference, dll.)
pytest tests/test_advanced.py -v
```

### Langkah 4: Verifikasi Forward Pass

```python
python -c "
import torch
from losion import LosionConfig, LosionForCausalLM

# Buat konfigurasi kecil untuk testing
config = LosionConfig(
    d_model=64,
    n_layers=2,
    vocab_size=100,
    max_seq_len=128,
)

# Buat model
model = LosionForCausalLM(config)
print(f'Model created with {sum(p.numel() for p in model.parameters()):,} parameters')

# Forward pass
input_ids = torch.randint(0, 100, (2, 32))  # batch=2, seq=32
output = model(input_ids)
print(f'Forward pass OK! Logits shape: {output.logits.shape}')
"
```

---

## Training Pertama

### Quick Training (1B Model)

```bash
# Pastikan di direktori Losion
cd losion

# Mulai training dengan konfigurasi default
python scripts/train.py --config configs/losion-1b.yaml
```

Ini akan memulai training Losion-1B dengan:
- 12 layer Tri-Jalur Router
- Batch size 32
- 100,000 training steps
- bf16 precision
- Training 4-fase otomatis

### Training dengan Custom Data

```bash
# Siapkan dataset (format JSONL)
# data/train.jsonl:
# {"text": "Contoh teks pertama"}
# {"text": "Contoh teks kedua"}

python scripts/train.py --config configs/losion-1b.yaml \
    --data_dir ./data \
    --output_dir ./my-first-model
```

### Training dengan Multi-GPU

```bash
# 2 GPU
torchrun --nproc_per_node=2 scripts/train.py --config configs/losion-1b.yaml

# 4 GPU
torchrun --nproc_per_node=4 scripts/train.py --config configs/losion-7b.yaml
```

### Resume Training

```bash
# Jika training terputus, resume dari checkpoint terakhir
python scripts/train.py --config configs/losion-1b.yaml \
    --resume ./checkpoints/checkpoint-step-5000
```

### Monitoring Training

```bash
# Dengan wandb (direkomendasikan)
# 1. Install wandb
pip install wandb
wandb login

# 2. Aktifkan di config atau command line
python scripts/train.py --config configs/losion-1b.yaml --use_wandb

# Atau edit config YAML:
# training:
#   use_wandb: true
#   wandb_project: "my-losion-experiment"
```

---

## Pre-Training Your First Model

Bagian ini memberikan contoh konkret untuk pre-training model dari awal,
dari persiapan data hingga training pertama.

### Langkah 1: Siapkan Data

```bash
# Buat direktori data
mkdir -p data/raw data/tokenized

# Format: JSONL dengan field "text"
# data/raw/train.jsonl:
# {"text": "Indonesia adalah negara kepulauan..."}
# {"text": "Python adalah bahasa pemrograman..."}
# {"text": "The quick brown fox jumps over..."}
```

### Langkah 2: Tokenisasi

```python
from losion.data import Tokenizer, PreprocessingPipeline

# Train tokenizer dari corpus (atau gunakan pre-trained)
tokenizer = Tokenizer.train(
    files=["data/raw/train.jsonl"],
    vocab_size=32000,
    model_type="BPE",
    special_tokens=["<pad>", "<s>", "</s>", "<unk>", "<mask>",
                    "<think_start>", "<think_end>"],
    min_frequency=2,
)

# Preprocessing pipeline
pipeline = PreprocessingPipeline(
    tokenizer_path="tokenizer-32k.json",
    max_seq_len=8192,
    pack_sequences=True,
    pack_separator="<s>",
    num_workers=8,
)

dataset = pipeline.process(
    input_pattern="data/raw/*.jsonl",
    output_dir="data/tokenized/",
)
```

### Langkah 3: Konfigurasi Model

```yaml
# configs/my-first-model.yaml
model:
  d_model: 768
  n_layers: 12
  vocab_size: 32000
  max_seq_len: 8192

  ssm:
    d_state: 64
    d_conv: 4
    expand: 2
    chunk_size: 256
    use_wkv: true
    use_delta_net: true
    interleaving_ratios: [4, 1, 1]

  attention:
    n_heads: 12
    d_kv: 64
    mla_latent_dim: 128
    use_irope: true
    irope_ratio: 3

  retrieval:
    num_experts: 16
    num_active_experts: 2
    d_ff: 1536
    use_engram: true
    engram_dim: 128

  router:
    top_k_pathways: 2
    use_thinking_toggle: true

training:
  batch_size: 32
  learning_rate: 3.0e-4
  weight_decay: 0.1
  warmup_steps: 2000
  max_steps: 100000
  grad_clip: 1.0
  precision: bf16

hardware:
  device: auto
  backend: auto
  compile_model: true
```

### Langkah 4: Mulai Pre-Training

```bash
# Single GPU
python scripts/train.py --config configs/my-first-model.yaml

# Multi-GPU (4×A100)
torchrun --nproc_per_node=4 \
    scripts/train.py --config configs/my-first-model.yaml \
    --output_dir ./my-first-pretrain

# Monitor dengan wandb
python scripts/train.py --config configs/my-first-model.yaml \
    --use_wandb \
    --wandb_project "my-first-losion"
```

### Langkah 5: Verifikasi Hasil

```python
import torch
from losion import LosionConfig, LosionForCausalLM

# Load checkpoint
config = LosionConfig.from_pretrained("./my-first-pretrain")
model = LosionForCausalLM.from_pretrained("./my-first-pretrain")

# Generate text
input_ids = torch.tensor([[1, 50, 100]])  # token IDs
output = model.generate(input_ids, max_new_tokens=50, temperature=0.8)
print(f"Generated tokens: {output[0].tolist()}")
```

> **Agent Context**
> ```
> Pre-training config: configs/losion-{1b,7b,48b}.yaml
> Data pipeline: losion/data/ — Tokenizer, PreprocessingPipeline
> Training entry: scripts/train.py → LosionTrainer.train()
> Chinchilla sizing: ~20 tokens per active parameter
> Detail pre-training: docs/TRAINING.md §2
> ```

---

## Menggunakan Reasoning Engine

Losion mengintegrasikan Reasoning Engine yang terinspirasi dari DeepMind
(AlphaZero, AlphaProof) dan Google (Gemini Deep Think). Engine ini memungkinkan
model untuk mengefaluasi beberapa jalur penalaran secara paralel dan memilih
yang terbaik.

### MCTS (Monte Carlo Tree Search)

MCTS mengeksplorasi pohon penalaran menggunakan Upper Confidence Bound (UCB):

```python
from losion import LosionConfig, LosionForCausalLM
from losion.core.reasoning.mcts import MCTSReasoner, MCTSConfig

# Load model
config = LosionConfig()
model = LosionForCausalLM(config)

# Konfigurasi MCTS
mcts_config = MCTSConfig(
    num_simulations=64,    # Jumlah simulasi per pencarian
    c_puct=1.5,            # Exploration constant
    max_depth=10,          # Kedalaman maksimum pohon
)

reasoner = MCTSReasoner(model, config=mcts_config)

# Jalankan MCTS reasoning
result = reasoner.search(
    prompt=input_ids,
    max_new_tokens=256,
)

print(f"Best output: {result.best_output}")
print(f"Value: {result.value:.4f}")
print(f"Simulations used: {result.num_simulations}")
```

### Parallel Thinking (Gemini Deep Think Style)

Evaluasi beberapa jalur penalaran secara simultan, lalu pilih yang terbaik:

```python
from losion.core.reasoning.parallel_thinking import ParallelThinker, ThinkingStrategy

thinker = ParallelThinker(
    model=model,
    num_paths=4,                          # 4 jalur penalaran paralel
    selection_strategy=ThinkingStrategy.BEST_OF_N,  # Pilih yang terbaik
)

# Generate dengan parallel thinking
result = thinker.think(
    prompt=input_ids,
    max_new_tokens=256,
    temperature=0.7,
)

print(f"Best path score: {result.best_score:.4f}")
print(f"Strategy used: {result.selection_strategy}")
print(f"All path scores: {result.all_scores}")
```

**Strategi seleksi yang tersedia**:

| Strategi | Deskripsi | Kapan Digunakan |
|----------|-----------|-----------------|
| `BEST_OF_N` | Pilih path dengan skor tertinggi | Umum, paling sederhana |
| `MAJORITY_VOTE` | Pilih path dengan konsistensi tertinggi | Output kategorial/faktual |
| `WEIGHTED_MERGE` | Gabungkan semua path dengan bobot softmax | Output kreatif/nuanced |
| `TOURNAMENT` | Turnamen eliminasi antar path | Kompetisi ketat, kualitas tinggi |

### Neuro-Symbolic Verification (AlphaProof Style)

Verifikasi formal langkah-langkah penalaran menggunakan symbolic rules:

```python
from losion.core.reasoning.neuro_symbolic import NeuroSymbolicVerifier

verifier = NeuroSymbolicVerifier(
    model=model,
    num_rules=16,                    # Jumlah symbolic rules
    max_revision_iterations=3,       # Iterasi revisi maksimum
    verification_threshold=0.8,      # Threshold verifikasi
)

# Verifikasi output
result = verifier.verify(
    prompt=input_ids,
    output=generated_output,
)

print(f"Status: {result.status}")         # VERIFIED / FAILED / NEEDS_REVISION
print(f"Confidence: {result.confidence:.2f}")
if result.feedback:
    print(f"Feedback: {result.feedback}")
```

### Menggunakan Reasoning Engine di Inference

```python
# Cara mudah: aktifkan reasoning mode saat generate
output = model.generate(
    input_ids,
    max_new_tokens=512,
    thinking_mode=True,         # Aktifkan Thinking Toggle
    use_mcts=True,              # Gunakan MCTS
    mcts_simulations=32,        # Jumlah simulasi
    temperature=0.7,
)
```

> **Agent Context**
> ```
> Reasoning Engine: losion/core/reasoning/
> MCTS: mcts.py → MCTSReasoner, MCTSConfig
> Parallel Thinking: parallel_thinking.py → ParallelThinker, ThinkingStrategy
> Neuro-Symbolic: neuro_symbolic.py → NeuroSymbolicVerifier
> Enable via generate(): thinking_mode=True, use_mcts=True
> Adaptive compute: budget scales with complexity_score from ThinkingToggle
> Detail: ARCHITECTURE.md §7
> ```

---

## Menggunakan Elastic Inference (Matryoshka)

Elastic Inference memungkinkan satu set bobot yang sudah dilatih untuk
menghasilkan beberapa submodel yang berbeda ukurannya — terinspirasi dari
Matryoshka Nested Transformer dan Gemma 3n.

### Konsep Dasar

```
Satu model terlatih → Banyak ukuran deployment

Full Model (100%)
├── Submodel 75% — Hampir sama baik, 25% lebih kecil
├── Submodel 50% — Kualitas moderat, 50% lebih kecil
└── Submodel 25% — Kualitas dasar, 75% lebih kecil
```

### Menggunakan Matryoshka

```python
from losion import LosionConfig, LosionForCausalLM
from losion.core.elastic.matryoshka import MatryoshkaModel

# Load model dengan Matryoshka support
config = LosionConfig()
model = LosionForCausalLM(config)

# Buat wrapper Matryoshka
matryoshka = MatryoshkaModel(model, granularity_factors=[0.25, 0.5, 0.75, 1.0])

# Deploy submodel ukuran tertentu
small_model = matryoshka.extract_submodel(factor=0.5)  # 50% dari ukuran penuh

# Generate dengan submodel kecil (lebih cepat, lebih sedikit memori)
output_small = small_model.generate(input_ids, max_new_tokens=100)

# Generate dengan model penuh (kualitas tertinggi)
output_full = model.generate(input_ids, max_new_tokens=100)
```

### Adaptive Sizing per Token

Matryoshka juga mendukung adaptive sizing — token sederhana menggunakan
submodel kecil, token kompleks menggunakan model penuh:

```python
# Aktifkan adaptive sizing saat inference
output = matryoshka.generate_adaptive(
    input_ids,
    max_new_tokens=100,
    adaptive=True,  # Otomatis pilih ukuran per token
)

# Lihat distribusi ukuran yang digunakan
stats = matryoshka.get_sizing_stats()
print(f"Token distribution: {stats.factor_distribution}")
# {0.25: 0.35, 0.5: 0.30, 0.75: 0.20, 1.0: 0.15}
```

### Mix'n'Match: Berbagai Ukuran per Layer

```python
# Layer awal kecil, layer akhir penuh
from losion.core.elastic.matryoshka import MixNMatchConfig

mix_config = MixNMatchConfig(
    layer_factors={
        0: 0.25,   # Layer 0: 25% ukuran
        1: 0.25,   # Layer 1: 25% ukuran
        2: 0.50,   # Layer 2: 50% ukuran
        3: 0.50,   # Layer 3: 50% ukuran
        4: 0.75,   # Layer 4: 75% ukuran
        5: 1.00,   # Layer 5: 100% ukuran (full)
    }
)

custom_model = matryoshka.extract_mixnmatch(mix_config)
```

### Training dengan Matryoshka Loss

Untuk memastikan semua submodel bekerja dengan baik, gunakan Matryoshka loss
saat training:

```yaml
# Dalam config training
training:
  use_matryoshka: true
  matryoshka:
    granularity_factors: [0.25, 0.5, 0.75, 1.0]
    loss_weight: 0.5    # Bobot Matryoshka loss relatif terhadap main loss
```

```python
# Atau secara programatik
from losion.core.elastic.matryoshka import MatryoshkaLoss

matryoshka_loss = MatryoshkaLoss(
    granularity_factors=[0.25, 0.5, 0.75, 1.0],
    weight=0.5,
)

# Dalam training loop
submodel_losses = []
for factor in [0.25, 0.5, 0.75, 1.0]:
    sub_out = matryoshka.forward_at_factor(x, factor=factor)
    sub_loss = ce_loss(sub_out, targets)
    submodel_losses.append(sub_loss)

total_loss = main_loss + matryoshka_loss(submodel_losses)
```

> **Agent Context**
> ```
> Elastic Inference: losion/core/elastic/matryoshka.py
> MatryoshkaModel: wrapper untuk extract submodels
> MatryoshkaLoss: training loss untuk semua granularity
> MixNMatchConfig: per-layer sizing
> Enable training: config.training.use_matryoshka = True
> Detail: ARCHITECTURE.md §8
> ```

---

## Memahami Konfigurasi

Losion menggunakan file YAML untuk konfigurasi. Berikut penjelasan
setiap bagian:

### Model Configuration

```yaml
model:
  d_model: 768          # Dimensi model — ukuran representasi internal
  n_layers: 12          # Jumlah layer — kedalaman model
  vocab_size: 32000     # Ukuran vocabulary — jumlah token unik
  max_seq_len: 32768    # Panjang sequence maksimum
```

**Panduan memilih**:
- `d_model`: Semakin besar = model lebih ekspresif, tapi lebih mahal
- `n_layers`: Semakin banyak = reasoning lebih dalam, tapi lebih lambat
- `vocab_size`: 32K untuk bahasa tunggal, 128K untuk multibahasa/code
- `max_seq_len`: 32K untuk umum, 131K+ untuk dokumen panjang

### SSM Configuration (Jalur 1)

```yaml
  ssm:
    d_state: 64          # Dimensi state SSM — kapasitas memori
    d_conv: 4            # Lebar konvolusi lokal — konteks lokal
    expand: 2            # Faktor ekspansi internal
    chunk_size: 256      # Ukuran chunk untuk paralelisasi
    use_wkv: true        # Aktifkan RWKV-7 WKV sub-layer
    use_delta_net: true  # Aktifkan Gated DeltaNet sub-layer
    interleaving_ratios: [4, 1, 1]  # SSD:WKV:Delta ratio
```

### Attention Configuration (Jalur 2)

```yaml
  attention:
    n_heads: 12          # Jumlah attention head
    d_kv: 64             # Dimensi per head
    mla_latent_dim: 128  # Dimensi latent MLA (kompresi KV)
    use_irope: true      # Interleaved RoPE
    irope_ratio: 3       # RoPE:NoPE ratio (3:1)
    base_interleaving_ratio: 5     # Local:Global ratio (normal)
    thinking_interleaving_ratio: 2 # Local:Global ratio (thinking)
```

### Retrieval Configuration (Jalur 3)

```yaml
  retrieval:
    num_experts: 16         # Total expert di MoE
    num_active_experts: 2   # Expert aktif per token (sparse)
    d_ff: 1536              # Dimensi FFN per expert
    use_engram: true        # Aktifkan Engram Memory
    engram_dim: 128         # Dimensi vektor engram
    use_shared_expert: true # Expert yang selalu aktif
```

### Router Configuration

```yaml
  router:
    top_k_pathways: 2       # Jalur aktif per token
    use_thinking_toggle: true  # Aktifkan thinking detection
    bias_lr: 0.01           # Learning rate khusus routing bias
    aux_loss_weight: 0.0    # 0.0 = aux-loss-free!
```

### Training Configuration

```yaml
training:
  batch_size: 32        # Batch size global
  learning_rate: 3.0e-4 # Learning rate awal
  weight_decay: 0.1     # Weight decay (regularisasi)
  warmup_steps: 2000    # Langkah warmup
  max_steps: 100000     # Total langkah training
  grad_clip: 1.0        # Norma gradient maksimum
  fp8_enabled: false    # FP8 (hanya H100/MI300X)
  precision: bf16       # Presisi: fp32, bf16, atau fp8
```

### Hardware Configuration

```yaml
hardware:
  device: auto          # auto = deteksi otomatis
  backend: auto         # auto = CUDA atau ROCm
  compile_model: true   # torch.compile() optimisasi
```

---

## Langkah Selanjutnya

Setelah berhasil menginstal dan menjalankan training pertama, berikut
langkah yang direkomendasikan:

### 1. Pelajari Arsitektur

Baca [ARCHITECTURE.md](ARCHITECTURE.md) untuk memahami bagaimana
Tri-Jalur Router bekerja secara detail.

### 2. Eksperimen dengan Konfigurasi

Coba ubah parameter konfigurasi dan amati efeknya:

```bash
# Eksperimen: Nonaktifkan Engram
python scripts/train.py --config configs/losion-1b.yaml \
    --override "model.retrieval.use_engram=false"

# Eksperimen: Ubah rasio interleaving
python scripts/train.py --config configs/losion-1b.yaml \
    --override "model.ssm.interleaving_ratios=[2,2,2]"
```

### 3. Gunakan Hardware yang Tepat

Baca [HARDWARE.md](HARDWARE.md) untuk panduan memilih dan mengonfigurasi
GPU, termasuk dukungan NVIDIA dan AMD.

### 4. Pelajari Training 4-Fase

Baca [TRAINING.md](TRAINING.md) untuk memahami training 4-fase Losion
dan cara mengoptimalkan hyperparameter.

### 5. Jelajahi Fitur Lanjutan

- **Reasoning Engine**: MCTS, Parallel Thinking, Neuro-Symbolic Verification
  — lihat [ARCHITECTURE.md §7](ARCHITECTURE.md)
- **Elastic Inference**: Matryoshka submodel extraction — lihat
  [ARCHITECTURE.md §8](ARCHITECTURE.md)
- **Advanced RLHF**: GRPO, Self-Play, Value Head — lihat
  [TRAINING.md §5–6](TRAINING.md)

### 6. Kontribusi

Baca [CONTRIBUTING.md](CONTRIBUTING.md) jika Anda ingin berkontribusi
ke pengembangan Losion.

### 7. Eksplorasi API

```python
from losion import (
    LosionConfig,
    LosionForCausalLM,
    LosionTrainer,
    GRPOTrainer,
    CurriculumScheduler,
)

# Buat konfigurasi custom
config = LosionConfig(d_model=256, n_layers=4, vocab_size=1000)

# Buat model
model = LosionForCausalLM(config)

# Hitung parameter
param_counts = model.count_parameters()
print(f"Total: {param_counts['total']:,} parameters")

# Generate text
input_ids = torch.tensor([[1, 50, 100]])
output = model.generate(input_ids, max_new_tokens=20, temperature=0.8)
print(f"Generated: {output}")
```

### 8. Baca Dokumentasi Tambahan

| Dokumen | Deskripsi |
|---------|-----------|
| [CHANGELOG.md](../CHANGELOG.md) | Riwayat perubahan per versi |
| [ROADMAP.md](../ROADMAP.md) | Rencana pengembangan ke depan |
| [SECURITY.md](../SECURITY.md) | Panduan keamanan dan vulnerability reporting |
| [CODE_OF_CONDUCT.md](../CODE_OF_CONDUCT.md) | Kode etik komunitas |

---

## FAQ

### Q: Apakah Losion berjalan di AMD GPU?

**Ya!** Losion dirancang hardware-agnostic. PyTorch mendukung ROCm,
sehingga Losion berjalan di AMD GPU tanpa perubahan kode. Cukup install
PyTorch dengan ROCm backend.

### Q: Berapa lama training 1B model?

Dengan RTX 4090: ~2-3 hari untuk 100K steps.
Dengan A100 80GB: ~1-2 hari untuk 100K steps.
Dengan CPU: **tidak direkomendasikan** (berminggu-minggu).

### Q: Apakah perlu dataset khusus?

Tidak, Losion menerima dataset teks standar (JSONL). Namun, untuk hasil
terbaik, gunakan dataset yang beragam dan berkualitas tinggi. Lihat
[TRAINING.md](TRAINING.md) untuk rekomendasi dataset.

### Q: Bagaimana cara mengubah ukuran model?

Buat file YAML baru atau salin dari konfigurasi yang sudah ada:

```bash
# Salin dan edit
cp configs/losion-1b.yaml configs/my-model.yaml
# Edit d_model, n_layers, dll.
python scripts/train.py --config configs/my-model.yaml
```

### Q: Error "CUDA out of memory", apa yang harus dilakukan?

1. Kurangi `batch_size` di config
2. Tambah `gradient_accumulation_steps`
3. Kurangi `max_seq_len`
4. Aktifkan `gradient_checkpointing: true`
5. Aktifkan Progressive KV Compression untuk inference panjang
6. Lihat [HARDWARE.md](HARDWARE.md) untuk panduan lengkap

### Q: Bagaimana menggunakan Reasoning Engine?

Aktifkan `thinking_mode=True` saat inference, atau gunakan MCTS/Parallel
Thinking langsung. Lihat [Menggunakan Reasoning Engine](#menggunakan-reasoning-engine).

### Q: Bagaimana menggunakan Elastic Inference?

Wrap model dengan `MatryoshkaModel` dan extract submodel berbagai ukuran.
Lihat [Menggunakan Elastic Inference](#menggunakan-elastic-inference-matryoshka).

### Q: Di mana melihat rencana pengembangan?

Lihat [ROADMAP.md](../ROADMAP.md) untuk rencana pengembangan dan
[CHANGELOG.md](../CHANGELOG.md) untuk riwayat perubahan.

### Q: Bagaimana melaporkan masalah keamanan?

Lihat [SECURITY.md](../SECURITY.md) untuk panduan vulnerability reporting.

---

*Dokumentasi ini ditulis untuk Losion v0.1.0. Jika mengalami masalah, silakan buka issue di repository GitHub.*
