# Panduan Hardware Losion

> Dokumentasi komprehensif tentang dukungan hardware untuk framework Losion,
> termasuk NVIDIA GPU (CUDA), AMD GPU (ROCm), CPU-only training, instalasi,
> perbandingan performa, dan troubleshooting.

> **Agent Context**
> ```
> Hardware detection APIs: losion.check_cuda(), losion.check_rocm(),
>   losion.get_device_info() — lihat losion/utils/hardware.py
> Config: HardwareConfig di losion/config.py (device, backend, compile_model, precision)
> FP8 training: config.training.fp8_enabled = True (hanya H100/MI300X)
> Memory estimation: lihat Rumus Memori di bawah + ARCHITECTURE.md §14
> Progressive KV Compression: losion/training/advanced_memory_data.py
> Attention Sinks: losion/training/advanced_memory_data.py
> Dynamic Expert Buffer: losion/training/advanced_memory_data.py
> Security best practices: SECURITY.md
> ```

---

## Daftar Isi

1. [Pesan Utama](#pesan-utama)
2. [Hardware Detection API](#hardware-detection-api)
3. [NVIDIA GPU Support](#nvidia-gpu-support)
4. [AMD GPU Support](#amd-gpu-support)
5. [CPU-only Training](#cpu-only-training)
6. [FP8 Training](#fp8-training)
7. [Panduan Instalasi](#panduan-instalasi)
8. [Perbandingan Performa](#perbandingan-performa)
9. [Kebutuhan Memori per Ukuran Model](#kebutuhan-memori-per-ukuran-model)
10. [Rumus Estimasi Memori](#rumus-estimasi-memori)
11. [Fitur Memori Lanjutan](#fitur-memori-lanjutan)
12. [Multi-GPU Setup](#multi-gpu-setup)
13. [Keamanan Hardware](#keamanan-hardware)
14. [Troubleshooting](#troubleshooting)
15. [Quick Reference](#quick-reference)

---

## Pesan Utama

> **Losion dirancang untuk hardware-agnostic.** PyTorch mendukung CUDA dan ROCm,
> sehingga Losion bisa di-training di NVIDIA maupun AMD tanpa perubahan kode.
> Satu codebase, dua platform.

Losion tidak menggunakan custom CUDA kernels — seluruh implementasi berbasis
**murni PyTorch**. Ini berarti:

- ✅ Berjalan di NVIDIA GPU (CUDA) — tested on A100, H100, RTX 4090
- ✅ Berjalan di AMD GPU (ROCm) — tested on MI300X, MI250, RX 7900 XTX
- ✅ Berjalan di CPU — untuk development dan testing (lambat)
- ✅ `torch.compile()` memberikan optimisasi otomatis di semua platform
- ✅ Tidak perlu mengubah kode saat berpindah platform

---

## Hardware Detection API

Losion menyediakan API deteksi hardware yang menyeluruh untuk mengidentifikasi
platform GPU, kemampuan FP8, dan estimasi memori secara otomatis.

### API Utama

```python
from losion import check_cuda, check_rocm, get_device_info

# Cek ketersediaan CUDA
if check_cuda():
    print("NVIDIA CUDA terdeteksi!")

# Cek ketersediaan ROCm
if check_rocm():
    print("AMD ROCm terdeteksi!")

# Info detail perangkat (semua platform)
info = get_device_info()
print(info)
# {'cuda_available': True, 'rocm_available': False,
#  'device_count': 1, 'device_name': 'NVIDIA A100-SXM4-80GB',
#  'device_memory_gb': 79.15, 'fp8_supported': False}
```

### API Lanjutan

```python
from losion.utils.hardware import (
    check_fp8_support,          # Apakah hardware mendukung FP8?
    get_compute_capability,     # CUDA compute capability (misal 9.0 untuk H100)
    estimate_memory_required,   # Estimasi VRAM berdasarkan config model
    get_optimal_batch_size,     # Rekomendasi batch size berdasarkan VRAM
)

# Cek dukungan FP8
if check_fp8_support():
    print("FP8 training tersedia!")

# Estimasi kebutuhan memori
from losion.config import LosionConfig
config = LosionConfig()
est = estimate_memory_required(config, batch_size=4, seq_len=2048)
print(f"Estimasi VRAM: {est.total_gb:.1f} GB")
print(f"Breakdown: params={est.params_gb:.1f} GB, "
      f"activations={est.activations_gb:.1f} GB, "
      f"kv_cache={est.kv_cache_gb:.1f} GB")

# Rekomendasi batch size
optimal_bs = get_optimal_batch_size(config, available_vram_gb=80)
print(f"Batch size optimal: {optimal_bs}")
```

> **Agent Context**
> ```
> Semua hardware detection: losion/utils/hardware.py
> check_cuda() / check_rocm() → bool
> get_device_info() → dict dengan fp8_supported field
> estimate_memory_required(config, batch_size, seq_len) → MemoryEstimate
> get_optimal_batch_size(config, available_vram_gb) → int
> ```

---

## NVIDIA GPU Support

### CUDA Compatibility

Losion memerlukan **CUDA 11.8+** (direkomendasikan CUDA 12.1+). PyTorch
mendukung CUDA melalui backend bawaan — tidak perlu instalasi CUDA toolkit
terpisah jika menggunakan PyTorch dari pip/conda.

| CUDA Version | PyTorch Version | Status |
|-------------|----------------|--------|
| CUDA 11.8 | PyTorch 2.1+ | Didukung |
| CUDA 12.1 | PyTorch 2.2+ | Direkomendasikan |
| CUDA 12.4 | PyTorch 2.4+ | Terbaru |

### GPU yang Direkomendasikan

#### Untuk Training

| GPU | VRAM | Model Size | Catatan |
|-----|------|-----------|---------|
| RTX 4090 | 24 GB | 1B (single GPU) | Entry-level training |
| RTX 6000 Ada | 48 GB | 1B-7B (single GPU) | Workstation |
| A100 40GB | 40 GB | 1B-7B | Cloud standard |
| A100 80GB | 80 GB | 7B (single) / 48B (multi) | Cloud recommended |
| H100 80GB | 80 GB | 7B-48B | Fastest training + FP8 |
| H200 141GB | 141 GB | 48B (single) | Memory-optimized |
| B200 192GB | 192 GB | 48B+ (single) | Next-gen + FP8 native |
| B200 192GB (QuantSpec) | 192 GB | 48B+ quantized | BitNet/QuantSpec aware, 3× throughput |

### B200 192GB GPU — Deep Dive

The NVIDIA B200 with 192GB HBM3e is a game-changer for Losion training:

- **192 GB VRAM**: Fits Losion-48B entirely in single-GPU memory, even with optimizer states
- **FP8 native**: Blackwell FP8 support for ~2x training throughput vs bf16
- **Quantization-aware considerations**:
  - **BitNet 1.58-bit**: With B200's large memory, BitNet quantized models can train
    at 3× throughput. Use `BitNetConfig(warmup_steps=2000)` for gradual quantization.
  - **QuantSpec**: Self-speculative decoding with hierarchical quantization uses
    quantized versions of the same model as draft models for >90% acceptance rates.
    Pathway-aware: SSM pathway quantized more aggressively (less sensitive to error).
  - **Recommended**: For 48B+ models on B200, enable `fp8_enabled: true` +
    `use_bitdistill: true` for joint quantization + distillation training.

#### Untuk Inference

| GPU | VRAM | Model Size | Throughput |
|-----|------|-----------|------------|
| RTX 4090 | 24 GB | 1B | ~200 tok/s |
| A100 80GB | 80 GB | 7B | ~150 tok/s |
| H100 80GB | 80 GB | 7B | ~300 tok/s |
| H100 80GB | 80 GB | 48B (MLA compressed) | ~50 tok/s |

### Fitur NVIDIA Spesifik

- **Tensor Cores**: Dimanfaatkan otomatis oleh PyTorch untuk bf16/fp16
- **FP8 Training**: Didukung di H100+ (lihat [FP8 Training](#fp8-training))
- **Flash Attention**: Kompatibel melalui `torch.nn.functional.scaled_dot_product_attention`
- **NVLink**: Penting untuk multi-GPU training (48B model)
- **torch.compile()**: Mengoptimalkan graph secara otomatis untuk arsitektur GPU spesifik
- **Transformer Engine**: Tersedia untuk FP8 mixed-precision training di H100+

---

## AMD GPU Support

### ROCm Compatibility

Losion mendukung **ROCm 5.7+** (direkomendasikan ROCm 6.0+). PyTorch untuk ROCm
tersedia melalui channel khusus atau Docker image resmi AMD.

| ROCm Version | PyTorch Version | Status |
|-------------|----------------|--------|
| ROCm 5.7 | PyTorch 2.1+ | Didukung |
| ROCm 6.0 | PyTorch 2.2+ | Direkomendasikan |
| ROCm 6.2 | PyTorch 2.4+ | Terbaru |

> **Catatan Penting**: Di PyTorch, ROCm GPU dilaporkan sebagai "CUDA" device
> melalui HIP compatibility layer. Artinya, `torch.cuda.is_available()` mengembalikan
> `True` di AMD GPU, dan kode CUDA standar berjalan tanpa modifikasi.

### GPU yang Direkomendasikan

#### Untuk Training

| GPU | VRAM | Model Size | Catatan |
|-----|------|-----------|---------|
| RX 7900 XTX | 24 GB | 1B (single GPU) | Consumer-grade |
| RX 7900 XT | 20 GB | 1B | Consumer alternative |
| MI210 | 64 GB | 1B-7B | Data center |
| MI250 | 128 GB (2×64) | 7B | Data center standard |
| MI250X | 128 GB (2×64) | 7B | Data center enhanced |
| MI300X | 192 GB | 7B-48B | Flagship AMD + FP8 |

#### Untuk Inference

| GPU | VRAM | Model Size | Throughput |
|-----|------|-----------|------------|
| RX 7900 XTX | 24 GB | 1B | ~150 tok/s |
| MI250 | 128 GB | 7B | ~120 tok/s |
| MI300X | 192 GB | 48B (MLA compressed) | ~45 tok/s |

### Deteksi Otomatis AMD GPU

Losion memiliki utilitas deteksi otomatis yang mengenali AMD GPU:

```python
from losion import check_rocm, get_device_info

# Cek apakah ROCm tersedia
if check_rocm():
    print("AMD ROCm terdeteksi!")

# Info detail perangkat
info = get_device_info()
print(info)
# {'cuda_available': True, 'rocm_available': True,
#  'device_count': 1, 'device_name': 'AMD Instinct MI300X',
#  'device_memory_gb': 192.0, 'fp8_supported': True}
```

### Fitur AMD Spesifik

- **Matrix Cores**: Dimanfaatkan otomatis oleh PyTorch untuk bf16
- **FP8 Training**: Didukung di MI300X (lihat [FP8 Training](#fp8-training))
- **Infinity Fabric**: Penting untuk multi-GPU di MI250/MI300X
- **ROCm Docker**: Image resmi tersedia untuk deployment yang mudah
- **MIOpen**: Kernel cache otomatis setelah first run

---

## CPU-only Training

### Kapan Memungkinkan?

CPU-only training **tidak direkomendasikan** untuk training produksi, tetapi
berguna untuk:

1. **Development & debugging**: Mengembangkan fitur baru tanpa GPU
2. **Testing**: Menjalankan test suite di CI/CD tanpa GPU
3. **Eksperimen kecil**: Model 1B dengan batch size sangat kecil
4. **Verifikasi forward pass**: Memastikan model berjalan dengan benar

### Batasan CPU-only

| Aspek | GPU | CPU |
|-------|-----|-----|
| Kecepatan training 1B | ~1 menit/step | ~30 menit/step |
| Kecepatan training 7B | ~5 menit/step | ~3 jam/step |
| Precision | bf16/fp8 | fp32/bf16 (terbatas) |
| Multi-threading | Parallel otomatis | Perlu konfigurasi manual |
| Memory bandwidth | ~2 TB/s (H100) | ~50 GB/s (DDR5) |

### Konfigurasi CPU

```yaml
# configs/losion-1b-cpu.yaml
hardware:
  device: cpu
  backend: cpu
  compile_model: false  # compile lambat di CPU
  precision: fp32       # bf16 terbatas di CPU
```

### Optimisasi CPU

```bash
# Gunakan semua core
export OMP_NUM_THREADS=$(nproc)
export MKL_NUM_THREADS=$(nproc)

# Intel oneDNN optimisasi
export DNNL_PRIMITIVE_CACHE_CAPACITY=1024

# Jalankan training
python scripts/train.py --config configs/losion-1b-cpu.yaml
```

---

## FP8 Training

FP8 (8-bit floating point) training mengurangi kebutuhan memori ~40% dan
meningkatkan throughput secara signifikan. Losion mendukung FP8 melalui
`transformer_engine` (NVIDIA) atau `torch._scaled_mm` (PyTorch native).

### Hardware yang Didukung

| Platform | GPU | Compute Capability | FP8 Format | Status |
|----------|-----|-------------------|------------|--------|
| NVIDIA | H100 SXM | 9.0 | E4M3 / E5M2 | Didukung penuh |
| NVIDIA | H100 PCIe | 9.0 | E4M3 / E5M2 | Didukung penuh |
| NVIDIA | H200 | 9.0 | E4M3 / E5M2 | Didukung penuh |
| NVIDIA | B200 | 9.0+ | E4M3 / E5M2 | Didukung penuh |
| AMD | MI300X | CDNA 3 | FP8 (DType) | Didukung penuh |
| AMD | MI325 | CDNA 3+ | FP8 (DType) | Didukung penuh |
| NVIDIA | A100 | 8.0 | — | ❌ Tidak didukung |
| NVIDIA | RTX 4090 | 8.9 | — | ❌ Tidak didukung |

> **Catatan**: FP8 memerlukan hardware dengan compute capability 9.0+
> (NVIDIA Hopper atau yang lebih baru) atau AMD CDNA 3+ (MI300X+).
> GPU konsumer (RTX 3090, 4090) **tidak mendukung** FP8.

### Konfigurasi FP8

```yaml
training:
  fp8_enabled: true
  precision: bf16  # FP8 untuk matmul, bf16 untuk akumulasi
  fp8_format: hybrid  # "hybrid" (E4M3 fwd + E5M2 bwd) atau "e4m3"

hardware:
  compile_model: true  # Direkomendasikan dengan FP8
```

### Cara Kerja FP8 di Losion

```
┌──────────────────────────────────────────────────┐
│               FP8 Mixed Precision                │
│                                                  │
│  Forward:                                        │
│    Input (bf16) → [FP8 Cast] → Matmul (FP8)     │
│                                   ↓              │
│                              Output (bf16)       │
│                                                  │
│  Backward:                                       │
│    Grad (bf16) → [FP8 Cast] → Matmul (FP8)      │
│                                    ↓             │
│                              Grad (bf16)         │
│                                                  │
│  Tidak menggunakan FP8:                          │
│    - Layer Norm / RMSNorm                        │
│    - Softmax                                     │
│    - Loss computation                            │
│    - Router bias update                          │
│    - Engram hash lookup                          │
└──────────────────────────────────────────────────┘
```

### FP8 Performance Boost

| Model | GPU | bf16 Throughput | FP8 Throughput | Speedup |
|-------|-----|----------------|----------------|---------|
| Losion-7B | H100 80GB | ~4200 tok/s | ~6800 tok/s | 1.62× |
| Losion-7B | MI300X 192GB | ~3600 tok/s | ~5500 tok/s | 1.53× |
| Losion-48B | 8×H100 | ~1200 tok/s | ~1900 tok/s | 1.58× |

### FP8 Limitations

- **Tidak semua operasi menggunakan FP8**: Layer norm, softmax, dan loss
  tetap menggunakan bf16/fp32 untuk akurasi
- **Requires scaling**: FP8 membutuhkan per-tensor scaling factors yang
  harus di-calibrate secara periodik
- **Training stability**: Pada model kecil (<1B), FP8 bisa kurang stabil.
  Direkomendasikan hanya untuk model 7B+
- **Gradient accumulation**: Pastikan akumulasi gradient tetap di bf16/fp32

### Cek Dukungan FP8

```python
from losion.utils.hardware import check_fp8_support

if check_fp8_support():
    print("FP8 tersedia! Aktifkan dengan fp8_enabled: true")
else:
    print("FP8 tidak didukung di hardware ini. Gunakan bf16.")
```

> **Agent Context**
> ```
> FP8 check: losion.utils.hardware.check_fp8_support()
> Enable: config.training.fp8_enabled = True
> Format: config.training.fp8_format = "hybrid" (default)
> Backend: transformer_engine (NVIDIA) atau torch._scaled_mm (PyTorch native)
> Hanya efektif untuk model 7B+ di H100/MI300X
> ```

---

## Panduan Instalasi

### Instalasi untuk NVIDIA (CUDA)

```bash
# 1. Buat virtual environment
python -m venv losion-env
source losion-env/bin/activate

# 2. Install PyTorch dengan CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 3. Install Losion
pip install -e .

# 4. (Opsional) Install Transformer Engine untuk FP8 di H100
pip install transformer-engine

# 5. Verifikasi
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
```

### Instalasi untuk AMD (ROCm)

```bash
# Opsi A: Native instalasi
# 1. Install ROCm 6.0 (lihat panduan AMD)
# 2. Buat virtual environment
python -m venv losion-env
source losion-env/bin/activate

# 3. Install PyTorch dengan ROCm 6.0
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.0

# 4. Install Losion
pip install -e .

# 5. Verifikasi
python -c "import torch; print(f'CUDA/ROCm: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

# Opsi B: Docker (direkomendasikan untuk AMD)
docker pull rocm/pytorch:latest
docker run -it --device=/dev/kfd --device=/dev/dri \
    --group-add=video --ipc=host --cap-add=SYS_PTRACE \
    --security-opt seccomp=unconfined \
    -v $(pwd):/workspace \
    rocm/pytorch:latest

# Di dalam container:
cd /workspace
pip install -e .
```

### Instalasi untuk CPU-only

```bash
# 1. Buat virtual environment
python -m venv losion-env
source losion-env/bin/activate

# 2. Install PyTorch (CPU version)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# 3. Install Losion
pip install -e .

# 4. Verifikasi
python -c "import torch; print(f'CPU only: {not torch.cuda.is_available()}')"
```

### Verifikasi Instalasi Lengkap

```bash
# Script verifikasi otomatis
python -c "
from losion import check_cuda, check_rocm, get_device_info

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

---

## Perbandingan Performa

### NVIDIA vs AMD — Training Throughput

Berdasarkan benchmark internal (Losion-7B, batch_size=4, seq_len=2048, bf16):

| Metric | H100 80GB | MI300X 192GB | Rasio |
|--------|-----------|-------------|-------|
| Training throughput | ~4200 tok/s | ~3600 tok/s | 0.86× |
| Peak memory usage | 68 GB | 62 GB | 0.91× |
| Time to 1K steps | ~2.5 jam | ~3.0 jam | 1.2× |
| FP8 throughput | ~6800 tok/s | ~5500 tok/s | 0.81× |

Berdasarkan benchmark internal (Losion-1B, batch_size=8, seq_len=2048, bf16):

| Metric | RTX 4090 | RX 7900 XTX | Rasio |
|--------|----------|-------------|-------|
| Training throughput | ~1800 tok/s | ~1400 tok/s | 0.78× |
| Peak memory usage | 18 GB | 16 GB | 0.89× |
| Time to 1K steps | ~1.5 jam | ~2.0 jam | 1.3× |

### Analisis Perbandingan

1. **H100 vs MI300X**: Performa MI300X ~80-86% dari H100. Keunggulan MI300X adalah VRAM lebih besar (192GB vs 80GB), memungkinkan batch size lebih besar atau model yang lebih besar tanpa offloading.

2. **RTX 4090 vs RX 7900 XTX**: Performa RX 7900 XTX ~78% dari RTX 4090. Gap ini terutama karena optimisasi software PyTorch yang lebih matang di CUDA.

3. **Tren**: Gap performa semakin mengecil dengan setiap rilis ROCm. AMD sangat aktif mengoptimalkan PyTorch untuk ROCm.

### Kapan Memilih AMD?

- ✅ Ketika VRAM besar diperlukan (MI300X 192GB vs H100 80GB)
- ✅ Ketika biaya per GB VRAM harus minimal
- ✅ Ketika vendor lock-in ingin dihindari
- ✅ Ketika model sangat besar (48B+) dan single-GPU deployment diinginkan

### Kapan Memilih NVIDIA?

- ✅ Ketika throughput maksimum diperlukan
- ✅ Ketika FP8 training adalah kebutuhan (implementasi lebih matang)
- ✅ Ketika ekosistem tooling (NCCL, TensorRT, DeepSpeed) diperlukan
- ✅ Ketika multi-GPU scaling di atas 8 GPU diperlukan

---

## Kebutuhan Memori per Ukuran Model

### Training Memory (bf16)

| Model | Parameter | Batch=1, Seq=2048 | Batch=8, Seq=2048 | Batch=32, Seq=8192 |
|-------|-----------|-------------------|-------------------|---------------------|
| Losion-1B | ~1B | ~6 GB | ~14 GB | ~38 GB |
| Losion-7B | ~7B | ~28 GB | ~52 GB | ~148 GB |
| Losion-48B | ~48B | ~96 GB | ~192 GB | ~580 GB |

### Inference Memory (bf16)

| Model | Parameter | Seq=2048 | Seq=32768 | Seq=131072 | Seq=1M |
|-------|-----------|----------|-----------|------------|--------|
| Losion-1B | ~1B | ~2.5 GB | ~3.2 GB | ~5.8 GB | ~28 GB |
| Losion-7B | ~7B | ~15 GB | ~18 GB | ~28 GB | ~120 GB |
| Losion-48B | ~48B | ~98 GB | ~106 GB | ~135 GB | ~480 GB |

> **Catatan**: Angka inference menggunakan MLA compressed KV-cache.
> Tanpa MLA, kebutuhan memori untuk Seq=1M akan 8× lebih besar.

### Rekomendasi GPU per Model

| Model | Single GPU | Multi-GPU (2×) | Multi-GPU (4×) | Multi-GPU (8×) |
|-------|-----------|----------------|----------------|----------------|
| Losion-1B | RTX 4090 24GB | - | - | - |
| Losion-7B | A100 80GB | 2× A100 40GB | 2× RTX 4090 | - |
| Losion-48B | - | 2× H100 80GB | 4× A100 80GB | 8× A100 40GB |

---

## Rumus Estimasi Memori

Untuk perhitungan memori yang lebih presisi, Losion menyediakan formula
estimasi berikut. Detail lengkap ada di [ARCHITECTURE.md §14 Parameter Computation](ARCHITECTURE.md).

### Training Memory (per GPU)

```
VRAM_training = Mem_param + Mem_gradient + Mem_optimizer + Mem_activation + Mem_kv

Mem_param      = 2 × N × bytes_per_param          # bf16 = 2 bytes
Mem_gradient   = 2 × N × bytes_per_param           # sama dengan params
Mem_optimizer  = 8 × N                              # AdamW: 2 state × 4 bytes
Mem_activation = batch × seq × d_model × n_layers × 4  # perkiraan
Mem_kv         = 2 × n_layers × n_heads × d_kv × seq × batch × bytes_per_param
```

Dengan MLA compression (8×):

```
Mem_kv_mla = 2 × n_layers × d_latent × seq × batch × bytes_per_param
```

### Inference Memory

```
VRAM_inference = Mem_param + Mem_kv + Mem_activation_infer

Mem_kv = Mem_kv_mla  (dengan MLA, 8× kompresi)
       atau Mem_kv_full (tanpa MLA)

# Dengan Progressive KV Compression (untuk seq panjang):
Mem_kv_progressive = Σ (window_size × compression_ratio × bytes_per_token)
```

### Contoh Perhitungan (Losion-7B, bf16, batch=4, seq=2048)

```python
# Parameter
N = 7_200_000_000  # active parameters (7B active dari 9.8B total)
bytes_per_param = 2  # bf16

# Training memory (perkiraan)
param_mem = 2 * N * bytes_per_param / 1e9           # ~28.8 GB
grad_mem = 2 * N * bytes_per_param / 1e9             # ~28.8 GB (tidak simultan dgn param)
optim_mem = 8 * N / 1e9                              # ~57.6 GB (fp32 states)
activation_mem = 4 * 2048 * 2048 * 12 * 4 / 1e9     # ~8.0 GB (approximate)

# Total dengan gradient checkpointing (hemat ~60% activation)
total = param_mem + optim_mem + activation_mem * 0.4
print(f"Estimasi VRAM training: ~{total:.0f} GB")
```

### Menggunakan API Estimasi

```python
from losion.utils.hardware import estimate_memory_required
from losion.config import LosionConfig

config = LosionConfig()
est = estimate_memory_required(config, batch_size=4, seq_len=2048)
print(f"Total: {est.total_gb:.1f} GB")
print(f"  Params: {est.params_gb:.1f} GB")
print(f"  Optimizer: {est.optimizer_gb:.1f} GB")
print(f"  Activations: {est.activations_gb:.1f} GB")
print(f"  KV Cache: {est.kv_cache_gb:.1f} GB")
```

> **Agent Context**
> ```
> Memory estimation: losion/utils/hardware.py → estimate_memory_required()
> Parameter computation formulas: ARCHITECTURE.md §14
> MoE: gunakan active_params, bukan total_params
> MLA: KV cache 8× lebih kecil
> FP8: matmul ~40% lebih kecil, akumulasi tetap bf16
> ```

---

## Fitur Memori Lanjutan

Losion mengintegrasikan beberapa teknik optimasi memori canggih yang secara
signifikan mengurangi kebutuhan VRAM, terutama untuk context panjang dan
model besar.

### Progressive KV Compression (Gemini LC)

Kompresi KV cache berdasarkan posisi — token baru mendapat fidelity penuh,
token lama dikompresi lebih agresif.

| Token Age | Compression | Memory |
|-----------|------------|--------|
| Recent (last 4K) | 1:1 (full) | 100% |
| Medium (4K–64K) | 4:1 | 25% |
| Old (64K+) | 16:1 | 6.25% |

**Overall**: ~10× memory reduction untuk 1M context dibanding uniform storage.

```python
from losion.training.advanced_memory_data import ProgressiveKVCompressor

compressor = ProgressiveKVCompressor(
    recent_window=4096,
    medium_window=65536,
    recent_ratio=1.0,
    medium_ratio=0.25,
    old_ratio=0.0625,
)

# Kompres KV cache
compressed_k, compressed_v = compressor.compress_kv(keys, values, current_length=100000)

# Estimasi penghematan memori
savings = compressor.estimate_memory_savings(seq_len=100000, bytes_per_element=2)
print(f"Memory savings: {savings['savings_ratio']:.1%}")
```

### Attention Sinks (Gemini LC)

Cadangkan 4 "sink tokens" di awal setiap sequence. Sink tokens tidak pernah
di-evict dari KV cache dan menerima attention weight yang disproportional,
menstabilkan streaming inference.

```python
from losion.training.advanced_memory_data import AttentionSinkManager

sink_manager = AttentionSinkManager(num_sink_tokens=4)

# Buat eviction mask (True = bisa di-evict)
eviction_mask = sink_manager.get_eviction_mask(seq_len=10000, device=device)

# Modifikasi attention mask agar selalu attend ke sink tokens
modified_mask = sink_manager.modify_attention_mask(attention_mask)
```

### Dynamic Expert Buffer Allocation (GShard)

Alih-alih over-provisioning buffer tetap untuk setiap MoE expert (menyebabkan
30–50% memory waste), alokasi secara dinamis berdasarkan predicted load.

```python
from losion.training.advanced_memory_data import DynamicExpertBufferAllocator

allocator = DynamicExpertBufferAllocator(
    num_experts=64,
    base_buffer_size=256,
    safety_margin=0.10,
)

# Alokasi buffer berdasarkan predicted load
predicted_loads = router.get_predicted_loads()  # [num_experts]
buffers = allocator.allocate_buffers(predicted_loads, total_tokens=4096)

# Bandingkan dengan alokasi tetap
savings = allocator.compute_memory_savings(predicted_loads, total_tokens=4096)
print(f"Memory savings: {savings['memory_savings_percent']:.1f}%")
```

### Ringkasan Optimasi Memori

| Teknik | Memory Savings | Kapan Digunakan |
|--------|---------------|-----------------|
| MLA KV Compression | 8× KV cache | Selalu (built-in) |
| Progressive KV Compression | ~10× untuk seq >64K | Long-context inference |
| Attention Sinks | Stabilitas streaming | Streaming / infinite generation |
| Dynamic Expert Buffer | 30–50% MoE buffer | MoE dengan expert >16 |
| Gradient Checkpointing | ~60% activation | Training, semua ukuran model |
| FP8 Training | ~40% matmul | H100/MI300X, model 7B+ |
| FSDP Sharding | ~70% per GPU | Multi-GPU training |

> **Agent Context**
> ```
> Semua teknik memori: losion/training/advanced_memory_data.py
> ProgressiveKVCompressor: kompresi posisi-dependent
> AttentionSinkManager: stabilisasi streaming inference
> DynamicExpertBufferAllocator: alokasi dinamis MoE buffer
> ModalityAwareLossWeighter: per-pathway loss weighting
> Enable di config: training.gradient_checkpointing = True
> FP8: training.fp8_enabled = True (lihat §FP8 Training)
> ```

---

## Multi-GPU Setup

### Distributed Data Parallel (DDP)

DDP mereplikasi model di setiap GPU dan menyinkronkan gradient:

```bash
# NVIDIA
torchrun --nproc_per_node=4 scripts/train.py --config configs/losion-7b.yaml

# AMD (pastikan ROCm terinstal)
torchrun --nproc_per_node=4 scripts/train.py --config configs/losion-7b.yaml
```

Konfigurasi YAML untuk DDP:

```yaml
training:
  batch_size: 128  # Global batch size
  # Effective batch per GPU = 128 / num_gpus

hardware:
  use_ddp: true
```

### Fully Sharded Data Parallel (FSDP)

FSDP mensharding parameter model di seluruh GPU, mengurangi kebutuhan memori per GPU:

```bash
# FSDP memerlukan lebih banyak GPU
torchrun --nproc_per_node=8 scripts/train.py --config configs/losion-48b.yaml
```

Konfigurasi YAML untuk FSDP:

```yaml
training:
  batch_size: 512
  use_fsdp: true

hardware:
  compile_model: false  # FSDP + compile belum stabil
```

### Multi-Node Setup

Untuk training skala besar (>8 GPU):

```bash
# Node 0 (master)
torchrun --nnodes=4 --nproc_per_node=8 \
    --master_addr=10.0.0.1 --master_port=29500 \
    scripts/train.py --config configs/losion-48b.yaml

# Node 1-3 (worker)
torchrun --nnodes=4 --nproc_per_node=8 \
    --master_addr=10.0.0.1 --master_port=29500 \
    scripts/train.py --config configs/losion-48b.yaml
```

### AMD Multi-GPU Spesifik

Untuk AMD MI250/MI300X dengan Infinity Fabric:

```bash
# Set HIP_VISIBLE_DEVICES untuk memilih GPU
HIP_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 \
    scripts/train.py --config configs/losion-7b.yaml

# Verifikasi topology
rocm-smi
rocminfo | grep "Agent"
```

---

## Keamanan Hardware

Keamanan hardware adalah aspek penting dalam deployment AI. Untuk panduan
lengkap, lihat [SECURITY.md](../SECURITY.md).

### Praktik Terbaik

- **Secure Boot**: Aktifkan secure boot di semua node training untuk mencegah
  unauthorized firmware
- **GPU Firmware**: Selalu update GPU firmware ke versi terbaru dari vendor
- **Memory Encryption**: Aktifkan hardware-level memory encryption (AMD SEV-SNP
  atau NVIDIA Confidential Computing) untuk data sensitif
- **Network Isolation**: Isolasi jaringan training dari internet publik
- **Access Control**: Batasi akses fisik dan remote ke GPU cluster
- **Audit Logging**: Aktifkan audit logging di semua GPU node

### Referensi

- [SECURITY.md](../SECURITY.md) — Panduan keamanan lengkap Losion
- [NVIDIA Security](https://www.nvidia.com/en-us/security/) — GPU security advisories
- [AMD Security](https://www.amd.com/en/resources/security-landing.html) — ROCm security

---

## Troubleshooting

### Masalah Umum NVIDIA

#### 1. CUDA Out of Memory

```
RuntimeError: CUDA out of memory. Tried to allocate 2.00 GiB
```

**Solusi**:
- Kurangi batch size: `training.batch_size: 16` (dari 32)
- Kurangi sequence length: `model.max_seq_len: 16384` (dari 32768)
- Aktifkan gradient accumulation: `training.gradient_accumulation_steps: 4`
- Aktifkan gradient checkpointing: `training.gradient_checkpointing: true`
- Gunakan FSDP untuk model besar
- Aktifkan Progressive KV Compression untuk inference panjang

#### 2. CUDA Version Mismatch

```
RuntimeError: CUDA version mismatch
```

**Solusi**:
```bash
# Cek versi CUDA
nvcc --version
python -c "import torch; print(torch.version.cuda)"

# Reinstall PyTorch dengan versi CUDA yang sesuai
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

#### 3. NCCL Timeout (Multi-GPU)

```
RuntimeError: NCCL error: unhandled system error
```

**Solusi**:
```bash
# Set environment variable
export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=eth0  # Ganti sesuai network interface
export NCCL_IB_DISABLE=1       # Jika InfiniBand tidak tersedia
```

#### 4. FP8 Training Error

```
RuntimeError: FP8 is not supported on this device
```

**Solusi**:
- Pastikan GPU mendukung FP8 (H100, H200, MI300X)
- Gunakan `check_fp8_support()` untuk verifikasi
- Nonaktifkan FP8: `training.fp8_enabled: false`

### Masalah Umum AMD

#### 1. ROCm Not Detected

```
CUDA not available (ROCm)
```

**Solusi**:
```bash
# Cek apakah ROCm terinstal
rocm-smi
/opt/rocm/bin/rocminfo

# Cek apakah user dalam group 'video' dan 'render'
groups $USER
sudo usermod -aG video,render $USER

# Re-login dan coba lagi
```

#### 2. HIP Memory Error

```
HIP error: hipErrorMemoryAllocation
```

**Solusi**:
- Kurangi batch size atau sequence length
- Gunakan `HSA_ENABLE_SDMA=0` environment variable
- Set `GPU_MAX_HEAP_SIZE=100` (persentase VRAM yang digunakan)

#### 3. MIOpen Kernel Cache

```
MIOpen: Cannot find kernel in cache
```

**Solusi**:
```bash
# Bersihkan cache MIOpen
rm -rf ~/.miopen/*
export MIOPEN_USER_DB_PATH=/tmp/miopen-db-$USER
```

#### 4. Slow First Run di AMD

Eksekusi pertama di AMD GPU lambat karena MIOpen kernel compilation.
Ini normal — kernel di-cache untuk eksekusi berikutnya.

```bash
# Pre-warm cache
export MIOPEN_FIND_MODE=NORMAL  # Lebih cepat daripada HYBRID untuk first run
```

### Masalah Umum (Kedua Platform)

#### 1. torch.compile() Error

```
RuntimeError: Compiled graph execution failed
```

**Solusi**:
```bash
# Nonaktifkan compile
# Dalam config YAML:
hardware:
  compile_model: false

# Atau environment variable
export TORCHDYNAMO_DISABLE=1
```

#### 2. bf16 Not Supported

```
RuntimeError: "bfloat16" is not supported on this device
```

**Solusi**:
- Gunakan fp32: `training.precision: fp32`
- Hanya GPU dengan compute capability 8.0+ (Ampere+) yang mendukung bf16

#### 3. Slow Data Loading

**Solusi**:
```bash
# Tingkatkan num_workers
# Dalam config:
training:
  dataloader_num_workers: 8
  dataloader_pin_memory: true

# Atau gunakan persistent_workers
# Di DataLoader:
DataLoader(dataset, num_workers=8, pin_memory=True, persistent_workers=True)
```

---

## Quick Reference

### Environment Variables Penting

| Variable | Platform | Keterangan |
|----------|----------|------------|
| `CUDA_VISIBLE_DEVICES` | NVIDIA | Pilih GPU yang terlihat |
| `HIP_VISIBLE_DEVICES` | AMD | Pilih GPU yang terlihat |
| `NCCL_DEBUG` | NVIDIA | Debug multi-GPU |
| `NCCL_SOCKET_IFNAME` | NVIDIA | Network interface untuk multi-node |
| `HSA_ENABLE_SDMA` | AMD | Disable SDMA untuk stabilitas |
| `MIOPEN_FIND_MODE` | AMD | Kernel find mode |
| `TORCHDYNAMO_DISABLE` | Keduanya | Disable torch.compile() |
| `OMP_NUM_THREADS` | CPU | Thread count untuk CPU training |
| `TRANSFORMER_ENGINE_FP8` | NVIDIA | Enable FP8 via Transformer Engine |

### Checklist Instalasi

- [ ] Python 3.10+ terinstal
- [ ] PyTorch terinstal dengan backend yang sesuai (CUDA/ROCm/CPU)
- [ ] `torch.cuda.is_available()` mengembalikan True (untuk GPU)
- [ ] `python -c "import losion"` berjalan tanpa error
- [ ] Losion test suite pass: `pytest tests/`
- [ ] Advanced feature tests pass: `pytest tests/test_advanced.py`
- [ ] Verifikasi hardware info: `python -c "from losion import get_device_info; print(get_device_info())"`
- [ ] Cek FP8 support (jika H100/MI300X): `python -c "from losion.utils.hardware import check_fp8_support; print(check_fp8_support())"`
- [ ] Review keamanan hardware: lihat [SECURITY.md](../SECURITY.md)

---

*Dokumentasi ini ditulis untuk Losion v1.9.0. Performa dapat bervariasi tergantung driver, firmware, dan versi software.*
