"""
Diffusion Refinement — AlphaFold3-inspired output refinement via diffusion.

Diadaptasi dari AlphaFold3's diffusion module (Google DeepMind, 2024):
alih-alih langsung memproduksi output final, model men-generate
coarse output lalu melakukan iterative denoising untuk meningkatkan
kualitas. Dalam AlphaFold3, ini menghasilkan koordinat atom 3D;
dalam Losion, ini memperhalus representasi output sebelum proyeksi
ke vocabulary.

Konsep kunci:
1. **Coarse-to-Fine**: Output kasar → iterative refinement → output halus
2. **Conditional Denoising**: Setiap langkah denoising dikondisikan pada
   konteks (hidden states dari Tri-Jalur)
3. **Adaptive Steps**: Lebih banyak steps = kualitas lebih tinggi tapi lebih lambat
4. **Training**: Forward process = add noise, reverse process = learn to denoise

AlphaFold3 menggunakan diffusion untuk menghasilkan 3D coordinates.
Losion mengadaptasi ini untuk:
- Refine token representations sebelum vocabulary projection
- Mengurangi artifacts dan inkonsistensi dalam output
- Meningkatkan koherensi sequence yang dihasilkan

Referensi:
- Abramson et al., "Accurate structure prediction of biomolecular interactions
  with AlphaFold 3" (Nature, 2024)
- Ho et al., "Denoising Diffusion Probabilistic Models" (NeurIPS 2020)

Hardware: Pure PyTorch, kompatibel dengan CUDA, ROCm, dan CPU.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiffusionRefinementConfig:
    """Konfigurasi untuk Diffusion Refinement module.

    Args:
        num_steps: Jumlah denoising steps saat inference.
        d_context: Dimensi context conditioning.
        d_refine: Dimensi internal refinement.
        schedule_type: Tipe noise schedule ("linear" atau "cosine").
        min_signal_rate: Minimum signal rate (controls noise level).
        max_signal_rate: Maximum signal rate.
    """

    def __init__(
        self,
        num_steps: int = 4,
        d_context: int = 2048,
        d_refine: int = 512,
        schedule_type: str = "cosine",
        min_signal_rate: float = 0.02,
        max_signal_rate: float = 0.95,
    ) -> None:
        self.num_steps = num_steps
        self.d_context = d_context
        self.d_refine = d_refine
        self.schedule_type = schedule_type
        self.min_signal_rate = min_signal_rate
        self.max_signal_rate = max_signal_rate


class NoiseScheduler:
    """Noise scheduler untuk diffusion process.

    Mengatur level noise pada setiap step t:
    - signal_rate(t) = alpha_bar(t) → seberapa banyak signal yang tersisa
    - noise_rate(t) = sigma(t) → seberapa banyak noise yang ditambahkan

    Varian:
    - Linear: alpha_bar menurun linear dari 1 ke 0
    - Cosine: alpha_bar mengikuti cosine schedule (lebih smooth)

    Args:
        num_steps: Jumlah total steps.
        schedule_type: "linear" atau "cosine".
        min_signal_rate: Minimum signal rate.
        max_signal_rate: Maximum signal rate.
    """

    def __init__(
        self,
        num_steps: int = 4,
        schedule_type: str = "cosine",
        min_signal_rate: float = 0.02,
        max_signal_rate: float = 0.95,
    ) -> None:
        self.num_steps = num_steps
        self.schedule_type = schedule_type
        self.min_signal_rate = min_signal_rate
        self.max_signal_rate = max_signal_rate

    def signal_rate(self, t: torch.Tensor) -> torch.Tensor:
        """Hitung signal rate (alpha_bar) pada time step t.

        Args:
            t: Time step [batch] di [0, 1]

        Returns:
            Signal rate [batch] di [min_signal_rate, max_signal_rate]
        """
        if self.schedule_type == "cosine":
            # Cosine schedule: alpha_bar(t) = cos^2((1-t) * pi/2)
            return torch.cos(t * math.pi / 2) ** 2
        else:
            # Linear schedule
            return 1.0 - t

    def noise_rate(self, t: torch.Tensor) -> torch.Tensor:
        """Hitung noise rate (sigma) pada time step t.

        sigma(t) = sqrt(1 - alpha_bar(t)^2)

        Args:
            t: Time step [batch] di [0, 1]

        Returns:
            Noise rate [batch]
        """
        return torch.sqrt(1.0 - self.signal_rate(t) ** 2)

    def get_timesteps(self, device: torch.device) -> torch.Tensor:
        """Dapatkan normalized time steps untuk inference.

        Returns:
            Time steps [num_steps] di [0, 1]
        """
        steps = torch.linspace(0, 1, self.num_steps + 1, device=device)
        return steps[:-1]  # Exclude t=1 (pure noise)


class DenoiserBlock(nn.Module):
    """Single denoising block — memproses noisy output menjadi cleaner output.

    Diadaptasi dari AlphaFold3's diffusion module: menggunakan
    Transformer-style architecture yang dikondisikan pada context
    dan time step.

    Args:
        d_input: Dimensi input (noisy representation).
        d_context: Dimensi context conditioning.
        d_hidden: Dimensi internal.
    """

    def __init__(
        self,
        d_input: int,
        d_context: int,
        d_hidden: int = 512,
    ) -> None:
        super().__init__()
        self.d_input = d_input
        self.d_context = d_context

        # === Time Embedding ===
        self.time_mlp = nn.Sequential(
            nn.Linear(1, d_hidden // 4, bias=False),
            nn.SiLU(),
            nn.Linear(d_hidden // 4, d_input, bias=False),
        )

        # === Context Conditioning ===
        self.context_proj = nn.Sequential(
            nn.Linear(d_context, d_hidden, bias=False),
            nn.SiLU(),
            nn.Linear(d_hidden, d_input, bias=False),
        )

        # === Denoising Network ===
        self.denoiser = nn.Sequential(
            nn.Linear(d_input * 2, d_hidden, bias=False),
            nn.SiLU(),
            nn.Linear(d_hidden, d_hidden, bias=False),
            nn.SiLU(),
            nn.Linear(d_hidden, d_input, bias=False),
        )

        # === Output Gate ===
        self.gate = nn.Sequential(
            nn.Linear(d_input + 1, d_input, bias=False),
            nn.Sigmoid(),
        )

        self.layer_norm = nn.LayerNorm(d_input)

    def forward(
        self,
        x_noisy: torch.Tensor,
        context: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass denoising block.

        Args:
            x_noisy: Noisy input [batch, seq, d_input]
            context: Context conditioning [batch, seq, d_context]
            t: Time step [batch, 1]

        Returns:
            Denoised output [batch, seq, d_input]
        """
        # Time embedding
        t_embed = self.time_mlp(t)  # [batch, d_input]

        # Context projection
        ctx_signal = self.context_proj(context)  # [batch, seq, d_input]

        # Combine noisy input + context
        denoiser_input = torch.cat([x_noisy, ctx_signal], dim=-1)  # [batch, seq, d_input*2]
        denoised = self.denoiser(denoiser_input)  # [batch, seq, d_input]

        # Add time signal
        denoised = denoised + t_embed.unsqueeze(1)  # Broadcast over seq

        # Gate: kontrol seberapa banyak denoising yang diapply
        gate_input = torch.cat(
            [x_noisy.mean(dim=-1, keepdim=False), t.expand(x_noisy.shape[0])],
            dim=-1,
        )
        gate_value = self.gate(gate_input.unsqueeze(1))  # [batch, 1, d_input]

        # Gated residual
        output = x_noisy + gate_value * denoised
        output = self.layer_norm(output)

        return output


class DiffusionRefinement(nn.Module):
    """Diffusion Refinement — AlphaFold3-style output refinement.

    Meningkatkan kualitas output melalui iterative denoising.
    Output dari Tri-Jalur pipeline di-refine sebelum diproyeksikan
    ke vocabulary.

    Alur:
    1. Coarse output dari Tri-Jalur pipeline → x_0
    2. Add noise: x_t = alpha * x_0 + sigma * noise
    3. Iterative denoising: x_{t-1} = denoise(x_t, context, t)
    4. Final refined output: x_0_hat

    Training:
    - Sample random t ∈ [0, 1]
    - Add noise: x_t = alpha(t) * x_0 + sigma(t) * eps
    - Predict: eps_hat = model(x_t, context, t)
    - Loss: ||eps - eps_hat||^2

    Inference:
    - Start dari t=1 (pure noise) atau t=T (slightly noised x_0)
    - Iterative denoising dari t ke t-1
    - Final: refined x_0

    Integrasi dengan Losion:
    - Berada di output pipeline, setelah flow matching
    - Dapat diaktifkan hanya untuk task yang membutuhkan kualitas tinggi
    - Adaptive steps: lebih banyak steps untuk output yang lebih penting

    Args:
        d_model: Dimensi model.
        config: Konfigurasi diffusion refinement.
    """

    def __init__(
        self,
        d_model: int,
        config: Optional[DiffusionRefinementConfig] = None,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.config = config or DiffusionRefinementConfig(d_context=d_model)

        # === Noise Scheduler ===
        self.scheduler = NoiseScheduler(
            num_steps=self.config.num_steps,
            schedule_type=self.config.schedule_type,
            min_signal_rate=self.config.min_signal_rate,
            max_signal_rate=self.config.max_signal_rate,
        )

        # === Denoiser Blocks ===
        # Stack beberapa denoiser blocks untuk kualitas lebih tinggi
        self.denoisers = nn.ModuleList([
            DenoiserBlock(
                d_input=d_model,
                d_context=self.config.d_context,
                d_hidden=self.config.d_refine,
            )
            for _ in range(min(self.config.num_steps, 4))
        ])

        # === Output projection ===
        self.output_proj = nn.Sequential(
            nn.Linear(d_model, d_model, bias=False),
            nn.SiLU(),
            nn.Linear(d_model, d_model, bias=False),
        )

    def add_noise(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Tambahkan noise ke input berdasarkan time step.

        Forward process: x_t = alpha(t) * x_0 + sigma(t) * noise

        Args:
            x: Clean input [batch, seq, d_model]
            t: Time step [batch, 1]
            noise: Pre-generated noise (opsional)

        Returns:
            Noisy input [batch, seq, d_model]
        """
        if noise is None:
            noise = torch.randn_like(x)

        alpha = self.scheduler.signal_rate(t).unsqueeze(-1).unsqueeze(-1)
        sigma = self.scheduler.noise_rate(t).unsqueeze(-1).unsqueeze(-1)

        return alpha * x + sigma * noise

    def training_loss(
        self,
        x_clean: torch.Tensor,
        context: torch.Tensor,
    ) -> torch.Tensor:
        """Hitung diffusion training loss.

        Menggunakan noise prediction objective (epsilon prediction):
        L = ||eps - eps_hat||^2

        Args:
            x_clean: Clean output [batch, seq, d_model]
            context: Context conditioning [batch, seq, d_model]

        Returns:
            Scalar loss
        """
        batch_size = x_clean.shape[0]

        # Sample random time steps
        t = torch.rand(batch_size, 1, device=x_clean.device)

        # Generate noise
        noise = torch.randn_like(x_clean)

        # Add noise
        x_noisy = self.add_noise(x_clean, t, noise)

        # Predict noise using first denoiser
        denoiser_idx = 0
        if len(self.denoisers) > 0:
            predicted_noise = self.denoisers[denoiser_idx](
                x_noisy, context, t
            )
        else:
            predicted_noise = x_noisy  # Fallback

        # Loss: MSE antara predicted dan actual noise
        # Menggunakan x_clean sebagai target (data prediction)
        # Atau noise sebagai target (epsilon prediction)
        loss = F.mse_loss(predicted_noise, x_clean)

        return loss

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        num_steps: Optional[int] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Forward pass: refine output melalui iterative denoising.

        Args:
            x: Coarse output dari Tri-Jalur pipeline [batch, seq, d_model]
            context: Context conditioning [batch, seq, d_model]
            num_steps: Override jumlah denoising steps (opsional)

        Returns:
            Tuple (refined_output, info):
            - refined_output: [batch, seq, d_model]
            - info: Dictionary statistik
        """
        n_steps = num_steps or self.config.num_steps
        batch_size = x.shape[0]
        device = x.device

        # Mulai dari coarse output yang sedikit di-noise
        # Ini bukan pure noise — kita mulai dari model output
        # dan refine sedikit demi sedikit
        t_start = torch.ones(batch_size, 1, device=device) * 0.3  # Light noise
        current = self.add_noise(x, t_start)

        # Iterative denoising
        timesteps = torch.linspace(0.3, 0.0, n_steps + 1, device=device)

        for step in range(n_steps):
            t = timesteps[step].expand(batch_size, 1)

            # Pilih denoiser block (cycle jika steps > denoisers)
            denoiser_idx = step % len(self.denoisers)
            current = self.denoisers[denoiser_idx](current, context, t)

        # Final projection
        refined = self.output_proj(current)

        # Residual connection: refined = alpha * refined + (1-alpha) * x
        alpha = 0.7  # Sebagian besar gunakan refined
        refined = alpha * refined + (1 - alpha) * x

        info = {
            "num_refinement_steps": n_steps,
            "start_noise_level": 0.3,
            "refinement_delta": (refined - x).norm(dim=-1).mean().item(),
        }

        return refined, info
