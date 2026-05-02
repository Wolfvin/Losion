"""
Evolutionary Program Search — FunSearch-inspired solution discovery.

Diadaptasi dari FunSearch (DeepMind, 2023, Nature, 1325+ cites):
menggunakan LLM sebagai mutator dalam evolutionary search untuk
menemukan solusi baru. FunSearch memasangkan LLM dengan systematic
evaluator untuk menemukan program yang memecahkan masalah matematika.

Konsep FunSearch:
1. Population: Kumpulan program/solusi saat ini
2. Mutation: LLM menghasilkan variasi dari program terbaik
3. Evaluation: Evaluator menilai kualitas setiap program
4. Selection: Program terbaik survive ke generasi berikutnya
5. Repeat: Iterasi sampai konvergensi

Adaptasi untuk Losion:
1. Population: Kumpulan reasoning paths/solutions
2. Mutation: Model menghasilkan variasi berdasarkan best solutions
3. Evaluation: Value network + symbolic verifier menilai kualitas
4. Selection: Solutions terbaik survive
5. Repeat: Iterasi sampai solusi memenuhi kriteria

Keunggulan:
- Menemukan solusi NOVEL yang tidak muncul dari single-pass generation
- LLM sebagai mutator memberikan variasi yang cerdas
- Evaluator memastikan kualitas
- Populasi-based → lebih robust daripada single solution

Hardware: Pure PyTorch, kompatibel dengan CUDA, ROCm, dan CPU.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Callable, Tuple
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Program:
    """Satu program/solusi dalam populasi evolutionary.

    Attributes:
        id: Identifikasi unik.
        content: Representasi program (embedding tensor).
        score: Skor evaluasi.
        generation: Generasi keberapa program ini dibuat.
        parent_id: ID parent program (None untuk seed).
    """

    id: int = 0
    content: Optional[torch.Tensor] = None
    score: float = -float("inf")
    generation: int = 0
    parent_id: Optional[int] = None

    @property
    def is_valid(self) -> bool:
        """Apakah program ini valid (punya content dan score)."""
        return self.content is not None and self.score > -float("inf")


@dataclass
class EvolutionaryConfig:
    """Konfigurasi untuk Evolutionary Search.

    Attributes:
        population_size: Ukuran populasi per generasi.
        num_elites: Jumlah elite yang langsung survive.
        mutation_rate: Probabilitas mutasi per individu.
        crossover_rate: Probabilitas crossover.
        max_generations: Maksimum generasi.
        score_threshold: Threshold skor untuk berhenti (jika tercapai).
        diversity_weight: Bobot keragaman dalam selection.
    """

    population_size: int = 16
    num_elites: int = 4
    mutation_rate: float = 0.7
    crossover_rate: float = 0.3
    max_generations: int = 10
    score_threshold: float = 0.95
    diversity_weight: float = 0.1


class LLMMutator(nn.Module):
    """LLM-based mutator — menghasilkan variasi dari program terbaik.

    Diadaptasi dari FunSearch: LLM bertindak sebagai mutator cerdas
    yang menghasilkan variasi dari program yang sudah ada. Berbeda
    dengan random mutation, LLM mutation mempertahankan struktur
    yang baik dan mengubah bagian yang perlu.

    Args:
        d_model: Dimensi model.
        mutation_strength: Kekuatan mutasi (0.0 = tidak ada, 1.0 = sangat berbeda).
    """

    def __init__(
        self,
        d_model: int,
        mutation_strength: float = 0.3,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.mutation_strength = mutation_strength

        # Mutation network: menghasilkan perturbasi berdasarkan parent
        self.mutation_net = nn.Sequential(
            nn.Linear(d_model, d_model, bias=False),
            nn.SiLU(),
            nn.Linear(d_model, d_model, bias=False),
        )

        # Context conditioning: memasukkan info tentang best solutions
        self.context_net = nn.Sequential(
            nn.Linear(d_model, d_model // 2, bias=False),
            nn.SiLU(),
            nn.Linear(d_model // 2, d_model, bias=False),
        )

    def forward(
        self,
        parent: torch.Tensor,
        context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Mutasi program parent menjadi child.

        Args:
            parent: Program parent [batch, d_model]
            context: Context dari best solutions [batch, d_model]

        Returns:
            Mutated program [batch, d_model]
        """
        # Base mutation: perturbasi dari parent
        mutation = self.mutation_net(parent)

        # Context-aware: jika ada info tentang best solutions
        if context is not None:
            ctx_signal = self.context_net(context)
            mutation = mutation + ctx_signal

        # Combine: parent + scaled perturbation
        child = parent + self.mutation_strength * mutation

        return child


class CrossoverOperator(nn.Module):
    """Crossover operator — menggabungkan dua program.

    Mengambil bagian dari dua parent dan menggabungkannya
    menjadi child. Diadaptasi dari genetic programming crossover.

    Args:
        d_model: Dimensi model.
    """

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.d_model = d_model

        # Learned crossover mask generator
        self.mask_generator = nn.Sequential(
            nn.Linear(d_model * 2, d_model // 4, bias=False),
            nn.SiLU(),
            nn.Linear(d_model // 4, d_model, bias=False),
            nn.Sigmoid(),  # [0, 1] per dimension — blending weight
        )

    def forward(
        self,
        parent_a: torch.Tensor,
        parent_b: torch.Tensor,
    ) -> torch.Tensor:
        """Crossover dua parent menjadi child.

        Args:
            parent_a: Parent A [batch, d_model]
            parent_b: Parent B [batch, d_model]

        Returns:
            Child [batch, d_model]
        """
        # Generate blending mask
        combined = torch.cat([parent_a, parent_b], dim=-1)
        mask = self.mask_generator(combined)  # [batch, d_model]

        # Blend: mask * A + (1 - mask) * B
        child = mask * parent_a + (1 - mask) * parent_b

        return child


class EvolutionarySearcher(nn.Module):
    """Evolutionary Searcher — FunSearch-style solution discovery.

    Menggabungkan:
    1. LLM Mutator: variasi cerdas dari solusi terbaik
    2. Crossover: kombinasi solusi yang menjanjikan
    3. Evaluation: scoring berdasarkan value network + verifier
    4. Selection: survival of the fittest

    Integrasi dengan Losion:
    - Dapat digunakan sebagai bagian dari thinking mode
    - Berguna untuk: mathematical reasoning, code generation, puzzle solving
    - Output: best solution setelah beberapa generasi evolusi

    Args:
        d_model: Dimensi model.
        config: Konfigurasi evolutionary search.
    """

    def __init__(
        self,
        d_model: int,
        config: Optional[EvolutionaryConfig] = None,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.config = config or EvolutionaryConfig()

        # === LLM Mutator ===
        self.mutator = LLMMutator(d_model)

        # === Crossover Operator ===
        self.crossover = CrossoverOperator(d_model)

        # === Evaluator (Value Network) ===
        self.evaluator = nn.Sequential(
            nn.Linear(d_model, d_model // 2, bias=False),
            nn.SiLU(),
            nn.Linear(d_model // 2, 1, bias=False),
            nn.Tanh(),  # [-1, 1]
        )

        # === Diversity Scorer ===
        # Mengukur seberapa berbeda program dari populasi
        self.diversity_scorer = nn.Sequential(
            nn.Linear(d_model, d_model // 4, bias=False),
            nn.SiLU(),
            nn.Linear(d_model // 4, 1, bias=False),
            nn.Sigmoid(),  # [0, 1]
        )

    def evaluate(
        self,
        programs: List[Program],
    ) -> List[Program]:
        """Evaluasi semua program dalam populasi.

        Args:
            programs: List Program yang akan dievaluasi.

        Returns:
            Program yang sudah di-update dengan skor.
        """
        with torch.no_grad():
            for prog in programs:
                if prog.content is not None:
                    score = self.evaluator(prog.content).item()
                    prog.score = score

        return programs

    def select(
        self,
        programs: List[Program],
    ) -> List[Program]:
        """Seleksi program terbaik untuk generasi berikutnya.

        Elitism + fitness-proportionate selection.

        Args:
            programs: List Program.

        Returns:
            Selected programs.
        """
        # Sort by score (descending)
        sorted_progs = sorted(programs, key=lambda p: p.score, reverse=True)

        # Elite: langsung survive
        elites = sorted_progs[:self.config.num_elites]

        return elites

    def forward(
        self,
        seed: torch.Tensor,
        external_evaluator: Optional[Callable] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Jalankan evolutionary search.

        Args:
            seed: Seed program [batch, d_model]
            external_evaluator: External evaluation function (opsional).

        Returns:
            Tuple (best_solution, info):
            - best_solution: [batch, d_model]
            - info: Dictionary statistik
        """
        batch_size = seed.shape[0]
        pop_size = self.config.population_size
        device = seed.device

        # === Initialize Population ===
        population: List[Program] = []
        prog_id = 0

        for b in range(batch_size):
            population.append(Program(
                id=prog_id,
                content=seed[b:b+1].detach(),
                score=0.0,
                generation=0,
            ))
            prog_id += 1

        # Add random variations to fill population
        while len(population) < pop_size * batch_size:
            parent_idx = len(population) % batch_size
            parent_content = population[parent_idx].content
            # Small random perturbation
            mutated = parent_content + torch.randn_like(parent_content) * 0.1
            population.append(Program(
                id=prog_id,
                content=mutated.detach(),
                score=0.0,
                generation=0,
                parent_id=population[parent_idx].id,
            ))
            prog_id += 1

        # === Evolution Loop ===
        best_score_history = []
        best_program = population[0]

        for gen in range(self.config.max_generations):
            # 1. Evaluate
            population = self.evaluate(population)

            # Override with external evaluator if provided
            if external_evaluator is not None:
                for prog in population:
                    if prog.content is not None:
                        ext_score = external_evaluator(prog.content)
                        prog.score = 0.5 * prog.score + 0.5 * ext_score

            # Track best
            current_best = max(population, key=lambda p: p.score)
            if current_best.score > best_program.score:
                best_program = current_best

            best_score_history.append(best_program.score)

            # Check termination
            if best_program.score >= self.config.score_threshold:
                break

            # 2. Select
            elites = self.select(population)

            # 3. Create next generation
            new_population: List[Program] = []

            # Add elites directly
            for elite in elites:
                new_population.append(copy.deepcopy(elite))

            # Fill rest with mutations and crossovers
            while len(new_population) < pop_size * batch_size:
                # Select parent (tournament selection)
                import random
                tournament = random.sample(
                    population, min(3, len(population))
                )
                parent = max(tournament, key=lambda p: p.score)

                if parent.content is None:
                    continue

                # Mutation
                if random.random() < self.config.mutation_rate:
                    # Context: mean of elite contents
                    elite_contents = torch.cat([
                        e.content for e in elites if e.content is not None
                    ], dim=0)
                    context = elite_contents.mean(dim=0, keepdim=True)

                    child_content = self.mutator(parent.content, context)
                    new_population.append(Program(
                        id=prog_id,
                        content=child_content.detach(),
                        score=0.0,
                        generation=gen + 1,
                        parent_id=parent.id,
                    ))
                    prog_id += 1

                # Crossover
                if (random.random() < self.config.crossover_rate
                        and len(elites) >= 2):
                    parent_a = random.choice(elites)
                    parent_b = random.choice(elites)
                    if parent_a.content is not None and parent_b.content is not None:
                        child_content = self.crossover(
                            parent_a.content, parent_b.content
                        )
                        new_population.append(Program(
                            id=prog_id,
                            content=child_content.detach(),
                            score=0.0,
                            generation=gen + 1,
                            parent_id=parent_a.id,
                        ))
                        prog_id += 1

            population = new_population

        # Final evaluation
        population = self.evaluate(population)
        final_best = max(population, key=lambda p: p.score)
        if final_best.score > best_program.score:
            best_program = final_best

        info = {
            "generations": len(best_score_history),
            "best_score": best_program.score,
            "score_history": best_score_history,
            "population_size": len(population),
            "converged": best_program.score >= self.config.score_threshold,
        }

        # Return best solution
        if best_program.content is not None:
            return best_program.content.expand(batch_size, -1), info
        else:
            return seed, info
