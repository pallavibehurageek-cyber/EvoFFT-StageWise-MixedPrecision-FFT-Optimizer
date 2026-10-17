"""
Optimization Utilities for Mixed-Precision FFT
CORRECTED VERSION: Handles large chromosomes efficiently
"""

import numpy as np
import random
from pymoo.core.sampling import Sampling
from pymoo.core.mutation import Mutation
from pymoo.core.crossover import Crossover


class MyCallback:
    """Callback to track evolution history"""
    def __init__(self):
        self.data = []

    def __call__(self, algorithm):
        F = algorithm.pop.get('F')
        self.data.append(F)


class SmartInitialSampling(Sampling):
    """
    Custom sampling for initial population using domain knowledge
    For large chromosomes, this is much better than pure random
    """
    
    def _do(self, problem, n_samples, **kwargs):
        """
        Generate initial population using smart strategies
        """
        from globalVariablesMixedFFT import (
            ENABLE_SMART_INITIALIZATION,
            generate_smart_initial_population,
            CURRENT_FFT_SIZE
        )
        
        if ENABLE_SMART_INITIALIZATION:
            # Use smart initialization
            pop = generate_smart_initial_population(CURRENT_FFT_SIZE, n_samples)
            X = np.array(pop)
        else:
            # Fallback to random
            X = np.random.randint(0, 2, size=(n_samples, problem.n_var))
        
        return X


class BlockwiseMutation(Mutation):
    """
    Mutation operator that respects FFT stage structure
    Instead of randomly mutating any gene, mutate entire butterflies or stages
    """
    
    def __init__(self, fft_size, **kwargs):
        super().__init__(**kwargs)
        self.fft_size = fft_size
        import math
        self.num_stages = int(math.log2(fft_size))
        self.butterflies_per_stage = fft_size // 2
        self.genes_per_butterfly = 2  # mult_prec, add_prec
    
    def _do(self, problem, X, **kwargs):
        from globalVariablesMixedFFT import MUTATION_RATE, CURRENT_GEN, GENERATIONS
        
        # Adaptive mutation rate (decreases over time)
        adaptive_rate = MUTATION_RATE * (1.0 - CURRENT_GEN / GENERATIONS)
        
        Xp = np.copy(X)
        
        for i in range(len(Xp)):
            # Strategy: Mutate entire butterflies, not individual genes
            # This maintains some structure
            
            num_butterflies = len(Xp[i]) // self.genes_per_butterfly
            
            for bf_idx in range(num_butterflies):
                if random.random() < adaptive_rate:
                    # Mutate this butterfly's precision
                    gene_start = bf_idx * self.genes_per_butterfly
                    
                    # Three mutation strategies (equal probability)
                    strategy = random.randint(0, 2)
                    
                    if strategy == 0:
                        # Flip multiplier precision
                        Xp[i][gene_start] = 1 - Xp[i][gene_start]
                    elif strategy == 1:
                        # Flip adder precision
                        Xp[i][gene_start + 1] = 1 - Xp[i][gene_start + 1]
                    else:
                        # Flip both
                        Xp[i][gene_start] = 1 - Xp[i][gene_start]
                        Xp[i][gene_start + 1] = 1 - Xp[i][gene_start + 1]
        
        return Xp


class StagewiseMutation(Mutation):
    """
    Mutation that operates on entire stages.
    With stage-level encoding each stage IS already a 2-gene block,
    so this is equivalent to BlockwiseMutation but with stage-granularity
    strategies (set all FP4, all FP8, flip, randomise).
    """

    def __init__(self, fft_size, **kwargs):
        super().__init__(**kwargs)
        self.fft_size = fft_size
        import math
        self.num_stages = int(math.log2(fft_size))
        self.butterflies_per_stage = fft_size // 2  # kept for informational consistency
        self.genes_per_stage = 2  # one mult gene + one add gene per stage (stage-level encoding)

    def _do(self, problem, X, **kwargs):
        from globalVariablesMixedFFT import MUTATION_RATE

        Xp = np.copy(X)

        for i in range(len(Xp)):
            if random.random() < MUTATION_RATE:
                stage_to_mutate = random.randint(0, self.num_stages - 1)

                gene_start = stage_to_mutate * self.genes_per_stage
                gene_end   = gene_start + self.genes_per_stage

                strategy = random.randint(0, 3)

                if strategy == 0:
                    Xp[i][gene_start:gene_end] = 0          # all FP4
                elif strategy == 1:
                    Xp[i][gene_start:gene_end] = 1          # all FP8
                elif strategy == 2:
                    for j in range(gene_start, gene_end):   # flip
                        Xp[i][j] = 1 - Xp[i][j]
                else:
                    for j in range(gene_start, gene_end):   # random
                        Xp[i][j] = random.randint(0, 1)

        return Xp


class TwoPointCrossover(Crossover):
    """
    Two-point crossover that respects butterfly boundaries
    """
    
    def __init__(self, fft_size, **kwargs):
        super().__init__(2, 2, **kwargs)  # 2 parents, 2 offspring
        self.fft_size = fft_size
        import math
        self.num_stages = int(math.log2(fft_size))
        self.butterflies_per_stage = fft_size // 2
        self.genes_per_butterfly = 2
    
    def _do(self, problem, X, **kwargs):
        from globalVariablesMixedFFT import CROSSOVER_RATE
        
        _, n_matings, n_var = X.shape
        Y = np.full_like(X, 0)
        
        for k in range(n_matings):
            if random.random() < CROSSOVER_RATE:
                # Perform crossover at butterfly boundaries
                num_butterflies = n_var // self.genes_per_butterfly
                
                # Select two crossover points (butterfly indices)
                point1 = random.randint(0, num_butterflies - 1)
                point2 = random.randint(point1, num_butterflies - 1)
                
                # Convert to gene indices
                gene_point1 = point1 * self.genes_per_butterfly
                gene_point2 = point2 * self.genes_per_butterfly
                
                # Parent 1 and Parent 2
                p1 = X[0, k]
                p2 = X[1, k]
                
                # Offspring 1: P1[0:p1] + P2[p1:p2] + P1[p2:]
                Y[0, k, :gene_point1] = p1[:gene_point1]
                Y[0, k, gene_point1:gene_point2] = p2[gene_point1:gene_point2]
                Y[0, k, gene_point2:] = p1[gene_point2:]
                
                # Offspring 2: P2[0:p1] + P1[p1:p2] + P2[p2:]
                Y[1, k, :gene_point1] = p2[:gene_point1]
                Y[1, k, gene_point1:gene_point2] = p1[gene_point1:gene_point2]
                Y[1, k, gene_point2:] = p2[gene_point2:]
            else:
                # No crossover, copy parents
                Y[0, k] = X[0, k]
                Y[1, k] = X[1, k]
        
        return Y


class StagewiseCrossover(Crossover):
    """
    Crossover that exchanges entire stages between parents.
    With stage-level encoding each stage is a 2-gene block.
    """

    def __init__(self, fft_size, **kwargs):
        super().__init__(2, 2, **kwargs)
        self.fft_size = fft_size
        import math
        self.num_stages = int(math.log2(fft_size))
        self.butterflies_per_stage = fft_size // 2  # kept for informational consistency
        self.genes_per_stage = 2  # one mult gene + one add gene per stage (stage-level encoding)

    def _do(self, problem, X, **kwargs):
        from globalVariablesMixedFFT import CROSSOVER_RATE

        _, n_matings, n_var = X.shape
        Y = np.full_like(X, 0)

        for k in range(n_matings):
            if random.random() < CROSSOVER_RATE:
                crossover_stage = random.randint(0, self.num_stages - 1)

                gene_start = crossover_stage * self.genes_per_stage
                gene_end   = gene_start + self.genes_per_stage

                p1 = X[0, k]
                p2 = X[1, k]

                # Offspring 1: P1[before] + P2[stage] + P1[after]
                Y[0, k, :gene_start]        = p1[:gene_start]
                Y[0, k, gene_start:gene_end] = p2[gene_start:gene_end]
                Y[0, k, gene_end:]          = p1[gene_end:]

                # Offspring 2: P2[before] + P1[stage] + P2[after]
                Y[1, k, :gene_start]        = p2[:gene_start]
                Y[1, k, gene_start:gene_end] = p1[gene_start:gene_end]
                Y[1, k, gene_end:]          = p2[gene_end:]
            else:
                Y[0, k] = X[0, k]
                Y[1, k] = X[1, k]

        return Y


def determineDecisionVariableLimit(fft_size):
    """
    Determine chromosome bounds for given FFT size (stage-level encoding).

    Returns:
        [lower_limits, upper_limits]  each of length num_stages * 2
    """
    import math
    num_stages = int(math.log2(fft_size))
    chromosome_length = 2 * num_stages
    return [[0] * chromosome_length, [1] * chromosome_length]


def analyze_population_diversity(population):
    """
    Analyze diversity in population
    Returns statistics about precision distribution
    """
    pop_array = np.array(population)
    
    stats = {
        'mean_fp8_ratio': np.mean(pop_array),
        'std_fp8_ratio': np.std(pop_array),
        'min_fp8_count': np.min(np.sum(pop_array, axis=1)),
        'max_fp8_count': np.max(np.sum(pop_array, axis=1)),
        'unique_solutions': len(np.unique(pop_array, axis=0))
    }
    
    return stats
