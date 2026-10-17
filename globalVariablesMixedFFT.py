"""
Global Variables for Mixed-Precision FFT Optimization
Stage-level precision control (Option A).
Now using actual Vivado synthesis timing for the latency objective.
"""

import random
import math
from multiprocessing.pool import ThreadPool

# ======================= NSGA-II Parameters =======================
POPULATION = 30
GENERATIONS = 100
SEED = 42
MUTATION_RATE = 0.05
CROSSOVER_RATE = 0.9
OBJECTIVES = 4               # Power, Area, Performance, Latency (from Vivado)

CURRENT_GEN = 0
SOLUTION_THREADS = 6

FITNESS = 'fitness.npy'
DPI = 200

random.seed(SEED)

# ======================= FFT Configuration =======================
FFT_SIZES = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
CURRENT_FFT_SIZE = 8

# ======================= Chromosome Size Calculation =======================
def calculate_chromosome_size(fft_size):
    num_stages = int(math.log2(fft_size))
    chromosome_length = 2 * num_stages
    return chromosome_length

print("Chromosome sizes for different FFT sizes:")
for size in [8, 16, 32, 64, 128, 256, 512, 1024]:
    chrom_size = calculate_chromosome_size(size)
    ns = int(math.log2(size))
    print(f"  FFT-{size:<4}: {ns:>2} stages x 2 = {chrom_size:>3} genes")

# ======================= Vivado Configuration =======================
VIVADO_PATH = '/tools/Xilinx/Vivado/2021.1/bin/vivado'
VIVADO_BATCH_MODE = True
CLOCK_PERIOD = 100.0
FPGA_DEVICE = 'xc7a35tcpg236-1'

# ======================= File Paths =======================
VERILOG_SOURCES_DIR = './verilog_sources'
GENERATED_DESIGNS_DIR = './generated_designs'
VIVADO_PROJECTS_DIR = './vivado_projects'
REPORTS_DIR = './reports'
SIMULATION_DIR = './sim'
RESULTS_DIR = './results'

# ======================= Optimization Weights =======================
WEIGHT_POWER = 1.0
WEIGHT_AREA = 0.001
WEIGHT_PERFORMANCE = 50.0
WEIGHT_LATENCY = 8.0          # Higher weight for actual timing objective

# ======================= Constraint Thresholds =======================
MAX_POWER_W = 3.0
MAX_AREA_LUTS = 10000
MIN_SQNR_DB = -10.0
MAX_LATENCY_NORM = 3.0        # Max acceptable normalized latency
MIN_FREQ_MHZ = 80.0

# Reference for normalization (target clock period)
REFERENCE_CLOCK_PERIOD_NS = CLOCK_PERIOD

# Model parameters kept only as fallback
FP8_MULT_DELAY_NS = 6.5
FP4_MULT_DELAY_NS = 3.5
FP8_ADD_DELAY_NS  = 4.0
FP4_ADD_DELAY_NS  = 2.0
STAGE_OVERHEAD_NS = 2.0

# ======================= Performance Metrics =======================
ENABLE_RESULT_CACHE = True
RESULT_CACHE = {}

# ======================= Optimization Strategies =======================
def generate_smart_initial_population(fft_size, pop_size):
    from fft_template_generator import FFTTemplateGenerator
    gen = FFTTemplateGenerator(fft_size)
    chrom_length = gen.get_chromosome_length()
    population = []

    population.append([0] * chrom_length)                    # All FP4
    population.append([1] * chrom_length)                    # All FP8

    progressive = []
    for stage in range(gen.num_stages):
        prec = 1 if stage < gen.num_stages // 2 else 0
        progressive.extend([prec, prec])
    population.append(progressive)

    progressive_inv = []
    for stage in range(gen.num_stages):
        prec = 0 if stage < gen.num_stages // 2 else 1
        progressive_inv.extend([prec, prec])
    population.append(progressive_inv)

    mult_fp8 = []
    for _ in range(gen.num_stages):
        mult_fp8.extend([1, 0])
    population.append(mult_fp8)

    mult_fp4 = []
    for _ in range(gen.num_stages):
        mult_fp4.extend([0, 1])
    population.append(mult_fp4)

    for fp4_prob in [0.7, 0.3]:
        individual = [0 if random.random() < fp4_prob else 1 for _ in range(chrom_length)]
        population.append(individual)

    strat9 = []
    for stage in range(gen.num_stages):
        prec = 1 if stage < 2 else 0
        strat9.extend([prec, prec])
    population.append(strat9)

    strat10 = []
    for _ in range(gen.num_stages):
        strat10.extend([1, 0])
    population.append(strat10)

    while len(population) < pop_size:
        population.append([random.randint(0, 1) for _ in range(chrom_length)])

    return population[:pop_size]

ENABLE_SMART_INITIALIZATION = True

# ======================= Logging =======================
VERBOSE = True
LOG_FILE = './optimization.log'
SAVE_ALL_DESIGNS = False

# ======================= Helper Functions =======================
def log_message(message, level='INFO'):
    import datetime
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    log_line = f"[{timestamp}] [{level}] {message}"
    if VERBOSE:
        print(log_line)
    with open(LOG_FILE, 'a') as f:
        f.write(log_line + '\n')

def initialize_directories():
    import os
    dirs = [VERILOG_SOURCES_DIR, GENERATED_DESIGNS_DIR, VIVADO_PROJECTS_DIR,
            REPORTS_DIR, SIMULATION_DIR, RESULTS_DIR]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
    log_message("Initialized directory structure")

# ======================= Chromosome Encoding Notes =======================
"""
CHROMOSOME ENCODING:
  [s0_mult, s0_add, s1_mult, s1_add, ..., sN_mult, sN_add]

4th OBJECTIVE: Actual latency from Vivado synthesis timing report.
"""

initialize_directories()
