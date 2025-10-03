import math
import random
from dataclasses import dataclass
from typing import Dict, List, NamedTuple, Sequence, Set, Tuple

PATTERN_VALUES: Tuple[str, str] = ("N", "O")


class AdjacentPair(NamedTuple):
    index: int
    orientation: str  # 'H' or 'V'
    a: Tuple[int, int]
    b: Tuple[int, int]


@dataclass
class GAParams:
    population_size: int = 60
    generations: int = 200
    tournament_k: int = 3
    crossover_rate: float = 0.9
    mutation_rate: float = 0.15
    inversion_rate: float = 0.15
    alpha: float = 100.0
    beta: float = 1.0
    gamma: float = 1.0
    Cmin: float = 1.0e-15
    Cmax: float = 30.0e-15
    seed: int = 42


@dataclass
class CPGenome:
    c_unit: float
    patterns: List[str]

    def clone(self) -> "CPGenome":
        return CPGenome(self.c_unit, self.patterns.copy())

    def as_labels(self) -> List[str]:
        return self.patterns.copy()


def clamp_c_unit(value: float, params: GAParams) -> float:
    return max(params.Cmin, min(params.Cmax, value))


def _mirror(coord: Tuple[int, int], rows: int, cols: int) -> Tuple[int, int]:
    r, c = coord
    return rows - 1 - r, cols - 1 - c


def build_cp_pairs(rows: int, cols: int) -> List[AdjacentPair]:
    if rows <= 0 or cols <= 0:
        raise ValueError("Array dimensions must be positive.")
    pairs: List[AdjacentPair] = []
    seen: Set[Tuple[Tuple[int, int], Tuple[int, int]]] = set()

    def try_add(a: Tuple[int, int], b: Tuple[int, int], orientation: str) -> None:
        canonical = tuple(sorted((a, b)))
        mirror_pair = tuple(sorted((_mirror(a, rows, cols), _mirror(b, rows, cols))))
        if canonical in seen or mirror_pair in seen:
            return
        seen.add(canonical)
        seen.add(mirror_pair)
        pairs.append(AdjacentPair(len(pairs), orientation, a, b))

    for r in range(rows):
        for c in range(cols - 1):
            try_add((r, c), (r, c + 1), "H")
    for c in range(cols):
        for r in range(rows - 1):
            try_add((r, c), (r + 1, c), "V")
    return pairs


def format_cp_sequence(genome: CPGenome, adjacency: Sequence[AdjacentPair]) -> List[str]:
    if len(genome.patterns) != len(adjacency):
        raise ValueError("Pattern length does not match adjacency description.")
    tokens = [f"{genome.c_unit:.6e}"]
    for pattern, pair in zip(genome.patterns, adjacency):
        tokens.append(
            f"{pattern}@{pair.orientation}({pair.a[0]},{pair.a[1]})-({pair.b[0]},{pair.b[1]})"
        )
    return tokens


def random_initial_genome(num_pairs: int, params: GAParams) -> CPGenome:
    c_unit = random.uniform(params.Cmin, params.Cmax)
    patterns = [random.choice(PATTERN_VALUES) for _ in range(num_pairs)]
    return CPGenome(c_unit=c_unit, patterns=patterns)


def evaluate_dnl_inl(
    genome: CPGenome,
    adjacency: Sequence[AdjacentPair],
    rows: int,
    cols: int,
) -> Tuple[float, float]:
    weight_map = {"O": 1.0, "N": 0.35}
    grid = [[0.0 for _ in range(cols)] for _ in range(rows)]
    for pattern, pair in zip(genome.patterns, adjacency):
        weight = weight_map.get(pattern, 0.0)
        r1, c1 = pair.a
        r2, c2 = pair.b
        grid[r1][c1] += weight
        grid[r2][c2] += weight
    total = sum(sum(row) for row in grid)
    if total == 0:
        return 0.5, 0.5

    center_r = (rows - 1) / 2.0
    center_c = (cols - 1) / 2.0

    moment_x = 0.0
    moment_y = 0.0
    edge_metric = 0.0
    for r in range(rows):
        for c in range(cols):
            v = grid[r][c]
            if v == 0.0:
                continue
            moment_x += v * (c - center_c)
            moment_y += v * (r - center_r)
            edge_metric += v * (abs(r - center_r) + abs(c - center_c))
    norm_moment = math.sqrt(moment_x * moment_x + moment_y * moment_y) / (total + 1e-12)
    edge_metric = edge_metric / ((rows + cols) * total + 1e-12)

    symmetry = 0.0
    visited: Set[Tuple[Tuple[int, int], Tuple[int, int]]] = set()
    for r in range(rows):
        for c in range(cols):
            mirror_r, mirror_c = _mirror((r, c), rows, cols)
            key = tuple(sorted(((r, c), (mirror_r, mirror_c))))
            if key in visited:
                continue
            visited.add(key)
            v1 = grid[r][c]
            v2 = grid[mirror_r][mirror_c]
            symmetry += abs(v1 - v2)
    symmetry /= (total + 1e-12)

    mismatch_metric = 0.5 * symmetry + 0.3 * norm_moment + 0.2 * edge_metric
    dnl_max = mismatch_metric * 0.6 + norm_moment * 0.2
    inl_max = mismatch_metric * 0.4 + edge_metric * 0.25
    return dnl_max, inl_max


def compute_fitness(c_unit: float, dnl_max: float, inl_max: float, params: GAParams) -> float:
    if dnl_max >= 0.5 or inl_max >= 0.5:
        return 0.0
    c_max = params.Cmax if params.Cmax > 0 else 1.0
    denom = (
        params.alpha * (c_unit / c_max)
        + params.beta * (dnl_max / 0.5)
        + params.gamma * (inl_max / 0.5)
    )
    if denom <= 0:
        return 0.0
    return 1.0 / denom


def fitness_value(
    genome: CPGenome,
    params: GAParams,
    rows: int,
    cols: int,
    adjacency: Sequence[AdjacentPair],
) -> float:
    dnl_max, inl_max = evaluate_dnl_inl(genome, adjacency, rows, cols)
    return compute_fitness(genome.c_unit, dnl_max, inl_max, params)


def evaluate_cp_sequence(
    c_unit: float,
    routing_labels: Sequence[str],
    rows: int,
    cols: int,
    params: GAParams,
) -> Dict[str, float]:
    adjacency = build_cp_pairs(rows, cols)
    patterns = [label.strip().upper() for label in routing_labels]
    if len(patterns) != len(adjacency):
        raise ValueError("Routing pattern count does not match adjacency count.")
    for label in patterns:
        if label not in PATTERN_VALUES:
            raise ValueError(f"Unsupported routing label '{label}'.")
    c_unit = clamp_c_unit(c_unit, params)
    genome = CPGenome(c_unit=c_unit, patterns=patterns)
    dnl_max, inl_max = evaluate_dnl_inl(genome, adjacency, rows, cols)
    fitness = compute_fitness(genome.c_unit, dnl_max, inl_max, params)
    return {
        "Cunit": genome.c_unit,
        "DNLmax": dnl_max,
        "INLmax": inl_max,
        "fitness": fitness,
    }


def tournament_select(
    population: Sequence[CPGenome],
    fitnesses: Sequence[float],
    k: int,
) -> CPGenome:
    if not population:
        raise ValueError("Population is empty.")
    k = max(1, min(k, len(population)))
    indices = random.sample(range(len(population)), k)
    best_index = max(indices, key=lambda idx: fitnesses[idx])
    return population[best_index].clone()


def one_point_crossover(a: CPGenome, b: CPGenome) -> Tuple[CPGenome, CPGenome]:
    n = len(a.patterns)
    if n < 2 or n != len(b.patterns):
        return a.clone(), b.clone()
    point = random.randint(1, n - 1)
    child1_patterns = a.patterns[:point] + b.patterns[point:]
    child2_patterns = b.patterns[:point] + a.patterns[point:]
    return (
        CPGenome(c_unit=a.c_unit, patterns=child1_patterns),
        CPGenome(c_unit=b.c_unit, patterns=child2_patterns),
    )


def two_point_crossover(a: CPGenome, b: CPGenome) -> Tuple[CPGenome, CPGenome]:
    n = len(a.patterns)
    if n < 3 or n != len(b.patterns):
        return a.clone(), b.clone()
    p1 = random.randint(0, n - 2)
    p2 = random.randint(p1 + 1, n - 1)
    child1_patterns = a.patterns[:p1] + b.patterns[p1:p2] + a.patterns[p2:]
    child2_patterns = b.patterns[:p1] + a.patterns[p1:p2] + b.patterns[p2:]
    return (
        CPGenome(c_unit=a.c_unit, patterns=child1_patterns),
        CPGenome(c_unit=b.c_unit, patterns=child2_patterns),
    )


def mutate(genome: CPGenome, params: GAParams) -> None:
    if not genome.patterns:
        genome.c_unit = random.uniform(params.Cmin, params.Cmax)
        return
    if random.random() < 0.5:
        genome.c_unit = clamp_c_unit(random.uniform(params.Cmin, params.Cmax), params)
        return
    count = len(genome.patterns)
    flips = max(1, int(round(0.1 * count)))
    flips = min(flips, count)
    indices = random.sample(range(count), flips)
    for idx in indices:
        genome.patterns[idx] = "O" if genome.patterns[idx] == "N" else "N"


def inversion(genome: CPGenome) -> None:
    n = len(genome.patterns)
    if n < 2:
        return
    i, j = sorted(random.sample(range(n), 2))
    genome.patterns[i : j + 1] = list(reversed(genome.patterns[i : j + 1]))


def run_ga(rows: int, cols: int, params: GAParams) -> Tuple[CPGenome, float, Dict[str, object]]:
    random.seed(params.seed)
    adjacency = build_cp_pairs(rows, cols)
    num_pairs = len(adjacency)

    population = [
        random_initial_genome(num_pairs, params) for _ in range(params.population_size)
    ]
    fitnesses = [fitness_value(g, params, rows, cols, adjacency) for g in population]
    best_idx = max(range(len(population)), key=lambda idx: fitnesses[idx])
    best = population[best_idx].clone()
    best_fit = fitnesses[best_idx]

    history: Dict[str, object] = {
        "best": [max(fitnesses)] if fitnesses else [0.0],
        "avg": [sum(fitnesses) / len(fitnesses) if fitnesses else 0.0],
    }

    for _ in range(params.generations):
        new_population: List[CPGenome] = []
        while len(new_population) < params.population_size:
            parent1 = tournament_select(population, fitnesses, params.tournament_k)
            parent2 = tournament_select(population, fitnesses, params.tournament_k)

            child1, child2 = parent1.clone(), parent2.clone()
            if random.random() < params.crossover_rate:
                if random.random() < 0.5:
                    child1, child2 = one_point_crossover(parent1, parent2)
                else:
                    child1, child2 = two_point_crossover(parent1, parent2)

            if random.random() < params.mutation_rate:
                mutate(child1, params)
            if random.random() < params.mutation_rate:
                mutate(child2, params)

            if random.random() < params.inversion_rate:
                inversion(child1)
            if random.random() < params.inversion_rate:
                inversion(child2)

            new_population.append(child1)
            if len(new_population) < params.population_size:
                new_population.append(child2)

        population = new_population
        fitnesses = [fitness_value(g, params, rows, cols, adjacency) for g in population]
        current_best_idx = max(range(len(population)), key=lambda idx: fitnesses[idx])
        current_best = fitnesses[current_best_idx]
        if current_best > best_fit:
            best_fit = current_best
            best = population[current_best_idx].clone()

        history["best"].append(current_best)
        history["avg"].append(sum(fitnesses) / len(fitnesses) if fitnesses else 0.0)

    history["global_best"] = best_fit
    history["adjacency"] = adjacency
    return best, best_fit, history


if __name__ == "__main__":
    rows, cols = 8, 8
    params = GAParams(
        population_size=80,
        generations=300,
        tournament_k=3,
        crossover_rate=0.9,
        mutation_rate=0.2,
        inversion_rate=0.2,
        alpha=100.0,
        beta=1.0,
        gamma=1.0,
        Cmin=2e-15,
        Cmax=20e-15,
        seed=7,
    )
    best, fitness, history = run_ga(rows, cols, params)
    adjacency = history["adjacency"]
    dnl_max, inl_max = evaluate_dnl_inl(best, adjacency, rows, cols)

    print("=== GA result ===")
    print(f"Best fitness: {fitness:.6f}")
    print(f"Cunit: {best.c_unit:.3e} F (range {params.Cmin:.2e} - {params.Cmax:.2e})")
    print(f"DNLmax: {dnl_max:.4f} LSB, INLmax: {inl_max:.4f} LSB")

    sequence_tokens = format_cp_sequence(best, adjacency)
    print(f"Routing entries: {len(best.patterns)}")
    preview = sequence_tokens[1 : min(len(sequence_tokens), 100)]
    print("Highlighted CP-sequence tokens:")
    for token in preview:
        print(f"  {token}")
    if len(sequence_tokens) > 100:
        remaining = len(sequence_tokens) - 100
        print(f"  ... {remaining} more routing tokens")


# Format: P@O(r1,c1)-(r2,c2)

# P: routing pattern — N = non‑overlapped wires, O = overlapped wires (more parasitic).
# O: adjacency orientation — H = horizontal neighbor, V = vertical neighbor.
# (r,c): 0‑based row/col of the two adjacent unit capacitors in the r×s array.
# Symmetry: Only one of each 180°-rotationally symmetric pair is kept to preserve common‑centroid routing; the mirrored counterpart is implied and not listed.

# Ordering: The sequence is built by sweeping all horizontals row‑by‑row, then all verticals column‑by‑column, after symmetry pruning.

# Your 4×5 example:

# Horizontal tokens cover rows 0–1: (0,0)-(0,1)…(0,3)-(0,4) and (1,0)-(1,1)…(1,3)-(1,4). Rows 2–3 are omitted as symmetric.
# Vertical tokens cover columns 0–2, top to bottom. Some bottommost pairs (e.g., (2,2)-(3,2)) are omitted because they mirror earlier top pairs (e.g., (0,2)-(1,2)). Columns 3–4 are omitted as symmetric to 1–0.
# Total here is 16 routing genes; the unlisted symmetric counterparts are implied.
# Full CP‑sequence: It starts with the unit capacitor size Cunit, followed by these routing tokens. Tweaking N/O changes local parasitics, which shifts DNL/INL; the GA evaluates fitness as 1 / [α·(Cunit/Cmax) + β·(DNLmax/0.5) + γ·(INLmax/0.5)], with ∞ penalty if DNL/INL ≥ 0.5 LSB.

# Example: O@V(1,2)-(2,2) means the vertical neighbor pair at rows 1–2, column 2 is routed with overlapped top/bottom wires (higher parasitic), and its 180°-rotated counterpart is implied, not repeated.