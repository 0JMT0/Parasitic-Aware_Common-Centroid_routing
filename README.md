# Parasitic-Aware Common-Centroid Toolkit

This repository bundles placement synthesis/evaluation utilities, routing visualizers, and a CP-sequence genetic algorithm for common-centroid capacitor arrays. Each script can run standalone, or you can drive the entire flow through `Parasitic-Aware.py`.

## Repository Map

| File | Description |
| --- | --- |
| `Parasitic-Aware.py` | End-to-end demo that combines placement synthesis/evaluation, top/bottom routing, and the GA optimizer. |
| `Common-Centroid_Placement.py` | Placement synthesizer/evaluator with CLI subcommands `synthesize` and `evaluate`. |
| `Placement_Evaluator.py` | Lightweight evaluator for an existing label grid (metrics only). |
| `MST_Routing_topToCP.py` | TOP-plate routing visualizer and CP-sequence generator (MST/BFS). |
| `MST_Routing_bottom.py` | BOTTOM-plate routing visualizer (MST/BFS). |
| `CP-seq_GA.py` | Genetic algorithm that optimizes N/O routing patterns. |
| `test1.py` | Scratchpad script for experimentation. |
| `best_layout.txt`, `code_placement.txt` | Sample placement label grids. |
| `backup/CCP_backup.py` | Archived placement logic snapshot. |

## Requirements

- Python 3.10+ (tested with 3.11/3.12/3.13)
- `numpy`
- (Optional) `matplotlib` for plotting

Install the dependencies:

```bash
python -m pip install numpy matplotlib
```

## Quick Start

### 1. Evaluate an existing layout

```bash
python Common-Centroid_Placement.py evaluate ./code_placement.txt --print-grid --alpha 0.2 --beta 0.2 --gamma 0.1 --neigh-th 1.5
```

### 2. Synthesize a placement from scratch

```bash
python Common-Centroid_Placement.py synthesize --nbits 6 --include-b0 --show --restarts 4 --sa-iters 8000 --w-sym 5.0 --out best_layout.txt
```

### 3. Route a layout (top plate example)

```bash
python MST_Routing_topToCP.py --layout-file best_layout.txt --method mst --mode random
```

### 4. Run the CP-sequence GA standalone

```bash
python CP-seq_GA.py
```

### 5. End-to-end demo

```bash
python Parasitic-Aware.py --layout best_layout.txt --top-mode both --bottom-mode random --show-plots
```

(omit `--layout` to let the demo synthesize a layout using `--nbits`.)

## Script Reference

### `Parasitic-Aware.py`

End-to-end orchestrator. Highlights:

- `--layout <txt>`: load an existing grid; otherwise synthesis runs.
- `--nbits <int>`: DAC resolution during synthesis.
- `--include-b0` / `--no-b0`: toggle the dummy capacitor.
- `--unit-cap`, `--rho-u`, `--l`, `--gx`, `--gy`, `--alpha`, `--beta`, `--gamma`, `--neigh-th`: placement evaluation knobs.
- `--top-mode` / `--bottom-mode`: routing pair mode (`overlapped`, `nonoverlapped`, `both`, `random`).
- `--top-method` / `--bottom-method`: spanning-tree builder (`mst`, `bfs`).
- `--skip-ga`: bypass GA stage.
- `--show-plots`: display top/bottom routing and GA pattern plots (requires matplotlib).
- `--ga-*`: tune GA population, generations, tournament size, crossover/mutation/inversion probabilities, fitness weights (`alpha`, `beta`, `gamma`), capacitance bounds (`cmin`, `cmax`), and seeds.

Output includes placement metrics, routing summaries, CP-sequence tokens, and GA statistics. When plotting is enabled three figures are shown sequentially (top routing, bottom routing, GA pattern overlay).

### `Common-Centroid_Placement.py`

Two subcommands:

- `evaluate <txt>`: report mismatch, overall correlation, and trunk wire metrics for a provided grid.
- `synthesize`: simulated annealing search that returns the best placement map plus metrics.

Shared flags cover gradient parameters, correlation weights, and neighborhood thresholds. Synthesis also exposes `--restarts`, `--sa-iters`, and cost weights (`--w-m`, `--w-rho`, `--w-wire`, `--w-sym`).

### `Placement_Evaluator.py`

Standalone evaluator mirroring the `evaluate` subcommand. Useful when you only need metrics without running the full toolkit.

### `MST_Routing_topToCP.py`

Routes top-plate nets grouped by capacitor ID.

- `--layout-file <txt>`: input grid (defaults to built-in sample).
- `--method {mst,bfs}`: spanning-tree algorithm per component.
- `--mode {overlapped,nonoverlapped,both,random}`: how mirrored pairs are handled.
- `--unit-cap <float>`: printed in the CP sequence output.
- `--no-plot`: disable plotting (default renders with matplotlib).

CLI emits legacy CP sequences, v2 tokens, and pattern histograms; plotting overlays routes on the layout.

### `MST_Routing_bottom.py`

Bottom-plate counterpart to the top router. Interface is identical but focuses on bottom wiring.

### `CP-seq_GA.py`

Implements the GA over N/O adjacency patterns.

- `GAParams`: configuration dataclass.
- `run_ga(rows, cols, params)`: main GA entry point.
- Utilities for formatting and evaluating sequences.

Running the script directly executes a sample GA on an 8×8 grid and prints best fitness/sequence information.

### `test1.py`

Exploratory script that demonstrates usage combinations of the modules.

## Plotting Tips

- Matplotlib is optional. Without it, plotting flags gracefully degrade.
- GUI windows block execution; close each figure to proceed.
- On headless systems, skip plotting (`--no-plot`, omit `--show-plots`) or set a non-interactive backend (e.g., `MPLBACKEND=Agg`).

## Data Files

- `best_layout.txt`: Sample synthesized layout.
- `code_placement.txt`: Layout matching the reference publication.

## Suggested Workflow

1. **Synthesize** a candidate layout (optional).
2. **Evaluate** mismatch/correlation/wire metrics.
3. **Route** top and bottom plates; inspect CP patterns.
4. **Optimize** parasitics with the GA.
5. **Visualize** final routing/parasitic patterns via `--show-plots`.

## Troubleshooting

- Missing `matplotlib` → install it or omit plot options.
- GA fitness of `0.0` indicates DNL/INL exceeded limits; adjust GA weights or allow more generations.
- Ensure layout text files are rectangular and contain integer labels `0..N` (negative values are treated as empty).

## License / Attribution

Refer to project documentation or contact the original authors for licensing details. The `backup/` directory preserves earlier placement logic for reference.
