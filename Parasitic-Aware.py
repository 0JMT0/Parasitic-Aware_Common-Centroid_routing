#!/usr/bin/env python3
"""Parasitic-aware placement, routing, and CP-sequence demo flow."""


# """
# ===============================================================
#  Parasitic-Aware Demo Flow 使用說明
# ===============================================================
#
# 本工具整合 placement 合成/評估、上下金屬 routing、以及 CP-sequence GA 最佳化，
# 由單一 CLI 完成 parasitic-aware common-centroid 設計流程：
#   1. 版圖來源：可讀取既有標籤矩陣 (--layout)，或僅指定 --nbits 交由合成器產生。
#   2. 佈局評估：內建梯度/相關性模型，計算 mismatch、overall rho、trunk wire 指標。
#   3. Routing：分別針對 top/bottom plate 建立 MST/BFS 連線並輸出 legacy/v2 CP 序列。
#   4. GA 優化：對 N/O routing pattern 執行遺傳演算法，回報最佳 fitness 與序列。 
#   5. Plotting：啟用 --show-plots 時顯示上下 routing 及 GA pattern 版圖 (需 matplotlib)。
#
# ---------------------------------------------------------------
# 【輸入選項重點】
# --layout <txt>        : (選) 匯入既有標籤矩陣；未指定時依 --nbits 自動合成。
# --nbits <int>         : 合成模式下的 DAC 位元數；搭配 --include-b0 決定單元個數。
# --unit-cap <float>    : 單位電容大小，沿用於評估與 CP 序列輸出。
# --rho-u / --l         : 單元相關性的基礎值與距離衰減；與 placement 工具相同。
# --gx / --gy           : X/Y 方向線性梯度 (%)，影響 mismatch 計算。
# --alpha/--beta/--gamma/--neigh-th : 設定 cross terms 權重與鄰近距離門檻。
# --top-mode / --bottom-mode : 控制 routing mirror 模式 (overlapped / nonoverlapped / both / random)。
# --top-method / --bottom-method : routing 樹建立方法 (mst 或 bfs)。
# --skip-ga             : 略過遺傳演算法，只執行合成、評估與 routing。
# --show-plots          : 若安裝 matplotlib，流程結束顯示上下 routing 與 GA pattern 版圖。
# --ga-*                : 調整 GA 的族群、代數、交叉/突變/反轉率與 fitness 權重、C 範圍等。
#
# ---------------------------------------------------------------
# 【實用指令範例】
# 1) 使用既有布局並查看 routing/GA 結果：
#    python Parasitic-Aware.py --layout .\best_layout.txt --top-mode both --bottom-mode random --show-plots
#
# 2) 由 Nbits=6 自動合成並執行 GA：
#    python Parasitic-Aware.py --nbits 6 --include-b0 --ga-generations 150 --ga-population 60
#
# 3) 僅評估既有布局 (無 GA/無繪圖)：
#    python Parasitic-Aware.py --layout .\code_placement.txt --skip-ga
#
# ---------------------------------------------------------------
# 【輸出摘要】
# - 佈局評估：Mismatch (%)、Overall Correlation、Trunk Wire 長度。
# - Routing：上下 plate 的 component/segment 數、CP 序列與 pattern 統計。
# - GA：最佳 fitness、Cunit、DNL/INL 最大值、序列 preview；若繪圖則佈局上標記 N/O 對應。
# """

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Set, Tuple

import numpy as np


MODULE_DIR = Path(__file__).resolve().parent


def _load_module(name: str, path: Path):
    """Import a standalone script as a module."""
    if not path.exists():
        raise FileNotFoundError(f"Missing dependency: {path}")
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module '{name}' from '{path}'")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    return module


CC_PLACEMENT = _load_module("ccp_module", MODULE_DIR / "Common-Centroid_Placement.py")
ROUTING_TOP = _load_module("routing_top_module", MODULE_DIR / "MST_Routing_topToCP.py")
ROUTING_BOTTOM = _load_module("routing_bottom_module", MODULE_DIR / "MST_Routing_bottom.py")
CP_GA = _load_module("cp_ga_module", MODULE_DIR / "CP-seq_GA.py")

synthesize_best_layout = CC_PLACEMENT.synthesize_best_layout
read_label_grid_from_txt = CC_PLACEMENT.read_label_grid_from_txt
GAParams = CP_GA.GAParams
RUN_GA = CP_GA.run_ga
EVALUATE_DNL_INL = CP_GA.evaluate_dnl_inl
FORMAT_CP_SEQUENCE = CP_GA.format_cp_sequence


def load_layout_from_file(path: Path) -> np.ndarray:
    grid = read_label_grid_from_txt(str(path))
    return np.array(grid, dtype=int)


def evaluate_layout(
    matrix: np.ndarray,
    unit_cap: float,
    rho_u: float,
    l_value: float,
    gradient_x: float,
    gradient_y: float,
    alpha: float,
    beta: float,
    gamma: float,
    neigh_thresh: float,
) -> Dict[str, float]:
    rows, cols = matrix.shape
    sanitized = np.where(matrix < 0, 0, matrix)
    positives = sanitized[sanitized > 0]
    if positives.size == 0:
        raise ValueError("Layout matrix does not contain any capacitor bits above zero.")
    nbits = int(positives.max())
    include_dummy = bool(np.any(matrix == 0))

    optimizer = CC_PLACEMENT.DACPlacementOptimizer(
        N_bits=nbits,
        max_row=rows,
        max_col=cols,
        rho_u=rho_u,
        l=l_value,
        gradient_x=gradient_x,
        gradient_y=gradient_y,
        unit_cap_size=unit_cap,
        include_dummy_b0=include_dummy,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        neigh_thresh=neigh_thresh,
    )

    placement_map = np.full((rows, cols), None, dtype=object)
    for r in range(rows):
        for c in range(cols):
            bit = int(matrix[r, c])
            if bit < 0:
                placement_map[r, c] = None
            else:
                placement_map[r, c] = CC_PLACEMENT.UnitCapacitor(bit, unit_cap)

    mismatch = optimizer.calculate_mismatch(placement_map)
    rho = optimizer.calculate_correlation(placement_map)
    wire = optimizer.calculate_trunk_wire_metric(placement_map)
    return {
        "nbits": float(nbits),
        "mismatch": float(mismatch),
        "correlation": float(rho),
        "trunk_wire": float(wire),
    }


def run_plate_routing(
    module,
    matrix: np.ndarray,
    method: str,
    mode: str,
    unit_cap: float,
    plate_label: str,
) -> Dict[str, Any]:
    routes = module.route_layout(matrix, method=method, base_mode=mode)
    legacy_sequence = module.build_cp_sequence(matrix, routes)
    edge_modes = module.collect_edge_modes(matrix, routes)
    cp_entries = module.enumerate_cp_pattern_entries(matrix, edge_modes)
    cp_sequence = module.build_cp_sequence_v2(unit_cap, cp_entries)
    histogram = module.pattern_histogram(cp_entries)
    segment_count = sum(len(route.paths) for route in routes)
    unique_components = {route.component_id for route in routes if route.component_id >= 0}
    return {
        "plate": plate_label,
        "routes": routes,
        "segments": segment_count,
        "components": len(unique_components),
        "legacy_sequence": legacy_sequence,
        "cp_sequence": cp_sequence,
        "histogram": dict(sorted(histogram.items())),
    }


def _preview_sequence(sequence, limit: int = 6) -> str:
    if not sequence:
        return "(empty)"
    items = [str(token) for token in sequence[:limit]]
    if len(sequence) > limit:
        items.append(f"... +{len(sequence) - limit} more")
    return ", ".join(items)


def make_ga_params(args) -> GAParams:
    seed = args.ga_seed if args.ga_seed is not None else args.seed
    return GAParams(
        population_size=args.ga_population,
        generations=args.ga_generations,
        tournament_k=args.ga_tournament,
        crossover_rate=args.ga_crossover,
        mutation_rate=args.ga_mutation,
        inversion_rate=args.ga_inversion,
        alpha=args.ga_alpha,
        beta=args.ga_beta,
        gamma=args.ga_gamma,
        Cmin=args.ga_cmin,
        Cmax=args.ga_cmax,
        seed=seed,
    )


def run_cp_ga(matrix: np.ndarray, params: GAParams) -> Dict[str, Any]:
    rows, cols = matrix.shape
    best, fitness, history = RUN_GA(rows, cols, params)
    adjacency = history.get("adjacency")
    if adjacency is None:
        raise RuntimeError("GA history is missing adjacency information.")
    dnl_max, inl_max = EVALUATE_DNL_INL(best, adjacency, rows, cols)
    sequence_tokens = FORMAT_CP_SEQUENCE(best, adjacency)
    preview = sequence_tokens[: min(len(sequence_tokens), 12)]
    return {
        "fitness": float(fitness),
        "c_unit": float(best.c_unit),
        "dnl_max": float(dnl_max),
        "inl_max": float(inl_max),
        "token_count": len(sequence_tokens),
        "preview": preview,
        "sequence": sequence_tokens,
        "history": {
            "best": history.get("best", []),
            "avg": history.get("avg", []),
        },
        "genome": best,
        "adjacency": adjacency,
    }

def display_routing_plots(matrix: np.ndarray, top_summary: Dict[str, Any], bottom_summary: Dict[str, Any], args) -> bool:
    try:
        ROUTING_TOP.require_matplotlib()
    except RuntimeError as exc:
        print(f'[WARN] Routing plots skipped: {exc}')
        return False
    except Exception as exc:
        print(f'[WARN] Routing plots unavailable: {exc}')
        return False

    def _plot(module, routes, title: str, plate_filter: Optional[str] = None) -> None:
        if not routes:
            return
        kwargs: Dict[str, Any] = {
            'matrix': matrix,
            'routes': routes,
            'title': title,
        }
        if plate_filter is not None:
            try:
                module.plot_routed_layout(**kwargs, plate_filter=plate_filter)
                return
            except TypeError:
                pass
        module.plot_routed_layout(**kwargs)

    top_title = f'Top Plate Routing via {args.top_method.upper()} (mode={args.top_mode})'
    _plot(ROUTING_TOP, top_summary.get('routes'), top_title, plate_filter='top')

    bottom_title = f'Bottom Plate Routing via {args.bottom_method.upper()} (mode={args.bottom_mode})'
    _plot(ROUTING_BOTTOM, bottom_summary.get('routes'), bottom_title, plate_filter='bottom')
    return True


def plot_ga_sequence(matrix: np.ndarray, ga_result: Dict[str, Any], args) -> bool:
    genome = ga_result.get('genome')
    adjacency = ga_result.get('adjacency')
    if genome is None or adjacency is None:
        return False

    try:
        ROUTING_TOP.require_matplotlib()
    except RuntimeError as exc:
        print(f'[WARN] GA plot skipped: {exc}')
        return False
    except Exception as exc:
        print(f'[WARN] GA plot unavailable: {exc}')
        return False

    try:
        from matplotlib import lines as mlines  # type: ignore
    except Exception as exc:
        print(f'[WARN] GA plot unavailable: {exc}')
        return False

    plt = ROUTING_TOP.plt
    mcolors = ROUTING_TOP.mcolors
    if plt is None or mcolors is None:
        print('[WARN] GA plot unavailable: matplotlib backend missing.')
        return False

    rows, cols = matrix.shape
    unique_components = sorted(np.unique(matrix))
    cmap = plt.colormaps.get_cmap('Paired')
    colors = cmap(np.linspace(0, 1, len(unique_components)))
    color_map = {comp: colors[idx] for idx, comp in enumerate(unique_components)}
    if -1 in color_map:
        color_map[-1] = mcolors.to_rgba('lightgrey')

    colored_matrix = np.zeros(matrix.shape + (4,))
    for comp_id in unique_components:
        if comp_id in color_map:
            colored_matrix[matrix == comp_id] = color_map[comp_id]

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(colored_matrix, interpolation='nearest')

    for r in range(rows):
        for c in range(cols):
            comp_id = int(matrix[r, c])
            if comp_id in color_map:
                text_color = 'white' if sum(color_map[comp_id][:3]) < 1.5 else 'black'
                ax.text(c, r, str(comp_id), ha='center', va='center', color=text_color, fontsize=10)

    pattern_styles = {
        'O': {'color': mcolors.to_rgba('crimson'), 'width': 2.5, 'linestyle': '-'},
        'N': {'color': mcolors.to_rgba('seagreen'), 'width': 2.0, 'linestyle': '--'},
    }

    visited_edges: Set[Tuple[Tuple[int, int], Tuple[int, int]]] = set()
    for pattern, pair in zip(genome.patterns, adjacency):
        segments = [(pair.a, pair.b)]
        mirror_a = CP_GA._mirror(pair.a, rows, cols)
        mirror_b = CP_GA._mirror(pair.b, rows, cols)
        mirror_segment = tuple(sorted((mirror_a, mirror_b)))
        original_segment = tuple(sorted((pair.a, pair.b)))
        if mirror_segment != original_segment:
            segments.append((mirror_a, mirror_b))

        style = pattern_styles.get(pattern, pattern_styles['N'])
        for start, end in segments:
            key = tuple(sorted((start, end)))
            if key in visited_edges:
                continue
            visited_edges.add(key)
            xs = [start[1], end[1]]
            ys = [start[0], end[0]]
            ax.plot(
                xs,
                ys,
                color=style['color'],
                linewidth=style['width'],
                linestyle=style['linestyle'],
                alpha=0.85,
            )

    ax.set_title(
        f"CP-Sequence GA Patterns (Cunit={ga_result['c_unit']:.3e} F)\n"
        f"DNLmax={ga_result['dnl_max']:.4f} LSB, INLmax={ga_result['inl_max']:.4f} LSB"
    , fontsize=14)
    ax.set_xticks(np.arange(-0.5, cols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, rows, 1), minor=True)
    ax.grid(which='minor', color='black', linestyle='-', linewidth=1)
    ax.tick_params(which='minor', size=0)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(-0.5, cols - 0.5)
    ax.set_ylim(rows - 0.5, -0.5)

    legend_handles = [
        mlines.Line2D([], [], color=pattern_styles['N']['color'], linestyle=pattern_styles['N']['linestyle'], linewidth=pattern_styles['N']['width'], label='N: non-overlapped'),
        mlines.Line2D([], [], color=pattern_styles['O']['color'], linestyle=pattern_styles['O']['linestyle'], linewidth=pattern_styles['O']['width'], label='O: overlapped'),
    ]
    ax.legend(handles=legend_handles, loc='upper right')

    plt.tight_layout()
    plt.show()
    return True


def prepare_layout(args) -> Tuple[np.ndarray, Dict[str, Any]]:
    if args.layout is not None:
        matrix = load_layout_from_file(args.layout)
        return matrix, {
            "origin": "file",
            "path": str(args.layout.resolve()),
        }

    optimizer, best_map, metrics, dims = synthesize_best_layout(
        N_bits=args.nbits,
        rho_u=args.rho_u,
        l=args.l,
        gx=args.gx,
        gy=args.gy,
        include_dummy_b0=args.include_b0,
        alpha=args.alpha,
        beta=args.beta,
        gamma=args.gamma,
        neigh_thresh=args.neigh_th,
        restarts=args.synth_restarts,
        sa_iters=args.synth_iters,
        seed=args.seed,
    )
    matrix = np.array(optimizer.to_label_grid(best_map), dtype=int)
    return matrix, {
        "origin": "synthesized",
        "nbits": args.nbits,
        "grid": tuple(int(x) for x in dims),
        "include_b0": bool(args.include_b0),
        "anneal_metrics": {
            "mismatch": float(metrics[0]),
            "correlation": float(metrics[1]),
            "trunk_wire": float(metrics[2]),
        },
    }


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run an end-to-end parasitic-aware placement and routing demonstration.",
    )
    parser.add_argument("--layout", type=Path, help="Existing layout txt (whitespace separated integers).")
    parser.add_argument("--nbits", type=int, default=6, help="Bits for synthesized placement when --layout is absent.")
    parser.add_argument("--unit-cap", type=float, default=1.0, help="Unit capacitor size used for evaluation and CP sequences.")
    parser.add_argument("--include-b0", dest="include_b0", action="store_true", help="Include dummy B0 during synthesis (default).")
    parser.add_argument("--no-b0", dest="include_b0", action="store_false", help="Exclude dummy B0 during synthesis.")
    parser.set_defaults(include_b0=True)
    parser.add_argument("--rho-u", type=float, default=0.9, help="Uncorrelated unit rho parameter.")
    parser.add_argument("--l", type=float, default=0.5, help="Correlation decay length parameter.")
    parser.add_argument("--gx", type=float, default=0.005, help="Gradient along x used for evaluation.")
    parser.add_argument("--gy", type=float, default=0.002, help="Gradient along y used for evaluation.")
    parser.add_argument("--alpha", type=float, default=0.2, help="Evaluator intra(Ci) weight.")
    parser.add_argument("--beta", type=float, default=0.2, help="Evaluator intra(Cj) weight.")
    parser.add_argument("--gamma", type=float, default=0.1, help="Evaluator neighbor cross weight.")
    parser.add_argument("--neigh-th", type=float, default=1.5, help="Evaluator neighbor distance threshold.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed for synthesis and as GA fallback.")
    parser.add_argument("--synth-restarts", type=int, default=3, help="Simulated annealing restarts for synthesis.")
    parser.add_argument("--synth-iters", type=int, default=6000, help="Simulated annealing iterations per restart.")
    parser.add_argument("--top-mode", choices=["overlapped", "nonoverlapped", "both", "random"], default="random", help="Routing mode for top plate pairs.")
    parser.add_argument("--bottom-mode", choices=["overlapped", "nonoverlapped", "both", "random"], default="random", help="Routing mode for bottom plate pairs.")
    parser.add_argument("--top-method", choices=["mst", "bfs"], default="mst", help="Routing tree builder for the top plate.")
    parser.add_argument("--bottom-method", choices=["mst", "bfs"], default="mst", help="Routing tree builder for the bottom plate.")
    parser.add_argument("--skip-ga", action="store_true", help="Skip the CP-sequence genetic algorithm stage.")
    parser.add_argument("--show-plots", action="store_true", help="Display routing plots at the end of the flow.")
    parser.add_argument("--ga-population", type=int, default=40, help="GA population size.")
    parser.add_argument("--ga-generations", type=int, default=80, help="GA generations.")
    parser.add_argument("--ga-tournament", type=int, default=3, help="GA tournament selection size.")
    parser.add_argument("--ga-crossover", type=float, default=0.9, help="GA crossover probability.")
    parser.add_argument("--ga-mutation", type=float, default=0.15, help="GA mutation probability.")
    parser.add_argument("--ga-inversion", type=float, default=0.15, help="GA inversion probability.")
    parser.add_argument("--ga-alpha", type=float, default=100.0, help="GA fitness alpha weight.")
    parser.add_argument("--ga-beta", type=float, default=1.0, help="GA fitness beta weight.")
    parser.add_argument("--ga-gamma", type=float, default=1.0, help="GA fitness gamma weight.")
    parser.add_argument("--ga-cmin", type=float, default=2e-15, help="Minimum unit capacitance for GA search.")
    parser.add_argument("--ga-cmax", type=float, default=20e-15, help="Maximum unit capacitance for GA search.")
    parser.add_argument("--ga-seed", type=int, help="Override GA random seed.")
    return parser.parse_args(argv)


def print_routing_summary(label: str, summary: Dict[str, Any]) -> None:
    print(f"\n--- {label.capitalize()} plate routing ---")
    print(f"Components    : {summary['components']}")
    print(f"Segments      : {summary['segments']}")
    print(f"Legacy tokens : {len(summary['legacy_sequence'])}")
    print(f"CP sequence   : {_preview_sequence(summary['cp_sequence'])}")
    histogram = summary['histogram']
    if histogram:
        formatted = ", ".join(f"{key}:{value}" for key, value in histogram.items())
        print(f"Pattern hist. : {formatted}")
    else:
        print("Pattern hist. : (empty)")


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    matrix, meta = prepare_layout(args)
    rows, cols = matrix.shape

    print("=== Parasitic-aware demo flow ===")
    if meta["origin"] == "file":
        print(f"Layout source : file -> {meta['path']}")
    else:
        grid_r, grid_c = meta["grid"]
        metrics = meta["anneal_metrics"]
        print(
            "Layout source : synthesized"
            f" (nbits={meta['nbits']}, grid={grid_r}x{grid_c}, include_b0={meta['include_b0']})"
        )
        print(
            "  SA metrics   : mismatch={metrics['mismatch']:.4f}% "
            f"rho={metrics['correlation']:.4f} wire={metrics['trunk_wire']:.4f}"
        )
    print(f"Array shape   : {rows} x {cols}")

    placement_metrics = evaluate_layout(
        matrix,
        unit_cap=args.unit_cap,
        rho_u=args.rho_u,
        l_value=args.l,
        gradient_x=args.gx,
        gradient_y=args.gy,
        alpha=args.alpha,
        beta=args.beta,
        gamma=args.gamma,
        neigh_thresh=args.neigh_th,
    )

    print("\n--- Placement evaluation ---")
    print(f"Mismatch (%)  : {placement_metrics['mismatch']:.6f}")
    print(f"Correlation   : {placement_metrics['correlation']:.6f}")
    print(f"Trunk wire    : {placement_metrics['trunk_wire']:.6f}")
    print(f"Inferred bits : {int(placement_metrics['nbits'])}")

    top_summary = run_plate_routing(
        ROUTING_TOP,
        matrix,
        method=args.top_method,
        mode=args.top_mode,
        unit_cap=args.unit_cap,
        plate_label="top",
    )
    print_routing_summary("top", top_summary)

    bottom_summary = run_plate_routing(
        ROUTING_BOTTOM,
        matrix,
        method=args.bottom_method,
        mode=args.bottom_mode,
        unit_cap=args.unit_cap,
        plate_label="bottom",
    )
    print_routing_summary("bottom", bottom_summary)

    ga_result: Optional[Dict[str, Any]] = None
    if args.skip_ga:
        print("\nCP-sequence GA skipped (per flag).")
    else:
        ga_params = make_ga_params(args)
        ga_result = run_cp_ga(matrix, ga_params)

        print("\n--- CP-sequence GA ---")
        print(f"Population    : {ga_params.population_size}, generations: {ga_params.generations}")
        print(f"Best fitness  : {ga_result['fitness']:.6f}")
        print(f"C unit (F)    : {ga_result['c_unit']:.3e}")
        print(f"DNLmax / INLmax (LSB): {ga_result['dnl_max']:.4f} / {ga_result['inl_max']:.4f}")
        print(f"Tokens        : {ga_result['token_count']}")
        print(f"Sequence prev.: {_preview_sequence(ga_result['preview'], limit=8)}")

    plots_ready = False
    if args.show_plots:
        plots_ready = display_routing_plots(matrix, top_summary, bottom_summary, args)
        if plots_ready and not args.skip_ga and ga_result is not None:
            plot_ga_sequence(matrix, ga_result, args)


if __name__ == "__main__":
    main()


