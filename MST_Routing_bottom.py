# -*- coding: utf-8 -*-
"""Utility to visualize and route common-centroid layouts using MST wiring (BOTTOM ONLY)."""

from __future__ import annotations

import argparse
import math
import heapq
import random
from collections import deque, defaultdict, Counter
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Set, Tuple, Union

import numpy as np

# Guard plotting imports so routing still works without matplotlib installed
try:  # pragma: no cover
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
except ModuleNotFoundError:  # pragma: no cover - handled gracefully at runtime
    plt = None  # type: ignore
    mcolors = None  # type: ignore

Coordinate = Tuple[int, int]
Group = List[Coordinate]
Shape = Tuple[int, int]


# =========================
# Dataclasses
# =========================
@dataclass
class RoutePath:
    """Represents a single routed segment as a sequence of coordinates."""
    start: Coordinate
    end: Coordinate
    points: List[Coordinate]


@dataclass
class RouteResult:
    """Holds routing data and a CP sequence for a single component ID."""
    component_id: int
    method: str
    mode: str
    plate: str
    paths: List[RoutePath]

    @property
    def cp_sequence(self) -> List[str]:
        entries: List[str] = []
        for idx, segment in enumerate(self.paths, start=1):
            entries.append(
                f"CID={self.component_id} step={idx} method={self.method.upper()} "
                f"mode={self.mode.upper()} plate={self.plate} path={segment.points}"
            )
        return entries


@dataclass
class ComponentInfo:
    """Metadata about a connected component of identical unit capacitors."""
    comp_id: int
    coords: List[Coordinate]
    key: Tuple[Coordinate, ...]
    mirror_key: Tuple[Coordinate, ...]
    centroid: Tuple[float, float]
    is_center: bool = False


@dataclass
class ComponentPair:
    """Pair of symmetric connected components (mirror may be None)."""
    primary: ComponentInfo
    mirror: Optional[ComponentInfo]


@dataclass
class ComponentBridge:
    """Single-wire connection between two connected components."""
    path: List[Coordinate]
    source: ComponentInfo
    target: ComponentInfo


# =========================
# Parsing helpers
# =========================
def parse_layout_text(text: str) -> np.ndarray:
    """Parse a layout matrix from whitespace-separated text."""
    rows: List[List[int]] = []
    for raw_line in text.strip().splitlines():
        line = raw_line.strip()
        if not line:
            continue
        row = [int(token) for token in line.split()]
        rows.append(row)

    if not rows:
        raise ValueError("Layout text must contain at least one row of integers.")

    width = len(rows[0])
    for row in rows:
        if len(row) != width:
            raise ValueError("All rows in the layout text must have the same length.")

    return np.array(rows, dtype=int)


def component_positions(matrix: np.ndarray) -> Dict[int, List[Coordinate]]:
    """Return coordinates for each non-negative component ID."""
    positions: Dict[int, List[Coordinate]] = defaultdict(list)
    for r in range(matrix.shape[0]):
        for c in range(matrix.shape[1]):
            comp_id = int(matrix[r, c])
            if comp_id >= 0:
                positions[comp_id].append((r, c))
    return dict(sorted(positions.items()))


def unit_cell_sizes(matrix: np.ndarray) -> Dict[int, int]:
    """Return the population of each component ID within the layout."""
    return {comp_id: len(points) for comp_id, points in component_positions(matrix).items()}


def build_cp_sequence(matrix: np.ndarray, routes: List[RouteResult]) -> List[Union[Dict[int, int], str]]:
    """Concatenate unit sizes and per-segment CP entries into a single list."""
    sequence: List[Union[Dict[int, int], str]] = [unit_cell_sizes(matrix)]
    for route in routes:
        sequence.extend(route.cp_sequence)
    return sequence


# =========================
# ⭐ Routing pattern & CP v2 (保留；僅 bottom 時通常不產生 O/N)
# =========================
def route_mode_to_label(route_mode: str) -> str:
    mode_lower = route_mode.lower()
    if 'single' in mode_lower:
        return 'S'
    if 'nonoverlapped' in mode_lower:
        return 'N'
    if 'overlapped' in mode_lower:
        return 'O'
    return '?'


def collect_edge_modes(
    matrix: np.ndarray,
    routes: List[RouteResult],
) -> Dict[Tuple[Coordinate, Coordinate], Dict[str, Set[str]]]:
    """Record routing labels for every grid edge per plate."""
    rows, cols = matrix.shape
    edge_modes: Dict[Tuple[Coordinate, Coordinate], Dict[str, Set[str]]] = {}

    for route in routes:
        label = route_mode_to_label(route.mode)
        if label == '?':
            continue
        plate_key = route.plate.lower()
        for segment in route.paths:
            for start, end in zip(segment.points, segment.points[1:]):
                if not (0 <= start[0] < rows and 0 <= start[1] < cols):
                    continue
                if not (0 <= end[0] < rows and 0 <= end[1] < cols):
                    continue
                if abs(start[0] - end[0]) + abs(start[1] - end[1]) != 1:
                    continue
                edge = normalize_edge(start, end)
                per_plate = edge_modes.setdefault(edge, {})
                per_plate.setdefault(plate_key, set()).add(label)

    return edge_modes


def mirror_coordinate(coord: Coordinate, rows: int, cols: int) -> Coordinate:
    r, c = coord
    return rows - 1 - r, cols - 1 - c


def edge_pattern(edge_data: Dict[str, Set[str]]) -> Optional[str]:
    """Return O/N when both plates contribute; bottom-only 時多半回傳 None。"""
    top_labels = edge_data.get("top")
    bottom_labels = edge_data.get("bottom")
    if not top_labels or not bottom_labels:
        return None
    if "O" in top_labels or "O" in bottom_labels:
        return "O"
    if "N" in top_labels or "N" in bottom_labels:
        return "N"
    return None


def enumerate_cp_pattern_entries(
    matrix: np.ndarray,
    edge_modes: Dict[Tuple[Coordinate, Coordinate], Dict[str, Set[str]]],
) -> List[Tuple[str, Tuple[Coordinate, Coordinate]]]:
    rows, cols = matrix.shape
    entries: List[Tuple[str, Tuple[Coordinate, Coordinate]]] = []

    def consider_edge(a: Coordinate, b: Coordinate) -> None:
        edge = normalize_edge(a, b)
        mirror_edge_ = normalize_edge(
            mirror_coordinate(a, rows, cols),
            mirror_coordinate(b, rows, cols),
        )
        if edge not in edge_modes:
            return
        if edge > mirror_edge_:  # avoid double-count mirrored duplicates
            return
        pattern = edge_pattern(edge_modes[edge])
        if pattern:
            entries.append((pattern, edge))

    for r in range(rows):
        for c in range(cols - 1):
            consider_edge((r, c), (r, c + 1))
    for r in range(rows - 1):
        for c in range(cols):
            consider_edge((r, c), (r + 1, c))

    return entries


def build_cp_sequence_v2(
    unit_cap_value: float,
    entries: List[Tuple[str, Tuple[Coordinate, Coordinate]]],
) -> List[str]:
    sequence: List[str] = [f"Cu={unit_cap_value:g}"]
    sequence.extend(pattern for pattern, _ in entries)
    return sequence


def pattern_histogram(entries: List[Tuple[str, Tuple[Coordinate, Coordinate]]]) -> Counter:
    return Counter(pattern for pattern, _ in entries)


# =========================
# Component discovery & pairing
# =========================
def find_component_infos(
    matrix: np.ndarray,
) -> Dict[Tuple[int, Tuple[Coordinate, ...]], ComponentInfo]:
    """
    Discover all connected components for each capacitor ID.
    ⭐ 中心 4 格會被視為一個特殊 component (ID = -2)
    """
    rows, cols = matrix.shape
    center_rows = {rows // 2} if rows % 2 else {rows // 2 - 1, rows // 2}
    center_cols = {cols // 2} if cols % 2 else {cols // 2 - 1, cols // 2}
    center_cells = {(r, c) for r in center_rows for c in center_cols}

    visited = np.zeros(matrix.shape, dtype=bool)
    info_map: Dict[Tuple[int, Tuple[Coordinate, ...]], ComponentInfo] = {}

    # 先標記中心為已訪問
    for r, c in center_cells:
        visited[r, c] = True

    # 非中心元件
    for r in range(rows):
        for c in range(cols):
            coord = (r, c)
            if coord in center_cells:
                continue

            comp_id = int(matrix[r, c])
            if comp_id < 0 or visited[r, c]:
                continue

            queue: deque[Coordinate] = deque([(r, c)])
            visited[r, c] = True
            coords: List[Coordinate] = []

            while queue:
                y, x = queue.popleft()
                coords.append((y, x))
                for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    ny, nx = y + dy, x + dx
                    nbr = (ny, nx)
                    if (
                        0 <= ny < rows
                        and 0 <= nx < cols
                        and not visited[ny, nx]
                        and nbr not in center_cells
                        and int(matrix[ny, nx]) == comp_id
                    ):
                        visited[ny, nx] = True
                        queue.append(nbr)

            if not coords:
                continue

            key = tuple(sorted(coords))
            mirror_key = tuple(sorted(mirror_coordinate(p, rows, cols) for p in coords))
            centroid = (
                sum(pt[0] for pt in coords) / len(coords),
                sum(pt[1] for pt in coords) / len(coords),
            )
            info_map[(comp_id, key)] = ComponentInfo(
                comp_id=comp_id,
                coords=coords,
                key=key,
                mirror_key=mirror_key,
                centroid=centroid,
            )

    # 中心元件（自己的鏡像）
    if center_cells:
        center_coords = sorted(list(center_cells))
        center_key = tuple(center_coords)
        center_centroid = (
            sum(pt[0] for pt in center_coords) / len(center_coords),
            sum(pt[1] for pt in center_coords) / len(center_coords),
        )
        info_map[(-2, center_key)] = ComponentInfo(
            comp_id=-2,
            coords=center_coords,
            key=center_key,
            mirror_key=center_key,
            centroid=center_centroid,
            is_center=True,
        )

    return info_map


def create_component_pairs(
    component_infos: Dict[Tuple[int, Tuple[Coordinate, ...]], ComponentInfo]
) -> List[ComponentPair]:
    """Group components with their mirrored counterparts (if any)."""
    pairs: List[ComponentPair] = []
    processed: Set[Tuple[int, Tuple[Coordinate, ...]]] = set()

    for (comp_id, key), info in sorted(
        component_infos.items(), key=lambda item: (item[0][0], item[0][1])
    ):
        identifier = (comp_id, key)
        if identifier in processed:
            continue

        if info.is_center:
            pairs.append(ComponentPair(primary=info, mirror=None))
            processed.add(identifier)
            continue

        mirror_identifier = (comp_id, info.mirror_key)
        mirror_info = component_infos.get(mirror_identifier)

        if mirror_info and mirror_info.key != info.key:
            pairs.append(ComponentPair(primary=info, mirror=mirror_info))
            processed.add(identifier)
            processed.add(mirror_identifier)
        else:
            pairs.append(ComponentPair(primary=info, mirror=None))
            processed.add(identifier)

    return pairs


# =========================
# Graph helpers
# =========================
def component_spanning_tree(coords: List[Coordinate]) -> List[Tuple[Coordinate, Coordinate]]:
    """Create a rectilinear spanning tree over the provided coordinates."""
    if not coords:
        return []

    coord_set = set(coords)
    visited: Set[Coordinate] = set()
    queue: deque[Coordinate] = deque([coords[0]])
    visited.add(coords[0])
    edges: List[Tuple[Coordinate, Coordinate]] = []

    while queue:
        node = queue.popleft()
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nbr = (node[0] + dy, node[1] + dx)
            if nbr in coord_set and nbr not in visited:
                visited.add(nbr)
                queue.append(nbr)
                edges.append((node, nbr))

    if len(visited) != len(coord_set):
        raise RuntimeError("Component connectivity could not be established via adjacency.")

    return edges


def mirror_edge(edge: Tuple[Coordinate, Coordinate], rows: int, cols: int) -> Tuple[Coordinate, Coordinate]:
    start, end = edge
    return mirror_coordinate(start, rows, cols), mirror_coordinate(end, rows, cols)


def closest_component_cells(
    info_a: ComponentInfo, info_b: ComponentInfo
) -> Tuple[Coordinate, Coordinate, int]:
    best: Optional[Tuple[Coordinate, Coordinate, int]] = None
    best_dist = math.inf
    for cell_a in info_a.coords:
        for cell_b in info_b.coords:
            dist = manhattan_distance(cell_a, cell_b)
            if dist < best_dist:
                best_dist = dist
                best = (cell_a, cell_b, dist)
    if best is None:
        raise RuntimeError("Unable to find closest cells between components")
    return best


# =========================
# Low-level geometry
# =========================
def manhattan_distance(a: Coordinate, b: Coordinate) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def normalize_edge(a: Coordinate, b: Coordinate) -> Tuple[Coordinate, Coordinate]:
    return tuple(sorted((a, b)))  # type: ignore[return-value]


def plate_coordinate(points: Iterable[Coordinate], plate: str, rows: int) -> Coordinate:
    """Return pseudo-coordinate representing the top/bottom plate."""
    cols = [c for _, c in points]
    avg_col = int(round(sum(cols) / len(cols))) if cols else 0
    if plate == "top":
        return (-1, avg_col)  # kept for completeness
    if plate == "bottom":
        return (rows, avg_col)
    raise ValueError(f"Unsupported plate side: {plate}")


def neighbors(node: Coordinate, rows: int, cols: int, plate: Coordinate) -> List[Coordinate]:
    """Return valid 4-neighbour coordinates including plate link."""
    r, c = node
    results: List[Coordinate] = []
    candidates = [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]

    for nr, nc in candidates:
        if (nr, nc) == plate:
            results.append((nr, nc))
        elif 0 <= nr < rows and 0 <= nc < cols:
            results.append((nr, nc))

    # allow transitions from plate to grid
    if node == plate:
        target_row = 0 if plate[0] < 0 else rows - 1
        if 0 <= plate[1] < cols:
            results.append((target_row, plate[1]))

    return results


def bfs_path(
    start: Coordinate,
    goal: Coordinate,
    rows: int,
    cols: int,
    plate: Coordinate,
    blocked_edges: Optional[Set[Tuple[Coordinate, Coordinate]]] = None,
) -> List[Coordinate]:
    blocked_edges = blocked_edges or set()
    queue: deque[Coordinate] = deque([start])
    parent: Dict[Coordinate, Optional[Coordinate]] = {start: None}

    while queue:
        node = queue.popleft()
        if node == goal:
            break
        for nbr in neighbors(node, rows, cols, plate):
            edge = normalize_edge(node, nbr)
            if edge in blocked_edges:
                continue
            if nbr not in parent:
                parent[nbr] = node
                queue.append(nbr)

    if goal not in parent:
        raise ValueError(f"No route found between {start} and {goal} with current constraints.")

    path: List[Coordinate] = []
    node = goal
    while node is not None:
        path.append(node)
        node = parent[node]
    path.reverse()
    return path


def manhattan_path(start: Coordinate, end: Coordinate) -> List[Coordinate]:
    r0, c0 = start
    r1, c1 = end
    path: List[Coordinate] = [(r0, c0)]

    r = r0
    while r != r1:
        r += 1 if r1 > r else -1
        path.append((r, c0))

    c = c0
    while c != c1:
        c += 1 if c1 > c else -1
        path.append((r1, c))

    return path


def build_mst(nodes: List[Coordinate]) -> List[Tuple[Coordinate, Coordinate]]:
    """Return MST edges connecting nodes using Manhattan distance."""
    if not nodes:
        return []

    remaining = set(nodes)
    start = nodes[0]
    remaining.remove(start)
    visited = {start}
    edges: List[Tuple[Coordinate, Coordinate]] = []

    while remaining:
        best_edge: Optional[Tuple[Coordinate, Coordinate]] = None
        best_dist = math.inf
        for u in visited:
            for v in remaining:
                dist = manhattan_distance(u, v)
                if dist < best_dist:
                    best_dist = dist
                    best_edge = (u, v)
        if best_edge is None:
            raise RuntimeError("Failed to build MST: disconnected nodes")
        u, v = best_edge
        edges.append((u, v))
        visited.add(v)
        remaining.remove(v)

    return edges


def build_bfs_tree(
    nodes: List[Coordinate], rows: int, cols: int, plate: Coordinate
) -> List[Tuple[Coordinate, Coordinate]]:
    """Return edges from BFS rooted at plate using grid connectivity."""
    node_set = set(nodes)
    root = nodes[0]
    queue: deque[Coordinate] = deque([root])
    visited = {root}
    edges: List[Tuple[Coordinate, Coordinate]] = []

    while queue:
        node = queue.popleft()
        for nbr in neighbors(node, rows, cols, plate):
            if nbr in visited or nbr not in node_set:
                continue
            visited.add(nbr)
            queue.append(nbr)
            edges.append((node, nbr))

    remaining = [n for n in nodes if n not in visited]
    for node in remaining:
        nearest = min(visited, key=lambda ref: manhattan_distance(ref, node))
        edges.append((nearest, node))
        visited.add(node)

    return edges


def dfs_order(edges: List[Tuple[Coordinate, Coordinate]], root: Coordinate) -> List[Coordinate]:
    adjacency: Dict[Coordinate, List[Coordinate]] = defaultdict(list)
    for a, b in edges:
        adjacency[a].append(b)
        adjacency[b].append(a)

    order: List[Coordinate] = []
    stack: List[Coordinate] = [root]
    seen: set[Coordinate] = set()

    while stack:
        node = stack.pop()
        if node in seen:
            continue
        seen.add(node)
        order.append(node)
        for nbr in sorted(adjacency[node], reverse=True):
            if nbr not in seen:
                stack.append(nbr)
    return order


def _assign_pair_modes(pairs: List[ComponentPair], base_mode: str) -> List[str]:
    """Determine the routing mode for each symmetric component pair."""
    if base_mode not in {"overlapped", "nonoverlapped", "both", "random"}:
        raise ValueError(f"Unsupported routing mode selection: {base_mode}")

    if base_mode == "overlapped":
        return ["overlapped"] * len(pairs)
    if base_mode == "nonoverlapped":
        return ["nonoverlapped"] * len(pairs)
    if base_mode == "both":
        result: List[str] = []
        toggle = True
        for _ in pairs:
            result.append("overlapped" if toggle else "nonoverlapped")
            toggle = not toggle
        return result
    return [random.choice(["overlapped", "nonoverlapped"]) for _ in pairs]


def _edges_to_paths(edges: List[Tuple[Coordinate, Coordinate]]) -> List[RoutePath]:
    return [RoutePath(start=a, end=b, points=[a, b]) for a, b in edges]


# =========================
# Routing (BOTTOM ONLY)
# =========================
def route_layout(
    matrix: np.ndarray,
    method: str = "mst",
    base_mode: str = "random",
) -> List[RouteResult]:
    """Generate bottom-only routing per the detailed routing spec."""
    rows, cols = matrix.shape
    component_infos = find_component_infos(matrix)
    pairs = create_component_pairs(component_infos)
    if not pairs:
        return []

    # 排除中心元件（只做普通元件的 bottom 內部連線）
    regular_pairs = [p for p in pairs if not p.primary.is_center]
    if not regular_pairs:
        return []

    pair_modes = _assign_pair_modes(regular_pairs, base_mode)
    mode_by_key: Dict[Tuple[Coordinate, ...], str] = {}
    component_edges: Dict[Tuple[Coordinate, ...], List[Tuple[Coordinate, Coordinate]]] = {}
    routes: List[RouteResult] = []

    # 計算普通 component 內部的連線 (spanning tree)
    for pair, mode in zip(regular_pairs, pair_modes):
        mode_by_key[pair.primary.key] = mode
        primary_edges = component_spanning_tree(pair.primary.coords)
        component_edges[pair.primary.key] = primary_edges

        if pair.mirror is not None:
            mode_by_key[pair.mirror.key] = mode
            mirrored_edges = [mirror_edge(edge, rows, cols) for edge in primary_edges]
            component_edges[pair.mirror.key] = mirrored_edges
        elif pair.primary.key not in component_edges:
            component_edges[pair.primary.key] = primary_edges

    # 僅 bottom plate 路由
    for pair in regular_pairs:
        primary_edges = component_edges.get(pair.primary.key, [])
        routes.append(
            RouteResult(
                component_id=pair.primary.comp_id,
                method=method,
                mode=mode_by_key[pair.primary.key],
                plate="bottom",
                paths=_edges_to_paths(primary_edges),
            )
        )
        if pair.mirror is not None:
            mirror_edges = component_edges.get(pair.mirror.key, [])
            routes.append(
                RouteResult(
                    component_id=pair.mirror.comp_id,
                    method=method,
                    mode=mode_by_key[pair.mirror.key],
                    plate="bottom",
                    paths=_edges_to_paths(mirror_edges),
                )
            )

    return routes


# =========================
# Plotting utilities
# =========================
def require_matplotlib() -> None:
    if plt is None or mcolors is None:
        raise RuntimeError("matplotlib is required for plotting but is not installed.")


def plot_routed_layout(
    matrix: np.ndarray,
    routes: List[RouteResult],
    title: str = "Routed Common-Centroid Layout (BOTTOM ONLY)",
) -> None:
    """Render the placement and overlay the routed paths (bottom only)."""
    require_matplotlib()

    if not routes:
        return

    unique_components = sorted(np.unique(matrix))
    cmap = plt.colormaps.get_cmap("Paired")
    colors = cmap(np.linspace(0, 1, len(unique_components)))
    color_map = {comp: colors[i] for i, comp in enumerate(unique_components)}

    if -1 in color_map:
        color_map[-1] = mcolors.to_rgba("lightgrey")

    colored_matrix = np.zeros(matrix.shape + (4,))
    for comp_id in unique_components:
        if comp_id in color_map:
            colored_matrix[matrix == comp_id] = color_map[comp_id]

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(colored_matrix, interpolation="nearest")

    for r in range(matrix.shape[0]):
        for c in range(matrix.shape[1]):
            component_id = int(matrix[r, c])
            if component_id in color_map:
                text_color = "white" if sum(color_map[component_id][:3]) < 1.5 else "black"
                ax.text(c, r, str(component_id), ha="center", va="center", color=text_color, fontsize=10)

    color_cycle = plt.colormaps.get_cmap("tab20")
    color_cache: Dict[Tuple[int, str], Tuple[float, float, float, float]] = {}

    def route_color(route: RouteResult) -> Tuple[float, float, float, float]:
        key = (route.component_id, route.mode.lower())
        if key not in color_cache:
            if route.component_id < 0 or "single" in route.mode:
                color_cache[key] = mcolors.to_rgba("black")
            else:
                mode_lower = route.mode.lower()
                raw_index = (route.component_id % 10) * 2
                if mode_lower == "nonoverlapped":
                    raw_index += 1
                normalized_index = raw_index / max(color_cycle.N - 1, 1)
                color_cache[key] = color_cycle(normalized_index)
        return color_cache[key]

    for route in routes:
        comp_color = route_color(route)
        is_single_wire_mode = "single" in route.mode
        for segment in route.paths:
            ys = [node[0] for node in segment.points]
            xs = [node[1] for node in segment.points]
            ax.plot(
                xs, ys,
                color=comp_color,
                linewidth=3 if is_single_wire_mode else 2,
                linestyle="--" if is_single_wire_mode else "-",
                marker="o",
                markersize=3,
                alpha=0.85,
                label=f"CID {route.component_id} ({route.mode})" if segment == route.paths[0] else None,
            )

    ax.set_title(title, fontsize=16)
    ax.set_xticks(np.arange(-0.5, matrix.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, matrix.shape[0], 1), minor=True)
    ax.grid(which="minor", color="black", linestyle="-", linewidth=1)
    ax.tick_params(which="minor", size=0)
    ax.set_xticks([])
    ax.set_yticks([])

    handles, labels = ax.get_legend_handles_labels()
    if labels:
        by_label = dict(zip(labels, handles))
        ax.legend(
            by_label.values(),
            by_label.keys(),
            bbox_to_anchor=(1.05, 1),
            loc="upper left",
            borderaxespad=0.0,
        )

    rows, cols = matrix.shape
    ax.set_ylim(rows - 0.5, -0.5)
    ax.set_xlim(-0.5, cols - 0.5)

    plt.tight_layout()
    plt.show()


# =========================
# CLI
# =========================
DEFAULT_LAYOUT = """
5 4 6 6 6 6 6 6
5 6 3 6 5 4 6 6
5 6 4 2 5 5 5 6
6 6 6 1 6 3 4 5
5 4 3 0 6 6 6 6
6 5 5 5 2 4 6 5
6 6 4 5 6 3 6 5
6 6 6 6 6 6 4 5
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Route a common-centroid layout using MST wiring (BOTTOM ONLY).")
    parser.add_argument(
        "--layout-file",
        type=str,
        help="Path to a text file containing a whitespace-separated layout matrix.",
    )
    parser.add_argument(
        "--method",
        choices=["mst", "bfs"],
        default="mst",
        help="Routing method to build the connectivity tree (intra-component).",
    )
    parser.add_argument(
        "--mode",
        choices=["overlapped", "nonoverlapped", "both", "random"],
        default="random",
        help="Routing mode selection applied to mirrored pairs (bottom only).",
    )
    parser.add_argument(
        "--unit-cap",
        type=float,
        default=1.0,
        help="Unit capacitor size used when emitting the CP sequence (emitted as 'Cu=<value>').",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip Matplotlib visualization (useful if matplotlib is not installed).",
    )

    args = parser.parse_args()

    # robust file read with fallback to DEFAULT_LAYOUT
    if args.layout_file:
        try:
            with open(args.layout_file, "r", encoding="utf-8") as fh:
                layout_text = fh.read()
            if not layout_text.strip():
                print(f"[WARN] Layout file is empty or whitespace-only: {args.layout_file!r}. Falling back to DEFAULT_LAYOUT.")
                layout_text = DEFAULT_LAYOUT
        except FileNotFoundError:
            print(f"[WARN] Layout file not found: {args.layout_file!r}. Falling back to DEFAULT_LAYOUT.")
            layout_text = DEFAULT_LAYOUT
    else:
        layout_text = DEFAULT_LAYOUT

    matrix = parse_layout_text(layout_text)

    # bottom-only routing
    routes = route_layout(
        matrix,
        method=args.method,
        base_mode=args.mode,
    )

    # Legacy CP sequence
    cp_sequence_legacy = build_cp_sequence(matrix, routes)

    # CP Sequence v2 + pattern 統計（bottom-only 通常無 O/N）
    edge_modes = collect_edge_modes(matrix, routes)
    cp_entries = enumerate_cp_pattern_entries(matrix, edge_modes)
    cp_sequence_v2 = build_cp_sequence_v2(args.unit_cap, cp_entries)
    histogram = pattern_histogram(cp_entries)

    print("=== Routing CP sequence (legacy format) ===")
    for entry in cp_sequence_legacy:
        print(entry)

    print("\n=== CP Sequence (Cu followed by routing patterns) ===")
    print(cp_sequence_v2)

    print("\n=== CP pattern entries with locations ===")
    if cp_entries:
        for pattern, edge in cp_entries:
            print(f"{pattern}: {edge}")
    else:
        print("No adjacent unit capacitors share both plate connections (bottom-only run).")

    print("\n=== Routing pattern histogram ===")
    if histogram:
        for key in sorted(histogram):
            print(f"{key}: {histogram[key]}")
    else:
        print("No patterns to report (bottom-only run).")

    if not args.no_plot:
        if plt is None:
            print("matplotlib is not installed; skipping visualization.")
        else:
            title = f"Routing via {args.method.upper()} (mode={args.mode}) - BOTTOM ONLY"
            plot_routed_layout(matrix, routes, title=title)


if __name__ == "__main__":  # pragma: no cover
    main()
