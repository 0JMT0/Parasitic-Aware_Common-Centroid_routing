#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
DAC Common-Centroid Toolkit
- synthesize: 只給 N_bits，自動產生貼近 paper 模型的最佳 placement
- evaluate  : 從 txt 讀取 0..N 的標籤矩陣，計算 Mismatch / Overall Rho / Trunk Wire

方程式：
  Eq.(4) Mismatch：線性製程梯度 (gx, gy)
  Eq.(5) rho_ab  ：平方距離版本  rho_ab = rho_u ^ ( ((Δr)^2+(Δs)^2) * l )
  Eq.(6) & (7)   ：支援 general cross terms（α, β, γ, 近鄰門檻），Overall ρ = Σ_{i<j} ρ_ij

注意：
- 0 = B0 (dummy)，不參與 M/ρ/wire計算
- 1..N = B1..BN
"""

import argparse
import os
import sys
import math
import random
import copy
import numpy as np

# =========================================================
#  基本結構
# =========================================================
class UnitCapacitor:
    def __init__(self, bit_index: int, size: float = 1.0):
        self.bit_index = bit_index   # 0 = B0(dummy), 1..N = 真實位元
        self.size = size

    def __repr__(self):
        return f"B{self.bit_index}"

# =========================================================
#  讀寫工具
# =========================================================
def read_label_grid_from_txt(path: str):
    """讀取以空白分隔的數字矩陣 txt；忽略空行；檢查是否為矩形。"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"檔案不存在: {path}")
    grid = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            row = [int(x) for x in parts]
            grid.append(row)
    if not grid:
        raise ValueError("檔案為空或沒有有效數據行。")
    w = len(grid[0])
    for i, r in enumerate(grid):
        if len(r) != w:
            raise ValueError(f"第 {i+1} 行長度 {len(r)} 與第一行 {w} 不同，非矩形。")
    return grid  # list[list[int]]

def write_label_grid_to_txt(path: str, label_grid):
    """將 2D 整數矩陣寫成空白分隔的 txt。"""
    with open(path, "w", encoding="utf-8") as f:
        for row in label_grid:
            f.write(" ".join(str(int(x)) for x in row))
            f.write("\n")

# =========================================================
#  評估器（Eq.4/5/6/7 + general cross terms）
# =========================================================
class PlacementEvaluator:
    def __init__(
        self,
        N_bits: int,
        rho_u: float = 0.9,
        l: float = 0.5,
        gradient_x: float = 0.005,
        gradient_y: float = 0.002,
        unit_cap_size: float = 1.0,
        alpha: float = 0.0,        # intra(Ci)
        beta: float = 0.0,         # intra(Cj)
        gamma: float = 0.0,        # neighbor cross
        neigh_thresh: float = 1.0  # 近鄰門檻（格點歐氏距離）
    ):
        self.N_bits = N_bits
        self.rho_u = rho_u
        self.l = l
        self.gx = gradient_x
        self.gy = gradient_y
        self.unit_cap_size = unit_cap_size
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.neigh_thresh = neigh_thresh

        self.placement_map = None
        self.rows = 0
        self.cols = 0

    def load_from_label_grid(self, label_grid):
        arr = np.array(label_grid, dtype=int)
        self.rows, self.cols = arr.shape
        pm = np.full((self.rows, self.cols), None, dtype=object)
        for r in range(self.rows):
            for c in range(self.cols):
                pm[r, c] = UnitCapacitor(int(arr[r, c]), self.unit_cap_size)
        self.placement_map = pm
        return pm

    def _get_unit_locations(self):
        """Collect coordinates for each bit index from the current placement_map.
        Returns a dict: {bit_index: [(r, c), ...]} including bit 0 (B0).
        """
        if self.placement_map is None:
            return {bit: [] for bit in range(0, self.N_bits + 1)}
        locs = {bit: [] for bit in range(0, self.N_bits + 1)}
        for r in range(self.rows):
            for c in range(self.cols):
                u = self.placement_map[r, c]
                if u is not None:
                    locs[u.bit_index].append((r, c))
        return locs

    def _precompute_rho_ab_and_dist(self):
        coords = [(r, c) for r in range(self.rows) for c in range(self.cols)
                  if self.placement_map[r, c] is not None]
        rho_ab = {}
        dist_e = {}
        for i in range(len(coords)):
            ra, ca = coords[i]
            for j in range(i, len(coords)):
                rb, cb = coords[j]
                dr, ds = ra - rb, ca - cb
                # Eq.(5): 平方距離版本
                dist_term = (dr*dr + ds*ds) * self.l
                val = (self.rho_u ** dist_term)
                rho_ab[(ra, ca, rb, cb)] = val
                rho_ab[(rb, cb, ra, ca)] = val
                de = math.hypot(dr, ds)  # 給近鄰門檻使用
                dist_e[(ra, ca, rb, cb)] = de
                dist_e[(rb, cb, ra, ca)] = de
        return rho_ab, dist_e

    def _intra_avg(self, bit, locs, rho_ab):
        pts = locs.get(bit, [])
        m = len(pts)
        if m < 2:
            return 0.0
        s = 0.0
        for i in range(m):
            ra, ca = pts[i]
            for j in range(i + 1, m):
                rb, cb = pts[j]
                s += rho_ab.get((ra, ca, rb, cb), 0.0)
        return (2.0 / (m * (m - 1))) * s

    # ---- 指標 ----
    def calculate_mismatch(self):
        locs = self._get_unit_locations()
        cr = (self.rows - 1) / 2.0
        cc = (self.cols - 1) / 2.0

        Cid = {}
        Cst = {}
        for bit in range(1, self.N_bits + 1):
            Cid[bit] = (2 ** (bit - 1)) * self.unit_cap_size
            s = 0.0
            for (r, c) in locs.get(bit, []):
                dev = 1 + self.gx * (c - cc) + self.gy * (r - cr)
                s += self.unit_cap_size * dev
            Cst[bit] = s

        max_m = 0.0
        for i in range(1, self.N_bits + 1):
            for j in range(1, self.N_bits + 1):
                if i == j: continue
                ideal_ratio = Cid[i] / Cid[j]
                star_ratio  = Cst[i] / Cst[j]
                m = abs(star_ratio / ideal_ratio - 1.0) * 100.0
                if m > max_m:
                    max_m = m
        return max_m

    def calculate_correlation(self):
        rho_ab, dist_e = self._precompute_rho_ab_and_dist()
        locs = self._get_unit_locations()

        def intra(bit):
            return self._intra_avg(bit, locs, rho_ab) if bit >= 1 else 0.0

        rho_ij_vals = []
        for i in range(1, self.N_bits + 1):
            for j in range(i + 1, self.N_bits + 1):
                Ci, Cj = locs.get(i, []), locs.get(j, [])
                mu, nu = len(Ci), len(Cj)
                if mu == 0 or nu == 0:
                    continue

                s_base, s_neigh = 0.0, 0.0
                for (ra, ca) in Ci:
                    for (rb, cb) in Cj:
                        v = rho_ab.get((ra, ca, rb, cb), 0.0)
                        s_base += v
                        if self.gamma != 0.0 and dist_e.get((ra, ca, rb, cb), 1e9) <= self.neigh_thresh:
                            s_neigh += v

                base_term  = s_base / (mu * nu)
                neigh_term = (s_neigh / (mu * nu)) if self.gamma != 0.0 else 0.0
                intra_i = intra(i) if self.alpha != 0.0 else 0.0
                intra_j = intra(j) if self.beta  != 0.0 else 0.0

                rho_ij = base_term + self.alpha * intra_i + self.beta * intra_j + self.gamma * neigh_term
                rho_ij_vals.append(rho_ij)

        return sum(rho_ij_vals)

    def calculate_trunk_wire_metric(self):
        """HPWL（僅 1..N 計算，忽略 B0）"""
        locs = self._get_unit_locations()
        total = 0
        for bit in range(1, self.N_bits + 1):
            coords = locs.get(bit, [])
            if len(coords) < 2: 
                continue
            min_r = min(r for r, c in coords)
            max_r = max(r for r, c in coords)
            min_c = min(c for r, c in coords)
            max_c = max(c for r, c in coords)
            total += (max_r - min_r) + (max_c - min_c)
        return total

    def evaluate(self):
        m = self.calculate_mismatch()
        rho = self.calculate_correlation()
        wire = self.calculate_trunk_wire_metric()
        return m, rho, wire

# =========================================================
#  優化器（自動生成 placement）
# =========================================================
def choose_grid_dims(total_units: int):
    """
    選一組近方形、偏偶數邊長的 (rows, cols) 使 rows*cols >= total_units
    """
    base = int(math.sqrt(total_units))
    candidates = []
    for rows in range(max(2, base - 2), base + 4):
        for cols in range(rows, rows + 6):  # cols >= rows，偏近方形
            if rows * cols >= total_units:
                penalty = (cols - rows) + (rows % 2) + (cols % 2)  # 偏好正方且偶數
                over = rows * cols - total_units
                score = penalty * 100 + over
                candidates.append((score, rows, cols))
    if not candidates:
        return base, base
    _, r, c = min(candidates, key=lambda x: x[0])
    return r, c

class DACPlacementOptimizer:
    def __init__(
        self,
        N_bits: int,
        max_row: int,
        max_col: int,
        rho_u: float = 0.9,
        l: float = 0.5,
        gradient_x: float = 0.005,
        gradient_y: float = 0.002,
        unit_cap_size: float = 1.0,
        include_dummy_b0: bool = True,
        # cross-terms（與 evaluator 同步）
        alpha: float = 0.0, beta: float = 0.0, gamma: float = 0.0, neigh_thresh: float = 1.0
    ):
        self.N_bits = N_bits
        self.max_row = max_row
        self.max_col = max_col
        self.rho_u = rho_u
        self.l = l
        self.gx = gradient_x
        self.gy = gradient_y
        self.unit_cap_size = unit_cap_size
        self.include_dummy_b0 = include_dummy_b0

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.neigh_thresh = neigh_thresh

        self.total_units = 2**N_bits - 1 + (1 if include_dummy_b0 else 0)
        if self.total_units > max_row * max_col:
            raise ValueError("網格尺寸不足以容納所有單位電容。")

        self.unit_list = self._create_unit_list()
        self.placement_map = None

        # 參考尺度（做簡單 normalization）
        self._m_ref = None
        self._rho_ref = None
        self._w_ref = None

        self._sym_ref = None

    def _create_unit_list(self):
        ul = []
        for bit in range(1, self.N_bits + 1):
            for _ in range(2 ** (bit - 1)):
                ul.append(UnitCapacitor(bit, self.unit_cap_size))
        if self.include_dummy_b0:
            ul.append(UnitCapacitor(0, self.unit_cap_size))
        ul.sort(key=lambda u: u.bit_index)  # B0 在前，之後 B1..BN
        return ul

    def _generate_spiral_from_center(self, unit_list):
        pm = np.full((self.max_row, self.max_col), None, dtype=object)
        if not unit_list:
            return pm
        r, c = self.max_row // 2, self.max_col // 2
        if self.max_row % 2 == 0: r -= 1
        if self.max_col % 2 == 0: c -= 1
        dr = [0, 1, 0, -1]
        dc = [1, 0, -1, 0]
        dir_idx = 0
        steps = 1
        turn = 0
        idx = 0
        placed = 0
        total = min(len(unit_list), self.max_row * self.max_col)
        while placed < total:
            for _ in range(steps):
                if 0 <= r < self.max_row and 0 <= c < self.max_col:
                    if pm[r, c] is None and idx < len(unit_list):
                        pm[r, c] = unit_list[idx]
                        idx += 1
                        placed += 1
                        if placed >= total:
                            break
                r += dr[dir_idx]; c += dc[dir_idx]
            if placed >= total: break
            dir_idx = (dir_idx + 1) % 4
            turn += 1
            if turn % 2 == 0:
                steps += 1
        return pm

    def _enumerate_sym_pairs(self, randomize=False):
        pairs = []
        seen = set()
        for r in range(self.max_row):
            for c in range(self.max_col):
                partner = self._symmetric_partner(r, c)
                if (r, c) in seen:
                    continue
                seen.add((r, c))
                seen.add(partner)
                if (r, c) == partner:
                    pairs.append([(r, c)])
                else:
                    primary, secondary = (r, c), partner
                    if primary > secondary:
                        primary, secondary = secondary, primary
                    pairs.append([primary, secondary])
        center_r = (self.max_row - 1) / 2.0
        center_c = (self.max_col - 1) / 2.0
        pairs.sort(key=lambda ps: sum((pr - center_r) ** 2 + (pc - center_c) ** 2 for pr, pc in ps) / len(ps))
        if randomize:
            random.shuffle(pairs)
        return pairs

    def _generate_symmetric_seed(self, unit_list, randomize=False):
        pm = np.full((self.max_row, self.max_col), None, dtype=object)
        pairs = self._enumerate_sym_pairs(randomize=randomize)
        per_bit = {}
        for unit in unit_list:
            per_bit.setdefault(unit.bit_index, []).append(unit)
        if randomize:
            for units in per_bit.values():
                random.shuffle(units)

        pair_indices = [idx for idx, pair in enumerate(pairs) if len(pair) >= 2]
        leftovers_indices = [idx for idx, pair in enumerate(pairs) if len(pair) < 2]
        pair_cursor = 0
        leftovers = []

        for bit in sorted(per_bit.keys(), reverse=True):
            units = per_bit[bit]
            while len(units) >= 2 and pair_cursor < len(pair_indices):
                pair = pairs[pair_indices[pair_cursor]]
                pair_cursor += 1
                if len(pair) < 2:
                    continue
                (r1, c1), (r2, c2) = pair
                pm[r1, c1] = units.pop()
                pm[r2, c2] = units.pop()
            if units:
                leftovers.extend(units)

        remaining_indices = pair_indices[pair_cursor:] + leftovers_indices
        if randomize:
            random.shuffle(remaining_indices)

        for idx in remaining_indices:
            pair = pairs[idx]
            for (r, c) in pair:
                if leftovers:
                    pm[r, c] = leftovers.pop()
                else:
                    break

        if leftovers:
            raise RuntimeError('Failed to place all capacitors in symmetric seed')

        return pm

    def _get_unit_locations(self, pm):
        locs = {bit: [] for bit in range(0, self.N_bits + 1)}
        for r in range(self.max_row):
            for c in range(self.max_col):
                u = pm[r, c]
                if u is not None:
                    locs[u.bit_index].append((r, c))
        return locs

    # ---- Eq.(5) 預先表 + 近鄰距離 ----
    def _precompute_rho_ab_and_dist(self, pm):
        coords = [(r, c) for r in range(self.max_row) for c in range(self.max_col) if pm[r, c] is not None]
        rho_ab, dist_e = {}, {}
        for i in range(len(coords)):
            ra, ca = coords[i]
            for j in range(i, len(coords)):
                rb, cb = coords[j]
                dr, ds = ra - rb, ca - cb
                dist_term = (dr*dr + ds*ds) * self.l
                val = (self.rho_u ** dist_term)
                rho_ab[(ra, ca, rb, cb)] = val
                rho_ab[(rb, cb, ra, ca)] = val
                de = math.hypot(dr, ds)
                dist_e[(ra, ca, rb, cb)] = de
                dist_e[(rb, cb, ra, ca)] = de
        return rho_ab, dist_e

    def _intra_avg(self, bit, locs, rho_ab):
        pts = locs.get(bit, [])
        m = len(pts)
        if m < 2: return 0.0
        s = 0.0
        for i in range(m):
            ra, ca = pts[i]
            for j in range(i + 1, m):
                rb, cb = pts[j]
                s += rho_ab.get((ra, ca, rb, cb), 0.0)
        return (2.0 / (m * (m - 1))) * s

    # ---- 指標 ----
    def calculate_mismatch(self, pm):
        locs = self._get_unit_locations(pm)
        cr = (self.max_row - 1) / 2.0
        cc = (self.max_col - 1) / 2.0
        Cid, Cst = {}, {}
        for bit in range(1, self.N_bits + 1):
            Cid[bit] = (2 ** (bit - 1)) * self.unit_cap_size
            s = 0.0
            for (r, c) in locs.get(bit, []):
                dev = 1 + self.gx * (c - cc) + self.gy * (r - cr)
                s += self.unit_cap_size * dev
            Cst[bit] = s
        max_m = 0.0
        for i in range(1, self.N_bits + 1):
            for j in range(1, self.N_bits + 1):
                if i == j: continue
                ideal_ratio = Cid[i] / Cid[j]
                star_ratio  = Cst[i] / Cst[j]
                m = abs(star_ratio / ideal_ratio - 1.0) * 100.0
                if m > max_m:
                    max_m = m
        return max_m

    def calculate_correlation(self, pm):
        rho_ab, dist_e = self._precompute_rho_ab_and_dist(pm)
        locs = self._get_unit_locations(pm)
        rho_ij_vals = []
        for i in range(1, self.N_bits + 1):
            for j in range(i + 1, self.N_bits + 1):
                Ci, Cj = locs.get(i, []), locs.get(j, [])
                mu, nu = len(Ci), len(Cj)
                if mu == 0 or nu == 0: 
                    continue
                s_base, s_neigh = 0.0, 0.0
                for (ra, ca) in Ci:
                    for (rb, cb) in Cj:
                        v = rho_ab.get((ra, ca, rb, cb), 0.0)
                        s_base += v
                        if self.gamma != 0.0 and dist_e.get((ra, ca, rb, cb), 1e9) <= self.neigh_thresh:
                            s_neigh += v
                base_term  = s_base  / (mu * nu)
                neigh_term = (s_neigh / (mu * nu)) if self.gamma != 0.0 else 0.0
                intra_i = self._intra_avg(i, locs, rho_ab) if self.alpha != 0.0 else 0.0
                intra_j = self._intra_avg(j, locs, rho_ab) if self.beta  != 0.0 else 0.0
                rho_ij = base_term + self.alpha*intra_i + self.beta*intra_j + self.gamma*neigh_term
                rho_ij_vals.append(rho_ij)
        return sum(rho_ij_vals)

    def calculate_trunk_wire_metric(self, pm):
        locs = self._get_unit_locations(pm)
        total = 0
        for bit in range(1, self.N_bits + 1):
            coords = locs.get(bit, [])
            if len(coords) < 2:
                continue
            min_r = min(r for r, c in coords)
            max_r = max(r for r, c in coords)
            min_c = min(c for r, c in coords)
            max_c = max(c for r, c in coords)
            total += (max_r - min_r) + (max_c - min_c)
        return total

    def calculate_symmetry_penalty(self, pm):
        pairs = self._enumerate_sym_pairs()
        total_pairs = 0
        penalty = 0.0
        for pair in pairs:
            if len(pair) < 2:
                continue
            (r1, c1), (r2, c2) = pair
            u1 = pm[r1, c1]
            u2 = pm[r2, c2]
            if u1 is None and u2 is None:
                continue
            total_pairs += 1
            if (u1 is None) != (u2 is None):
                penalty += 1.0
            elif u1.bit_index != u2.bit_index:
                penalty += 1.0 + 0.1 * abs(u1.bit_index - u2.bit_index)
        if total_pairs == 0:
            return 0.0
        return penalty / total_pairs

    def calculate_objective(self, pm, w_m, w_rho, w_wire, w_sym=0.0):
        m = self.calculate_mismatch(pm)
        rho = self.calculate_correlation(pm)
        wire = self.calculate_trunk_wire_metric(pm)

        sym_pen = 0.0
        sym_n = 0.0
        if w_sym != 0.0:
            sym_pen = self.calculate_symmetry_penalty(pm)
            if self._sym_ref is None:
                self._sym_ref = max(1e-6, sym_pen)
            sym_n = sym_pen / (1.0 + self._sym_ref)

        # ²�� normalization�G�H��l�Ѭ��ѦҤثס]lazy ��l�ơ^
        if self._m_ref is None:   self._m_ref   = max(1e-6, m)
        if self._rho_ref is None: self._rho_ref = max(1e-6, rho)
        if self._w_ref is None:   self._w_ref   = max(1.0,  wire)

        m_n   = m   / (1.0 + self._m_ref)
        rho_n = rho / (1.0 + self._rho_ref)
        w_n   = wire/ (1.0 + self._w_ref)

        cost = w_m * m_n - w_rho * rho_n + w_wire * w_n
        if w_sym != 0.0:
            cost += w_sym * sym_n
        return cost, m, rho, wire


    # ---- 特製擾動 ----
    def _swap_cells(self, pm, p1, p2):
        r1, c1 = p1; r2, c2 = p2
        npm = copy.deepcopy(pm)
        npm[r1, c1], npm[r2, c2] = npm[r2, c2], npm[r1, c1]
        return npm

    def _symmetric_partner(self, r, c):
        return (self.max_row - 1 - r, self.max_col - 1 - c)

    def _sym_swap(self, pm):
        """中心對稱交換：挑一格 (r,c) 與其對稱點互換。"""
        r = random.randrange(self.max_row)
        c = random.randrange(self.max_col)
        r2, c2 = self._symmetric_partner(r, c)
        if (r, c) == (r2, c2):
            return None
        return self._swap_cells(pm, (r, c), (r2, c2))

    def _rotate_2x2(self, pm):
        """隨機 2×2 區塊順時針旋轉。"""
        if self.max_row < 2 or self.max_col < 2:
            return None
        r = random.randrange(self.max_row - 1)
        c = random.randrange(self.max_col - 1)
        npm = copy.deepcopy(pm)
        a, b = npm[r, c],   npm[r, c+1]
        c2,d = npm[r+1, c], npm[r+1, c+1]
        # 旋轉： [a b; c d] -> [c a; d b]
        npm[r, c], npm[r, c+1], npm[r+1, c+1], npm[r+1, c] = c2, a, b, d
        return npm

    def optimize(self,
                 start_map,
                 iterations=20000,
                 T_initial=50.0,
                 T_final=0.05,
                 alpha=0.999,
                 w_m=1.0, w_rho=3.0, w_wire=0.05, w_sym=0.0,
                 move_mix=(0.3, 0.3, 0.4),  # sym_swap, rot2x2, random_swap
                 seed=None):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        current = copy.deepcopy(start_map)
        best = copy.deepcopy(current)

        # 初始化參考尺度
        _ = self.calculate_objective(current, w_m, w_rho, w_wire, w_sym)

        # 佔用格清單
        occ = [(r, c) for r in range(self.max_row) for c in range(self.max_col) if current[r, c] is not None]

        cur_cost, _, _, _ = self.calculate_objective(current, w_m, w_rho, w_wire, w_sym)
        best_cost = cur_cost

        p_sym, p_rot, p_swap = move_mix
        T = T_initial

        for _ in range(iterations):
            r = random.random()
            if r < p_sym:
                cand = self._sym_swap(current)
                if cand is None: 
                    continue
            elif r < p_sym + p_rot:
                cand = self._rotate_2x2(current)
                if cand is None:
                    continue
            else:
                # random swap
                (r1, c1), (r2, c2) = random.sample(occ, 2)
                cand = self._swap_cells(current, (r1, c1), (r2, c2))

            new_cost, _, _, _ = self.calculate_objective(cand, w_m, w_rho, w_wire, w_sym)
            delta = new_cost - cur_cost
            if delta < 0 or random.random() < math.exp(-delta / max(1e-9, T)):
                current = cand
                cur_cost = new_cost
                if cur_cost < best_cost:
                    best = copy.deepcopy(current)
                    best_cost = cur_cost

            T *= alpha
            if T < T_final:
                break

        return best, best_cost

    # ---- 便利 ----
    def to_label_grid(self, pm):
        out = np.full((self.max_row, self.max_col), -1, dtype=int)
        for r in range(self.max_row):
            for c in range(self.max_col):
                u = pm[r, c]
                out[r, c] = u.bit_index if u is not None else -1
        return out

    def print_placement(self, pm):
        print("\n--- Placement Map ---")
        for r in range(self.max_row):
            row = []
            for c in range(self.max_col):
                u = pm[r, c]
                row.append(f"B{u.bit_index:<2}" if u else "-- ")
            print(" ".join(row))
        print("---------------------\n")

# =========================================================
#  一鍵合成
# =========================================================
def synthesize_best_layout(
    N_bits: int,
    rho_u=0.9, l=0.5,
    gx=0.005, gy=0.002,
    include_dummy_b0=True,
    alpha=0.2, beta=0.2, gamma=0.1, neigh_thresh=1.5,
    restarts=6,
    sa_iters=None,
    w_m=1.0, w_rho=3.0, w_wire=0.05,
    w_sym=5.0,
    move_mix=(0.3, 0.3, 0.4),
    seed=None
):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    total_units = 2**N_bits - 1 + (1 if include_dummy_b0 else 0)
    rows, cols = choose_grid_dims(total_units)
    if sa_iters is None:
        sa_iters = int(4000 + 2000 * max(0, N_bits - 4))

    optimizer = DACPlacementOptimizer(
        N_bits=N_bits,
        max_row=rows, max_col=cols,
        rho_u=rho_u, l=l,
        gradient_x=gx, gradient_y=gy,
        include_dummy_b0=include_dummy_b0,
        alpha=alpha, beta=beta, gamma=gamma, neigh_thresh=neigh_thresh
    )

    # 基準：排序 unit_list 的螺旋放置
    start = optimizer._generate_symmetric_seed(optimizer.unit_list)
    best_map, best_cost = optimizer.optimize(
        start_map=start,
        iterations=sa_iters,
        T_initial=50.0, T_final=0.05, alpha=0.999,
        w_m=w_m, w_rho=w_rho, w_wire=w_wire,
        w_sym=w_sym,
        move_mix=move_mix,
        seed=seed
    )

    # 多重起點：將 B1..BN 隨機打散（保留 B0）
    for _ in range(max(0, restarts - 1)):
        start2 = optimizer._generate_symmetric_seed(optimizer.unit_list, randomize=True)
        cand_map, cand_cost = optimizer.optimize(
            start_map=start2,
            iterations=sa_iters,
            T_initial=50.0, T_final=0.05, alpha=0.999,
            w_m=w_m, w_rho=w_rho, w_wire=w_wire, w_sym=w_sym,
            move_mix=move_mix,
            seed=seed
        )
        if cand_cost < best_cost:
            best_map, best_cost = cand_map, cand_cost

    # 回傳數據
    m = optimizer.calculate_mismatch(best_map)
    rho = optimizer.calculate_correlation(best_map)
    wire = optimizer.calculate_trunk_wire_metric(best_map)
    return optimizer, best_map, (m, rho, wire), (rows, cols)

# =========================================================
#  CLI
# =========================================================
def main():
    parser = argparse.ArgumentParser(description="DAC Common-Centroid Placement Toolkit")
    sub = parser.add_subparsers(dest="cmd", required=True)

    # evaluate
    pe = sub.add_parser("evaluate", help="從 txt 讀取 0..N 矩陣，計算 Mismatch / Overall Rho / Trunk Wire")
    pe.add_argument("txt_path", help="txt 檔（空白分隔整數；0=B0，1..N=B位元）")
    pe.add_argument("--nbits", type=int, default=None, help="若不指定，取檔案中的最大值")
    pe.add_argument("--rho_u", type=float, default=0.9)
    pe.add_argument("--l", type=float, default=0.5)
    pe.add_argument("--gx", type=float, default=0.005)
    pe.add_argument("--gy", type=float, default=0.002)
    pe.add_argument("--unit", type=float, default=1.0)
    pe.add_argument("--alpha", type=float, default=0.0)
    pe.add_argument("--beta",  type=float, default=0.0)
    pe.add_argument("--gamma", type=float, default=0.0)
    pe.add_argument("--neigh-th", type=float, default=1.0)
    pe.add_argument("--print-grid", action="store_true")

    # synthesize
    ps = sub.add_parser("synthesize", help="只給 N_bits，自動產生最佳 placement")
    ps.add_argument("--nbits", type=int, required=True)
    ps.add_argument("--rho_u", type=float, default=0.9)
    ps.add_argument("--l", type=float, default=0.5)
    ps.add_argument("--gx", type=float, default=0.005)
    ps.add_argument("--gy", type=float, default=0.002)
    ps.add_argument("--include-b0", action="store_true", help="加入一顆 B0 (dummy)（預設不加）")
    ps.add_argument("--alpha", type=float, default=0.2)
    ps.add_argument("--beta",  type=float, default=0.2)
    ps.add_argument("--gamma", type=float, default=0.1)
    ps.add_argument("--neigh-th", type=float, default=1.5)
    ps.add_argument("--restarts", type=int, default=6)
    ps.add_argument("--sa-iters", type=int, default=None)
    ps.add_argument("--w-m", type=float, default=1.0)
    ps.add_argument("--w-rho", type=float, default=3.0)
    ps.add_argument("--w-wire", type=float, default=0.05)
    ps.add_argument("--w-sym", type=float, default=5.0)
    ps.add_argument("--seed", type=int, default=None)
    ps.add_argument("--out", type=str, default=None, help="輸出 placement txt（0..N）路徑")
    ps.add_argument("--show", action="store_true", help="在終端列印 B 標籤版圖")

    args = parser.parse_args()

    if args.cmd == "evaluate":
        label_grid = read_label_grid_from_txt(args.txt_path)
        arr = np.array(label_grid, dtype=int)
        max_label = int(arr.max())
        N_bits = args.nbits if args.nbits is not None else max_label
        if arr.min() < 0:
            print("[錯誤] 讀到負數標籤。", file=sys.stderr); sys.exit(1)
        if arr.max() > N_bits:
            print(f"[錯誤] 檔案含有標籤 {arr.max()} 超過 N_bits={N_bits}。", file=sys.stderr); sys.exit(1)

        if args.print_grid:
            print("\n=== 讀入的標籤矩陣 (0=B0, 1..N=位元) ===")
            for row in label_grid:
                print(" ".join(str(x) for x in row))

        evaluator = PlacementEvaluator(
            N_bits=N_bits,
            rho_u=args.rho_u, l=args.l,
            gradient_x=args.gx, gradient_y=args.gy,
            unit_cap_size=args.unit,
            alpha=args.alpha, beta=args.beta, gamma=args.gamma, neigh_thresh=args.neigh_th
        )
        evaluator.load_from_label_grid(label_grid)
        M, RHO, WIRE = evaluator.evaluate()
        rows, cols = np.array(label_grid).shape
        print(f"\n=== Placement 評估結果 ===")
        print(f"Grid Size        : {rows} x {cols}")
        print(f"N_bits           : {N_bits}")
        print(f"rho_u, l         : {args.rho_u}, {args.l}")
        print(f"grad_x, grad_y   : {args.gx}, {args.gy}")
        print(f"Unit Cap         : {args.unit}")
        print(f"alpha, beta, gamma, neigh_th : {args.alpha}, {args.beta}, {args.gamma}, {args.neigh_th}")
        print(f"Mismatch (M, %)  : {M:.6f}")
        print(f"Overall Rho      : {RHO:.6f}")
        print(f"Trunk Wire (HPWL): {WIRE:.6f}")

    elif args.cmd == "synthesize":
        optimizer, best_map, (M, RHO, WIRE), (r, c) = synthesize_best_layout(
            N_bits=args.nbits,
            rho_u=args.rho_u, l=args.l,
            gx=args.gx, gy=args.gy,
            include_dummy_b0=args.include_b0,
            alpha=args.alpha, beta=args.beta, gamma=args.gamma, neigh_thresh=args.neigh_th,
            restarts=args.restarts,
            sa_iters=args.sa_iters,
            w_m=args.w_m, w_rho=args.w_rho, w_wire=args.w_wire,
            w_sym=args.w_sym,
            seed=args.seed
        )
        print(f"\n=== 自動合成結果 (N_bits={args.nbits}, grid={r}x{c}) ===")
        print(f"Mismatch (M, %)  : {M:.6f}")
        print(f"Overall Rho      : {RHO:.6f}")
        print(f"Trunk Wire (HPWL): {WIRE:.6f}")

        if args.show:
            optimizer.print_placement(best_map)

        if args.out:
            label_grid = optimizer.to_label_grid(best_map)
            write_label_grid_to_txt(args.out, label_grid)
            print(f"[已輸出] {args.out}")

if __name__ == "__main__":
    main()


# """
# ===============================================================
#  DAC Common-Centroid Placement Toolkit 使用說明與公式
# ===============================================================

# 本程式提供兩大功能：
# 1. evaluate   : 從文字檔讀取已知的電容陣列 (0..N 標籤)，計算性能指標
# 2. synthesize : 只給 Nbits，自動合成一個最佳化的共質心電容陣列

# ---------------------------------------------------------------
# 【輸入格式】
# - evaluate 模式讀取的文字檔 (txt) 必須是矩形矩陣
# - 每格為整數：
#     0 = B0 (dummy capacitor，不參與 Mismatch/Correlation/Wire 計算)
#     1..N = 對應 DAC 的位元電容 (B1 = LSB, BN = MSB)
# - 例如：
#     5 4 6 6 6 6 6 6
#     5 6 3 6 5 4 6 6
#     5 6 4 2 5 5 5 6
#     6 6 6 1 6 3 4 5
#     5 4 3 0 6 6 6 6
#     6 5 5 5 2 4 6 5
#     6 6 4 5 6 3 6 5
#     6 6 6 6 6 6 4 5

# ---------------------------------------------------------------
# 【數學公式】

# (Equation 4) 製程梯度失配 (Mismatch M):
#     M = max | (Ci*/Cj*) / (Ci/Cj) - 1 | × 100%,   ∀ i, j
#     其中:
#       Ci  = 理想第 i 位元的總電容
#       Ci* = 考慮線性梯度 (gx, gy) 後的第 i 位元實際總電容

# (Equation 5) 單元電容相關係數 (ρ_ab):
#     ρ_ab = ρ_u ^ ( ((ra - rb)^2 + (sa - sb)^2) × l )
#     其中:
#       (ra, sa) = 單元 ua 的座標
#       (rb, sb) = 單元 ub 的座標
#       ρ_u      = 單位距離下的基礎相關性
#       l        = 距離衰減因子

# (Equation 6) 電容對間相關係數 (ρ_ij):
#     ρ_ij = (1 / (μ × ν)) × ( Σ Σ ρ_ab + cross terms )
#     其中:
#       μ = Ci 的單元數
#       ν = Cj 的單元數
#       cross terms 可包含:
#         - intra(Ci) 項: Ci 內部單元間的平均相關性
#         - intra(Cj) 項: Cj 內部單元間的平均相關性
#         - neighbor 項 : 若 Ci 與 Cj 單元距離小於門檻的額外加權

# (Equation 7) 整體相關係數 (ρ):
#     ρ = Σ Σ ρ_ij,   for all i < j

# ---------------------------------------------------------------
# 【指令用法】

# 1) 評估現有版圖：
#    python Common-Centroid_Placement.py evaluate <txt檔路徑> [選項]

#    常用選項：
#    --print-grid         顯示讀入的矩陣
#    --nbits <int>        指定 DAC 位元數，若省略則取矩陣中最大標籤值
#    --rho_u <float>      基礎單位相關係數 (預設 0.9)
#    --l <float>          距離衰減因子 (預設 0.5)
#    --gx <float>         X 方向製程梯度 (%/unit) (預設 0.005)
#    --gy <float>         Y 方向製程梯度 (%/unit) (預設 0.002)
#    --alpha <float>      cross terms 參數 (intra Ci)
#    --beta <float>       cross terms 參數 (intra Cj)
#    --gamma <float>      cross terms 參數 (近鄰 cross)
#    --neigh-th <float>   近鄰距離閾值

#    範例：
#    python Common-Centroid_Placement.py evaluate .\code_placement.txt --print-grid --alpha 0.2 --beta 0.2 --gamma 0.1 --neigh-th 1.5

# 2) 自動生成最佳佈局：
#    python Common-Centroid_Placement.py synthesize --nbits <int> [選項]

#    常用選項：
#    --include-b0         是否加入一顆 B0 (dummy)
#    --show               在終端列印 placement 版圖 (B標籤)
#    --out <檔名>         輸出最佳版圖到 txt 檔 (矩陣形式)
#    --restarts <int>     多起點隨機嘗試次數 (預設 6)
#    --sa-iters <int>     模擬退火迭代次數 (建議 Nbits>=6 時設定 >10000)
#    --w-m <float>        成本函式權重：Mismatch
#    --w-rho <float>      成本函式權重：Correlation
#    --w-wire <float>     成本函式權重：Wire
#    --alpha/beta/gamma/neigh-th 與 evaluate 相同

#    範例：
# python Common-Centroid_placement.py synthesize --nbits 6 --include-b0 --show --restarts 4 --sa-iters 8000 --w-sym 5.0
# ---------------------------------------------------------------
# 【輸出結果】
# - Placement 評估結果會輸出：
#     Grid Size        : <rows> x <cols>
#     N_bits           : <int>
#     rho_u, l         : <float>, <float>
#     grad_x, grad_y   : <float>, <float>
#     Mismatch (M, %)  : 最大製程梯度失配
#     Overall Rho      : 整體相關係數
#     Trunk Wire (HPWL): 繞線主幹線近似指標
# - synthesize 模式若有指定 --out，則會輸出最佳 placement 的標籤矩陣到 txt 檔。

# ---------------------------------------------------------------
