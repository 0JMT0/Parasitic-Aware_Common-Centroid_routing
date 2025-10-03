import numpy as np
import math
import random
import copy
from tqdm import tqdm

# -----------------------------
# 單位電容
# -----------------------------
class UnitCapacitor:
    def __init__(self, bit_index, initial_size=1.0):
        self.bit_index = bit_index  # 0 為 B0 (dummy)；1..Nbits 為實際位元
        self.size = initial_size

    def __repr__(self):
        return f"B{self.bit_index}"

# -----------------------------
# 版圖優化器
# -----------------------------
class DACPlacementOptimizer:
    """
    根據指定的規格優化 DAC 電容陣列佈局。
    """
    def __init__(self, N_bits, max_row, max_col, rho_u, l, unit_cap_size=1.0, 
                 gradient_x=0.001, gradient_y=0.001, include_dummy_b0=True):
        self.N_bits = N_bits
        self.max_row = max_row
        self.max_col = max_col
        self.rho_u = rho_u
        self.l = l
        self.unit_cap_size = unit_cap_size
        self.gradient_x = gradient_x
        self.gradient_y = gradient_y
        self.include_dummy_b0 = include_dummy_b0
        
        self.total_units = 2**N_bits - 1 + (1 if include_dummy_b0 else 0)
        if self.total_units > max_row * max_col:
            raise ValueError("網格尺寸不足以容納所有單位電容。")
            
        self.unit_list = self._create_unit_list()
        self.placement_map = None

    def _create_unit_list(self):
        unit_list = []
        for bit in range(1, self.N_bits + 1):
            count = 2**(bit - 1)
            for _ in range(count):
                unit_list.append(UnitCapacitor(bit, self.unit_cap_size))
        if self.include_dummy_b0:
            unit_list.append(UnitCapacitor(0, self.unit_cap_size))
        # 預設排序（B0, B1..BN）；若要隨機起點，之後會複製並 shuffle
        unit_list.sort(key=lambda u: u.bit_index)
        return unit_list

    def _generate_spiral_from_center(self, unit_list):
        """以螺旋方式，從中心往外放置 unit_list 順序的單位。"""
        placement_map = np.full((self.max_row, self.max_col), None, dtype=object)
        if not unit_list:
            return placement_map

        r, c = self.max_row // 2, self.max_col // 2
        if self.max_row % 2 == 0: r -= 1
        if self.max_col % 2 == 0: c -= 1

        dr = [0, 1, 0, -1]  # 右, 下, 左, 上
        dc = [1, 0, -1, 0]
        dir_idx = 0
        steps_in_dir = 1
        turn_count = 0
        
        unit_idx = 0
        placed_count = 0
        total_to_place = min(len(unit_list), self.max_row * self.max_col)

        while placed_count < total_to_place:
            for _ in range(steps_in_dir):
                if 0 <= r < self.max_row and 0 <= c < self.max_col:
                    if placement_map[r, c] is None and unit_idx < len(unit_list):
                        placement_map[r, c] = unit_list[unit_idx]
                        unit_idx += 1
                        placed_count += 1
                        if placed_count >= total_to_place:
                            break
                r, c = r + dr[dir_idx], c + dc[dir_idx]
            if placed_count >= total_to_place:
                break

            dir_idx = (dir_idx + 1) % 4
            turn_count += 1
            if turn_count % 2 == 0:
                steps_in_dir += 1

        return placement_map

    def _get_unit_locations(self, placement_map):
        locations = {bit: [] for bit in range(0, self.N_bits + 1)}
        for r in range(self.max_row):
            for c in range(self.max_col):
                unit = placement_map[r, c]
                if unit:
                    locations[unit.bit_index].append((r, c))
        return locations

    def _iter_real_bits(self):
        return range(1, self.N_bits + 1)

    # ------------------ 評估函式 ------------------
    def calculate_mismatch(self, placement_map):
        locations = self._get_unit_locations(placement_map)
        center_r, center_c = (self.max_row - 1) / 2.0, (self.max_col - 1) / 2.0
        
        C_ideal, C_star = {}, {}

        for bit in self._iter_real_bits():
            C_ideal[bit] = (2**(bit - 1)) * self.unit_cap_size
            c_sum = 0.0
            for r, c in locations.get(bit, []):
                deviation = 1 + self.gradient_x * (c - center_c) + self.gradient_y * (r - center_r)
                c_sum += self.unit_cap_size * deviation
            C_star[bit] = c_sum
        
        max_mismatch = 0.0
        for i in self._iter_real_bits():
            for j in self._iter_real_bits():
                if i == j: 
                    continue
                ideal_ratio = C_ideal[i] / C_ideal[j]
                star_ratio  = C_star[i]  / C_star[j]
                if ideal_ratio != 0:
                    mismatch = abs(star_ratio / ideal_ratio - 1) * 100.0
                    if mismatch > max_mismatch:
                        max_mismatch = mismatch
        return max_mismatch

    def calculate_correlation(self, placement_map):
        # 建立所有單位間 rho_ab 表
        coords = [(r, c) for r in range(self.max_row) for c in range(self.max_col)
                  if placement_map[r, c] is not None]
        rho_ab = {}
        for i in range(len(coords)):
            ra, ca = coords[i]
            for j in range(i, len(coords)):
                rb, cb = coords[j]
                d = math.sqrt((ra - rb)**2 + (ca - cb)**2)
                val = self.rho_u**(d * self.l)
                rho_ab[(ra, ca, rb, cb)] = val
                rho_ab[(rb, cb, ra, ca)] = val

        # 逐對位元計算 rho_ij（忽略 B0）
        locations = self._get_unit_locations(placement_map)
        rho_ij_vals = []
        for i in self._iter_real_bits():
            for j in range(i + 1, self.N_bits + 1):
                ui = locations.get(i, [])
                uj = locations.get(j, [])
                if not ui or not uj:
                    continue
                s = 0.0
                for (ra, ca) in ui:
                    for (rb, cb) in uj:
                        s += rho_ab.get((ra, ca, rb, cb), 0.0)
                rho_ij = s / (len(ui) * len(uj))
                rho_ij_vals.append(rho_ij)
        return sum(rho_ij_vals)

    def calculate_trunk_wire_metric(self, placement_map):
        locations = self._get_unit_locations(placement_map)
        total_hpwl = 0
        for bit in self._iter_real_bits():
            coords = locations.get(bit, [])
            if len(coords) < 2:
                continue
            min_r = min(r for r, c in coords)
            max_r = max(r for r, c in coords)
            min_c = min(c for r, c in coords)
            max_c = max(c for r, c in coords)
            total_hpwl += (max_r - min_r) + (max_c - min_c)
        return total_hpwl

    def calculate_objective(self, placement_map, w_m, w_rho, w_wire):
        m = self.calculate_mismatch(placement_map)
        rho = self.calculate_correlation(placement_map)
        wire = self.calculate_trunk_wire_metric(placement_map)
        cost = w_m * m - w_rho * rho + w_wire * wire
        return cost, m, rho, wire

    # ------------------ 優化器 ------------------
    def optimize(self, placement_map, iterations=10000, T_initial=50.0, T_final=0.1, alpha=0.999, 
                 w_m=1.0, w_rho=10.0, w_wire=0.1, freeze_labels=None, verbose=False):
        """
        模擬退火；可提供現成的 placement_map 作為起點。
        freeze_labels: set[int]；若提供，這些 bit 不允許與別的 bit 交換。
        """
        current = copy.deepcopy(placement_map)
        best = copy.deepcopy(current)

        occupied = [(r, c) for r in range(self.max_row) for c in range(self.max_col)
                    if current[r, c] is not None]

        def can_swap(u1, u2):
            if freeze_labels is None:
                return True
            return (u1.bit_index not in freeze_labels) and (u2.bit_index not in freeze_labels)

        cost, m, rho, wire = self.calculate_objective(current, w_m, w_rho, w_wire)
        cur_cost = cost
        best_cost = cur_cost

        if verbose:
            print(f"Init: Cost={best_cost:.4f}, M={m:.4f}%, Rho={rho:.4f}, Wire={wire:.1f}")

        T = T_initial
        for _ in tqdm(range(iterations), disable=not verbose, desc="SA"):
            # 抽兩格（符合凍結規則）
            for _try in range(50):
                r1, c1 = random.choice(occupied)
                r2, c2 = random.choice(occupied)
                if (r1, c1) == (r2, c2): 
                    continue
                u1 = current[r1, c1]
                u2 = current[r2, c2]
                if can_swap(u1, u2):
                    break
            else:
                break

            new_map = copy.deepcopy(current)
            new_map[r1, c1], new_map[r2, c2] = new_map[r2, c2], new_map[r1, c1]

            new_cost, _, _, _ = self.calculate_objective(new_map, w_m, w_rho, w_wire)
            delta = new_cost - cur_cost

            if delta < 0 or random.random() < math.exp(-delta / T):
                current = new_map
                cur_cost = new_cost
                if cur_cost < best_cost:
                    best = copy.deepcopy(current)
                    best_cost = cur_cost

            T *= alpha
            if T < T_final:
                break

        return best, best_cost

    # ------------------ 便利工具 ------------------
    def print_placement(self, placement_map=None):
        placement_map = placement_map if placement_map is not None else self.placement_map
        if placement_map is None:
            print("尚未生成佈局。")
            return
        print("\n--- Placement Map ---")
        for r in range(self.max_row):
            row = []
            for c in range(self.max_col):
                u = placement_map[r, c]
                row.append(f"B{u.bit_index:<2}" if u else "--- ")
            print(" ".join(row))
        print("---------------------\n")

    def to_label_grid(self, placement_map=None):
        """輸出 2D 數字矩陣（0..N）便於存檔/比對。"""
        placement_map = placement_map if placement_map is not None else self.placement_map
        out = np.full((self.max_row, self.max_col), -1, dtype=int)
        for r in range(self.max_row):
            for c in range(self.max_col):
                u = placement_map[r, c]
                out[r, c] = u.bit_index if u else -1
        return out

    # 允許外部載入目標標籤
    def load_architecture_from_labels(self, label_grid):
        arr = np.array(label_grid)
        if arr.shape != (self.max_row, self.max_col):
            raise ValueError(f"label_grid 尺寸應為 {(self.max_row, self.max_col)}，但得到 {arr.shape}")
        if arr.min() < 0 or arr.max() > self.N_bits:
            raise ValueError(f"label_grid 中的標籤必須介於 0..{self.N_bits}（含）。")
        placement = np.full((self.max_row, self.max_col), None, dtype=object)
        for r in range(self.max_row):
            for c in range(self.max_col):
                placement[r, c] = UnitCapacitor(int(arr[r, c]), self.unit_cap_size)
        self.placement_map = placement
        return placement

# -----------------------------
# 自動尺寸估算與一鍵合成
# -----------------------------
def _choose_grid_dims(total_units):
    """
    自動挑選接近方形、偏偶數邊長的網格(rows, cols)，確保 rows*cols >= total_units。
    策略：
      1) 先找接近 sqrt(total) 的整數
      2) 往上擴張直到 rows*cols >= total
      3) 偏好偶數（共質心中心更對稱）
    """
    base = int(math.sqrt(total_units))
    candidates = []
    for rows in range(max(2, base - 2), base + 3):
        for cols in range(rows, rows + 6):  # 近方形且 cols >= rows
            if rows * cols >= total_units:
                penalty = (cols - rows) + (rows % 2) + (cols % 2)  # 偏好正方形與偶數
                area_over = rows * cols - total_units
                score = penalty * 100 + area_over  # 先滿足形狀，再考慮剩餘格
                candidates.append((score, rows, cols))
    if not candidates:
        return base, base
    _, r, c = min(candidates, key=lambda x: x[0])
    return r, c

def synthesize_best_layout(
    N_bits,
    rho_u=0.9,
    l=0.5,
    gradient_x=0.005,
    gradient_y=0.002,
    include_dummy_b0=True,
    restarts=6,
    sa_iters_per_restart=None,
    w_m=1.0,
    w_rho=10.0,
    w_wire=0.1,
    verbose=False,
    seed=None
):
    """
    一鍵合成最佳佈局：只要給 N_bits。
    會：
      1) 自動挑 rows, cols
      2) 多重起點（隨機打亂 unit_list）、螺旋擺放
      3) 模擬退火優化，挑出全域最佳
    回傳：
      optimizer, best_map, best_cost, (M, rho, wire)
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    total_units = 2**N_bits - 1 + (1 if include_dummy_b0 else 0)
    rows, cols = _choose_grid_dims(total_units)

    # 預設 SA 迭代次數隨位元數上升
    if sa_iters_per_restart is None:
        sa_iters_per_restart = int(3000 + 1500 * max(0, N_bits - 4))

    optimizer = DACPlacementOptimizer(
        N_bits=N_bits,
        max_row=rows,
        max_col=cols,
        rho_u=rho_u,
        l=l,
        gradient_x=gradient_x,
        gradient_y=gradient_y,
        include_dummy_b0=include_dummy_b0
    )

    # 基準：使用排序後的 unit_list 做一次
    base_map = optimizer._generate_spiral_from_center(optimizer.unit_list)
    best_map, best_cost = optimizer.optimize(
        base_map,
        iterations=sa_iters_per_restart,
        T_initial=50.0,
        T_final=0.1,
        alpha=0.999,
        w_m=w_m,
        w_rho=w_rho,
        w_wire=w_wire,
        freeze_labels=None,
        verbose=verbose
    )

    # 多重起點：隨機打亂 unit_list（保留 B0 僅 1 顆）
    for _ in range(max(0, restarts - 1)):
        ul = copy.deepcopy(optimizer.unit_list)
        # 保證 B0 還在（若有），但其餘 B1..BN 隨機打散
        ul_nonzero = [u for u in ul if u.bit_index != 0]
        random.shuffle(ul_nonzero)
        ul = [u for u in ul if u.bit_index == 0] + ul_nonzero

        start_map = optimizer._generate_spiral_from_center(ul)
        cand_map, cand_cost = optimizer.optimize(
            start_map,
            iterations=sa_iters_per_restart,
            T_initial=50.0,
            T_final=0.1,
            alpha=0.999,
            w_m=w_m,
            w_rho=w_rho,
            w_wire=w_wire,
            freeze_labels=None,
            verbose=verbose
        )
        if cand_cost < best_cost:
            best_map, best_cost = cand_map, cand_cost

    optimizer.placement_map = best_map
    _, M, RHO, WIRE = optimizer.calculate_objective(best_map, w_m, w_rho, w_wire)
    return optimizer, best_map, best_cost, (M, RHO, WIRE)

# -----------------------------
# 範例：只給 N_bits，一鍵合成
# -----------------------------
if __name__ == "__main__":
    N = 6 
    optimizer, best_map, best_cost, (M, RHO, WIRE) = synthesize_best_layout(
        N_bits=N,
        rho_u=0.9,
        l=0.5,
        gradient_x=0.005,
        gradient_y=0.002,
        include_dummy_b0=True,
        restarts=8,                 # 多重起點
        sa_iters_per_restart=None,  # 自動依 Nbits 設定
        w_m=1.0,
        w_rho=10.0,
        w_wire=0.1,
        verbose=False,
        seed=42
    )

    print(f"\n=== 自動合成結果 (N_bits={N}, grid={optimizer.max_row}x{optimizer.max_col}) ===")
    optimizer.print_placement(best_map)
    print(f" Cost : {best_cost:.4f}")
    print(f" M    : {M:.6f}%")
    print(f" Rho  : {RHO:.6f}")
    print(f" Wire : {WIRE:.1f}")

    # 如果你要拿到數字矩陣（0..N）：
    label_grid = optimizer.to_label_grid(best_map)
    print("\nLabel grid (0=B0, 1..N=B位元)：")
    print(label_grid)
