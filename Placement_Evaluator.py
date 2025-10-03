import argparse
import os
import sys
import math
import numpy as np

# -----------------------------
# 基本資料結構
# -----------------------------
class UnitCapacitor:
    def __init__(self, bit_index: int, size: float = 1.0):
        # 0 = B0 (dummy), 1..N = 實際位元
        self.bit_index = bit_index
        self.size = size

# -----------------------------
# 評估器
# -----------------------------
class PlacementEvaluator:
    def __init__(
        self,
        N_bits: int,
        rho_u: float = 0.9,
        l: float = 0.5,
        gradient_x: float = 0.005,
        gradient_y: float = 0.002,
        unit_cap_size: float = 1.0,
    ):
        self.N_bits = N_bits
        self.rho_u = rho_u
        self.l = l
        self.gradient_x = gradient_x
        self.gradient_y = gradient_y
        self.unit_cap_size = unit_cap_size
        self.placement_map = None
        self.rows = 0
        self.cols = 0

    def load_from_label_grid(self, label_grid):
        """label_grid: 2D list/ndarray，值域 0..N（0=B0, 其餘為 B1..BN）"""
        arr = np.array(label_grid, dtype=int)
        rows, cols = arr.shape
        placement = np.full((rows, cols), None, dtype=object)
        for r in range(rows):
            for c in range(cols):
                bit_idx = int(arr[r, c])
                placement[r, c] = UnitCapacitor(bit_idx, self.unit_cap_size)
        self.placement_map = placement
        self.rows, self.cols = rows, cols
        return placement

    def _get_unit_locations(self):
        locs = {bit: [] for bit in range(0, self.N_bits + 1)}
        for r in range(self.rows):
            for c in range(self.cols):
                u = self.placement_map[r, c]
                if u is not None:
                    locs[u.bit_index].append((r, c))
        return locs

    def calculate_mismatch(self):
        """
        Equation (4):
        M = max_{i,j} | ( (Ci*/Cj*) / (Ci/Cj) - 1 ) | * 100%
        僅對 1..N_bits 計算（忽略 B0）
        """
        locs = self._get_unit_locations()
        center_r = (self.rows - 1) / 2.0
        center_c = (self.cols - 1) / 2.0

        C_ideal = {}
        C_star = {}
        for bit in range(1, self.N_bits + 1):
            C_ideal[bit] = (2 ** (bit - 1)) * self.unit_cap_size
            s = 0.0
            for (r, c) in locs.get(bit, []):
                dev = 1 + self.gradient_x * (c - center_c) + self.gradient_y * (r - center_r)
                s += self.unit_cap_size * dev
            C_star[bit] = s

        max_m = 0.0
        for i in range(1, self.N_bits + 1):
            for j in range(1, self.N_bits + 1):
                if i == j:
                    continue
                ideal_ratio = C_ideal[i] / C_ideal[j]
                star_ratio = C_star[i] / C_star[j]
                m = abs(star_ratio / ideal_ratio - 1.0) * 100.0
                if m > max_m:
                    max_m = m
        return max_m

    def calculate_correlation(self):
        """
        Equation (5)(6)(7) 實作：
        - ρ_ab = ρ_u ^ ( ((ra-rb)^2 + (sa-sb)^2) * l )
        - ρ_ij = 1/(μν) * Σ Σ ρ_ab (+ cross terms，目前只算 Ci×Cj 部分)
        - Overall ρ = Σ_i Σ_j ρ_ij
        """
        coords = [(r, c) for r in range(self.rows) for c in range(self.cols)
                if self.placement_map[r, c] is not None]

        # 預計算所有單元間的 ρ_ab
        rho_ab = {}
        for i in range(len(coords)):
            ra, ca = coords[i]
            for j in range(i, len(coords)):
                rb, cb = coords[j]
                dr = ra - rb
                ds = ca - cb
                # ★ 按照 paper 的 Eq.(5) (平方距離，不取根號)
                dist_term = (dr * dr + ds * ds) * self.l
                val = self.rho_u ** dist_term
                rho_ab[(ra, ca, rb, cb)] = val
                rho_ab[(rb, cb, ra, ca)] = val

        # 計算 ρ_ij
        locs = self._get_unit_locations()
        rho_ij_vals = []
        for i in range(1, self.N_bits + 1):
            for j in range(i + 1, self.N_bits + 1):
                ui = locs.get(i, [])
                uj = locs.get(j, [])
                if not ui or not uj:
                    continue
                s = 0.0
                for (ra, ca) in ui:
                    for (rb, cb) in uj:
                        s += rho_ab.get((ra, ca, rb, cb), 0.0)
                rho_ij = s / (len(ui) * len(uj))
                rho_ij_vals.append(rho_ij)

        # Overall ρ
        return sum(rho_ij_vals)


    def calculate_trunk_wire_metric(self):
        """
        Trunk wire cost 以 HPWL 估算（每個位元群的包絡盒寬高和）
        僅對 1..N_bits 計算（忽略 B0）
        """
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

# -----------------------------
# 輔助：讀 txt -> 2D int 矩陣
# -----------------------------
def read_label_grid_from_txt(path: str):
    """
    讀取以空白分隔的數字矩陣 txt。
    允許多個空白、前後空白；忽略完全空白行。
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"檔案不存在: {path}")

    grid = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            try:
                row = [int(x) for x in parts]
            except ValueError as e:
                raise ValueError(f"行內容無法轉為整數：{line}") from e
            grid.append(row)

    if not grid:
        raise ValueError("檔案為空或沒有有效數據行。")

    # 檢查是否為矩形
    width = len(grid[0])
    for idx, r in enumerate(grid):
        if len(r) != width:
            raise ValueError(f"第 {idx+1} 行長度 {len(r)} 與第一行 {width} 不同，非矩形。")

    return grid  # list[list[int]]

# -----------------------------
# 主程式
# -----------------------------
def main():
    parser = argparse.ArgumentParser(
        description="從 txt 讀取 DAC 版圖標籤矩陣（0..N），並計算 Mismatch / Overall Rho / Trunk Wire。"
    )
    parser.add_argument("txt_path", help="版圖矩陣 txt 檔路徑（空白分隔整數）")
    parser.add_argument("--nbits", type=int, default=None, help="DAC 位元數（若未指定，將以檔案最大值推斷）")
    parser.add_argument("--rho_u", type=float, default=0.9, help="單元基礎相關係數 rho_u")
    parser.add_argument("--l", type=float, default=0.5, help="距離衰減因子 l")
    parser.add_argument("--gx", type=float, default=0.005, help="X 向線性梯度（每單位距離百分比）")
    parser.add_argument("--gy", type=float, default=0.002, help="Y 向線性梯度（每單位距離百分比）")
    parser.add_argument("--unit", type=float, default=1.0, help="單位電容名目值（僅用於比值，預設 1.0）")
    parser.add_argument("--print-grid", action="store_true", help="印出載入的標籤矩陣")
    args = parser.parse_args()

    try:
        label_grid = read_label_grid_from_txt(args.txt_path)
    except Exception as e:
        print(f"[讀檔錯誤] {e}", file=sys.stderr)
        sys.exit(1)

    arr = np.array(label_grid, dtype=int)
    max_label = int(arr.max())
    if args.nbits is None:
        N_bits = max_label
    else:
        N_bits = args.nbits

    # 合法性檢查：元素需在 0..N_bits
    if arr.min() < 0:
        print("[錯誤] 讀到負數標籤，請確認檔案內容。", file=sys.stderr)
        sys.exit(1)
    if arr.max() > N_bits:
        print(f"[錯誤] 檔案中存在標籤 {arr.max()}，超過 N_bits={N_bits}。請修正檔案或用 --nbits 指定更大的位元數。", file=sys.stderr)
        sys.exit(1)

    if args.print_grid:
        print("\n=== 讀入的標籤矩陣 (0=B0, 1..N=位元) ===")
        for row in label_grid:
            print(" ".join(str(x) for x in row))

    evaluator = PlacementEvaluator(
        N_bits=N_bits,
        rho_u=args.rho_u,
        l=args.l,
        gradient_x=args.gx,
        gradient_y=args.gy,
        unit_cap_size=args.unit,
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
    print(f"Mismatch (M, %)  : {M:.6f}")
    print(f"Overall Rho      : {RHO:.6f}")
    print(f"Trunk Wire (HPWL): {WIRE:.6f}")

if __name__ == "__main__":
    main() #python Placement_Evaluator.py ".\my_placement.txt" --print-grid

