import random
import math
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy

# --- 1. 基本資料結構與輔助類別 ---
class Individual:
    """
    代表一個完整的佈局方案，是遺傳演算法中的一個「個體」。
    """
    def __init__(self, unit_size, placement_matrix):
        self.unit_size = unit_size  # 單位電容尺寸 (fF)
        self.placement_matrix = placement_matrix  # N x N 的 NumPy 陣列

    def __repr__(self):
        matrix_shape = self.placement_matrix.shape
        return f"Individual(C_unit={self.unit_size:.2f} fF, Layout={matrix_shape})"

# --- 2. Placement & Routing 流程 ---

def place_common_centroid_binary_array(n_bits, matrix_size):
    """
    生成一個二進位加權電容陣列的共心佈局。
    -1 代表空位。
    數字 k 代表這個單元屬於 bit k。
    """
    # 計算每個 bit 需要的單位電容數量
    units_needed = [2**i for i in range(n_bits)]
    total_units = sum(units_needed)
    
    if total_units > matrix_size * matrix_size:
        raise ValueError("Matrix size is too small for the given number of bits.")

    # 建立所有單元的列表 (e.g., [7, 7, ..., 6, 6, ..., 0])
    unit_list = []
    for bit, count in enumerate(units_needed):
        unit_list.extend([bit] * count)
    
    # 隨機打亂以實現分散性
    random.shuffle(unit_list)
    
    # 填滿剩餘空間為空位
    unit_list.extend([-1] * (matrix_size * matrix_size - total_units))
    
    # 以螺旋方式從中心向外放置，實現共心和對稱
    matrix = np.full((matrix_size, matrix_size), -1, dtype=int)
    x, y = matrix_size // 2, matrix_size // 2
    dx, dy = 0, -1
    
    for i in range(matrix_size**2):
        if (-matrix_size/2 < x <= matrix_size/2) and (-matrix_size/2 < y <= matrix_size/2):
            if matrix[y, x] == -1 and unit_list:
                matrix[y, x] = unit_list.pop(0)

        if x == y or (x < 0 and x == -y) or (x > 0 and x == 1-y):
            dx, dy = -dy, dx # 轉彎
        x, y = x + dx, y + dy
        
    return matrix

def build_routing_patterns_from_matrix(matrix):
    """
    根據 2D 佈局矩陣，掃描相鄰單元對以生成路由模式。
    這是一個簡化的模型，實際繞線更複雜。
    - 'O' (Overlapped): 相鄰且屬於同一個 bit。
    - 'N' (Non-overlapped): 相鄰但屬於不同 bit。
    - 'S' (Single): 旁邊是空位或邊界。
    """
    rows, cols = matrix.shape
    patterns = []
    # 橫向掃描
    for r in range(rows):
        for c in range(cols - 1):
            cell1 = matrix[r, c]
            cell2 = matrix[r, c+1]
            if cell1 == -1 or cell2 == -1:
                patterns.append('S')
            elif cell1 == cell2:
                patterns.append('O')
            else:
                patterns.append('N')
    # 縱向掃描
    for c in range(cols):
        for r in range(rows - 1):
            cell1 = matrix[r, c]
            cell2 = matrix[r+1, c]
            if cell1 == -1 or cell2 == -1:
                patterns.append('S')
            elif cell1 == cell2:
                patterns.append('O')
            else:
                patterns.append('N')
    return patterns

# --- 3. 物理模型與公式實作 ---

def calculate_parasitics_from_patterns(patterns, tech_params):
    """
    將 O/N/S 模式列表轉換為寄生電容值列表。
    """
    c_area = tech_params['c_area']
    c_fringe = tech_params['c_fringe']
    
    parasitics = []
    for p in patterns:
        if p == 'O':
            parasitics.append(c_area + c_fringe)
        elif p == 'N':
            parasitics.append(c_fringe)
        elif p == 'S':
            parasitics.append(c_area + c_fringe) # 假設單線也有面積和邊緣效應
    return parasitics

def simulate_dac_output_binary(individual, tech_params):
    """
    模擬二進位加權 DAC 的輸出電壓，包含寄生電容。
    """
    n_bits = tech_params['n_bits']
    v_ref = tech_params['v_ref']
    c_unit = individual.unit_size
    matrix = individual.placement_matrix
    
    # 1. 建立每個 bit 的單位電容數量映射
    bit_counts = {i: np.count_nonzero(matrix == i) for i in range(n_bits)}
    
    # 2. 簡化寄生模型：假設每個單元平均分配到的寄生電容
    patterns = build_routing_patterns_from_matrix(matrix)
    parasitics = calculate_parasitics_from_patterns(patterns, tech_params)
    total_parasitic_cap = sum(parasitics)
    total_units = sum(bit_counts.values())
    avg_parasitic_per_unit = total_parasitic_cap / total_units if total_units > 0 else 0

    # 假設 C_TS 與總面積成正比
    c_ts = tech_params['c_ts_factor'] * total_units * c_unit
    
    # 3. 遍歷所有數位碼計算 V_OUT
    v_out_all_codes = np.zeros(2**n_bits)
    
    for code in range(2**n_bits):
        c_on = 0
        c_off = 0
        
        for bit in range(n_bits):
            num_units_for_bit = bit_counts[bit]
            if (code >> bit) & 1: # 檢查 bit 是否為 1
                c_on += num_units_for_bit * c_unit
            else:
                c_off += num_units_for_bit * c_unit

        # 分配寄生電容到 ON 和 OFF 組
        num_on_units = c_on / c_unit if c_unit > 0 else 0
        num_off_units = c_off / c_unit if c_unit > 0 else 0
        
        c_tb_on = num_on_units * avg_parasitic_per_unit
        c_tb_off = num_off_units * avg_parasitic_per_unit
        
        denominator = c_on + c_off + c_tb_on + c_tb_off + c_ts
        if denominator == 0:
            v_out_all_codes[code] = 0
        else:
            numerator = c_on + c_tb_on
            v_out_all_codes[code] = v_ref * (numerator / denominator)
            
    return v_out_all_codes

def calculate_dnl_inl(v_out, tech_params):
    """根據 DAC 輸出電壓計算 DNL 和 INL (與 V1 相同)。"""
    n_bits = tech_params['n_bits']
    v_ref = tech_params['v_ref']
    v_lsb = v_ref / (2**n_bits)
    if v_lsb == 0: return 0, 0

    # DNL
    dnl = [(v_out[i+1] - v_out[i]) / v_lsb - 1 for i in range(2**n_bits - 1)]
    
    # INL
    v_out_ideal = np.arange(2**n_bits) * v_lsb
    inl = (v_out - v_out_ideal) / v_lsb
    
    max_abs_dnl = np.max(np.abs(dnl)) if dnl else 0
    max_abs_inl = np.max(np.abs(inl)) if len(inl) > 0 else 0
    
    return max_abs_dnl, max_abs_inl

# --- 4. Genetic Algorithm 核心 ---

def calculate_fitness(individual, tech_params):
    """適應度函數。"""
    # 模擬 DNL/INL
    v_out = simulate_dac_output_binary(individual, tech_params)
    dnl, inl = calculate_dnl_inl(v_out, tech_params)

    # 懲罰項
    penalty_dnl = max(0, dnl - 0.5) * tech_params['beta']
    penalty_inl = max(0, inl - 0.5) * tech_params['gamma']
    
    # C_unit 正規化成本 (面積成本)
    cost_cunit = (individual.unit_size / tech_params['c_unit_max']) * tech_params['alpha']
    
    total_cost = cost_cunit + penalty_dnl + penalty_inl
    return 1 / (total_cost + 1e-9)

# --- 5. 遺傳演算法運算子與主流程 ---

def initialize_population(pop_size, n_bits, matrix_size, tech_params):
    """初始化族群。"""
    population = []
    for _ in range(pop_size):
        unit_size = random.uniform(tech_params['c_unit_min'], tech_params['c_unit_max'])
        matrix = place_common_centroid_binary_array(n_bits, matrix_size)
        population.append(Individual(unit_size, matrix))
    return population

def selection(population, fitnesses):
    """錦標賽選擇。"""
    selected = []
    tournament_size = 3
    for _ in range(len(population)):
        contenders_indices = random.sample(range(len(population)), tournament_size)
        winner_index = max(contenders_indices, key=lambda i: fitnesses[i])
        selected.append(population[winner_index])
    return selected

def crossover(parent1, parent2):
    """
    對佈局矩陣進行均勻交配 (Uniform Crossover)。
    對 unit_size 進行平均。
    """
    matrix1 = parent1.placement_matrix
    matrix2 = parent2.placement_matrix
    rows, cols = matrix1.shape
    
    child1_matrix = np.copy(matrix1)
    child2_matrix = np.copy(matrix2)
    
    for r in range(rows):
        for c in range(cols):
            if random.random() < 0.5:
                # 交換此位置的電容
                child1_matrix[r, c], child2_matrix[r, c] = child2_matrix[r, c], child1_matrix[r, c]

    # TODO: 確保交配後的子代矩陣中各 bit 電容數量仍然正確。
    # 為簡化，此處暫不校正，但在實際應用中此為必要步驟。
    
    child_unit_size = (parent1.unit_size + parent2.unit_size) / 2
    
    child1 = Individual(child_unit_size, child1_matrix)
    child2 = Individual(child_unit_size, child2_matrix)
    
    return child1, child2
    
def mutation(individual, mutation_rate, tech_params):
    """突變：隨機交換兩個電容位置，或改變 unit_size。"""
    mutated_ind = deepcopy(individual)
    
    # 突變 placement_matrix: 交換兩個非空位的位置
    if random.random() < mutation_rate:
        rows, cols = mutated_ind.placement_matrix.shape
        non_empty_cells = np.argwhere(mutated_ind.placement_matrix != -1)
        if len(non_empty_cells) >= 2:
            idx1, idx2 = random.sample(range(len(non_empty_cells)), 2)
            r1, c1 = non_empty_cells[idx1]
            r2, c2 = non_empty_cells[idx2]
            
            val1 = mutated_ind.placement_matrix[r1, c1]
            val2 = mutated_ind.placement_matrix[r2, c2]
            mutated_ind.placement_matrix[r1, c1] = val2
            mutated_ind.placement_matrix[r2, c2] = val1

    # 突變 unit_size
    if random.random() < mutation_rate:
         mutated_ind.unit_size = random.uniform(tech_params['c_unit_min'], tech_params['c_unit_max'])

    return mutated_ind

# Inversion 在 2D 矩陣上較不直觀，此處省略以簡化模型。

def shielding_ilp(individual):
    """
    (佔位函數) ILP 屏蔽線分配。
    真實實現中，此函數會：
    1. 根據 individual 的佈局和路由模式，建立一個寄生網路模型。
    2. 定義一個 ILP 問題：
       - 變數: 每條潛在的屏蔽線是否被啟用 (0 或 1)。
       - 目標函數: 最小化 DNL/INL 相關的寄生失配，或最小化屏蔽線總長度。
       - 約束: 佈線資源限制、時序要求等。
    3. 呼叫 ILP 求解器 (如 PuLP, Gurobi)。
    4. 返回帶有屏蔽線分配的優化後佈局。
    """
    # print("Skipping ILP for shielding assignment in this version.")
    pass
    return individual # 目前直接返回原樣

def run_genetic_algorithm(tech_params):
    """GA 主流程。"""
    pop_size = tech_params['population_size']
    n_gens = tech_params['max_generations']
    n_bits = tech_params['n_bits']
    matrix_size = tech_params['matrix_size']
    elitism_count = tech_params['elitism_count']
    mutation_rate = tech_params['mutation_rate']

    # 初始化
    population = initialize_population(pop_size, n_bits, matrix_size, tech_params)
    best_fitness_history = []
    best_individual_ever = None
    best_fitness_ever = -1

    print("--- 開始執行遺傳演算法 (二進位加權 DAC) ---")
    for generation in range(n_gens):
        fitnesses = [calculate_fitness(ind, tech_params) for ind in population]
        
        current_best_idx = np.argmax(fitnesses)
        if fitnesses[current_best_idx] > best_fitness_ever:
            best_fitness_ever = fitnesses[current_best_idx]
            best_individual_ever = deepcopy(population[current_best_idx])
            
        best_fitness_history.append(best_fitness_ever)

        if (generation + 1) % 10 == 0:
            print(f"世代 {generation+1}/{n_gens} - 最佳適應度: {best_fitness_ever:.4f}")

        # 產生下一代
        new_population = []
        
        # 精英主義
        sorted_indices = np.argsort(fitnesses)[::-1]
        for i in range(elitism_count):
            new_population.append(population[sorted_indices[i]])

        # 選擇、交配、突變
        selected = selection(population, fitnesses)
        
        while len(new_population) < pop_size:
            p1, p2 = random.sample(selected, 2)
            c1, c2 = crossover(p1, p2)
            c1 = mutation(c1, mutation_rate, tech_params)
            c2 = mutation(c2, mutation_rate, tech_params)
            
            # (可選) 執行屏蔽優化
            c1 = shielding_ilp(c1)
            c2 = shielding_ilp(c2)
            
            new_population.extend([c1, c2])
        
        population = new_population[:pop_size]

    print("--- 演算法結束 ---")
    return best_individual_ever, best_fitness_history

# --- 6. 結果分析與視覺化 ---

def analyze_and_report_results(best_individual, history, tech_params):
    """對最佳結果進行分析、報告並視覺化。"""
    print("\n--- 最佳化結果報告 ---")
    
    v_out = simulate_dac_output_binary(best_individual, tech_params)
    dnl, inl = calculate_dnl_inl(v_out, tech_params)
    fitness = calculate_fitness(best_individual, tech_params)
    
    print("\n性能指標:")
    print(f"  - 最終適應度: {fitness:.6f}")
    print(f"  - 單位電容 (C_unit): {best_individual.unit_size:.4f} fF")
    print(f"  - 最大 DNL: {dnl:.4f} LSB")
    print(f"  - 最大 INL: {inl:.4f} LSB")

    # 繪製適應度演化圖
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history)
    plt.title("Fitness Evolution over Generations")
    plt.xlabel("Generation")
    plt.ylabel("Best Fitness")
    plt.grid(True)

    # 繪製最佳佈局圖
    plt.subplot(1, 2, 2)
    matrix = best_individual.placement_matrix
    # 使用 -1 作為空位的顏色
    cmap = plt.cm.get_cmap('viridis', tech_params['n_bits'] + 1)
    plt.imshow(matrix, cmap=cmap, interpolation='nearest')
    plt.colorbar(ticks=range(-1, tech_params['n_bits']), label="Bit assignment (-1 for empty)")
    plt.title("Best Common-Centroid Placement")
    plt.xlabel("X coordinate")
    plt.ylabel("Y coordinate")
    
    plt.tight_layout()
    plt.show()

# --- 主程式執行區塊 ---
if __name__ == "__main__":
    
    N_BITS = 6 # 位元數 (8-bit 會需要 255 個單元, 矩陣至少 16x16, 計算較慢)
    MATRIX_SIDE = 10 # 矩陣邊長 (10x10=100 cells, for 6-bit DAC we need 2^6-1=63 cells)

    TECH_PARAMETERS = {
        # DAC & 物理參數
        'n_bits': N_BITS,
        'v_ref': 1.0,               # V
        'c_area': 0.1,              # fF
        'c_fringe': 0.05,           # fF
        'c_ts_factor': 0.02,        # C_TS = factor * C_total
        'c_unit_min': 0.5,          # fF
        'c_unit_max': 5.0,          # fF
        
        # 適應度函數權重
        'alpha': 1.0,
        'beta': 100.0,
        'gamma': 100.0,
        
        # GA 演算法參數
        'matrix_size': MATRIX_SIDE,
        'population_size': 50,
        'max_generations': 100,
        'mutation_rate': 0.1,
        'elitism_count': 3,
    }
    
    best_solution, fitness_history = run_genetic_algorithm(TECH_PARAMETERS)
    
    if best_solution:
        analyze_and_report_results(best_solution, fitness_history, TECH_PARAMETERS)
    else:
        print("未能找到有效解。")