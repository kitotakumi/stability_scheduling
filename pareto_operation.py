"""
各種遺伝子操作を格納するモジュール
def initialize:遺伝子操作を登録する関数
def crossover:交叉の関数
def mutation:突然変異の関数
"""

import random
import array
import copy
import numpy as np

import gantt_chart_operation
from deap import base, creator, tools

"""
=============================== 操作の登録 ==============================================================
"""
def initialize_pareto(jm_table, cx, mut, sel, original_individual, fixed_gantt, reschedule_time): # fmt: skip
    # 最小化は-1.0
    creator.create("FitnessMin", base.Fitness, weights=(-1.0, -1.0))
    creator.create("Individual", list, typecode="b", fitness=creator.FitnessMin, weighted_fitness=float)

    toolbox = base.Toolbox()
    toolbox.register("individual", tools.initIterate, creator.Individual, lambda: create_individual_reactive(original_individual),)  # fmt: skip
    # リスケジューリング対象のジョブ集合を遺伝子で表現したものをoriginalとして登録
    toolbox.register("original_individual", tools.initIterate, creator.Individual, lambda: original_individual,)  # fmt: skip
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", objective_function_pareto, jm_table, fixed_gantt, reschedule_time)
    # 局所探索の登録
    toolbox.register("local_search", local_search_swap, jm_table, fixed_gantt, reschedule_time)  # fmt: skip
    # 交叉法の登録
    if cx == "hirano":
        toolbox.register("crossover", crossover_hirano)
    elif cx == "pmx":
        toolbox.register("crossover", crossover_pmx)
    # 突然変異法の登録
    if mut == "hirano":
        toolbox.register("mutate", mutation_hirano)
    elif mut == "inversion":
        toolbox.register("mutate", mutation_inversion)
    # 選択法の登録
    if sel == "Tournament":
        toolbox.register("select", tools.selTournament)
    elif sel == "WeightedTournament":
        toolbox.register("select", selWeightedTournament)

    return toolbox

# リスケジューリングでランダムな遺伝子を作成
def create_individual_reactive(individual):
    src = copy.deepcopy(individual)
    random.shuffle(src)
    return src

"""
=============================== 二目的最適化値の計算 ==============================================================
"""
# 正規化した効率値と安定値を返す。メイン関数で使用。
def objective_function_pareto(
    jm_table, fixed_gantt, reschedule_time, individual,
    ):  # fmt: skip
    # 評価値の取得
    efficiency = makespan_reactive2(jm_table, fixed_gantt, reschedule_time, individual)[0]
    # stability = stability_function_v3(jm_table, fixed_gantt, reschedule_time, individual)  # fmt:skip
    stability = stability_function_v3(jm_table, fixed_gantt, reschedule_time, individual)[0]  # fmt:skip
    return efficiency, stability

# 染色体を引数に、正規化して重み付き評価値を計算する
# 主に局所探索で使用
def weight_function(
    jm_table, fixed_gantt, reschedule_time, individual, w,
    max_efficiency, min_efficiency, max_stability
    ):  # fmt: skip
    # 評価値の取得
    efficiency = makespan_reactive2(jm_table, fixed_gantt, reschedule_time, individual)
    # stability = stability_function_v3(jm_table, fixed_gantt, reschedule_time, individual)  # fmt:skip
    stability = stability_function_v3(jm_table, fixed_gantt, reschedule_time, individual)
    # 評価値が0になり片方の目的が無視されるのを防ぐため、意図的に[1, 2]の範囲に正規化する
    norm_efficiency = 1 + (efficiency[0] - min_efficiency) / (max_efficiency - min_efficiency)  # fmt: skip
    # stabilityの最小値は常に0であるため、考慮しない
    norm_stability = 1 + (stability[0]) / (max_stability)  # fmt: skip
    # 重みパラメータ法
    objective_function = w[0] * norm_efficiency + w[1] * norm_stability
    return objective_function

# 計算済みのフィットネス値を引数に、正規化された重み付き評価値を計算する。
# 主にメイン関数でバケットを評価値でソートするために使用
def calculate_weighted_fitness(fitness_values, w, max_efficiency, min_efficiency, max_stability):
    efficiency, stability = fitness_values
    # 意図的な[1, 2]への正規化
    norm_efficiency = 1 + (efficiency - min_efficiency) / (max_efficiency - min_efficiency)
    # 安定性はminが0であるため、maxのみで正規化
    norm_stability = 1 + stability / max_stability
    
    return w[0] * norm_efficiency + w[1] * norm_stability


"""
=============================== 効率性, 安定性の評価 ==============================================================
"""
# リスケ用の評価関数メイクスパンの取得。
def makespan_reactive2(jm_table, fixed_gantt, reschedule_time, individual):
    gantt = gantt_chart_operation.get_gantt_reactive(
        jm_table, individual, fixed_gantt, reschedule_time
    )
    # 最後の作業終了時刻からメイクスパンを取得する
    makespan = 0
    for row in gantt:
        # rowは[start, end, job_num]のリスト
        makespan = max(makespan, row[-1][1])
    return (makespan,)


# 安定性関数ver3（順位偏差×順位的距離）
def stability_function_v3(jm_table, fixed_gantt, reschedule_time, individual):
    gantt = gantt_chart_operation.get_gantt_reactive(
        jm_table, individual, fixed_gantt, reschedule_time
    )
    delayed_gantt = jm_table.delayed_gantt()
    init_changed_gantt = create_changed_gantt(delayed_gantt, fixed_gantt)
    after_changed_gantt = create_changed_gantt(gantt, fixed_gantt)
    rank_diff_function = 0
    for machine in range(len(after_changed_gantt)):
        for operation in range(len(after_changed_gantt[machine])):
            job_number = init_changed_gantt[machine][operation][2]
            for operation2 in range(len(after_changed_gantt[machine])):
                if job_number == after_changed_gantt[machine][operation2][2]:
                    rank_diff = operation - operation2
                    rank_diff_function += abs(rank_diff) ** 1 / (operation2 + 1) ** 1.25
    return (rank_diff_function,)

# すでに完了しているジョブを除外してリスケ対象のジョブを抽出する
# 安定性関数内の計算のみで使用
def create_changed_gantt(gantt, fixed_gantt):
    changed_gantt = []
    for gantt_job, fixed_job in zip(gantt, fixed_gantt):
        # Remove elements in fixed_job from gantt_job
        changed_job = [
            operation for operation in gantt_job if operation not in fixed_job
        ]
        changed_gantt.append(changed_job)
    return changed_gantt

# 世代から効率性関数と安定性関数の最小値と最大値を求める
def get_max_min(
    jm_table, fixed_gantt, reschedule_time, population, 
    max_efficiency, min_efficiency, max_stability
    ):  # fmt:skip
    # init_gantt = jm_table.initial_gantt()
    eff, sta = [], []
    for ind in population:
        # 個体の評価値を求める
        efficiency = makespan_reactive2(jm_table, fixed_gantt, reschedule_time, ind)
        stability = stability_function_v3(jm_table, fixed_gantt, reschedule_time, ind)
        # 最大値と最小値を更新
        eff.append(efficiency[0])
        sta.append(stability[0])
        max_efficiency = max(efficiency[0], max_efficiency)
        min_efficiency = min(efficiency[0], min_efficiency)
        max_stability = max(stability[0], max_stability)
    statistics = [np.mean(eff), np.std(eff, ddof=0), np.mean(sta), np.std(sta, ddof=0), eff, sta]  # fmt: skip
    return (max_efficiency, min_efficiency, max_stability, statistics)
"""
=============================== 選択 ==============================================================
"""
# 反応型スケジューリング用の重み付きトーナメント選択
def selWeightedTournament(individuals, k, tournsize):
    chosen = []
    for i in range(k):
        # トーナメント出場者をランダムに選択
        aspirants = tools.selRandom(individuals, tournsize)
        
        # トーナメント出場者の中で、重み付け評価値が最も良い個体を選ぶ
        # calculate_weighted_fitnessは最小化問題なので、min() を使用
        winner = min(aspirants, key=lambda ind: ind.weighted_fitness)
        chosen.append(winner)
    
    return chosen


"""
=============================== 局所探索 ==============================================================
"""
# 隣接する染色体を一箇所入れ替える局所探索
def local_search_swap(jm_table, fixed_gantt, reschedule_time, indiv, w, max_ms, min_ms, max_st, iters=10):
    # 深いコピーして best を保持
    best = indiv[:]
    # 初期評価
    best_val = weight_function(
        jm_table, fixed_gantt, reschedule_time, best, w, max_ms, min_ms, max_st
    )
    for _ in range(iters):
        cand = best[:]
        i = random.randrange(len(cand)-1)   # 0 ～ n-2 の間
        j = i + 1                           # 常に隣接
        cand[i], cand[j] = cand[j], cand[i]
        # 評価
        fval = weight_function(
            jm_table, fixed_gantt, reschedule_time, cand, w, max_ms, min_ms, max_st
        )
        # 改善なら採用
        if fval < best_val:
            best, best_val = cand, fval

    return best


"""
=============================== 交叉 ==============================================================
"""
# PMX交叉
def crossover_pmx(parent1, parent2):
    size = len(parent1)
    # ランダムに2つの交叉点を選ぶ
    cx_point1 = random.randint(0, size - 1)
    cx_point2 = random.randint(0, size - 1)
    # 交叉点をソートして範囲を確定
    if cx_point1 > cx_point2:
        cx_point1, cx_point2 = cx_point2, cx_point1
    # 交叉範囲内で入れ替えを行う
    for ind in range(cx_point1, cx_point2 + 1):
        if parent1[ind] != parent2[ind]:
            x, y = parent1[ind], parent2[ind]
            # parent1のindの位置の遺伝子の入れ替え相手をランダムに探索
            ran = ind
            while parent1[ran] != y:
                ran = random.randint(0, size - 1)
            parent1[ind], parent1[ran] = parent1[ran], parent1[ind]
            # parent2のindの位置の遺伝子の入れ替え相手をランダムに探索
            ran = ind
            while parent2[ran] != x:
                ran = random.randint(0, size - 1)
            parent2[ind], parent2[ran] = parent2[ran], parent2[ind]

    return parent1, parent2


# JSP用の2点交叉処理
def crossover_hirano(ind1, ind2):
    # ind1, ind2の長さは同じ
    size = len(ind1)
    cxpoint1, cxpoint2 = 0, 0
    while cxpoint1 == cxpoint2:
        cxpoint1 = random.randrange(0, size)
        cxpoint2 = random.randrange(0, size)
    # 2点目が1点目より前なら入れ替える
    if cxpoint2 < cxpoint1:
        cxpoint1, cxpoint2 = cxpoint2, cxpoint1
    # 部分遺伝子を取得
    sub1 = array.array(ind1.typecode, ind1[cxpoint1:cxpoint2])
    sub2 = array.array(ind2.typecode, ind2[cxpoint1:cxpoint2])
    new_sub1 = array.array(ind1.typecode, [-1] * (cxpoint2 - cxpoint1))
    new_sub2 = array.array(ind2.typecode, [-1] * (cxpoint2 - cxpoint1))
    for s1_idx, s1 in enumerate(ind1[cxpoint1:cxpoint2]):
        # ind1の部分遺伝子の要素がind2の部分遺伝子にみつからない
        if s1 not in sub2:
            continue
        # みつかったら相手のnew_subに位置を保存してコピー
        s2_idx = sub2.index(s1)
        new_sub1[s2_idx] = s1
        new_sub2[s1_idx] = s1
        # コピーした要素は-1にしておく
        sub1[s1_idx] = -1
        sub2[s2_idx] = -1
    # コピーしなかった要素を順序を保存して戻す
    for s1 in sub1:
        # -1の場合コピー済み
        if s1 == -1:
            continue
        # new_sub1の未コピー位置を取得
        idx = new_sub1.index(-1)
        new_sub1[idx] = s1
    for s2 in sub2:
        if s2 == -1:
            continue
        idx = new_sub2.index(-1)
        new_sub2[idx] = s2
    # 部分遺伝子を個体にセット
    ind1[cxpoint1:cxpoint2] = new_sub1
    ind2[cxpoint1:cxpoint2] = new_sub2
    return ind1, ind2


"""
=============================== 突然変異 ==============================================================
"""
# 逆位
def mutation_inversion(ind):
    # gene = copy.deepcopy(individual)
    # ランダムな2点を選ぶ
    point1 = random.randint(0, len(ind) - 1)
    point2 = random.randint(0, len(ind) - 1)

    while point1 == point2:
        point2 = random.randint(0, len(ind) - 1)

    # 2点の順序を確保する
    if point1 > point2:
        point1, point2 = point2, point1

    # 部分配列を反転させる
    ind[point1 : point2 + 1] = ind[point1 : point2 + 1][::-1]

    return ind


# JSP用の突然変異処理
def mutation_hirano(ind):
    size = len(ind)
    # 理由は不明だがオリジナルソースでは2回実施している
    for _ in range(2):
        pos1 = random.randrange(0, size)
        pos2 = random.randrange(0, size)
        ind[pos1], ind[pos2] = ind[pos2], ind[pos1]
    return ind