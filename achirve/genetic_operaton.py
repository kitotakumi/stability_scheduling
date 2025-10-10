"""
各種遺伝子操作を格納するモジュール
def initialize:遺伝子操作を登録する関数
def crossover:交叉の関数
def mutation:突然変異の関数
"""

import random
import array
import copy
from tracemalloc import start
from webbrowser import get
import numpy as np

# from numpy import diff
import job_shop_scheduling
import gantt_chart_operation
from deap import base, creator, tools


def initialize_pareto(jm_table, cx, mut, sel, original_individual, fixed_gantt, reschedule_time): # fmt: skip
    # 最小化は-1.0
    creator.create("FitnessMin", base.Fitness, weights=(-1.0, -1.0))
    creator.create("Individual", list, typecode="b", fitness=creator.FitnessMin)

    toolbox = base.Toolbox()
    toolbox.register("individual", tools.initIterate, creator.Individual, lambda: create_individual_reactive(original_individual),)  # fmt: skip
    # リスケジューリング対象のジョブ集合を遺伝子で表現したものをoriginalとして登録
    toolbox.register("original_individual", tools.initIterate, creator.Individual, lambda: original_individual,)  # fmt: skip
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", objective_function_pareto, jm_table, fixed_gantt, reschedule_time)  # fmt: skip
    toolbox.register("get_max_min", get_max_min, jm_table, fixed_gantt, reschedule_time)  # fmt: skip
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
    elif sel == "Roulette":
        toolbox.register("select", tools.selRoulette)

    return toolbox

# 正規化して重みパラメータ法
def objective_function_pareto(
    jm_table, fixed_gantt, reschedule_time, individual,
    ):  # fmt: skip
    # 評価値の取得
    efficiency = makespan_reactive2(jm_table, fixed_gantt, reschedule_time, individual)
    # stability = stability_function_v3(jm_table, fixed_gantt, reschedule_time, individual)  # fmt:skip
    stability = rank_diff(jm_table, fixed_gantt, reschedule_time, individual)  # fmt:skip
    return efficiency, stability

# 安定関数用の対象問題の取得とGAの操作手法の登録
def initialize_stability(jm_table, cx, mut, sel, original_individual, fixed_gantt, reschedule_time):  # fmt: skip
    # 最小化は-1.0
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", array.array, typecode="b", fitness=creator.FitnessMin)

    toolbox = base.Toolbox()
    toolbox.register("individual", tools.initIterate, creator.Individual, lambda: create_individual_reactive(original_individual),)  # fmt: skip
    # リスケジューリング対象のジョブ集合を遺伝子で表現したものをoriginalとして登録
    toolbox.register("original_individual", tools.initIterate, creator.Individual, lambda: original_individual,)  # fmt: skip
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", objective_function_v3, jm_table, fixed_gantt, reschedule_time)  # fmt: skip
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
    elif sel == "Roulette":
        toolbox.register("select", tools.selRoulette)

    return toolbox


# ランダムな遺伝子を生成
def create_individual(job_num, machine_num):
    # 0からmachine_numまでの数がそれぞれjob_numあるリストを作成しシャッフルする
    src = list(range(job_num)) * machine_num
    random.shuffle(src)
    return src


# リスケジューリングでランダムな遺伝子を作成
def create_individual_reactive(individual):
    src = copy.deepcopy(individual)
    random.shuffle(src)
    return src


# 評価関数メイクスパンの取得。最大化にするために逆数
def makespan(jm_table, individual):
    gantt = gantt_chart_operation.get_gantt(jm_table, individual)
    # 最後の作業終了時刻からメイクスパンを取得する
    makespan = 0
    for row in gantt:
        # rowは[start, end, job_num]のリスト
        makespan = max(makespan, row[-1][1])
    return (1 / makespan,)


# リスケ用の評価関数メイクスパンの取得。最大化にするために逆数
def makespan_reactive(jm_table, fixed_gantt, reschedule_time, individual):
    gantt = gantt_chart_operation.get_gantt_reactive_natural(
        jm_table, individual, fixed_gantt, reschedule_time
    )
    # 最後の作業終了時刻からメイクスパンを取得する
    makespan = 0
    for row in gantt:
        # rowは[start, end, job_num]のリスト
        makespan = max(makespan, row[-1][1])
    return (1 / makespan,)


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


# 順位偏差
def rank_diff(jm_table, fixed_gantt, reschedule_time, individual):
    gantt = gantt_chart_operation.get_gantt_reactive_natural(
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
                    rank_diff_function += abs(rank_diff)
    return (rank_diff_function,)


def create_changed_gantt(gantt, fixed_gantt):
    changed_gantt = []
    for gantt_job, fixed_job in zip(gantt, fixed_gantt):
        # Remove elements in fixed_job from gantt_job
        changed_job = [
            operation for operation in gantt_job if operation not in fixed_job
        ]
        changed_gantt.append(changed_job)
    return changed_gantt


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


# 安定性関数（順位偏差×順位的距離）を統計情報とともに返す
def stability_function_stat(jm_table, fixed_gantt, reschedule_time, individual):
    gantt = gantt_chart_operation.get_gantt_reactive(
        jm_table, individual, fixed_gantt, reschedule_time
    )
    delayed_gantt = jm_table.delayed_gantt()
    init_changed_gantt = create_changed_gantt(delayed_gantt, fixed_gantt)
    after_changed_gantt = create_changed_gantt(gantt, fixed_gantt)
    rank_diff_function = 0
    changed_sum = 0
    changed_first = changed_secound = changed_third = changed_4th = changed_5th = 0  # fmt: skip
    changed_6th = changed_7th = changed_8th = changed_9th = changed_10th = 0  # fmt: skip
    distance_mean = 0
    distance_sum = 0
    for machine in range(len(after_changed_gantt)):
        for operation in range(len(after_changed_gantt[machine])):
            job_number = init_changed_gantt[machine][operation][2]
            for operation2 in range(len(after_changed_gantt[machine])):
                if job_number == after_changed_gantt[machine][operation2][2]:
                    rank_diff = operation - operation2
                    rank_diff_function += abs(rank_diff) ** 1 / (operation2 + 1) ** 1.25
                    if rank_diff != 0:
                        changed_sum += 1
                        distance_mean += abs(rank_diff)
                        distance_sum += abs(rank_diff)
                        if (operation2 + 1) == 1:
                            changed_first += abs(rank_diff)
                        if (operation2 + 1) == 2:
                            changed_secound += abs(rank_diff)
                        if (operation2 + 1) == 3:
                            changed_third += abs(rank_diff)
                        if (operation2 + 1) == 4:
                            changed_4th += abs(rank_diff)
                        if (operation2 + 1) == 5:
                            changed_5th += abs(rank_diff)
                        if (operation2 + 1) == 6:
                            changed_6th += abs(rank_diff)
                        if (operation2 + 1) == 7:
                            changed_7th += abs(rank_diff)
                        if (operation2 + 1) == 8:
                            changed_8th += abs(rank_diff)
                        if (operation2 + 1) == 9:
                            changed_9th += abs(rank_diff)
                        if (operation2 + 1) == 10:
                            changed_10th += abs(rank_diff)
    # safe division: changed_sum が 0 のときは mean を 0 に
    if changed_sum > 0:
        distance_mean = distance_sum / changed_sum
    else:
        distance_mean = 0
    return (rank_diff_function, changed_sum, distance_mean, distance_sum, changed_first, changed_secound, changed_third, changed_4th, changed_5th, changed_6th, changed_7th, changed_8th, changed_9th, changed_10th)  # fmt:skip


# 正規化して重みパラメータ法
def objective_function_v3(
    jm_table, fixed_gantt, reschedule_time, individual,
    max_efficiency, min_efficiency, max_stability, min_stability, statistics
    ):  # fmt: skip
    # 評価値の取得
    efficiency = makespan_reactive2(jm_table, fixed_gantt, reschedule_time, individual)
    # stability = stability_function_v3(jm_table, fixed_gantt, reschedule_time, individual)  # fmt:skip
    stability = rank_diff(jm_table, fixed_gantt, reschedule_time, individual)  # fmt:skip
    # 正規化
    norm_efficiency = 1 + (efficiency[0] - 1019) / (max_efficiency - min_efficiency)  # fmt: skip
    norm_stability = 1 + (stability[0] - 0.01) / (max_stability - min_stability)  # fmt: skip
    # 重みパラメータ法
    objective_function = 1 * norm_efficiency + 0 * norm_stability
    # print(norm_efficiency, norm_stability)
    # print(efficiency[0], stability[0])
    # print(max_efficiency, min_efficiency, max_stability, min_stability)
    # print(norm_efficiency)
    return (objective_function,)


# 世代から効率性関数と安定性関数の最小値と最大値を求める
def get_max_min(
    jm_table, fixed_gantt, reschedule_time, population, 
    max_efficiency, min_efficiency, max_stability, min_stability
    ):  # fmt:skip
    # init_gantt = jm_table.initial_gantt()
    eff, sta = [], []
    for ind in population:
        # 個体の評価値を求める
        efficiency = makespan_reactive2(jm_table, fixed_gantt, reschedule_time, ind)
        stability = rank_diff(jm_table, fixed_gantt, reschedule_time, ind)
        # 最大値と最小値を更新
        eff.append(efficiency[0])
        sta.append(stability[0])
        max_efficiency = max(efficiency[0], max_efficiency)
        min_efficiency = min(efficiency[0], min_efficiency)
        max_stability = max(stability[0], max_stability)
        min_stability = min(stability[0], min_stability)
    statistics = [np.mean(eff), np.std(eff, ddof=0), np.mean(sta), np.std(sta, ddof=0), eff, sta]  # fmt: skip
    return (max_efficiency, min_efficiency, max_stability, min_stability, statistics)


# 開始時刻偏差
def starting_time_diviation_v2(jm_table, fixed_gantt, reschedule_time, individual):
    gantt = gantt_chart_operation.get_gantt_reactive_natural(
        jm_table, individual, fixed_gantt, reschedule_time
    )
    init_gantt = jm_table.initial_gantt()
    # 初期のガントチャートと更新後のガントチャートの開始時刻の差を計算する
    starting_time_diviation_sum = 0
    reschedule_time = 423
    for machine in range(len(gantt)):
        machine_diff = []
        for operation in range(len(gantt[machine])):
            job_number = init_gantt[machine][operation][2]
            for operation2 in range(len(gantt[machine])):
                if job_number == gantt[machine][operation2][2]:
                    time_diff = (
                        init_gantt[machine][operation][0]
                        - gantt[machine][operation2][0]
                    )
                    starting_time_diviation_sum += abs(time_diff)
    return (starting_time_diviation_sum,)


# ルーレット選択
def selection_roulette(population, num):
    # 適応度のリストのタプルの1つ目の要素を抽出
    fitness_values = [(1 / ind.fitness.values[0]) for ind in population]
    max_num = max(fitness_values)
    min_num = min(fitness_values)
    fit = [(max_num + min_num - x) ** 80 for x in fitness_values]
    # print(fitness_values)
    # print(fit)
    # 適応度の合計を計算
    total_fitness = sum(fit)
    # 適応度に基づいて各個体が選択される確率を計算
    selection_probs = [x / total_fitness for x in fit]  # fmt: skip
    # print(selection_probs)
    # ルーレット選択を実行
    children = random.choices(
        population, weights=selection_probs, k=num)  # fmt: skip

    return children


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


# 初期スケジュール作成時の対象問題の取得とGAの操作手法の登録
def initialize(jsp, cx, mut, sel):
    jm_table = job_shop_scheduling.get_jm_table(jsp)
    MAX_JOBS = jm_table.get_job_count()
    MAX_MACHINES = jm_table.get_machine_count()
    # 適応度は最大化にするために逆数
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", array.array, typecode="b", fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    toolbox.register("individual", tools.initIterate, creator.Individual, lambda: create_individual(MAX_JOBS, MAX_MACHINES),)  # fmt: skip
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", makespan, jm_table)
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
    elif sel == "Roulette":
        toolbox.register("select", tools.selRoulette)

    return toolbox, jm_table


# リスケ用の対象問題の取得とGAの操作手法の登録
def initialize_reactive(jm_table, cx, mut, sel, original_individual, fixed_gantt, reschedule_time):  # fmt: skip
    # 適応度は最大化にするために逆数
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", array.array, typecode="b", fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    toolbox.register("individual", tools.initIterate, creator.Individual, lambda: create_individual_reactive(original_individual),)  # fmt: skip
    # リスケジューリング対象のジョブ集合を遺伝子で表現したものをoriginalとして登録
    toolbox.register("original_individual", tools.initIterate, creator.Individual, lambda: original_individual,)  # fmt: skip
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate",makespan_reactive, jm_table, fixed_gantt, reschedule_time)  # fmt: skip
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
    elif sel == "Roulette":
        toolbox.register("select", tools.selRoulette)

    return toolbox
