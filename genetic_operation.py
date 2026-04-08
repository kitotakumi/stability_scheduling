"""
遺伝的アルゴリズムの操作モジュール（リスケジューリング用）

交叉・突然変異・選択・個体生成・評価関数の登録を行う。
評価関数は evaluation.py の共通関数を使用する。
"""

import random
import array
import copy
import numpy as np

import job_shop_scheduling
import gantt_chart_operation
import evaluation


# ========== 個体生成 ==========

def create_individual_reactive(individual):
    """リスケジューリング用のランダム個体を生成"""
    src = copy.deepcopy(individual)
    random.shuffle(src)
    return src


# ========== 評価関数 ==========

def compute_makespan(jm_table, fixed_gantt, reschedule_time, individual):
    """個体のメイクスパンを計算"""
    gantt = gantt_chart_operation.get_gantt_reactive(
        jm_table, individual, fixed_gantt, reschedule_time)
    return evaluation.compute_makespan_from_gantt(gantt)


def compute_stability(jm_table, fixed_gantt, reschedule_time, individual):
    """個体の安定性を計算（evaluation.pyの共通関数を使用）"""
    gantt = gantt_chart_operation.get_gantt_reactive(
        jm_table, individual, fixed_gantt, reschedule_time)
    delayed_gantt = jm_table.delayed_gantt()
    return evaluation.compute_stability_from_gantt(delayed_gantt, gantt, fixed_gantt)


def compute_stability_stat(jm_table, fixed_gantt, reschedule_time, individual):
    """安定性の詳細統計を返す"""
    gantt = gantt_chart_operation.get_gantt_reactive(
        jm_table, individual, fixed_gantt, reschedule_time)
    delayed_gantt = jm_table.delayed_gantt()
    return evaluation.compute_stability_stat(delayed_gantt, gantt, fixed_gantt)


def objective_function(jm_table, fixed_gantt, reschedule_time, individual,
                       weights, norm_params):
    """重み付き正規化目的関数

    Parameters:
        weights: [効率性の重み, 安定性の重み]
        norm_params: dict with 'min_eff', 'max_eff', 'max_stab'
    """
    makespan = compute_makespan(jm_table, fixed_gantt, reschedule_time, individual)
    stability = compute_stability(jm_table, fixed_gantt, reschedule_time, individual)
    score = evaluation.weighted_objective(makespan, stability, weights, norm_params)
    return (score,)


# ========== 正規化パラメータ推定 ==========

def estimate_normalization_params(jm_table, fixed_gantt, reschedule_time, population):
    """集団から正規化パラメータを推定"""
    eff_list, stab_list = [], []
    for ind in population:
        ms = compute_makespan(jm_table, fixed_gantt, reschedule_time, ind)
        st = compute_stability(jm_table, fixed_gantt, reschedule_time, ind)
        eff_list.append(ms)
        stab_list.append(st)

    eff_list.sort()
    p90_idx = int(len(eff_list) * 0.9)
    return {
        'min_eff': min(eff_list),
        'max_eff': eff_list[p90_idx] if p90_idx < len(eff_list) else max(eff_list),
        'max_stab': max(stab_list) if stab_list else 1e-6,
    }


# ========== 交叉 ==========

def crossover_hirano(ind1, ind2):
    """JSP用の2点交叉"""
    size = len(ind1)
    cxpoint1, cxpoint2 = 0, 0
    while cxpoint1 == cxpoint2:
        cxpoint1 = random.randrange(0, size)
        cxpoint2 = random.randrange(0, size)
    if cxpoint2 < cxpoint1:
        cxpoint1, cxpoint2 = cxpoint2, cxpoint1

    sub1 = array.array(ind1.typecode, ind1[cxpoint1:cxpoint2])
    sub2 = array.array(ind2.typecode, ind2[cxpoint1:cxpoint2])
    new_sub1 = array.array(ind1.typecode, [-1] * (cxpoint2 - cxpoint1))
    new_sub2 = array.array(ind2.typecode, [-1] * (cxpoint2 - cxpoint1))
    for s1_idx, s1 in enumerate(ind1[cxpoint1:cxpoint2]):
        if s1 not in sub2:
            continue
        s2_idx = sub2.index(s1)
        new_sub1[s2_idx] = s1
        new_sub2[s1_idx] = s1
        sub1[s1_idx] = -1
        sub2[s2_idx] = -1
    for s1 in sub1:
        if s1 == -1:
            continue
        idx = new_sub1.index(-1)
        new_sub1[idx] = s1
    for s2 in sub2:
        if s2 == -1:
            continue
        idx = new_sub2.index(-1)
        new_sub2[idx] = s2
    ind1[cxpoint1:cxpoint2] = new_sub1
    ind2[cxpoint1:cxpoint2] = new_sub2
    return ind1, ind2


def crossover_pmx(parent1, parent2):
    """PMX交叉"""
    size = len(parent1)
    cx_point1 = random.randint(0, size - 1)
    cx_point2 = random.randint(0, size - 1)
    if cx_point1 > cx_point2:
        cx_point1, cx_point2 = cx_point2, cx_point1
    for ind in range(cx_point1, cx_point2 + 1):
        if parent1[ind] != parent2[ind]:
            x, y = parent1[ind], parent2[ind]
            ran = ind
            while parent1[ran] != y:
                ran = random.randint(0, size - 1)
            parent1[ind], parent1[ran] = parent1[ran], parent1[ind]
            ran = ind
            while parent2[ran] != x:
                ran = random.randint(0, size - 1)
            parent2[ind], parent2[ran] = parent2[ran], parent2[ind]
    return parent1, parent2


# ========== 突然変異 ==========

def mutation_inversion(ind):
    """逆位突然変異"""
    point1 = random.randint(0, len(ind) - 1)
    point2 = random.randint(0, len(ind) - 1)
    while point1 == point2:
        point2 = random.randint(0, len(ind) - 1)
    if point1 > point2:
        point1, point2 = point2, point1
    ind[point1:point2 + 1] = ind[point1:point2 + 1][::-1]
    return ind


def mutation_hirano(ind):
    """JSP用の突然変異（2回スワップ）"""
    size = len(ind)
    for _ in range(2):
        pos1 = random.randrange(0, size)
        pos2 = random.randrange(0, size)
        ind[pos1], ind[pos2] = ind[pos2], ind[pos1]
    return ind


# ========== DEAP初期化 ==========

def initialize(jm_table, cx, mut, sel, original_individual, fixed_gantt,
               reschedule_time, weights):
    """DEAPツールボックスの初期化

    Parameters:
        weights: [効率性の重み, 安定性の重み]
    """
    from deap import base, creator, tools

    if not hasattr(creator, "FitnessMin"):
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    if not hasattr(creator, "Individual"):
        creator.create("Individual", array.array, typecode="b",
                       fitness=creator.FitnessMin)

    toolbox = base.Toolbox()
    toolbox.register("individual", tools.initIterate, creator.Individual,
                     lambda: create_individual_reactive(original_individual))
    toolbox.register("original_individual", tools.initIterate,
                     creator.Individual, lambda: original_individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # 交叉
    cx_func = crossover_hirano if cx == "hirano" else crossover_pmx
    toolbox.register("crossover", cx_func)
    # 突然変異
    mut_func = mutation_inversion if mut == "inversion" else mutation_hirano
    toolbox.register("mutate", mut_func)
    # 選択
    if sel == "Tournament":
        toolbox.register("select", tools.selTournament)
    elif sel == "Roulette":
        toolbox.register("select", tools.selRoulette)

    return toolbox
