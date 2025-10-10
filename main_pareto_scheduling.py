import random
from deap import tools
import job_shop_scheduling
import pareto_operation
import gantt_chart_operation
import analysis
import sys
import numpy as np


# パラメータの設定
def parameters():
    jsp = "MT10_10"
    num_weights = 10  # 重みベクトル本数
    bucket_size = 30  # CS バケットの最大容量のサイズ
    ngen = 10
    cx = "hirano"  # hirano or pmx
    mut = "inversion"  # hirano or inversion
    sel = "Tournament"  # Tournament or Roulette ルーレットはスケーリングしないと解が改善しない
    cxpb = 0.85
    mutpb = 0.1
    return jsp, num_weights, bucket_size, ngen, cx, mut, sel, cxpb, mutpb # fmt: skip

# 目的関数の数（dim）に応じて0-1の単体上に一様に分布する重みベクトルを生成
def generate_weight_vectors(num_weights):
    weights = []
    for i in range(num_weights):
        w0 = i / (num_weights - 1)
        w1 = 1.0 - w0
        weights.append(np.array([w0, w1]))
    return weights

# wを1-0.5の間で恣意的に生成。効率性×w+安定性×(1-w)
def generate_weight_vectors_arbitrary():
    weights = []
    for w0 in np.arange(1.0, 0.45, -0.05):  # 終点値を0.45に変更
        w0 = np.round(w0, 2)  # np.roundを使用
        w1 = np.round(1.0 - w0, 2)  # np.roundを使用
        weights.append([w0, w1])
    return weights


# reactive scheduling systemのメイン関数
def main():
    jsp, num_weights, bucket_size, ngen, cx, mut, sel, cxpb, mutpb = parameters() # fmt: skip
    jm_table = job_shop_scheduling.get_jm_table(jsp)
    # init_ganttは初期スケジュール
    init_gantt = jm_table.initial_gantt()
    # delayed_ganttはある作業に遅延が発生しているスケジュール
    delayed_gantt = jm_table.delayed_gantt()
    # fixed_ganttはリスケ対象ではない作業のスケジュール、reschedule_ganttはリスケ対象の作業のスケジュール
    fixed_gantt, reschedule_gantt, reschedule_time, message = (
        gantt_chart_operation.check_disturbance(init_gantt, delayed_gantt))  # fmt: skip
    if reschedule_time == 0:
        print(message)
        analysis.plot_gantt(init_gantt, jm_table.get_job_count(), jm_table.get_machine_count(), jsp)  # fmt: skip
        sys.exit()
    rsr_gantt, rescheduled_rsr_gantt = gantt_chart_operation.create_rsr_gantt(fixed_gantt, reschedule_gantt)  # fmt:skip
    # original_individualは_ganttを遺伝子で表現したもの
    original_individual = gantt_chart_operation.get_gene(rescheduled_rsr_gantt)
    print(message)
    # 色々プロットする
    gantt = gantt_chart_operation.get_gantt_reactive(
        jm_table, original_individual, fixed_gantt, reschedule_time)  # fmt: skip

    toolbox = pareto_operation.initialize_pareto(
        jm_table, cx, mut, sel, original_individual, fixed_gantt, reschedule_time)  # fmt: skip
    # analysis.plot_gantt(rescheduled_rsr_gantt, jm_table.get_job_count(), jm_table.get_machine_count(), "resceduled")  # fmt: skip
    # analysis.plot_gantt(rsr_gantt, jm_table.get_job_count(), jm_table.get_machine_count(), "rsr")  # fmt: skip
    # analysis.plot_gantt(init_gantt, jm_table.get_job_count(), jm_table.get_machine_count(), "")  # fmt: skip
    # analysis.plot_gantt(fixed_gantt, jm_table.get_job_count(), jm_table.get_machine_count(), "fixed", reschedule_time,)  # fmt: skip
    # analysis.plot_gantt(reschedule_gantt, jm_table.get_job_count(), jm_table.get_machine_count(), "object to be rescheduled", reschedule_time,)  # fmt: skip
    # analysis.plot_gantt(gantt, jm_table.get_job_count(), jm_table.get_machine_count(), "RSR", reschedule_time,)  # fmt: skip
    # all_fitnesses = 全個体の評価値 optimal_individuals = 各世代の最良個体

    # 重みベクトルの生成
    weights = generate_weight_vectors_arbitrary()  # [array([w0, w1]), ...]
    # weights = generate_weight_vectors(num_weights)
    num_weights = len(weights)

    # current setと Pareto Archiveの初期化
    CS = {i: [] for i in range(num_weights)}  # キー：重みベクトル番号、値：個体のリスト [individulal, ...] # fmt: skip
    All_solution = []  # Pareto Archive [individual, ...]
    PF = tools.ParetoFront()  # Pareto Front

    # ランダムに100個体を生成して正規化用にmax_minを取得
    sample_population = toolbox.population(n=100)  # 1個体の初期集団
    max_efficiency, max_stability = float("-inf"), float("-inf")
    min_efficiency = float("inf")
    max_efficiency, min_efficiency, max_stability, statistics = (
        pareto_operation.get_max_min(jm_table, fixed_gantt, reschedule_time, sample_population, max_efficiency, min_efficiency, max_stability)
    )  # fmt:skip
    
    """
    =============================== バケットの初期化 ==============================================================
    """
    for ind, w in enumerate(weights):
        # 初期個体の生成と局所探索
        population = toolbox.population(n=20)  # 1個体の初期集団。多めに生成して局所探索後に数を絞る
        population[0] = toolbox.original_individual()  # original個体をpopulationの1つ目に格納
        
        # ローカル探索と評価値の登録
        for i in population:
            best = toolbox.local_search(
                i, w, max_efficiency, min_efficiency, max_stability
            )
            i[:] = best[:]  # 個体の遺伝子を更新
            i.fitness.values = toolbox.evaluate(i)  # fmt: skip
            CS[ind].append(i)  # 個体と評価値をCSに登録
            # 重みパラメータでソート
            All_solution.append(i)  # 個体と評価値をPEに登録
        
        CS[ind].sort(key=lambda ind: pareto_operation.calculate_weighted_fitness(
            ind.fitness.values, w, max_efficiency, min_efficiency, max_stability
        ))
        CS[ind] = CS[ind][:bucket_size]  # バケットサイズを維持
        
        PF.update(CS[ind])  # PFに更新

    """
    =============================== ループ探索 ==============================================================
    """
    for gen in range(ngen):
        idx = random.randrange(num_weights)  # 重みベクトルのインデックスをランダムに選択
        w = weights[idx]  # 選択した重みベクトル
        bucket = CS[idx]  # 選択した重みベクトルに対応するCSバケット
        parents = toolbox.select(bucket, 2, 4)
        child1, child2 = map(toolbox.clone, parents) # copyして参照を切ってるらしい
        # 交叉確率に基づいて交叉・突然変異を行う
        if random.random() < cxpb:
            toolbox.crossover(child1, child2)
        if random.random() < mutpb:
            toolbox.mutate(child1)
        if random.random() < mutpb:
            toolbox.mutate(child2)

        # 5) 局所探索（Min–Max 正規化ヒルクライム）＆ 評価
        for child in (child1, child2):
            # in-place に局所探索
            refined = toolbox.local_search(
                child, w, max_efficiency, min_efficiency, max_stability
            )
            child[:] = refined  # 個体を更新
            # DEAP 評価関数呼び出し
            child.fitness.values = toolbox.evaluate(child)

            # 6) CS 更新
            CS[idx].append(child)
        # 重みパラメータでソート
        CS[idx].sort(key=lambda ind: pareto_operation.calculate_weighted_fitness(
            ind.fitness.values, w, max_efficiency, min_efficiency, max_stability
        ))
        CS[idx] = CS[idx][:bucket_size]  # バケットサイズを維持
        # 7) PE 更新
        All_solution.append(child)
        PF.update([child])  # PFに更新
        print(f"Generation {gen+1}/{ngen} done.")
        
    analysis.plot_cs_pe(CS, PF, title="CS & PF Scatter")  # fmt: skip

if __name__ == "__main__":
    main()
