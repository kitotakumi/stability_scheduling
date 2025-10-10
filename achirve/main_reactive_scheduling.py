import random
from deap import tools
import job_shop_scheduling
import genetic_operaton
import gantt_chart_operation
import analysis


# パラメータの設定
def parameters():
    jsp = "MT10_10"
    ngen = 150
    cx = "pmx"  # hirano or pmx
    mut = "inversion"  # hirano or inversion
    sel = "Tournament"  # Tournament or Roulette ルーレットはスケーリングしないと解が改善しない
    cxpb = 0.85
    mutpb = 0.1
    pop_size = 50
    return jsp, ngen, cx, mut, sel, cxpb, mutpb, pop_size


# reactive scheduling systemのメイン関数
def main():
    jsp, ngen, cx, mut, sel, cxpb, mutpb, pop_size = parameters()
    jm_table = job_shop_scheduling.get_jm_table(jsp)
    # init_ganttは初期スケジュール
    init_gantt = jm_table.initial_gantt()
    # delayed_ganttは有る作業に遅延が発生しているスケジュール
    delayed_gantt = jm_table.delayed_gantt()
    # fixed_ganttはリスケ対象ではない作業のスケジュール、reschedule_ganttはリスケ対象の作業のスケジュール
    fixed_gantt, reschedule_gantt, reschedule_time, message = (
        gantt_chart_operation.check_disturbance(init_gantt, delayed_gantt))  # fmt: skip
    # original_individualはreschedule_ganttを遺伝子で表現したもの
    original_individual = gantt_chart_operation.get_gene(reschedule_gantt)
    toolbox = genetic_operaton.initialize_reactive(
        jm_table, cx, mut, sel, original_individual, fixed_gantt, reschedule_time)  # fmt: skip
    print(message)
    # 色々プロットする
    gantt = gantt_chart_operation.get_gantt_reactive(
        jm_table, original_individual, fixed_gantt, reschedule_time)  # fmt: skip
    # analysis.plot_gantt(init_gantt, jm_table.get_job_count(), jm_table.get_machine_count(), jsp)  # fmt: skip
    # analysis.plot_gantt(fixed_gantt, jm_table.get_job_count(), jm_table.get_machine_count(), "fixed", reschedule_time,)  # fmt: skip
    # analysis.plot_gantt(reschedule_gantt, jm_table.get_job_count(), jm_table.get_machine_count(), "object to be rescheduled", reschedule_time,)  # fmt: skip
    analysis.plot_gantt(gantt, jm_table.get_job_count(), jm_table.get_machine_count(), "RSR", reschedule_time,)  # fmt: skip
    # all_fitnesses = 全個体の評価値 optimal_individuals = 各世代の最良個体
    all_fitnesses, optimal_individuals = [], []

    for gen in range(ngen):
        # 初期世代の生成
        if gen == 0:
            population = toolbox.population(n=pop_size)
            # original個体をpopulationの1つ目に格納
            population[0] = toolbox.original_individual()
            # 評価値の登録 fitnessは単目的でもtupleとして格納されている
            for ind in population:
                ind.fitness.values = toolbox.evaluate(ind)
            optimal_individuals.append(tools.selBest(population, 1)[0])
            all_fitnesses.append([1 / ind.fitness.values[0] for ind in population])
            offspring = population[:]
            continue
        else:
            # 世代を更新
            population = offspring[:]
            offspring.clear()
        # populationから遺伝子を選択
        for child in range(int(pop_size / 2)):
            if sel == "Tournament":
                children = toolbox.select(population, 2, 4)
            elif sel == "Roulette":
                children = toolbox.select(population, 2)
            children = list(map(toolbox.clone, children))  # Copyして参照を切ってるらしい  # fmt: skip
            # 交叉確率に基づいて交叉を行う
            if random.random() < cxpb:
                toolbox.crossover(children[0], children[1])
            offspring.extend(children)

        for mutant in offspring:
            # 突然変異確率に基づいて突然変異を行う
            if random.random() < mutpb:
                toolbox.mutate(mutant)
            # 評価値を削除して更新する
            del mutant.fitness.values
            mutant.fitness.values = toolbox.evaluate(mutant)

        optimal_individuals.append(tools.selBest(offspring, 1)[0])
        # エリート選択（１個体）
        offspring[0] = tools.selBest(optimal_individuals, 1)[0]
        all_fitnesses.append([1 / ind.fitness.values[0] for ind in offspring])
        print(gen)

    # 最良個体を選択
    best_individual = tools.selBest(optimal_individuals, 1)[0]
    # 散布図と最良個体ガントチャート
    analysis.plot_scatter(ngen, pop_size, all_fitnesses)
    gantt = gantt_chart_operation.get_gantt_reactive(
        jm_table, best_individual, fixed_gantt, reschedule_time)  # fmt: skip
    analysis.plot_gantt(gantt, jm_table.get_job_count(), jm_table.get_machine_count(), "after rescheduled", reschedule_time,)  # fmt: skip


if __name__ == "__main__":
    main()
