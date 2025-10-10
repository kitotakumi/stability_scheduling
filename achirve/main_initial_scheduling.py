import random
from deap import tools
import gantt_chart_operation
import genetic_operaton
import analysis


# パラメータの設定
def parameters():
    jsp = "MT10_10"
    ngen = 150
    cx = "pmx"  # hirano or pmx
    mut = "inversion"  # hirano or inversion
    sel = "Tournament"  # Tournament or Roulette or MyRoulette
    cxpb = 0.9
    mutpb = 0.1
    pop_size = 50
    return jsp, ngen, cx, mut, sel, cxpb, mutpb, pop_size


# 初期スケジュール探索のメインループ
def main():
    jsp, ngen, cx, mut, sel, cxpb, mutpb, pop_size = parameters()
    toolbox, jm_table = genetic_operaton.initialize(jsp, cx, mut, sel)
    # all_fitnesses = 全個体の評価値 optimal_individuals = 各世代の最良個体
    all_fitnesses, optimal_individuals = [], []

    for gen in range(ngen):
        # 初期世代の生成
        if gen == 0:
            population = toolbox.population(n=pop_size)
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
            elif sel == "MyRoulette":
                children = genetic_operaton.selection_roulette(population, 2)
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
    gantt = gantt_chart_operation.get_gantt(jm_table, best_individual)
    print(gantt)
    analysis.plot_gantt(
        gantt, jm_table.get_job_count(), jm_table.get_machine_count(), jsp)  # fmt: skip


if __name__ == "__main__":
    main()
