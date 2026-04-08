"""
GA (遺伝的アルゴリズム) による再スケジューリング

解表現: GT法の遺伝子（ジョブ番号混合列）
デコード: get_gantt_reactive（active schedule、左詰め挿入）
評価: メイクスパン（効率性）+ 順位偏差（安定性）の重み付き和
"""

import random
import sys
import time

import job_shop_scheduling
import gantt_chart_operation
import genetic_operation
import evaluation
import analysis


class GASolver:
    """遺伝的アルゴリズムによるジョブショップ再スケジューリングソルバー"""

    def __init__(self, jm_table, fixed_gantt, reschedule_time, weights,
                 cx="hirano", mut="inversion", sel="Tournament",
                 cxpb=0.85, mutpb=0.1, pop_size=50):
        self.jm_table = jm_table
        self.fixed_gantt = fixed_gantt
        self.reschedule_time = reschedule_time
        self.weights = weights
        self.cxpb = cxpb
        self.mutpb = mutpb
        self.pop_size = pop_size

        # 初期遺伝子の取得
        reschedule_gantt = self._get_reschedule_gantt()
        rsr_gantt, rescheduled_rsr_gantt = gantt_chart_operation.create_rsr_gantt(
            fixed_gantt, reschedule_gantt)
        self.original_individual = gantt_chart_operation.get_gene(rescheduled_rsr_gantt)

        # DEAPツールボックスの初期化
        self.toolbox = genetic_operation.initialize(
            jm_table, cx, mut, sel, self.original_individual,
            fixed_gantt, reschedule_time, weights)

    def _get_reschedule_gantt(self):
        """外乱検知からreschedule_ganttを取得（check_disturbanceと同じ結果）"""
        init_gantt = self.jm_table.initial_gantt()
        delayed_gantt = self.jm_table.delayed_gantt()
        _, reschedule_gantt, _, _ = gantt_chart_operation.check_disturbance(
            init_gantt, delayed_gantt)
        return reschedule_gantt

    def run(self, ngen=300, verbose=True, norm_params=None):
        """GAメインループ

        Parameters:
            ngen: 世代数
            verbose: 詳細出力
            norm_params: 事前計算済み正規化パラメータ（Noneの場合は従来方式で推定）

        Returns:
            best_individual, best_makespan, best_stability, convergence_info, history
            convergence_info: 最良解到達時点の計算量指標
            history: 世代ごとの記録リスト [{cpu_time, evaluations, generation,
                     best_makespan, best_stability, best_score}]
        """
        from deap import tools

        start_time = time.time()
        eval_count = 0
        history = []

        global_best_score = float('inf')
        best_gen = 0
        best_cpu_time = 0.0
        best_eval_count = 0
        global_best_ms = None
        global_best_st = None

        for gen in range(ngen):
            if gen == 0:
                population = self.toolbox.population(n=self.pop_size)
                population[0] = self.toolbox.original_individual()
                if norm_params is None:
                    norm_params = genetic_operation.estimate_normalization_params(
                        self.jm_table, self.fixed_gantt, self.reschedule_time,
                        population)
                for ind in population:
                    ind.fitness.values = genetic_operation.objective_function(
                        self.jm_table, self.fixed_gantt, self.reschedule_time,
                        ind, self.weights, norm_params)
                    eval_count += 1
                offspring = population[:]

                best = tools.selBest(offspring, 1)[0]
                score = best.fitness.values[0]
                ms = genetic_operation.compute_makespan(
                    self.jm_table, self.fixed_gantt, self.reschedule_time, best)
                st = genetic_operation.compute_stability(
                    self.jm_table, self.fixed_gantt, self.reschedule_time, best)
                if score < global_best_score:
                    global_best_score = score
                    global_best_ms = ms
                    global_best_st = st
                    best_gen = 0
                    best_cpu_time = time.time() - start_time
                    best_eval_count = eval_count

                history.append({
                    'cpu_time': time.time() - start_time,
                    'evaluations': eval_count,
                    'generation': 0,
                    'best_makespan': global_best_ms,
                    'best_stability': global_best_st,
                    'best_score': global_best_score,
                })
                continue

            population = offspring[:]
            offspring.clear()

            for _ in range(int(self.pop_size / 2)):
                children = self.toolbox.select(population, 2, 4)
                children = list(map(self.toolbox.clone, children))
                if random.random() < self.cxpb:
                    self.toolbox.crossover(children[0], children[1])
                offspring.extend(children)

            for mutant in offspring:
                if random.random() < self.mutpb:
                    self.toolbox.mutate(mutant)

            for ind in offspring:
                del ind.fitness.values
                ind.fitness.values = genetic_operation.objective_function(
                    self.jm_table, self.fixed_gantt, self.reschedule_time,
                    ind, self.weights, norm_params)
                eval_count += 1

            best_prev = tools.selBest(population, 1)[0]
            best_prev.fitness.values = genetic_operation.objective_function(
                self.jm_table, self.fixed_gantt, self.reschedule_time,
                best_prev, self.weights, norm_params)
            eval_count += 1
            offspring[0] = tools.selBest(offspring + [best_prev], 1)[0]

            best = tools.selBest(offspring, 1)[0]
            score = best.fitness.values[0]
            ms = genetic_operation.compute_makespan(
                self.jm_table, self.fixed_gantt, self.reschedule_time, best)
            st = genetic_operation.compute_stability(
                self.jm_table, self.fixed_gantt, self.reschedule_time, best)
            if score < global_best_score:
                global_best_score = score
                global_best_ms = ms
                global_best_st = st
                best_gen = gen
                best_cpu_time = time.time() - start_time
                best_eval_count = eval_count

            history.append({
                'cpu_time': time.time() - start_time,
                'evaluations': eval_count,
                'generation': gen,
                'best_makespan': global_best_ms,
                'best_stability': global_best_st,
                'best_score': global_best_score,
            })

            if verbose and (gen % 50 == 0 or gen == ngen - 1):
                elapsed = time.time() - start_time
                print(f"  Gen {gen}: Makespan={global_best_ms}, Stability={global_best_st:.2f} ({elapsed:.1f}s)")

        elapsed = time.time() - start_time
        best = tools.selBest(offspring, 1)[0]
        ms = genetic_operation.compute_makespan(
            self.jm_table, self.fixed_gantt, self.reschedule_time, best)
        st = genetic_operation.compute_stability(
            self.jm_table, self.fixed_gantt, self.reschedule_time, best)
        if verbose:
            print(f"\nGA完了: {ngen}世代, {elapsed:.2f}秒")

        convergence_info = {
            'cpu_time': best_cpu_time,
            'evaluations': best_eval_count,
            'generation': best_gen,
            'total_cpu_time': elapsed,
            'total_evaluations': eval_count,
            'total_generations': ngen,
        }
        if verbose:
            print(f"最良解到達: 世代={best_gen}, 評価回数={best_eval_count}, "
                  f"CPU時間={best_cpu_time:.2f}秒")

        return best, ms, st, convergence_info, history


if __name__ == "__main__":
    problem_name = "mt10"
    scenario_name = "mt10_delay60"
    weights = [0.5, 0.5]

    jm_table = job_shop_scheduling.get_jm_table(problem_name, scenario_name)
    init_gantt = jm_table.initial_gantt()
    delayed_gantt = jm_table.delayed_gantt()

    fixed_gantt, reschedule_gantt, reschedule_time, msg = \
        gantt_chart_operation.check_disturbance(init_gantt, delayed_gantt)

    if reschedule_time == 0:
        print("リスケジューリング不要です")
        sys.exit()

    print(f"問題: {problem_name} / シナリオ: {scenario_name}")
    print(msg)
    print(f"リスケジュール時刻: {reschedule_time}")
    print(f"重みベクトル: 効率性={weights[0]}, 安定性={weights[1]}")

    solver = GASolver(jm_table, fixed_gantt, reschedule_time, weights)

    # 初期解の評価
    init_ms = genetic_operation.compute_makespan(
        jm_table, fixed_gantt, reschedule_time, solver.original_individual)
    print(f"\n初期解: Makespan={init_ms}")

    # GA実行
    best_ind, makespan, stability, conv_info, history = solver.run(ngen=300)

    # 結果表示
    print(f"\n===== 最終結果 =====")
    print(f"Makespan:  {init_ms} → {makespan} (改善: {init_ms - makespan})")
    print(f"Stability: {stability:.4f}")

    # 詳細統計
    stat = genetic_operation.compute_stability_stat(
        jm_table, fixed_gantt, reschedule_time, best_ind)
    print(f"詳細統計: {stat}")

    # ガントチャート表示
    gantt = gantt_chart_operation.get_gantt_reactive(
        jm_table, best_ind, fixed_gantt, reschedule_time)
    analysis.plot_gantt(
        gantt, jm_table.get_job_count(), jm_table.get_machine_count(),
        "GA Result", reschedule_time)
