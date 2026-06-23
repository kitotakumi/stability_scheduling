"""
Memetic GA による再スケジューリング

アーキテクチャ:
  GA の交叉・突然変異 (広域探索) × N5 局所探索 (個体精緻化)
  × repair キック (高安定性方向への引き戻し)
  × path relinking キック (初期解方向への系統的探索)

解表現の橋渡し:
  GA 遺伝子 (GT法ジョブ列) ←→ ILS machine_orders (semi-active)
  変換を通じて両者の探索インフラを共用する。
"""

import random
import time

import evaluation
import gantt_chart_operation
import genetic_operation
from ga_scheduling import GASolver
from ils_scheduling import ILSSolver


class MemeticGASolver:
    """Memetic GA: GA 広域探索 × N5 局所探索 × kick (repair / PR)

    個体精緻化のキックモードは kick_mode で選択する。

    Parameters
    ----------
    kick_mode : str
        'none'   : LS のみ (キックなし)
        'repair' : LS → repair → LS
        'pr'     : LS → path relinking → LS
        'random' : LS → ランダム方向 direct swap → LS（repair の強度を揃えた方向ランダム
                   対照。利得が S_p 誘導由来か一般的多様化由来かを分離する内的妥当性の対照）
    kick_prob : float
        キックを各個体に確率的に適用する確率。
    repair_strength : int
        kick_mode='repair' 時の direct swap 回数を [1, cap] で一様サンプリングする際の cap の天井。
        cap は基本「経路長（improved→初期解の不一致数）」。repair_strength<=0（デフォルト）なら
        cap=経路長（フル）、>0 なら cap=min(経路長, repair_strength)（深さを制限）。
    """

    def __init__(self, jm_table, fixed_gantt, reschedule_gantt, reschedule_time, weights,
                 cx="ppx", mut="inversion", sel="Tournament",
                 cxpb=0.85, mutpb=0.1, pop_size=50,
                 kick_mode='repair', kick_prob=0.5,
                 repair_strength=0, ls_strategy='best', pr_step_strategy='random',
                 pr_ls_top_k=3):  # 既定: PRキックは top-k=3（2026-06-12確定, param_sweep_v1/RESULTS.md §1。kick_mode='pr'時のみ作用）
        self.jm_table = jm_table
        self.fixed_gantt = fixed_gantt
        self.reschedule_time = reschedule_time
        self.weights = weights
        self.pop_size = pop_size
        self.cxpb = cxpb
        self.mutpb = mutpb
        self.kick_mode = kick_mode
        self.kick_prob = kick_prob
        self.repair_strength = repair_strength
        self.ls_strategy = ls_strategy
        self.pr_step_strategy = pr_step_strategy
        self.pr_ls_top_k = pr_ls_top_k

        # GA: crossover / mutation / selection の toolbox と original_individual を借りる
        self._ga = GASolver(
            jm_table, fixed_gantt, reschedule_time, weights,
            cx, mut, sel, cxpb, mutpb, pop_size,
        )
        self.toolbox = self._ga.toolbox
        self.original_individual = self._ga.original_individual

        # ILS: N5 LS と repair のインフラを借りる
        self._ils = ILSSolver(
            jm_table, fixed_gantt, reschedule_gantt, reschedule_time, weights,
            active_schedule=False, taillard_acceleration=True,
        )

    # ========== 解表現の変換 ==========

    def _gene_to_machine_orders(self, ind):
        """GA 遺伝子 → ILS machine_orders

        GT法でデコードした full gantt から、固定済みタスクを除いた
        reschedule 部分だけを ILS の machine_orders 形式に変換する。
        """
        full_gantt = gantt_chart_operation.get_gantt_reactive(
            self.jm_table, ind, self.fixed_gantt, self.reschedule_time)

        # fixed_gantt のタスクを (start, end, job_id) のセットで把握する
        fixed_keys = {
            (t[0], t[1], t[2])
            for m_tasks in self.fixed_gantt
            for t in m_tasks
        }

        reschedule_gantt = [
            [t for t in m_tasks if (t[0], t[1], t[2]) not in fixed_keys]
            for m_tasks in full_gantt
        ]
        return self._ils._gantt_to_machine_orders(reschedule_gantt)

    def _machine_orders_to_gene(self, machine_orders):
        """ILS machine_orders → GA 遺伝子

        semi-active スケジュールを構築し、op_times から直接 reschedule_gantt を
        組み立てて get_gene に渡す。None を返した場合は変換失敗。
        """
        op_times = self._ils.build_gantt(machine_orders)
        if op_times is None:
            return None

        num_machines = self.jm_table.get_machine_count()
        reschedule_gantt = [[] for _ in range(num_machines)]
        for (job_id, _op_idx), (start, end, m_idx) in op_times.items():
            reschedule_gantt[m_idx].append([start, end, job_id])
        for m in reschedule_gantt:
            m.sort(key=lambda x: x[0])

        _, rsr_gantt = gantt_chart_operation.create_rsr_gantt(
            self.fixed_gantt, reschedule_gantt)
        return gantt_chart_operation.get_gene(rsr_gantt)

    # ========== 個体精緻化 ==========

    def _refine_individual(self, ind, norm_params, skip_first_ls=False):
        """N5 LS (+ repair / PR キック) を適用し、遺伝子と fitness を更新する。

        フロー: gene → [LS →] [repair → LS] | [PR → LS] → gene 書き戻し
        repair / PR キック後の解は無条件採用し、質の判断を tournament selection に委ねる。

        skip_first_ls=True: 個体は既に局所最適のため最初の LS をスキップする。
          キックも発火しなければ何も変更せず (None, None) を返す。
        """
        if skip_first_ls and self.kick_mode == 'none':
            return None, None

        machine_orders = self._gene_to_machine_orders(ind)

        # UEA 記録ポリシー: LS 開始点（交叉/突変直後・LS 前）も記録する。kick_point
        # （LS 前のキック出力）だけ記録すると、キック手法にだけ LS 前の点が計上される
        # 非対称が生じるため（2026-06-12 監査）。skip_first_ls の個体は前世代の記録済み
        # 局所最適解そのものなので記録しない。
        ind.prels_point = None
        if not skip_first_ls:
            ind.prels_point = self._ils.evaluate_pareto(machine_orders)
            improved, _, _ = self._ils.local_search(machine_orders, strategy=self.ls_strategy)
        else:
            improved = machine_orders  # 前世代のLS済み局所最適解なのでそのまま使う

        # kick: repair または path relinking を確率的に適用し、結果は無条件採用
        ind.kick_point = None
        ind.kick_raw_point = None
        kick_applied = False
        if self.kick_mode != 'none' and random.random() < self.kick_prob:
            if self.kick_mode == 'repair':
                # 経路長（improved→初期解の不一致数）を上限に repair 強度をランダムサンプリング。
                # repair_strength<=0 なら cap=経路長（フル）、>0 なら cap=min(経路長, repair_strength)。
                n_diff = self._ils._count_diffs(improved, self._ils.initial_machine_orders)
                cap = n_diff if self.repair_strength <= 0 else min(n_diff, self.repair_strength)
                cap = max(1, cap)
                depth = random.randint(1, cap)
                # 機構統計 (per-call): 発動時の経路長（不一致数）と適用 depth を記録
                if not hasattr(self._ils, 'repair_call_stats'):
                    self._ils.repair_call_stats = []
                self._ils.repair_call_stats.append((n_diff, depth))
                kicked = self._ils.perturb(improved, 'repair', depth)
            elif self.kick_mode == 'random':
                # 内的妥当性の対照: repair と同じ強度分布（depth ∈ [1, cap], cap=経路長）で
                # 方向だけランダム化した direct swap を適用する。depth のサンプリングは repair
                # と完全一致させ、機構統計も同じ repair_call_stats に積む（強度分布の一致を
                # 監査できるようにする）。「S_p 誘導 vs 一般的多様化」の分離が目的。
                n_diff = self._ils._count_diffs(improved, self._ils.initial_machine_orders)
                cap = n_diff if self.repair_strength <= 0 else min(n_diff, self.repair_strength)
                cap = max(1, cap)
                depth = random.randint(1, cap)
                if not hasattr(self._ils, 'repair_call_stats'):
                    self._ils.repair_call_stats = []
                self._ils.repair_call_stats.append((n_diff, depth))
                kicked = self._ils._perturb_random_swap(improved, depth)
            else:  # 'pr'
                # 集団ベースの memetic では path_relinking はデフォルト
                # (return_intermediate=False = 始点 S_best を返す) のままで良い。
                # ILS と違い個体が多様なため、初期解へ向かう経路に「始点より良い
                # 改善中間解」が存在することが多く、その場合 S_best がその中間解に
                # なる。エリート保存で best は保護されるので悪化もしない。
                # → memetic+PR は no-op ではなく Pareto 域を広げる有効なキックになる。
                # (実験: 2026-06-02, memetic+PR は memetic-LS に HV で有意勝ち。
                #  始点・終点除外しても結果は変わらず、むしろ大幅減速するため不採用。)
                if self.pr_ls_top_k and self.pr_ls_top_k > 1:
                    # top-k LS variant: 経路上上位 k 中間解に LS をかけ最良を返す
                    kicked, _ = self._ils.path_relinking(
                        improved, self._ils.initial_machine_orders,
                        ls_strategy='best', step_strategy=self.pr_step_strategy,
                        escape_infeasible=True, ls_top_k=self.pr_ls_top_k)
                    # k=1 が kick_point として記録する「生の最良中間解」を top-k でも記録
                    # （記録対称性。path_relinking が stash した評価値を回収）
                    ind.kick_raw_point = self._ils._pr_last_raw
                else:
                    kicked, _ = self._ils.path_relinking(
                        improved, self._ils.initial_machine_orders,
                        ls_strategy=None, step_strategy=self.pr_step_strategy,
                        escape_infeasible=True)
            kick_ms, kick_st = self._ils.evaluate_pareto(kicked)
            ind.kick_point = (kick_ms, kick_st)
            improved, _, _ = self._ils.local_search(kicked, strategy=self.ls_strategy)
            kick_applied = True

        if skip_first_ls and not kick_applied:
            return None, None

        # machine_orders → 遺伝子に書き戻す (DEAP Individual は array 型の場合があるため要素単位で代入)
        new_gene = self._machine_orders_to_gene(improved)
        if new_gene is not None and len(new_gene) == len(ind):
            for i, v in enumerate(new_gene):
                ind[i] = v

        ms, st = self._ils.evaluate_pareto(improved)
        score = evaluation.weighted_objective(ms, st, self.weights, norm_params)
        ind.fitness.values = (score,)
        ind.cached_ms = ms
        ind.cached_st = st
        return ms, st

    # ========== メインループ ==========

    def run(self, ngen=300, verbose=True, norm_params=None, track_population=False, patience=None):
        """Memetic GA メインループ

        GASolver.run と同じインタフェース。
        各個体評価を _refine_individual (N5 LS + repair) に差し替えた版。

        Returns
        -------
        best_individual, best_makespan, best_stability, convergence_info, history
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

        def snapshot(gen, offspring):
            entry = {
                'cpu_time': time.time() - start_time,
                'evaluations': eval_count,
                'generation': gen,
                'best_makespan': global_best_ms,
                'best_stability': global_best_st,
                'best_score': global_best_score,
            }
            if track_population:
                entry['pop_points'] = [
                    [ind.cached_ms, ind.cached_st] for ind in offspring
                ]
                entry['kick_points'] = [
                    [ind.kick_point[0], ind.kick_point[1]]
                    for ind in offspring
                    if getattr(ind, 'kick_point', None) is not None
                ]
                # LS 開始点（交叉/突変直後・LS 前）と top-k 時の生最良中間解。
                # UEA 記録の対称性のため（memetic_ls にも prels_points が入る）
                entry['prels_points'] = [
                    [ind.prels_point[0], ind.prels_point[1]]
                    for ind in offspring
                    if getattr(ind, 'prels_point', None) is not None
                ]
                entry['kick_raw_points'] = [
                    [ind.kick_raw_point[0], ind.kick_raw_point[1]]
                    for ind in offspring
                    if getattr(ind, 'kick_raw_point', None) is not None
                ]
            history.append(entry)

        def snapshot_intra(gen):
            history.append({
                'cpu_time': time.time() - start_time,
                'evaluations': eval_count,
                'generation': gen,
                'best_makespan': global_best_ms,
                'best_stability': global_best_st,
                'best_score': global_best_score,
            })

        sub_every = max(1, self.pop_size // 4)

        # 正規化パラメータ: GA と ILS で共有する
        shared_norm = norm_params

        for gen in range(ngen):
            if gen == 0:
                population = self.toolbox.population(n=self.pop_size)
                population[0] = self.toolbox.original_individual()

                # 正規化パラメータの推定 (未指定の場合)
                if shared_norm is None:
                    shared_norm = genetic_operation.estimate_normalization_params(
                        self.jm_table, self.fixed_gantt, self.reschedule_time,
                        population)
                self._ils.set_normalization_params(shared_norm)

                # RSR baseline: GA の __init__ で計算済みの値を再利用
                self.baseline_ms = self._ga.baseline_rsr_ms
                self.baseline_st = self._ga.baseline_rsr_st  # 0.0
                self.baseline_score = evaluation.weighted_objective(
                    self.baseline_ms, self.baseline_st, self.weights, shared_norm)

                # active-decoded baseline: GA と同じ方式（N5 LS 適用前）
                active_ms = genetic_operation.compute_makespan(
                    self.jm_table, self.fixed_gantt, self.reschedule_time, population[0])
                active_st = genetic_operation.compute_stability(
                    self.jm_table, self.fixed_gantt, self.reschedule_time, population[0])
                self.baseline_active_ms = active_ms
                self.baseline_active_st = active_st
                self.baseline_active_score = evaluation.weighted_objective(
                    active_ms, active_st, self.weights, shared_norm)

                for k_ind, ind in enumerate(population):
                    self._refine_individual(ind, shared_norm)
                    eval_count += 1
                    if (k_ind + 1) % sub_every == 0 and (k_ind + 1) < len(population):
                        partial = population[:k_ind + 1]
                        cur_best = tools.selBest(partial, 1)[0]
                        cur_score = cur_best.fitness.values[0]
                        if cur_score < global_best_score:
                            global_best_score = cur_score
                            global_best_ms = cur_best.cached_ms
                            global_best_st = cur_best.cached_st
                            best_gen = 0
                            best_cpu_time = time.time() - start_time
                            best_eval_count = eval_count
                        snapshot_intra(0)

                offspring = population[:]
                best = tools.selBest(offspring, 1)[0]
                if best.fitness.values[0] < global_best_score:
                    global_best_score = best.fitness.values[0]
                    global_best_ms = best.cached_ms
                    global_best_st = best.cached_st
                    best_gen = 0
                    best_cpu_time = time.time() - start_time
                    best_eval_count = eval_count

                snapshot(0, offspring)
                continue

            population = offspring[:]
            offspring.clear()

            for _ in range(self.pop_size // 2):
                children = self.toolbox.select(population, 2, 4)
                children = list(map(self.toolbox.clone, children))
                parents_identical = children[0] == children[1]
                if not parents_identical and random.random() < self.cxpb:
                    self.toolbox.crossover(children[0], children[1])
                    children[0].modified = True
                    children[1].modified = True
                else:
                    children[0].modified = False
                    children[1].modified = False
                offspring.extend(children)

            for mutant in offspring:
                if random.random() < self.mutpb:
                    self.toolbox.mutate(mutant)
                    mutant.modified = True

            for k_ind, ind in enumerate(offspring):
                if ind.modified:
                    del ind.fitness.values
                    self._refine_individual(ind, shared_norm)
                    eval_count += 1
                else:
                    ms, st = self._refine_individual(ind, shared_norm, skip_first_ls=True)
                    if ms is not None:
                        eval_count += 1
                if (k_ind + 1) % sub_every == 0 and (k_ind + 1) < len(offspring):
                    partial = offspring[:k_ind + 1]
                    cur_best = tools.selBest(partial, 1)[0]
                    cur_score = cur_best.fitness.values[0]
                    if cur_score < global_best_score:
                        global_best_score = cur_score
                        global_best_ms = cur_best.cached_ms
                        global_best_st = cur_best.cached_st
                        best_gen = gen
                        best_cpu_time = time.time() - start_time
                        best_eval_count = eval_count
                    snapshot_intra(gen)

            # エリート保存: 前世代最良をそのまま競合させる
            best_prev = tools.selBest(population, 1)[0]
            offspring[0] = tools.selBest(offspring + [best_prev], 1)[0]

            best = tools.selBest(offspring, 1)[0]
            if best.fitness.values[0] < global_best_score:
                global_best_score = best.fitness.values[0]
                global_best_ms = best.cached_ms
                global_best_st = best.cached_st
                best_gen = gen
                best_cpu_time = time.time() - start_time
                best_eval_count = eval_count

            snapshot(gen, offspring)

            if verbose and (gen % 50 == 0 or gen == ngen - 1):
                elapsed = time.time() - start_time
                print(f"  Gen {gen}: Makespan={global_best_ms}, "
                      f"Stability={global_best_st:.2f} ({elapsed:.1f}s)")

            if patience is not None and gen - best_gen >= patience:
                if verbose:
                    print(f"  早期終了: {patience}世代改善なし (gen={gen})")
                break

        elapsed = time.time() - start_time
        best = tools.selBest(offspring, 1)[0]
        ms, st = best.cached_ms, best.cached_st
        if verbose:
            print(f"\nMemetic GA 完了: {gen + 1}世代/{ngen}, {elapsed:.2f}秒")

        convergence_info = {
            'cpu_time': best_cpu_time,
            'evaluations': best_eval_count,
            'generation': best_gen,
            'total_cpu_time': elapsed,
            'total_evaluations': eval_count,
            'total_generations': gen + 1,
        }
        if verbose:
            print(f"最良解到達: 世代={best_gen}, 評価回数={best_eval_count}, "
                  f"CPU時間={best_cpu_time:.2f}秒")

        return best, ms, st, convergence_info, history


if __name__ == "__main__":
    import sys
    import job_shop_scheduling

    problem_name = "mt10"
    scenario_name = "mt10_delay60"
    weights = [0.5, 0.5]

    jm_table = job_shop_scheduling.get_jm_table(problem_name, scenario_name)
    init_gantt = jm_table.initial_gantt()
    delayed_gantt = jm_table.delayed_gantt()

    fixed_gantt, reschedule_gantt, reschedule_time, msg = \
        gantt_chart_operation.check_disturbance(init_gantt, delayed_gantt)

    if reschedule_time == 0:
        print("リスケジューリング不要")
        sys.exit()

    print(f"問題: {problem_name} / シナリオ: {scenario_name}")
    print(msg)
    print(f"重みベクトル: 効率性={weights[0]}, 安定性={weights[1]}")

    solver = MemeticGASolver(
        jm_table, fixed_gantt, reschedule_gantt, reschedule_time, weights,
        kick_mode='repair', kick_prob=0.3, repair_strength=4,
    )

    best_ind, makespan, stability, conv_info, history = solver.run(ngen=100)

    print(f"\n===== 最終結果 =====")
    print(f"Makespan:  {makespan}")
    print(f"Stability: {stability:.4f}")
