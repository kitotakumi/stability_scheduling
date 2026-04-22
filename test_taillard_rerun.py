"""Taillardフィルタ改修前後の実験結果比較

ProcessPoolExecutorを使わず逐次実行で、旧結果と新結果を比較する。
"""

import os
import sys
import random
import json
import time

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'experiments'))

import job_shop_scheduling
import gantt_chart_operation
import ils_scheduling
import evaluation

N_TRIALS = 10
ILS_MAX_ITER = 800


def get_problem(problem_name="mt10", scenario_name="mt10_delay60"):
    jm_table = job_shop_scheduling.get_jm_table(problem_name, scenario_name)
    init_gantt = jm_table.initial_gantt()
    delayed_gantt = jm_table.delayed_gantt()
    fixed_gantt, reschedule_gantt, reschedule_time, msg = \
        gantt_chart_operation.check_disturbance(init_gantt, delayed_gantt)
    return jm_table, fixed_gantt, reschedule_gantt, reschedule_time


def compute_norm_params(problem_name="mt10", scenario_name="mt10_delay60"):
    jm_table, fixed_gantt, reschedule_gantt, reschedule_time = get_problem(
        problem_name, scenario_name)
    delayed_gantt = jm_table.delayed_gantt()
    _, rescheduled_rsr_gantt = gantt_chart_operation.create_rsr_gantt(
        fixed_gantt, reschedule_gantt)
    base_gene = gantt_chart_operation.get_gene(rescheduled_rsr_gantt)
    return evaluation.estimate_normalization_params(
        jm_table, fixed_gantt, reschedule_time,
        delayed_gantt, base_gene, n_samples=200)


def run_single_ils(weights, seed, perturb_method, norm_params,
                   problem_name="mt10", scenario_name="mt10_delay60"):
    jm_table, fixed_gantt, reschedule_gantt, reschedule_time = get_problem(
        problem_name, scenario_name)
    random.seed(seed)
    solver = ils_scheduling.ILSSolver(
        jm_table, fixed_gantt, reschedule_gantt, reschedule_time, weights,
        taillard_acceleration=True)
    solver.estimate_normalization_params(n_samples=100, norm_params=norm_params)
    best_orders, best_score, conv_info, history = solver.run(
        max_iterations=ILS_MAX_ITER, perturb_method=perturb_method, verbose=False)
    ms, st = solver.evaluate_pareto(best_orders)
    return ms, st, best_score, conv_info.get('cpu_time', 0)


def run_experiment(weights_list, methods, norm_params):
    """全weight × 全手法 × N_TRIALS を逐次実行"""
    results = {}

    for weights in weights_list:
        w_key = f"eff={weights[0]}_stab={weights[1]}"
        results[w_key] = {}

        for method in methods:
            ms_list, st_list = [], []
            print(f"\n  {w_key} / {method}:", end=" ", flush=True)
            t0 = time.time()

            for trial in range(N_TRIALS):
                seed = trial * 100 + 7
                ms, st, score, cpu = run_single_ils(
                    weights, seed, method, norm_params)
                ms_list.append(ms)
                st_list.append(st)
                print(f"T{trial}(MS={ms},St={st:.2f})", end=" ", flush=True)

            elapsed = time.time() - t0
            import numpy as np
            results[w_key][method] = {
                'ms_mean': np.mean(ms_list), 'ms_std': np.std(ms_list),
                'st_mean': np.mean(st_list), 'st_std': np.std(st_list),
                'ms_list': ms_list, 'st_list': st_list,
                'elapsed': elapsed,
            }
            print(f"\n    -> MS={np.mean(ms_list):.1f}±{np.std(ms_list):.2f}, "
                  f"St={np.mean(st_list):.2f}±{np.std(st_list):.2f} "
                  f"({elapsed:.1f}s)")

    return results


def load_old_results():
    """旧結果を読み込む"""
    old = {}

    # GA vs ILS (0.95, 0.05)
    path = "experiments/ga_vs_ils/results/run_20260402_225852/comparison_eff=0.95_stab=0.05.json"
    if os.path.exists(path):
        with open(path) as f:
            data = json.load(f)
        for method in ['ILS_swap', 'ILS_insert']:
            if method in data:
                ms_list = [t['makespan'] for t in data[method]]
                st_list = [t['stability'] for t in data[method]]
                import numpy as np
                key = f"eff=0.95_stab=0.05/{method.split('_')[1]}"
                old[key] = {
                    'ms_mean': np.mean(ms_list), 'ms_std': np.std(ms_list),
                    'st_mean': np.mean(st_list), 'st_std': np.std(st_list),
                }

    # ILS analysis (0.9, 0.1) and (0.8, 0.2)
    for w in ['0.9_stab=0.1', '0.8_stab=0.2']:
        path = f"experiments/ils_analysis/results/run_20260404_130456/ils_comparison_eff={w}.json"
        if os.path.exists(path):
            with open(path) as f:
                data = json.load(f)
            for method in ['ILS_swap', 'ILS_insert']:
                if method in data:
                    ms_list = [t['makespan'] for t in data[method]]
                    st_list = [t['stability'] for t in data[method]]
                    import numpy as np
                    key = f"eff={w}/{method.split('_')[1]}"
                    old[key] = {
                        'ms_mean': np.mean(ms_list), 'ms_std': np.std(ms_list),
                        'st_mean': np.mean(st_list), 'st_std': np.std(st_list),
                    }

    return old


def main():
    print("=" * 70)
    print("Taillardフィルタ改修後の実験結果比較")
    print("  旧: est_ms <= current_ms (MSのみでフィルタ)")
    print("  新: score_lb <= current_score (合成スコア下界でフィルタ)")
    print("=" * 70)

    # 旧結果
    old = load_old_results()
    if old:
        print("\n旧結果 (読み込み済み):")
        for key, v in sorted(old.items()):
            print(f"  {key}: MS={v['ms_mean']:.1f}±{v['ms_std']:.2f}, "
                  f"St={v['st_mean']:.2f}±{v['st_std']:.2f}")

    # 正規化パラメータ
    print("\n正規化パラメータ推定中...")
    norm_params = compute_norm_params()
    print(f"  {norm_params}")

    # 新実験
    weights_list = [(0.95, 0.05), (0.9, 0.1), (0.8, 0.2)]
    methods = ['swap', 'insert']

    print("\n新実験開始 (逐次実行)")
    new_results = run_experiment(weights_list, methods, norm_params)

    # 比較表示
    import numpy as np
    print("\n" + "=" * 70)
    print("比較結果")
    print("=" * 70)
    print(f"{'条件':<30} {'旧MS':>12} {'新MS':>12} {'旧St':>12} {'新St':>12} {'MS差':>8}")
    print("-" * 86)

    for w_key, methods_data in new_results.items():
        for method, new_v in methods_data.items():
            old_key = f"{w_key}/{method}"
            label = f"{w_key}/{method}"

            new_ms = f"{new_v['ms_mean']:.1f}±{new_v['ms_std']:.1f}"
            new_st = f"{new_v['st_mean']:.2f}±{new_v['st_std']:.2f}"

            if old_key in old:
                old_v = old[old_key]
                old_ms = f"{old_v['ms_mean']:.1f}±{old_v['ms_std']:.1f}"
                old_st = f"{old_v['st_mean']:.2f}±{old_v['st_std']:.2f}"
                ms_diff = new_v['ms_mean'] - old_v['ms_mean']
                diff_str = f"{ms_diff:+.1f}"
            else:
                old_ms = "N/A"
                old_st = "N/A"
                diff_str = "-"

            print(f"{label:<30} {old_ms:>12} {new_ms:>12} {old_st:>12} {new_st:>12} {diff_str:>8}")


if __name__ == '__main__':
    main()
