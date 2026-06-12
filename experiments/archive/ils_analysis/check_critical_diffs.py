"""
ILS最良解の不一致位置がクリティカルパス上かどうかを調べる。
PRで「クリティカルパス外の変更だけ戻す」戦略に意味があるか確認。
"""

import copy
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from experiment_utils import compute_shared_norm_params, get_problem
import ils_scheduling


PROBLEMS = [
    ('mt10', 'mt10_delay60'),
    ('mt10', 'mt10_delay90'),
    ('la21', 'la21_delay147'),
    ('la36', 'la36_delay148'),
    ('la40', 'la40_delay148'),
]


def analyze(problem_name, scenario_name, weights, seed, n_warmup=200):
    jm_table, fixed_gantt, reschedule_gantt, reschedule_time = get_problem(
        problem_name, scenario_name)
    norm_params = compute_shared_norm_params(problem_name, scenario_name)

    random.seed(seed)
    solver = ils_scheduling.ILSSolver(
        jm_table, fixed_gantt, reschedule_gantt, reschedule_time, weights)
    solver.estimate_normalization_params(n_samples=100, norm_params=norm_params)

    best, best_score, _, _ = solver.run(
        max_iterations=n_warmup, perturb_method='swap', verbose=False)
    best_ms, best_st = solver.evaluate_pareto(best)

    op_times = solver.build_gantt(best)
    critical_path = solver.find_critical_path(op_times, best)

    init = solver.initial_machine_orders

    # 各機械の不一致位置を分類
    total_diffs = 0
    on_critical = 0
    off_critical = 0
    off_critical_details = []

    for m, cur_ops in best.items():
        ref_ops = init.get(m, cur_ops)
        for i in range(min(len(cur_ops), len(ref_ops))):
            if cur_ops[i] != ref_ops[i]:
                total_diffs += 1
                # この位置の操作がクリティカルパス上か？
                op_on_cp = cur_ops[i] in critical_path
                ref_op_on_cp = ref_ops[i] in critical_path
                if op_on_cp or ref_op_on_cp:
                    on_critical += 1
                else:
                    off_critical += 1
                    off_critical_details.append((m, i, cur_ops[i], ref_ops[i]))

    return {
        'ms': best_ms, 'st': best_st,
        'total_diffs': total_diffs,
        'on_critical': on_critical,
        'off_critical': off_critical,
        'off_critical_details': off_critical_details,
        'critical_path_size': len(critical_path),
    }


def main():
    seeds = [7, 107, 207]
    for problem_name, scenario_name in PROBLEMS:
        for weights in [[0.95, 0.05], [0.8, 0.2]]:
            prob = f"{problem_name}_{scenario_name}"
            print(f"\n{'='*70}")
            print(f"{prob} w={weights}")
            print(f"{'='*70}")
            for seed in seeds:
                r = analyze(problem_name, scenario_name, weights, seed)
                print(f"  seed={seed}: MS={r['ms']}, St={r['st']:.2f}, "
                      f"CP_size={r['critical_path_size']}")
                print(f"    diffs={r['total_diffs']}: "
                      f"on_CP={r['on_critical']}, off_CP={r['off_critical']}")
                if r['off_critical_details']:
                    for m, i, cur_op, ref_op in r['off_critical_details']:
                        print(f"      M{m} pos{i}: {cur_op} -> {ref_op} (both off CP)")


if __name__ == "__main__":
    main()
