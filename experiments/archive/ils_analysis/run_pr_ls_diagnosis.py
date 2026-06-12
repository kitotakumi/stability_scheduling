"""
PR+LS 診断: PRの各ステップ後にLSをかけた場合の効果を確認する。
PR単体 vs PR+LS(best) を同一条件で比較。
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
    ('la21', 'la21_delay147'),
    ('la36', 'la36_delay148'),
    ('la40', 'la40_delay148'),
]

WEIGHT_SETS = [
    [0.95, 0.05],
    [0.8, 0.2],
]


def run_one(problem_name, scenario_name, weights, seed, n_warmup=200):
    jm_table, fixed_gantt, reschedule_gantt, reschedule_time = get_problem(
        problem_name, scenario_name)
    norm_params = compute_shared_norm_params(problem_name, scenario_name)

    random.seed(seed)
    solver = ils_scheduling.ILSSolver(
        jm_table, fixed_gantt, reschedule_gantt, reschedule_time, weights)
    solver.estimate_normalization_params(n_samples=100, norm_params=norm_params)

    # ILSウォームアップ (swap)
    best, best_score, _, _ = solver.run(
        max_iterations=n_warmup, perturb_method='swap', verbose=False)
    best_ms, best_st = solver.evaluate_pareto(best)

    # PR単体
    _, pr_score, pr_trace = solver.path_relinking(
        best, solver.initial_machine_orders, stall_limit=10,
        ls_strategy=None, trace=True)

    # PR+LS(best)
    _, prls_score, prls_trace = solver.path_relinking(
        best, solver.initial_machine_orders, stall_limit=10,
        ls_strategy='best', trace=True)

    # insert warmup版も
    random.seed(seed)
    solver2 = ils_scheduling.ILSSolver(
        jm_table, fixed_gantt, reschedule_gantt, reschedule_time, weights)
    solver2.estimate_normalization_params(n_samples=100, norm_params=norm_params)
    best2, best_score2, _, _ = solver2.run(
        max_iterations=n_warmup, perturb_method='insert', verbose=False)
    best_ms2, best_st2 = solver2.evaluate_pareto(best2)

    _, pr_score2, pr_trace2 = solver2.path_relinking(
        best2, solver2.initial_machine_orders, stall_limit=10,
        ls_strategy=None, trace=True)
    _, prls_score2, prls_trace2 = solver2.path_relinking(
        best2, solver2.initial_machine_orders, stall_limit=10,
        ls_strategy='best', trace=True)

    return {
        'swap': {
            'ils': {'ms': best_ms, 'st': best_st, 'score': best_score},
            'pr': extract_result(pr_trace),
            'prls': extract_result(prls_trace),
        },
        'insert': {
            'ils': {'ms': best_ms2, 'st': best_st2, 'score': best_score2},
            'pr': extract_result(pr_trace2),
            'prls': extract_result(prls_trace2),
        },
    }


def extract_result(trace):
    result = [t for t in trace if t['type'] == 'result']
    steps = [t for t in trace if t['type'] == 'step']
    if result:
        r = result[0]
        return {
            'ms': r['best_makespan'], 'st': r['best_stability'],
            'score': r['best_score'], 'n_steps': r['total_steps'],
            'n_improving': sum(1 for s in steps if s['improved']),
        }
    return {'ms': None, 'st': None, 'score': None, 'n_steps': 0, 'n_improving': 0}


def print_comparison(label, ils, pr, prls):
    print(f"    {label:12s}  ILS: MS={ils['ms']:5d} St={ils['st']:5.2f} Sc={ils['score']:.4f}")
    pr_acc = "YES" if pr['score'] < ils['score'] else "no"
    prls_acc = "YES" if prls['score'] < ils['score'] else "no"
    print(f"    {'':12s}  PR:  MS={pr['ms']:5d} St={pr['st']:5.2f} Sc={pr['score']:.4f} "
          f"({pr['n_steps']}steps, {pr['n_improving']}impr) acc={pr_acc}")
    print(f"    {'':12s}  PRLS:MS={prls['ms']:5d} St={prls['st']:5.2f} Sc={prls['score']:.4f} "
          f"({prls['n_steps']}steps, {prls['n_improving']}impr) acc={prls_acc}")


def main():
    seeds = [7, 107, 207]
    for problem_name, scenario_name in PROBLEMS:
        for weights in WEIGHT_SETS:
            prob_label = f"{problem_name}_{scenario_name}"
            print(f"\n{'='*80}")
            print(f"{prob_label} w={weights}")
            print(f"{'='*80}")
            for seed in seeds:
                r = run_one(problem_name, scenario_name, weights, seed)
                print(f"\n  seed={seed}")
                print_comparison("swap->", r['swap']['ils'], r['swap']['pr'], r['swap']['prls'])
                print_comparison("insert->", r['insert']['ils'], r['insert']['pr'], r['insert']['prls'])

    print("\n完了")


if __name__ == "__main__":
    main()
