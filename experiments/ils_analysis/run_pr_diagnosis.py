"""
Path Relinking 診断実験

PRの内部動作を詳細にトレースし、何が起きているかを可視化する。
- ILSを途中まで回してからPRを単発実行し、経路上の解の推移を見る
- PR結果がILSのbestと比べてどうか、なぜ受理されない/されるかを分析
"""

import copy
import json
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from experiment_utils import (
    setup_output_dir, compute_shared_norm_params, get_problem, get_initial_makespan,
    ILS_MAX_ITER,
)
import ils_scheduling


PROBLEMS = [
    ('mt10', 'mt10_delay60'),
    ('la40', 'la40_delay148'),
    ('la36', 'la36_delay148'),
]

WEIGHT_SETS = [
    [0.95, 0.05],
    [0.8, 0.2],
]


def run_diagnosis(problem_name, scenario_name, weights, seed, n_ils_warmup=200):
    """ILSをn_ils_warmupステップ回した後、PRを単発実行してトレースする"""
    jm_table, fixed_gantt, reschedule_gantt, reschedule_time = get_problem(
        problem_name, scenario_name)
    random.seed(seed)

    solver = ils_scheduling.ILSSolver(
        jm_table, fixed_gantt, reschedule_gantt, reschedule_time, weights)
    norm_params = compute_shared_norm_params(problem_name, scenario_name)
    solver.estimate_normalization_params(n_samples=100, norm_params=norm_params)

    # ILSをウォームアップ（PRなし）
    best, best_score, conv_info, history = solver.run(
        max_iterations=n_ils_warmup, perturb_method='swap', verbose=False,
        path_relink_mode=False)
    best_ms, best_st = solver.evaluate_pareto(best)

    # 初期解の情報
    init_ms, init_st = solver.evaluate_pareto(solver.initial_machine_orders)
    init_score = solver.evaluate(solver.initial_machine_orders)

    # PRを単発実行（trace=True）
    pr_result, pr_score, trace_log = solver.path_relinking(
        best, solver.initial_machine_orders, stall_limit=10, trace=True)
    pr_ms, pr_st = solver.evaluate_pareto(pr_result)

    # insert摂動でウォームアップした場合も試す
    random.seed(seed)
    solver2 = ils_scheduling.ILSSolver(
        jm_table, fixed_gantt, reschedule_gantt, reschedule_time, weights)
    solver2.estimate_normalization_params(n_samples=100, norm_params=norm_params)
    best2, best_score2, _, _ = solver2.run(
        max_iterations=n_ils_warmup, perturb_method='insert', verbose=False,
        path_relink_mode=False)
    best_ms2, best_st2 = solver2.evaluate_pareto(best2)

    pr_result2, pr_score2, trace_log2 = solver2.path_relinking(
        best2, solver2.initial_machine_orders, stall_limit=10, trace=True)
    pr_ms2, pr_st2 = solver2.evaluate_pareto(pr_result2)

    return {
        'problem': f"{problem_name}_{scenario_name}",
        'weights': weights,
        'seed': seed,
        'norm_params': norm_params,
        'init': {'makespan': init_ms, 'stability': init_st, 'score': init_score},
        'swap_warmup': {
            'ils_best': {'makespan': best_ms, 'stability': best_st, 'score': best_score},
            'pr_result': {'makespan': pr_ms, 'stability': pr_st, 'score': pr_score},
            'pr_accepted': pr_score < best_score,
            'trace': trace_log,
        },
        'insert_warmup': {
            'ils_best': {'makespan': best_ms2, 'stability': best_st2, 'score': best_score2},
            'pr_result': {'makespan': pr_ms2, 'stability': pr_st2, 'score': pr_score2},
            'pr_accepted': pr_score2 < best_score2,
            'trace': trace_log2,
        },
    }


def print_trace(label, warmup_data):
    """トレースログを見やすく表示"""
    ils = warmup_data['ils_best']
    pr = warmup_data['pr_result']
    trace = warmup_data['trace']
    accepted = warmup_data['pr_accepted']

    print(f"\n  [{label}]")
    print(f"    ILS best:  MS={ils['makespan']}, St={ils['stability']:.2f}, Score={ils['score']:.4f}")
    print(f"    PR result: MS={pr['makespan']}, St={pr['stability']:.2f}, Score={pr['score']:.4f}")
    print(f"    Accepted:  {accepted}")

    # 経路上の推移
    steps = [t for t in trace if t['type'] == 'step']
    if not steps:
        print(f"    (PRステップなし - 候補が生成されなかった)")
        return

    init_entry = [t for t in trace if t['type'] == 'init'][0]
    result_entry = [t for t in trace if t['type'] == 'result'][0]

    print(f"    初期不一致: {init_entry['diffs_to_ref']}位置")
    print(f"    最終不一致: {result_entry['final_diffs']}位置")
    print(f"    総ステップ: {result_entry['total_steps']}")

    print(f"\n    Step | MS    | Stab  | Score  | BestSc | Cands | Feas | Infeas | Diffs | Improved")
    print(f"    {'─'*95}")
    print(f"    init | {init_entry['makespan']:5d} | {init_entry['stability']:5.2f} | "
          f"{init_entry['score']:.4f} | {init_entry['best_score']:.4f} | "
          f"{'':>5} | {'':>4} | {'':>6} | {init_entry['diffs_to_ref']:5d} |")

    for s in steps:
        mark = "***" if s['improved'] else ""
        print(f"    {s['step']:4d} | {s['makespan']:5d} | {s['stability']:5.2f} | "
              f"{s['score']:.4f} | {s['best_score']:.4f} | "
              f"{s['n_candidates']:5d} | {s['n_feasible']:4d} | {s['n_infeasible']:6d} | "
              f"{s['diffs_to_ref']:5d} | {mark}")

    # スコア分布の分析
    improving_steps = [s for s in steps if s['improved']]
    worsening_steps = [s for s in steps if not s['improved']]
    print(f"\n    改善ステップ: {len(improving_steps)}/{len(steps)}")
    if improving_steps:
        print(f"    改善時のMS範囲: {min(s['makespan'] for s in improving_steps)}"
              f"〜{max(s['makespan'] for s in improving_steps)}")
        print(f"    改善時のSt範囲: {min(s['stability'] for s in improving_steps):.2f}"
              f"〜{max(s['stability'] for s in improving_steps):.2f}")
    if worsening_steps:
        print(f"    悪化時のスコア範囲: {min(s['score'] for s in worsening_steps):.4f}"
              f"〜{max(s['score'] for s in worsening_steps):.4f}")

    # feasibility分析
    total_cands = sum(s['n_candidates'] for s in steps)
    total_infeasible = sum(s['n_infeasible'] for s in steps)
    if total_cands > 0:
        print(f"\n    総候補数: {total_cands}, 実行不能: {total_infeasible} "
              f"({total_infeasible/total_cands*100:.1f}%)")


def plot_pr_trace(result, out_dir):
    """PR経路の推移をプロット"""
    prob = result['problem']
    w = result['weights']
    seed = result['seed']

    for warmup_label, warmup_key in [('swap', 'swap_warmup'), ('insert', 'insert_warmup')]:
        data = result[warmup_key]
        trace = data['trace']
        steps = [t for t in trace if t['type'] in ('init', 'step')]
        if len(steps) <= 1:
            continue

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        step_nums = [s['step'] for s in steps]
        ms_vals = [s['makespan'] for s in steps]
        st_vals = [s['stability'] for s in steps]
        sc_vals = [s['score'] for s in steps]
        best_vals = [s['best_score'] for s in steps]
        diff_vals = [s['diffs_to_ref'] for s in steps]

        ils_ms = data['ils_best']['makespan']
        ils_st = data['ils_best']['stability']
        ils_sc = data['ils_best']['score']

        axes[0][0].plot(step_nums, ms_vals, 'b-o', markersize=3, label='PR current')
        axes[0][0].axhline(ils_ms, color='r', ls='--', label=f'ILS best ({ils_ms})')
        axes[0][0].axhline(result['init']['makespan'], color='g', ls=':', label=f"Init ({result['init']['makespan']})")
        axes[0][0].set_ylabel('Makespan'); axes[0][0].legend(); axes[0][0].set_title('Makespan')
        axes[0][0].grid(True, alpha=0.3)

        axes[0][1].plot(step_nums, st_vals, 'b-o', markersize=3, label='PR current')
        axes[0][1].axhline(ils_st, color='r', ls='--', label=f'ILS best ({ils_st:.2f})')
        axes[0][1].axhline(0, color='g', ls=':', label='Init (0.00)')
        axes[0][1].set_ylabel('Stability'); axes[0][1].legend(); axes[0][1].set_title('Stability')
        axes[0][1].grid(True, alpha=0.3)

        axes[1][0].plot(step_nums, sc_vals, 'b-o', markersize=3, label='PR current score')
        axes[1][0].plot(step_nums, best_vals, 'r-', linewidth=2, label='PR best score')
        axes[1][0].axhline(ils_sc, color='orange', ls='--', label=f'ILS best ({ils_sc:.4f})')
        axes[1][0].set_ylabel('Score'); axes[1][0].legend(); axes[1][0].set_title('Weighted Score')
        axes[1][0].set_xlabel('PR Step'); axes[1][0].grid(True, alpha=0.3)

        axes[1][1].plot(step_nums, diff_vals, 'b-o', markersize=3)
        axes[1][1].set_ylabel('Diffs to Ref'); axes[1][1].set_xlabel('PR Step')
        axes[1][1].set_title('Distance to Reference'); axes[1][1].grid(True, alpha=0.3)

        fig.suptitle(f"PR Trace: {prob} w={w} seed={seed} warmup={warmup_label}", fontsize=12)
        fig.tight_layout()
        fname = f"pr_trace_{prob}_w{w[0]}_{warmup_label}_s{seed}.png"
        fig.savefig(os.path.join(out_dir, fname), dpi=150)
        plt.close(fig)


def main():
    out_dir = setup_output_dir("pr_diagnosis", base_dir=os.path.dirname(__file__))
    print(f"出力先: {out_dir}")

    seeds = [7, 107, 207]

    all_results = []
    for problem_name, scenario_name in PROBLEMS:
        for weights in WEIGHT_SETS:
            for seed in seeds:
                prob_label = f"{problem_name}_{scenario_name}"
                w_label = f"w={weights}"
                print(f"\n{'='*80}")
                print(f"{prob_label} {w_label} seed={seed}")
                print(f"{'='*80}")

                result = run_diagnosis(problem_name, scenario_name, weights, seed)
                all_results.append(result)

                print(f"\n  初期解: MS={result['init']['makespan']}, "
                      f"St={result['init']['stability']:.2f}, "
                      f"Score={result['init']['score']:.4f}")

                print_trace("swap → PR", result['swap_warmup'])
                print_trace("insert → PR", result['insert_warmup'])

                plot_pr_trace(result, out_dir)

    # サマリーJSONを保存（traceは大きいので省略）
    summary = []
    for r in all_results:
        entry = {
            'problem': r['problem'], 'weights': r['weights'], 'seed': r['seed'],
            'init': r['init'],
        }
        for key in ['swap_warmup', 'insert_warmup']:
            d = r[key]
            # traceからステップ数などだけ抽出
            steps = [t for t in d['trace'] if t['type'] == 'step']
            result_entry = [t for t in d['trace'] if t['type'] == 'result']
            entry[key] = {
                'ils_best': d['ils_best'],
                'pr_result': d['pr_result'],
                'pr_accepted': d['pr_accepted'],
                'n_steps': len(steps),
                'n_improving': sum(1 for s in steps if s['improved']),
                'total_candidates': sum(s['n_candidates'] for s in steps) if steps else 0,
                'total_infeasible': sum(s['n_infeasible'] for s in steps) if steps else 0,
                'initial_diffs': result_entry[0]['initial_diffs'] if result_entry else 0,
                'final_diffs': result_entry[0]['final_diffs'] if result_entry else 0,
            }
        summary.append(entry)

    with open(os.path.join(out_dir, "pr_diagnosis_summary.json"), 'w') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # 全体の傾向分析
    print(f"\n\n{'='*80}")
    print("全体傾向分析")
    print(f"{'='*80}")

    for key_label, key in [("swap→PR", "swap_warmup"), ("insert→PR", "insert_warmup")]:
        accepted_count = sum(1 for r in all_results if r[key]['pr_accepted'])
        total = len(all_results)
        print(f"\n  {key_label}: 受理 {accepted_count}/{total}")

        # PR結果がILS bestと比べてどう違うか
        ms_diffs = []
        st_diffs = []
        sc_diffs = []
        for r in all_results:
            d = r[key]
            ms_diffs.append(d['pr_result']['makespan'] - d['ils_best']['makespan'])
            st_diffs.append(d['pr_result']['stability'] - d['ils_best']['stability'])
            sc_diffs.append(d['pr_result']['score'] - d['ils_best']['score'])

        print(f"    PR - ILS (MS):   mean={np.mean(ms_diffs):+.1f}, "
              f"min={np.min(ms_diffs):+.0f}, max={np.max(ms_diffs):+.0f}")
        print(f"    PR - ILS (Stab): mean={np.mean(st_diffs):+.2f}, "
              f"min={np.min(st_diffs):+.2f}, max={np.max(st_diffs):+.2f}")
        print(f"    PR - ILS (Score):mean={np.mean(sc_diffs):+.4f}, "
              f"min={np.min(sc_diffs):+.4f}, max={np.max(sc_diffs):+.4f}")

    print(f"\n完了。結果は {out_dir}")


if __name__ == "__main__":
    main()
