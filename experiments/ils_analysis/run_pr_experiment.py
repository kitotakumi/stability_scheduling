"""
Path Relinking (direct swap) 実験 — 複数問題対応

新しいpath_relinking()メソッド（設計書準拠のdirect swap）の効果を検証する。
比較:
  - ILS(swap): ベースライン
  - ILS(swap+PR trigger=50): 停滞50反復でPR発動
  - ILS(swap+PR trigger=200): 停滞200反復でPR発動
"""

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from experiment_utils import (
    setup_output_dir, compute_shared_norm_params,
    get_initial_makespan, run_ils,
    print_method_summary, plot_iteration_trace, plot_trajectory,
    ILS_MAX_ITER,
)


# 問題/シナリオの組み合わせ
PROBLEMS = [
    ('mt10', 'mt10_delay60'),
    ('mt10', 'mt10_delay90'),
    ('la21', 'la21_delay147'),
    ('la36', 'la36_delay148'),
    ('la40', 'la40_delay148'),
]

# (perturb_method, path_relink_mode, relink_trigger)
ILS_METHODS = {
    'ILS_swap':              ('swap',   False, 0),
    'ILS_insert':            ('insert', False, 0),
    'ILS_swap_relink50':     ('swap',   True,  50),
    'ILS_swap_relink200':    ('swap',   True,  200),
    'ILS_insert_relink50':   ('insert', True,  50),
    'ILS_insert_relink200':  ('insert', True,  200),
}

METHOD_LABELS = {
    'ILS_swap': 'ILS(swap)',
    'ILS_insert': 'ILS(insert)',
    'ILS_swap_relink50': 'ILS(swap+PR t=50)',
    'ILS_swap_relink200': 'ILS(swap+PR t=200)',
    'ILS_insert_relink50': 'ILS(insert+PR t=50)',
    'ILS_insert_relink200': 'ILS(insert+PR t=200)',
}

METHOD_COLORS = {
    'ILS_swap': 'tab:blue',
    'ILS_insert': 'tab:green',
    'ILS_swap_relink50': 'tab:orange',
    'ILS_swap_relink200': 'tab:red',
    'ILS_insert_relink50': 'tab:cyan',
    'ILS_insert_relink200': 'tab:purple',
}


def _run_method(method_key, weights, seed, norm_params, problem_name, scenario_name):
    perturb, relink_mode, trigger = ILS_METHODS[method_key]
    return run_ils(weights, seed, perturb, ILS_MAX_ITER, norm_params,
                   path_relink_mode=relink_mode, relink_trigger=trigger,
                   problem_name=problem_name, scenario_name=scenario_name)


def run_comparison(weights, methods, norm_params, n_trials, problem_name, scenario_name):
    w_label = f"eff={weights[0]}_stab={weights[1]}"
    prob_label = f"{problem_name}_{scenario_name}"
    print(f"\n[{prob_label}] PR比較 (weights={weights})")
    print("=" * 70)

    futures = {}
    with ProcessPoolExecutor() as executor:
        for trial in range(n_trials):
            seed = trial * 100 + 7
            for mk in methods:
                f = executor.submit(_run_method, mk, weights, seed, norm_params,
                                    problem_name, scenario_name)
                futures[f] = (mk, trial, seed)

    all_results = {'weights': weights, 'problem': prob_label}
    for mk in methods:
        all_results[mk] = [None] * n_trials
        all_results[f'{mk}_histories'] = [None] * n_trials

    for future in as_completed(futures):
        mk, trial, seed = futures[future]
        r = future.result()
        all_results[mk][trial] = {
            'trial': trial, 'seed': seed,
            'makespan': r['makespan'], 'stability': r['stability'],
            'convergence': r['convergence'],
        }
        all_results[f'{mk}_histories'][trial] = r['history']

    # トライアル順に整列して表示
    for mk in methods:
        for trial in range(n_trials):
            r = all_results[mk][trial]
            print(f"  Trial {trial:2d} {METHOD_LABELS[mk]:25s}: "
                  f"MS={r['makespan']}, Stab={r['stability']:.2f}, "
                  f"CPU={r['convergence']['cpu_time']:.2f}s")

    return all_results, w_label, prob_label


def print_detailed_summary(all_results, methods, init_ms, prob_label):
    print(f"\n{'=' * 70}")
    print(f"サマリー [{prob_label}] (初期MS={init_ms})")
    print(f"{'=' * 70}")
    for mk in methods:
        results = all_results[mk]
        ms_list = [r['makespan'] for r in results]
        st_list = [r['stability'] for r in results]
        cpu_list = [r['convergence']['cpu_time'] for r in results]
        iter_list = [r['convergence']['iteration'] for r in results]
        print(f"\n--- {METHOD_LABELS[mk]} ---")
        print(f"  Makespan:  mean={np.mean(ms_list):.1f}, std={np.std(ms_list):.1f}, "
              f"min={np.min(ms_list)}, max={np.max(ms_list)}")
        print(f"  Stability: mean={np.mean(st_list):.2f}, std={np.std(st_list):.2f}, "
              f"min={np.min(st_list):.2f}, max={np.max(st_list):.2f}")
        print(f"  BestIter:  mean={np.mean(iter_list):.0f}")
        print(f"  BestCPU:   mean={np.mean(cpu_list):.2f}s")
        if init_ms:
            gap = (np.mean(ms_list) - init_ms) / init_ms * 100
            print(f"  MS gap from initial: {gap:+.2f}%")


def print_cross_problem_table(all_summaries, methods):
    """全問題を横断した比較テーブル"""
    print(f"\n{'=' * 90}")
    print("全問題横断サマリー")
    print(f"{'=' * 90}")
    header = f"{'Problem':30s}"
    for mk in methods:
        header += f" | {METHOD_LABELS[mk]:25s}"
    print(header)
    print("-" * len(header))

    for prob_label, init_ms, summaries in all_summaries:
        row = f"{prob_label + f' (init={init_ms})':30s}"
        for mk in methods:
            ms_mean = summaries[mk]['ms_mean']
            st_mean = summaries[mk]['st_mean']
            row += f" | MS={ms_mean:7.1f} St={st_mean:5.2f}    "
        print(row)


def main():
    parser = argparse.ArgumentParser(description="Path Relinking実験（複数問題）")
    parser.add_argument('--weights', nargs='+', type=str,
                        default=['0.95,0.05', '0.8,0.2'])
    parser.add_argument('--methods', nargs='+',
                        default=list(ILS_METHODS.keys()),
                        choices=list(ILS_METHODS.keys()))
    parser.add_argument('--trials', type=int, default=5)
    parser.add_argument('--problems', nargs='+', type=str, default=None,
                        help='問題を絞る場合 (例: mt10_delay60 la21_delay147)')
    args = parser.parse_args()

    n_trials = args.trials
    weight_list = [[float(x) for x in w.split(',')] for w in args.weights]
    methods = args.methods

    # 問題フィルタ
    if args.problems:
        problems = [(p, s) for p, s in PROBLEMS
                    if f"{p}_{s}" in args.problems or p in args.problems]
    else:
        problems = PROBLEMS

    out_dir = setup_output_dir("pr_multi", base_dir=os.path.dirname(__file__))
    print(f"出力先: {out_dir}")
    print(f"問題: {[f'{p}_{s}' for p,s in problems]}")
    print(f"手法: {[METHOD_LABELS[m] for m in methods]}")
    print(f"重み: {args.weights}")
    print(f"試行数: {n_trials}")

    for weights in weight_list:
        w_label = f"eff={weights[0]}_stab={weights[1]}"
        all_summaries = []

        for problem_name, scenario_name in problems:
            prob_label = f"{problem_name}_{scenario_name}"

            print(f"\n--- {prob_label}: 正規化パラメータ推定中... ---")
            norm_params = compute_shared_norm_params(problem_name, scenario_name)
            init_ms = get_initial_makespan(problem_name, scenario_name)
            print(f"  init_ms={init_ms}, norm={norm_params}")

            all_results, _, _ = run_comparison(
                weights, methods, norm_params, n_trials, problem_name, scenario_name)

            print_detailed_summary(all_results, methods, init_ms, prob_label)

            # 集計
            summaries = {}
            for mk in methods:
                results = all_results[mk]
                summaries[mk] = {
                    'ms_mean': np.mean([r['makespan'] for r in results]),
                    'st_mean': np.mean([r['stability'] for r in results]),
                    'ms_std': np.std([r['makespan'] for r in results]),
                    'st_std': np.std([r['stability'] for r in results]),
                }
            all_summaries.append((prob_label, init_ms, summaries))

            # JSON保存
            save_results = {k: v for k, v in all_results.items()
                           if not k.endswith('_histories')}
            fname = f"pr_{prob_label}_{w_label}.json"
            with open(os.path.join(out_dir, fname), 'w') as f:
                json.dump(save_results, f, indent=2, ensure_ascii=False)

        # 全問題横断テーブル
        print_cross_problem_table(all_summaries, methods)

        # 横断テーブルをJSONでも保存
        table_data = []
        for prob_label, init_ms, summaries in all_summaries:
            entry = {'problem': prob_label, 'init_ms': init_ms}
            for mk in methods:
                entry[mk] = summaries[mk]
            table_data.append(entry)
        with open(os.path.join(out_dir, f"pr_cross_summary_{w_label}.json"), 'w') as f:
            json.dump(table_data, f, indent=2, ensure_ascii=False)

    print(f"\n完了。結果は {out_dir} に保存されました。")


if __name__ == "__main__":
    main()
