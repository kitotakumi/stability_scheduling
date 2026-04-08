"""
ILS詳細分析実験

摂動手法の比較: swap, insert, path_relinkモード（swap+relink, insert+relink）
path_relinkの安定性修復効果を検証する。
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


N_TRIALS = 10

# 手法定義: (perturb_method, path_relink_mode, relink_trigger)
ILS_METHODS = {
    'ILS_swap':           ('swap',   False, 0),
    'ILS_insert':         ('insert', False, 0),
    'ILS_swap_relink':    ('swap',   True,  200),
    'ILS_insert_relink':  ('insert', True,  200),
}

METHOD_LABELS = {
    'ILS_swap': 'ILS(swap)',
    'ILS_insert': 'ILS(insert)',
    'ILS_swap_relink': 'ILS(swap+relink)',
    'ILS_insert_relink': 'ILS(insert+relink)',
}

METHOD_COLORS = {
    'ILS_swap': 'tab:blue',
    'ILS_insert': 'tab:green',
    'ILS_swap_relink': 'tab:cyan',
    'ILS_insert_relink': 'tab:purple',
}


def _run_method(method_key, weights, seed, norm_params):
    perturb, relink_mode, trigger = ILS_METHODS[method_key]
    return run_ils(weights, seed, perturb, ILS_MAX_ITER, norm_params,
                   path_relink_mode=relink_mode, relink_trigger=trigger)


def run_comparison(weights, methods, norm_params):
    w_label = f"eff={weights[0]}_stab={weights[1]}"
    print(f"\nILS詳細比較 (weights={weights}, methods={list(methods)})")
    print("=" * 70)

    futures = {}
    with ProcessPoolExecutor() as executor:
        for trial in range(N_TRIALS):
            seed = trial * 100 + 7
            for mk in methods:
                f = executor.submit(_run_method, mk, weights, seed, norm_params)
                futures[f] = (mk, trial, seed)

    all_results = {'weights': weights}
    for mk in methods:
        all_results[mk] = [None] * N_TRIALS
        all_results[f'{mk}_histories'] = [None] * N_TRIALS

    for future in as_completed(futures):
        mk, trial, seed = futures[future]
        r = future.result()
        all_results[mk][trial] = {
            'trial': trial, 'seed': seed,
            'makespan': r['makespan'], 'stability': r['stability'],
            'convergence': r['convergence'],
        }
        all_results[f'{mk}_histories'][trial] = r['history']
        print(f"  Trial {trial} {METHOD_LABELS[mk]}: MS={r['makespan']}, "
              f"Stab={r['stability']:.2f}, BestCPU={r['convergence']['cpu_time']:.2f}s")

    return all_results, w_label


def plot_cpu_comparison(all_results, methods, w_label, out_dir):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for mk in methods:
        if f'{mk}_histories' not in all_results:
            continue
        color = METHOD_COLORS[mk]
        histories = all_results[f'{mk}_histories']
        for i, history in enumerate(histories):
            times = [h['cpu_time'] for h in history]
            score_vals = [h['best_score'] for h in history]
            ms_vals = [h['best_makespan'] for h in history]
            st_vals = [h['best_stability'] for h in history]
            is_first = (i == 0)
            alpha = 0.8 if is_first else 0.15
            lw = 1.5 if is_first else 0.8
            lbl = METHOD_LABELS[mk] if is_first else None
            axes[0].plot(times, score_vals, color=color, alpha=alpha, linewidth=lw, label=lbl)
            axes[1].plot(times, ms_vals, color=color, alpha=alpha, linewidth=lw, label=lbl)
            axes[2].plot(times, st_vals, color=color, alpha=alpha, linewidth=lw, label=lbl)

    for ax, ylabel, title in zip(axes,
            ['Weighted Objective', 'Makespan', 'Stability'],
            [f'Objective ({w_label})', 'Makespan vs CPU Time', 'Stability vs CPU Time']):
        ax.set_xlabel('CPU Time (s)'); ax.set_ylabel(ylabel)
        ax.set_title(title); ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"ils_cpu_comparison_{w_label}.png"), dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="ILS詳細分析実験")
    parser.add_argument('--weights', nargs='+', type=str, default=['1.0,0.0', '0.9,0.1'],
                        help='比較する重み')
    parser.add_argument('--methods', nargs='+',
                        default=list(ILS_METHODS.keys()),
                        choices=list(ILS_METHODS.keys()),
                        help='実行する手法')
    args = parser.parse_args()

    weight_list = [[float(x) for x in w.split(',')] for w in args.weights]
    methods = args.methods

    out_dir = setup_output_dir("run", base_dir=os.path.dirname(__file__))
    print(f"出力先: {out_dir}")

    init_ms = get_initial_makespan()
    print(f"初期解メイクスパン: {init_ms}")

    print("\n共通正規化パラメータを推定中...")
    norm_params = compute_shared_norm_params()

    for weights in weight_list:
        all_results, w_label = run_comparison(weights, methods, norm_params)

        # サマリー出力
        for mk in methods:
            print_method_summary(METHOD_LABELS[mk], all_results[mk], init_ms)

        # JSON保存
        save_results = {k: v for k, v in all_results.items() if not k.endswith('_histories')}
        with open(os.path.join(out_dir, f"ils_comparison_{w_label}.json"), 'w') as f:
            json.dump(save_results, f, indent=2, ensure_ascii=False)

        # 可視化
        for mk in methods:
            plot_iteration_trace(all_results[f'{mk}_histories'],
                                 METHOD_LABELS[mk], w_label, out_dir)
            plot_trajectory(all_results[f'{mk}_histories'],
                            METHOD_LABELS[mk], w_label, out_dir)
        plot_cpu_comparison(all_results, methods, w_label, out_dir)

    print(f"\n完了。結果は {out_dir} に保存されました。")


if __name__ == "__main__":
    main()
