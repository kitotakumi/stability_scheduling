"""
GA vs ILS 比較実験

GA, ILS(swap), ILS(insert) の3手法を指定した重みで10試行比較する。
重みスイープで決定した最適重みを使用する想定。
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
    setup_output_dir, get_problem, compute_shared_norm_params,
    get_initial_makespan, run_ga, run_ils,
    print_method_summary, plot_iteration_trace, plot_trajectory,
    GA_NGEN, ILS_MAX_ITER, _trial_color,
)


N_TRIALS = 10

METHOD_NAMES = ['GA', 'ILS_swap', 'ILS_insert']


def _dispatch(method_name, weights, seed, norm_params):
    if method_name == 'GA':
        return run_ga(weights, seed, GA_NGEN, norm_params)
    elif method_name == 'ILS_swap':
        return run_ils(weights, seed, 'swap', ILS_MAX_ITER, norm_params)
    elif method_name == 'ILS_insert':
        return run_ils(weights, seed, 'insert', ILS_MAX_ITER, norm_params)

METHOD_LABELS = {'GA': 'GA', 'ILS_swap': 'ILS(swap)', 'ILS_insert': 'ILS(insert)'}
METHOD_COLORS = {'GA': 'tab:red', 'ILS_swap': 'tab:blue', 'ILS_insert': 'tab:green'}


def run_comparison(weights, norm_params):
    w_label = f"eff={weights[0]}_stab={weights[1]}"
    print(f"\n10回比較実験 (weights={weights})")
    print("=" * 70)

    futures = {}
    with ProcessPoolExecutor() as executor:
        for trial in range(N_TRIALS):
            seed = trial * 100 + 7
            for method_name in METHOD_NAMES:
                f = executor.submit(_dispatch, method_name, weights, seed, norm_params)
                futures[f] = (method_name, trial, seed)

    all_results = {'weights': weights}
    for mk in METHOD_NAMES:
        all_results[mk] = [None] * N_TRIALS
        all_results[f'{mk}_histories'] = [None] * N_TRIALS

    for future in as_completed(futures):
        method, trial, seed = futures[future]
        r = future.result()
        all_results[method][trial] = {
            'trial': trial, 'seed': seed,
            'makespan': r['makespan'], 'stability': r['stability'],
            'convergence': r['convergence'],
        }
        all_results[f'{method}_histories'][trial] = r['history']
        print(f"  Trial {trial} {method}: MS={r['makespan']}, Stab={r['stability']:.2f}, "
              f"BestCPU={r['convergence']['cpu_time']:.2f}s")

    return all_results, w_label


def plot_ga_generation_trace(all_results, w_label, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    n = len(all_results['GA_histories'])
    for i, history in enumerate(all_results['GA_histories']):
        gens = [h['generation'] for h in history]
        ms_vals = [h['best_makespan'] for h in history]
        st_vals = [h['best_stability'] for h in history]
        axes[0].plot(gens, ms_vals, color=_trial_color(i, n), alpha=0.7, linewidth=1.0,
                     label=f'Trial {i} (MS={ms_vals[-1]})')
        axes[1].plot(gens, st_vals, color=_trial_color(i, n), alpha=0.7, linewidth=1.0,
                     label=f'Trial {i}')
    axes[0].set_xlabel('Generation'); axes[0].set_ylabel('Best Makespan')
    axes[0].set_title('GA: Best Makespan per Generation'); axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=7, loc='upper right')
    axes[1].set_xlabel('Generation'); axes[1].set_ylabel('Best Stability')
    axes[1].set_title('GA: Best Stability per Generation'); axes[1].grid(True, alpha=0.3)
    fig.suptitle(f'GA Generation Trace ({w_label})', fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"ga_trace_{w_label}.png"), dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_cpu_time_comparison(all_results, w_label, out_dir):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for method_key in METHOD_NAMES:
        if f'{method_key}_histories' not in all_results:
            continue
        color = METHOD_COLORS[method_key]
        histories = all_results[f'{method_key}_histories']
        for i, history in enumerate(histories):
            times = [h['cpu_time'] for h in history]
            score_vals = [h['best_score'] for h in history]
            ms_vals = [h['best_makespan'] for h in history]
            st_vals = [h['best_stability'] for h in history]
            is_first = (i == 0)
            alpha = 0.8 if is_first else 0.15
            lw = 1.5 if is_first else 0.8
            lbl = METHOD_LABELS[method_key] if is_first else None
            axes[0].plot(times, score_vals, color=color, alpha=alpha, linewidth=lw, label=lbl)
            axes[1].plot(times, ms_vals, color=color, alpha=alpha, linewidth=lw, label=lbl)
            axes[2].plot(times, st_vals, color=color, alpha=alpha, linewidth=lw, label=lbl)

    for ax, ylabel, title in zip(axes,
            ['Weighted Objective', 'Makespan', 'Stability'],
            [f'Objective ({w_label})', 'Makespan vs CPU Time', 'Stability vs CPU Time']):
        ax.set_xlabel('CPU Time (s)'); ax.set_ylabel(ylabel)
        ax.set_title(title); ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"cpu_comparison_{w_label}.png"), dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="GA vs ILS 比較実験")
    parser.add_argument('--weights', nargs='+', type=str, default=['1.0,0.0', '0.9,0.1'],
                        help='比較する重み (例: "1.0,0.0" "0.9,0.1")')
    args = parser.parse_args()

    weight_list = [[float(x) for x in w.split(',')] for w in args.weights]

    out_dir = setup_output_dir("run", base_dir=os.path.dirname(__file__))
    print(f"出力先: {out_dir}")

    init_ms = get_initial_makespan()
    print(f"初期解メイクスパン: {init_ms}")

    print("\n共通正規化パラメータを推定中...")
    norm_params = compute_shared_norm_params()

    for weights in weight_list:
        all_results, w_label = run_comparison(weights, norm_params)

        # サマリー出力
        for mk in METHOD_NAMES:
            print_method_summary(METHOD_LABELS[mk], all_results[mk], init_ms)

        # JSON保存
        save_results = {k: v for k, v in all_results.items() if not k.endswith('_histories')}
        with open(os.path.join(out_dir, f"comparison_{w_label}.json"), 'w') as f:
            json.dump(save_results, f, indent=2, ensure_ascii=False)

        # 可視化
        plot_ga_generation_trace(all_results, w_label, out_dir)
        for mk in ['ILS_swap', 'ILS_insert']:
            plot_iteration_trace(all_results[f'{mk}_histories'], METHOD_LABELS[mk], w_label, out_dir)
            plot_trajectory(all_results[f'{mk}_histories'], METHOD_LABELS[mk], w_label, out_dir)
        plot_cpu_time_comparison(all_results, w_label, out_dir)

    print(f"\n完了。結果は {out_dir} に保存されました。")


if __name__ == "__main__":
    main()
