"""
GA vs ILS 複数問題セット比較実験

GA, ILS(swap), ILS(insert) を複数の問題セットで比較する。
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
    get_initial_makespan, run_ga, run_ils,
    print_method_summary, plot_iteration_trace, plot_trajectory,
    GA_NGEN, ILS_MAX_ITER, _trial_color,
)


N_TRIALS = 10

PROBLEM_SETS = [
    ('la21', 'la21_delay147'),
    ('la36', 'la36_delay148'),
    ('la40', 'la40_delay148'),
]

METHOD_NAMES = ['GA', 'ILS_swap', 'ILS_insert']
METHOD_LABELS = {'GA': 'GA', 'ILS_swap': 'ILS(swap)', 'ILS_insert': 'ILS(insert)'}
METHOD_COLORS = {'GA': 'tab:red', 'ILS_swap': 'tab:blue', 'ILS_insert': 'tab:green'}


def _dispatch(method_name, weights, seed, norm_params, problem_name, scenario_name):
    if method_name == 'GA':
        return run_ga(weights, seed, GA_NGEN, norm_params,
                      problem_name=problem_name, scenario_name=scenario_name)
    elif method_name == 'ILS_swap':
        return run_ils(weights, seed, 'swap', ILS_MAX_ITER, norm_params,
                       problem_name=problem_name, scenario_name=scenario_name)
    elif method_name == 'ILS_insert':
        return run_ils(weights, seed, 'insert', ILS_MAX_ITER, norm_params,
                       problem_name=problem_name, scenario_name=scenario_name)


def run_problem_experiment(problem_name, scenario_name, weights, out_dir):
    """1つの問題セットに対してGA vs ILS比較を実行"""
    prob_label = f"{problem_name}_{scenario_name}"
    w_label = f"eff={weights[0]}_stab={weights[1]}"
    print(f"\n{'='*70}")
    print(f"問題: {prob_label}, weights={weights}")
    print(f"{'='*70}")

    print("  正規化パラメータ推定中...")
    norm_params = compute_shared_norm_params(problem_name, scenario_name)

    init_ms = get_initial_makespan(problem_name, scenario_name)
    print(f"  初期解メイクスパン: {init_ms}")

    # 並列実行
    futures = {}
    with ProcessPoolExecutor() as executor:
        for trial in range(N_TRIALS):
            seed = trial * 100 + 7
            for method_name in METHOD_NAMES:
                f = executor.submit(_dispatch, method_name, weights, seed, norm_params,
                                    problem_name, scenario_name)
                futures[f] = (method_name, trial, seed)

    all_results = {'problem': problem_name, 'scenario': scenario_name,
                   'weights': weights, 'init_makespan': init_ms}
    for mk in METHOD_NAMES:
        all_results[mk] = [None] * N_TRIALS
        all_results[f'{mk}_histories'] = [None] * N_TRIALS

    for future in as_completed(futures):
        method, trial, seed = futures[future]
        try:
            r = future.result()
            all_results[method][trial] = {
                'trial': trial, 'seed': seed,
                'makespan': r['makespan'], 'stability': r['stability'],
                'convergence': r['convergence'],
            }
            all_results[f'{method}_histories'][trial] = r['history']
            print(f"  Trial {trial} {METHOD_LABELS[method]}: MS={r['makespan']}, "
                  f"Stab={r['stability']:.2f}, CPU={r['convergence']['cpu_time']:.2f}s")
        except Exception as e:
            print(f"  Trial {trial} {METHOD_LABELS[method]}: ERROR - {e}")
            import traceback
            traceback.print_exc()
            all_results[method][trial] = {
                'trial': trial, 'seed': seed, 'error': str(e)
            }

    # サマリー出力
    summary_lines = [f"\n問題: {prob_label}, weights={weights}, 初期MS={init_ms}"]
    for mk in METHOD_NAMES:
        valid = [d for d in all_results[mk] if d is not None and 'error' not in d]
        if valid:
            text = print_method_summary(METHOD_LABELS[mk], valid, init_ms)
            summary_lines.append(text)

    # 問題別サブディレクトリに保存
    prob_dir = os.path.join(out_dir, prob_label)
    os.makedirs(prob_dir, exist_ok=True)

    # JSON保存
    save_results = {k: v for k, v in all_results.items() if not k.endswith('_histories')}
    with open(os.path.join(prob_dir, f"comparison_{w_label}.json"), 'w') as f:
        json.dump(save_results, f, indent=2, ensure_ascii=False)

    # 可視化
    # GA世代トレース
    ga_histories = [h for h in all_results.get('GA_histories', []) if h is not None]
    if ga_histories:
        plot_ga_generation_trace(ga_histories, prob_label, w_label, prob_dir)

    # ILSトレース
    for mk in ['ILS_swap', 'ILS_insert']:
        histories = [h for h in all_results.get(f'{mk}_histories', []) if h is not None]
        if histories:
            plot_iteration_trace(histories, f"{prob_label} {METHOD_LABELS[mk]}",
                                 w_label, prob_dir)
            plot_trajectory(histories, f"{prob_label} {METHOD_LABELS[mk]}",
                            w_label, prob_dir)

    # CPU時間比較
    plot_cpu_time_comparison(all_results, prob_label, w_label, prob_dir)

    # MS/Stability箱ひげ図
    plot_box_comparison(all_results, prob_label, w_label, prob_dir, init_ms)

    return all_results, summary_lines


def plot_ga_generation_trace(histories, prob_label, w_label, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    n = len(histories)
    for i, history in enumerate(histories):
        gens = [h['generation'] for h in history]
        ms_vals = [h['best_makespan'] for h in history]
        st_vals = [h['best_stability'] for h in history]
        axes[0].plot(gens, ms_vals, color=_trial_color(i, n), alpha=0.7, linewidth=1.0,
                     label=f'Trial {i} (MS={ms_vals[-1]})')
        axes[1].plot(gens, st_vals, color=_trial_color(i, n), alpha=0.7, linewidth=1.0,
                     label=f'Trial {i}')
    axes[0].set_xlabel('Generation'); axes[0].set_ylabel('Best Makespan')
    axes[0].set_title(f'{prob_label} GA: Makespan'); axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=7, loc='upper right')
    axes[1].set_xlabel('Generation'); axes[1].set_ylabel('Best Stability')
    axes[1].set_title(f'{prob_label} GA: Stability'); axes[1].grid(True, alpha=0.3)
    fig.suptitle(f'{prob_label} GA Trace ({w_label})', fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"ga_trace_{w_label}.png"), dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_cpu_time_comparison(all_results, prob_label, w_label, out_dir):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for method_key in METHOD_NAMES:
        if f'{method_key}_histories' not in all_results:
            continue
        color = METHOD_COLORS[method_key]
        histories = all_results[f'{method_key}_histories']
        for i, history in enumerate(histories):
            if history is None:
                continue
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
        ax.set_title(f'{prob_label} {title}'); ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"cpu_comparison_{w_label}.png"), dpi=150)
    plt.close(fig)


def plot_box_comparison(all_results, prob_label, w_label, out_dir, init_ms):
    """MS, Stability, CPU時間の箱ひげ図で手法比較"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    ms_data, st_data, cpu_data = [], [], []
    labels = []
    for mk in METHOD_NAMES:
        valid = [d for d in all_results[mk] if d is not None and 'error' not in d]
        if not valid:
            continue
        labels.append(METHOD_LABELS[mk])
        ms_data.append([d['makespan'] for d in valid])
        st_data.append([d['stability'] for d in valid])
        cpu_data.append([d['convergence']['total_cpu_time'] for d in valid])

    colors = [METHOD_COLORS[mk] for mk in METHOD_NAMES if
              any(d is not None and 'error' not in d for d in all_results[mk])]

    for ax, data, ylabel, title in zip(axes,
            [ms_data, st_data, cpu_data],
            ['Makespan', 'Stability', 'Total CPU Time (s)'],
            ['Makespan', 'Stability', 'CPU Time']):
        bp = ax.boxplot(data, labels=labels, patch_artist=True)
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        ax.set_ylabel(ylabel)
        ax.set_title(f'{prob_label} {title}')
        ax.grid(True, alpha=0.3, axis='y')
        if ylabel == 'Makespan':
            ax.axhline(y=init_ms, color='gray', linestyle='--', alpha=0.7, label=f'Initial ({init_ms})')
            ax.legend()

    fig.suptitle(f'{prob_label} GA vs ILS ({w_label})', fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"box_comparison_{w_label}.png"), dpi=150, bbox_inches='tight')
    plt.close(fig)


def write_cross_problem_summary(all_summaries, out_dir):
    with open(os.path.join(out_dir, "cross_problem_summary.txt"), 'w', encoding='utf-8') as f:
        f.write("GA vs ILS Multi-Problem Comparison\n")
        f.write("=" * 70 + "\n")
        for lines in all_summaries:
            for line in lines:
                f.write(line + "\n")
            f.write("\n")
    print(f"\n横断サマリー: {os.path.join(out_dir, 'cross_problem_summary.txt')}")


def main():
    parser = argparse.ArgumentParser(description="GA vs ILS 複数問題比較実験")
    parser.add_argument('--problems', nargs='+', type=str, default=None,
                        help='問題 (例: la21:la21_delay147 la36:la36_delay148)')
    parser.add_argument('--weights', nargs='+', type=str, default=['1.0,0.0', '0.9,0.1'],
                        help='比較する重み')
    args = parser.parse_args()

    if args.problems:
        problem_sets = [(p.split(':')[0], p.split(':')[1]) for p in args.problems]
    else:
        problem_sets = PROBLEM_SETS

    weight_list = [[float(x) for x in w.split(',')] for w in args.weights]

    out_dir = setup_output_dir("ga_vs_ils", base_dir=os.path.dirname(__file__))
    print(f"出力先: {out_dir}")
    print(f"問題セット: {problem_sets}")
    print(f"重み: {weight_list}")

    all_summaries = []
    for problem_name, scenario_name in problem_sets:
        for weights in weight_list:
            try:
                _, summary_lines = run_problem_experiment(
                    problem_name, scenario_name, weights, out_dir)
                all_summaries.append(summary_lines)
            except Exception as e:
                print(f"\nERROR: {problem_name}/{scenario_name} weights={weights}: {e}")
                import traceback
                traceback.print_exc()
                all_summaries.append([f"ERROR: {problem_name}/{scenario_name}: {e}"])

    write_cross_problem_summary(all_summaries, out_dir)
    print(f"\n全実験完了。結果は {out_dir} に保存されました。")


if __name__ == "__main__":
    main()
