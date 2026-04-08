"""
ILS性能検証: 複数問題セットでの実験

mt10以外の新規問題セット (la21, la36, la40) を含めて
ILSの性能を検証する。
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

# 問題セット定義: (problem_name, scenario_name)
PROBLEM_SETS = [
    ('mt10', 'mt10_delay60'),
    ('la21', 'la21_delay147'),
    ('la36', 'la36_delay148'),
    ('la40', 'la40_delay148'),
]

# ILS手法定義
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


def _run_method(method_key, weights, seed, norm_params, problem_name, scenario_name):
    perturb, relink_mode, trigger = ILS_METHODS[method_key]
    return run_ils(weights, seed, perturb, ILS_MAX_ITER, norm_params,
                   path_relink_mode=relink_mode, relink_trigger=trigger,
                   problem_name=problem_name, scenario_name=scenario_name)


def run_problem_experiment(problem_name, scenario_name, weights, methods, out_dir):
    """1つの問題セットに対してILS比較実験を実行"""
    prob_label = f"{problem_name}_{scenario_name}"
    w_label = f"eff={weights[0]}_stab={weights[1]}"
    print(f"\n{'='*70}")
    print(f"問題: {prob_label}, weights={weights}")
    print(f"{'='*70}")

    # 正規化パラメータ推定
    print("  正規化パラメータ推定中...")
    norm_params = compute_shared_norm_params(problem_name, scenario_name)

    # 初期解メイクスパン
    init_ms = get_initial_makespan(problem_name, scenario_name)
    print(f"  初期解メイクスパン: {init_ms}")

    # 並列実行
    futures = {}
    with ProcessPoolExecutor() as executor:
        for trial in range(N_TRIALS):
            seed = trial * 100 + 7
            for mk in methods:
                f = executor.submit(_run_method, mk, weights, seed, norm_params,
                                    problem_name, scenario_name)
                futures[f] = (mk, trial, seed)

    all_results = {'problem': problem_name, 'scenario': scenario_name,
                   'weights': weights, 'init_makespan': init_ms}
    for mk in methods:
        all_results[mk] = [None] * N_TRIALS
        all_results[f'{mk}_histories'] = [None] * N_TRIALS

    for future in as_completed(futures):
        mk, trial, seed = futures[future]
        try:
            r = future.result()
            all_results[mk][trial] = {
                'trial': trial, 'seed': seed,
                'makespan': r['makespan'], 'stability': r['stability'],
                'convergence': r['convergence'],
            }
            all_results[f'{mk}_histories'][trial] = r['history']
            print(f"  Trial {trial} {METHOD_LABELS[mk]}: MS={r['makespan']}, "
                  f"Stab={r['stability']:.2f}, CPU={r['convergence']['cpu_time']:.2f}s")
        except Exception as e:
            print(f"  Trial {trial} {METHOD_LABELS[mk]}: ERROR - {e}")
            all_results[mk][trial] = {
                'trial': trial, 'seed': seed, 'error': str(e)
            }

    # サマリー出力
    summary_lines = [f"\n問題: {prob_label}, weights={weights}"]
    for mk in methods:
        valid = [d for d in all_results[mk] if d is not None and 'error' not in d]
        if valid:
            text = print_method_summary(METHOD_LABELS[mk], valid, init_ms)
            summary_lines.append(text)

    # 問題別サブディレクトリに保存
    prob_dir = os.path.join(out_dir, prob_label)
    os.makedirs(prob_dir, exist_ok=True)

    # JSON保存
    save_results = {k: v for k, v in all_results.items() if not k.endswith('_histories')}
    with open(os.path.join(prob_dir, f"results_{w_label}.json"), 'w') as f:
        json.dump(save_results, f, indent=2, ensure_ascii=False)

    # 可視化
    for mk in methods:
        histories = all_results.get(f'{mk}_histories', [])
        valid_histories = [h for h in histories if h is not None]
        if valid_histories:
            plot_iteration_trace(valid_histories, f"{prob_label} {METHOD_LABELS[mk]}",
                                 w_label, prob_dir)
            plot_trajectory(valid_histories, f"{prob_label} {METHOD_LABELS[mk]}",
                            w_label, prob_dir)

    # CPU時間比較プロット
    plot_cpu_comparison(all_results, methods, w_label, prob_dir, prob_label)

    return all_results, summary_lines


def plot_cpu_comparison(all_results, methods, w_label, out_dir, prob_label=""):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for mk in methods:
        if f'{mk}_histories' not in all_results:
            continue
        color = METHOD_COLORS[mk]
        histories = all_results[f'{mk}_histories']
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
            lbl = METHOD_LABELS[mk] if is_first else None
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


def write_cross_problem_summary(all_summaries, out_dir):
    """全問題の横断サマリーをファイルに出力"""
    with open(os.path.join(out_dir, "cross_problem_summary.txt"), 'w', encoding='utf-8') as f:
        f.write("ILS Multi-Problem Experiment Summary\n")
        f.write("=" * 70 + "\n")
        for lines in all_summaries:
            for line in lines:
                f.write(line + "\n")
            f.write("\n")
    print(f"\n横断サマリー: {os.path.join(out_dir, 'cross_problem_summary.txt')}")


def main():
    parser = argparse.ArgumentParser(description="ILS複数問題セット実験")
    parser.add_argument('--problems', nargs='+', type=str, default=None,
                        help='実行する問題 (例: mt10:mt10_delay60 la21:la21_delay147)')
    parser.add_argument('--weights', nargs='+', type=str, default=['1.0,0.0', '0.9,0.1'],
                        help='比較する重み')
    parser.add_argument('--methods', nargs='+',
                        default=list(ILS_METHODS.keys()),
                        choices=list(ILS_METHODS.keys()),
                        help='実行する手法')
    args = parser.parse_args()

    # 問題セット解析
    if args.problems:
        problem_sets = []
        for p in args.problems:
            parts = p.split(':')
            problem_sets.append((parts[0], parts[1]))
    else:
        problem_sets = PROBLEM_SETS

    weight_list = [[float(x) for x in w.split(',')] for w in args.weights]
    methods = args.methods

    out_dir = setup_output_dir("multi_problem", base_dir=os.path.dirname(__file__))
    print(f"出力先: {out_dir}")
    print(f"問題セット: {problem_sets}")
    print(f"重み: {weight_list}")
    print(f"手法: {methods}")

    all_summaries = []
    for problem_name, scenario_name in problem_sets:
        for weights in weight_list:
            try:
                _, summary_lines = run_problem_experiment(
                    problem_name, scenario_name, weights, methods, out_dir)
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
