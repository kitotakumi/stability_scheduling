"""
アクティブスケジュール vs セミアクティブスケジュール 比較実験

実験内容:
1. ILS(swap) でアクティブ/セミアクティブ/セミアクティブ+Taillard高速化 を比較
2. 重みベクトル2パターン × 10試行
3. メイクスパン・安定性・CPU時間の比較
"""

import argparse
import json
import os
import random
import sys
import time

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

import job_shop_scheduling
import gantt_chart_operation
import ils_scheduling
import evaluation

# ========== 設定 ==========
JSP_NAME = "MT10_10"
N_TRIALS = 10
ILS_MAX_ITER = 800

COMPARISON_WEIGHTS = [
    [1.0, 0.0],
    [0.9, 0.1],
]


def get_problem():
    jm_table = job_shop_scheduling.get_jm_table(JSP_NAME)
    init_gantt = jm_table.initial_gantt()
    delayed_gantt = jm_table.delayed_gantt()
    fixed_gantt, reschedule_gantt, reschedule_time, msg = \
        gantt_chart_operation.check_disturbance(init_gantt, delayed_gantt)
    return jm_table, fixed_gantt, reschedule_gantt, reschedule_time


def compute_shared_norm_params():
    jm_table, fixed_gantt, reschedule_gantt, reschedule_time = get_problem()
    delayed_gantt = jm_table.delayed_gantt()
    _, rescheduled_rsr_gantt = gantt_chart_operation.create_rsr_gantt(
        fixed_gantt, reschedule_gantt)
    base_gene = gantt_chart_operation.get_gene(rescheduled_rsr_gantt)
    return evaluation.estimate_normalization_params(
        jm_table, fixed_gantt, reschedule_time,
        delayed_gantt, base_gene, n_samples=200)


def _run_ils(weights, seed, active_schedule, taillard_acceleration,
             max_iterations, norm_params):
    jm_table, fixed_gantt, reschedule_gantt, reschedule_time = get_problem()
    random.seed(seed)
    solver = ils_scheduling.ILSSolver(
        jm_table, fixed_gantt, reschedule_gantt, reschedule_time, weights,
        active_schedule=active_schedule,
        taillard_acceleration=taillard_acceleration)
    solver.estimate_normalization_params(n_samples=100, norm_params=norm_params)
    best_orders, _, conv_info, history = solver.run(
        max_iterations=max_iterations, perturb_method='swap', verbose=False)
    ms, st = solver.evaluate_pareto(best_orders)
    return {'makespan': ms, 'stability': st, 'convergence': conv_info, 'history': history}


def run_experiment(weights, norm_params, out_dir):
    w_label = f"eff={weights[0]}_stab={weights[1]}"
    print(f"\n{'='*60}")
    print(f"重み: {weights}, {N_TRIALS}試行")
    print(f"{'='*60}")

    configs = {
        'active':         {'active_schedule': True,  'taillard_acceleration': False},
        'semi_active':    {'active_schedule': False, 'taillard_acceleration': False},
        'semi_taillard':  {'active_schedule': False, 'taillard_acceleration': True},
    }
    labels = {
        'active': 'Active (direct GT)',
        'semi_active': 'Semi-Active',
        'semi_taillard': 'Semi-Active + Taillard',
    }

    futures = {}
    with ProcessPoolExecutor() as executor:
        for config_key, cfg in configs.items():
            for trial in range(N_TRIALS):
                seed = trial * 100 + 7
                f = executor.submit(
                    _run_ils, weights, seed,
                    cfg['active_schedule'], cfg['taillard_acceleration'],
                    ILS_MAX_ITER, norm_params)
                futures[f] = (config_key, trial, seed)

    results = {k: [None] * N_TRIALS for k in configs}
    histories = {k: [None] * N_TRIALS for k in configs}

    for future in as_completed(futures):
        config_key, trial, seed = futures[future]
        r = future.result()
        results[config_key][trial] = {
            'trial': trial, 'seed': seed,
            'makespan': r['makespan'], 'stability': r['stability'],
            'convergence': r['convergence'],
        }
        histories[config_key][trial] = r['history']
        c = r['convergence']
        print(f"  {labels[config_key]:25s} Trial {trial}: MS={r['makespan']}, "
              f"Stab={r['stability']:.2f}, CPU={c['total_cpu_time']:.2f}s")

    # --- サマリー出力 ---
    lines = []
    lines.append(f"\n{'='*100}")
    lines.append(f"アクティブ vs セミアクティブ比較 (weights={weights})")
    lines.append(f"{'='*100}")

    for config_key in configs:
        data = results[config_key]
        ms_list = [d['makespan'] for d in data]
        st_list = [d['stability'] for d in data]
        cpu_best = [d['convergence']['cpu_time'] for d in data]
        cpu_total = [d['convergence']['total_cpu_time'] for d in data]
        eval_total = [d['convergence']['total_evaluations'] for d in data]

        lines.append(f"\n--- {labels[config_key]} ---")
        lines.append(f"  Makespan:  平均={np.mean(ms_list):.1f}, 最良={min(ms_list)}, "
                     f"最悪={max(ms_list)}, 標準偏差={np.std(ms_list):.1f}")
        lines.append(f"  Stability: 平均={np.mean(st_list):.2f}, 最良={min(st_list):.2f}, "
                     f"最悪={max(st_list):.2f}")
        lines.append(f"  最良解到達CPU時間: 平均={np.mean(cpu_best):.2f}s")
        lines.append(f"  全体CPU時間:       平均={np.mean(cpu_total):.2f}s")
        lines.append(f"  全体評価回数:       平均={np.mean(eval_total):.0f}")
        lines.append(f"\n  Trial | Makespan | Stability | BestCPU(s) | TotalCPU(s)")
        lines.append(f"  ------|----------|-----------|------------|------------")
        for d in data:
            c = d['convergence']
            lines.append(f"  {d['trial']:5d} | {d['makespan']:8d} | {d['stability']:9.2f} | "
                        f"{c['cpu_time']:10.2f} | {c['total_cpu_time']:10.2f}")

    text = "\n".join(lines)
    print(text)
    with open(os.path.join(out_dir, f"active_vs_semi_{w_label}.txt"), 'w', encoding='utf-8') as f:
        f.write(text)

    # --- 可視化: CPU時間ベース比較 ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors = {'active': 'tab:red', 'semi_active': 'tab:blue', 'semi_taillard': 'tab:green'}

    for config_key in configs:
        for i, history in enumerate(histories[config_key]):
            times = [h['cpu_time'] for h in history]
            ms_vals = [h['best_makespan'] for h in history]
            st_vals = [h['best_stability'] for h in history]
            is_first = (i == 0)
            alpha = 0.8 if is_first else 0.15
            lw = 1.5 if is_first else 0.8
            lbl = labels[config_key] if is_first else None
            axes[0].plot(times, ms_vals, color=colors[config_key],
                        alpha=alpha, linewidth=lw, label=lbl)
            axes[1].plot(times, st_vals, color=colors[config_key],
                        alpha=alpha, linewidth=lw, label=lbl)

    axes[0].set_xlabel('CPU Time (s)'); axes[0].set_ylabel('Makespan')
    axes[0].set_title(f'Makespan vs CPU Time ({w_label})'); axes[0].legend(); axes[0].grid(True, alpha=0.3)
    axes[1].set_xlabel('CPU Time (s)'); axes[1].set_ylabel('Stability')
    axes[1].set_title(f'Stability vs CPU Time ({w_label})'); axes[1].legend(); axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"active_vs_semi_{w_label}.png"), dpi=150)
    plt.close(fig)

    # --- 可視化: 反復ベース比較 ---
    fig, ax = plt.subplots(figsize=(10, 5))
    for config_key in configs:
        for i, history in enumerate(histories[config_key]):
            iters = [h['iteration'] for h in history]
            ms_vals = [h['best_makespan'] for h in history]
            is_first = (i == 0)
            alpha = 0.8 if is_first else 0.15
            lbl = labels[config_key] if is_first else None
            ax.plot(iters, ms_vals, color=colors[config_key],
                   alpha=alpha, linewidth=1.5 if is_first else 0.8, label=lbl)
    ax.set_xlabel('Iteration'); ax.set_ylabel('Best Makespan')
    ax.set_title(f'Makespan per Iteration ({w_label})'); ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"active_vs_semi_iter_{w_label}.png"), dpi=150)
    plt.close(fig)

    # JSON保存
    save_data = {k: results[k] for k in configs}
    with open(os.path.join(out_dir, f"active_vs_semi_{w_label}.json"), 'w') as f:
        json.dump(save_data, f, indent=2, ensure_ascii=False)

    return results


def main():
    parser = argparse.ArgumentParser(description="アクティブ vs セミアクティブ比較実験")
    parser.add_argument('--weights', nargs='+', type=str, default=None,
                       help='重み (例: "1.0,0.0")')
    parser.add_argument('--out-dir', type=str, default=None)
    args = parser.parse_args()

    if args.out_dir:
        out_dir = args.out_dir
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = os.path.join(os.path.dirname(__file__), "results", f"active_vs_semi_{timestamp}")
    os.makedirs(out_dir, exist_ok=True)

    if args.weights:
        comp_weights = [[float(x) for x in w.split(',')] for w in args.weights]
    else:
        comp_weights = COMPARISON_WEIGHTS

    print(f"出力先: {out_dir}")
    print("共通正規化パラメータ推定中...")
    norm_params = compute_shared_norm_params()
    print(f"正規化パラメータ: {norm_params}")

    for weights in comp_weights:
        run_experiment(weights, norm_params, out_dir)

    print(f"\n実験完了。結果は {out_dir} に保存されました。")


if __name__ == "__main__":
    main()
