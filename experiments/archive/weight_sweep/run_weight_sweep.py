"""
重みベクトルスイープ実験

GA / ILS(swap) / ILS(insert) で安定性重みを0〜1の範囲で変化させ、
各手法の最適な重みと重み耐性を調査する。
"""

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
    GA_NGEN, ILS_MAX_ITER,
)


# ========== 設定 ==========

SEEDS = [7, 107, 207, 307, 407, 507, 607]  # 7シード

# 0.05刻みの細かいスイープ
WEIGHT_CANDIDATES = [
    [1.0, 0.0],
    [0.95, 0.05],
    [0.9, 0.1],
    [0.85, 0.15],
    [0.8, 0.2],
    [0.75, 0.25],
    [0.7, 0.3],
    [0.6, 0.4],
    [0.5, 0.5],
    [0.3, 0.7],
    [0.0, 1.0],
]

METHOD_NAMES = ['GA', 'ILS_swap', 'ILS_insert']


def _dispatch(method_name, weights, seed, norm_params):
    if method_name == 'GA':
        return run_ga(weights, seed, 200, norm_params)
    elif method_name == 'ILS_swap':
        return run_ils(weights, seed, 'swap', 400, norm_params)
    elif method_name == 'ILS_insert':
        return run_ils(weights, seed, 'insert', 400, norm_params)


def run_sweep(out_dir, norm_params, init_ms=1080):
    print(f"\n重みベクトルスイープ ({len(WEIGHT_CANDIDATES)}重み × {len(SEEDS)}シード × {len(METHOD_NAMES)}手法)")
    print("=" * 70)

    futures = {}
    with ProcessPoolExecutor() as executor:
        for weights in WEIGHT_CANDIDATES:
            w_key = f"eff={weights[0]},stab={weights[1]}"
            for seed in SEEDS:
                for method_name in METHOD_NAMES:
                    f = executor.submit(_dispatch, method_name, weights, seed, norm_params)
                    futures[f] = (method_name, w_key, weights, seed)

    # 収集
    raw = {m: {} for m in METHOD_NAMES}
    for future in as_completed(futures):
        method, w_key, weights, seed = futures[future]
        r = future.result()
        if w_key not in raw[method]:
            raw[method][w_key] = {'weights': weights, 'runs': []}
        raw[method][w_key]['runs'].append({
            'seed': seed,
            'makespan': r['makespan'],
            'stability': r['stability'],
            'cpu_time': r['convergence']['total_cpu_time'],
        })

    # 集約（改善成功試行のみの統計 + 改善成功率）
    results = {m: [] for m in METHOD_NAMES}
    for method in METHOD_NAMES:
        for w_key in sorted(raw[method].keys(), reverse=True):
            data = raw[method][w_key]
            runs = data['runs']
            improved_runs = [r for r in runs if r['makespan'] < init_ms]
            n_improved = len(improved_runs)
            n_total = len(runs)

            entry = {
                'weights': data['weights'],
                'w_stab': data['weights'][1],
                'n_seeds': n_total,
                'n_improved': n_improved,
                'improve_rate': n_improved / n_total,
                'makespan_best': min(r['makespan'] for r in runs),
            }
            if n_improved > 0:
                entry['makespan_mean'] = float(np.mean([r['makespan'] for r in improved_runs]))
                entry['makespan_std'] = float(np.std([r['makespan'] for r in improved_runs]))
                entry['stability_mean'] = float(np.mean([r['stability'] for r in improved_runs]))
                entry['stability_std'] = float(np.std([r['stability'] for r in improved_runs]))
                entry['cpu_time_mean'] = float(np.mean([r['cpu_time'] for r in improved_runs]))
            else:
                entry['makespan_mean'] = float(init_ms)
                entry['makespan_std'] = 0.0
                entry['stability_mean'] = 0.0
                entry['stability_std'] = 0.0
                entry['cpu_time_mean'] = 0.0

            results[method].append(entry)
            rate_str = f"{n_improved}/{n_total}"
            print(f"  {method:12s} [{w_key}]: 成功={rate_str:>3s}, "
                  f"MS={entry['makespan_mean']:.1f}±{entry['makespan_std']:.1f}, "
                  f"Stab={entry['stability_mean']:.2f}±{entry['stability_std']:.2f}")

        results[method].sort(key=lambda x: x['w_stab'])

    with open(os.path.join(out_dir, "weight_sweep.json"), 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    return results


def plot_sweep(results, out_dir, init_ms):
    """重みスイープの可視化: MS平均, Stability平均, 改善率 vs w_stab"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    colors = {'GA': 'tab:red', 'ILS_swap': 'tab:blue', 'ILS_insert': 'tab:green'}

    for method, entries in results.items():
        w_stabs = [e['w_stab'] for e in entries]
        ms_means = [e['makespan_mean'] for e in entries]
        ms_stds = [e['makespan_std'] for e in entries]
        st_means = [e['stability_mean'] for e in entries]
        improved_rates = [100 * sum(1 for _ in range(e['n_seeds'])
                          if e['makespan_best'] < init_ms) / e['n_seeds']
                          if e['makespan_mean'] < init_ms else 0
                          for e in entries]
        # 改善率を正確に計算するにはrun単位の情報が必要だが、meanで近似
        improve_approx = [100.0 if e['makespan_mean'] < init_ms - 1 else
                          (0.0 if e['makespan_mean'] >= init_ms - 0.5 else 50.0)
                          for e in entries]

        c = colors[method]
        axes[0].errorbar(w_stabs, ms_means, yerr=ms_stds, color=c, marker='o',
                         label=method, capsize=3, linewidth=1.5)
        axes[1].plot(w_stabs, st_means, color=c, marker='s', label=method, linewidth=1.5)

    axes[0].axhline(y=init_ms, color='gray', linestyle='--', alpha=0.5, label=f'初期解 ({init_ms})')
    axes[0].set_xlabel('w_stab'); axes[0].set_ylabel('Makespan (mean±std)')
    axes[0].set_title('Makespan vs Weight'); axes[0].legend(); axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel('w_stab'); axes[1].set_ylabel('Stability (mean)')
    axes[1].set_title('Stability vs Weight'); axes[1].legend(); axes[1].grid(True, alpha=0.3)

    # MS-Stability トレードオフ
    for method, entries in results.items():
        ms_means = [e['makespan_mean'] for e in entries]
        st_means = [e['stability_mean'] for e in entries]
        c = colors[method]
        axes[2].plot(ms_means, st_means, color=c, marker='D', label=method, linewidth=1.5)
        for e in entries:
            axes[2].annotate(f"w={e['w_stab']:.2f}",
                             (e['makespan_mean'], e['stability_mean']),
                             fontsize=6, alpha=0.7)

    axes[2].set_xlabel('Makespan (mean)'); axes[2].set_ylabel('Stability (mean)')
    axes[2].set_title('Makespan-Stability Tradeoff'); axes[2].legend(); axes[2].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "weight_sweep_plot.png"), dpi=150)
    plt.close(fig)


def main():
    out_dir = setup_output_dir("run", base_dir=os.path.dirname(__file__))
    print(f"出力先: {out_dir}")

    init_ms = get_initial_makespan()
    print(f"初期解メイクスパン: {init_ms}")

    print("\n共通正規化パラメータを推定中...")
    norm_params = compute_shared_norm_params()

    results = run_sweep(out_dir, norm_params, init_ms)
    plot_sweep(results, out_dir, init_ms)

    # テーブル形式で出力（改善成功試行のみの統計）
    print("\n" + "=" * 100)
    print("重みスイープ結果サマリー（改善成功試行のみの統計）")
    print("=" * 100)
    for method in METHOD_NAMES:
        print(f"\n  {method}:")
        print(f"  {'w_stab':>6s} | {'成功率':>7s} | {'MS平均':>8s} | {'MS±std':>10s} | {'MS最良':>6s} | {'Stab平均':>8s} | {'CPU(s)':>7s}")
        print(f"  {'-'*6}-+-{'-'*7}-+-{'-'*8}-+-{'-'*10}-+-{'-'*6}-+-{'-'*8}-+-{'-'*7}")
        for e in results[method]:
            rate = f"{e['n_improved']}/{e['n_seeds']}"
            if e['n_improved'] > 0:
                print(f"  {e['w_stab']:6.2f} | {rate:>7s} | {e['makespan_mean']:8.1f} | "
                      f"{e['makespan_mean']:.0f}±{e['makespan_std']:.1f} | "
                      f"{e['makespan_best']:6d} | {e['stability_mean']:8.2f} | "
                      f"{e['cpu_time_mean']:7.2f}")
            else:
                print(f"  {e['w_stab']:6.2f} | {rate:>7s} |      --- |        --- |    --- |      --- |     ---")

    print(f"\n完了。結果は {out_dir} に保存されました。")


if __name__ == "__main__":
    main()
