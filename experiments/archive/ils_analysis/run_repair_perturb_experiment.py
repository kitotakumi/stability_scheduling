#!/usr/bin/env python3
"""
P-1 (Mini-PR kick / repair キック) 検証実験

=== 位置づけ ===
repair 摂動は主摂動を置き換えるのではなく、停滞時に発動する副摂動として
機能させる設計。`repair_mode=True` + `repair_trigger` で制御し、
PR とは独立して動作する。

=== 実験の目的 ===
以下の4条件を比較し、repair キックが追加価値を生むかを検証する:

  (A) ILS_swap           : swap のみ（ベースライン）
  (B) ILS_insert         : insert のみ（ベースライン）
  (C) ILS_swap_repair    : swap + repair-on-stagnation
  (D) ILS_insert_repair  : insert + repair-on-stagnation

=== 判断基準 ===
  実験対象の重みは [0.9, 0.1] と [0.8, 0.2]（stab={0.1, 0.2}）。
  stab=0 は検証価値なし、stab>=0.5 は初期解に強く引っ張られて探索が trivial
  化するため、「安定性が目的関数に効くが支配的ではない」レンジに絞る。

  - C/D が A/B の合成スコア平均を下回れば成功
  - Stability 改善 ＞ MS 悪化 のトレードオフが weighted score で正になる想定

=== 結論の解釈ガイド ===
  C/D 優位 → repair キックが副摂動として有効 → P-2/P-3 に進む
  C/D 劣位 → 停滞時のキックとしても repair は機能しない → 設計を再考
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
    print_method_summary,
    ILS_MAX_ITER,
)


N_TRIALS = 10
REPAIR_TRIGGER_DEFAULT = 30
REPAIR_STRENGTH_DEFAULT = 2

PROBLEM_SETS = [
    ('mt10', 'mt10_delay60'),
    ('la21', 'la21_delay147'),
    ('la36', 'la36_delay148'),
    ('la40', 'la40_delay148'),
]

WEIGHTS_LIST = [
    [0.9, 0.1],   # 安定性の影響小（repair の効果が出始めるレンジ）
    [0.8, 0.2],   # 安定性の影響中（repair が活きる想定の中心レンジ）
]

# 各手法の (perturb_method, repair_mode) 設定
ILS_METHODS = {
    'ILS_swap':          {'perturb': 'swap',   'repair_mode': False},
    'ILS_insert':        {'perturb': 'insert', 'repair_mode': False},
    'ILS_swap_repair':   {'perturb': 'swap',   'repair_mode': True},
    'ILS_insert_repair': {'perturb': 'insert', 'repair_mode': True},
}

METHOD_LABELS = {
    'ILS_swap':          'swap',
    'ILS_insert':        'insert',
    'ILS_swap_repair':   'swap+repair',
    'ILS_insert_repair': 'insert+repair',
}

METHOD_COLORS = {
    'ILS_swap':          'tab:blue',
    'ILS_insert':        'tab:orange',
    'ILS_swap_repair':   'tab:cyan',
    'ILS_insert_repair': 'tab:red',
}

# 対応するベース vs repair ペア
REPAIR_PAIRS = [
    ('ILS_swap',   'ILS_swap_repair',   'swap'),
    ('ILS_insert', 'ILS_insert_repair', 'insert'),
]


# ========== 個別実行 ==========

def _run_method(method_key, weights, seed, norm_params, problem_name, scenario_name,
                repair_trigger, repair_strength):
    cfg = ILS_METHODS[method_key]
    return run_ils(
        weights, seed, cfg['perturb'], ILS_MAX_ITER, norm_params,
        strategy='best',
        repair_mode=cfg['repair_mode'],
        repair_trigger=repair_trigger,
        repair_strength=repair_strength,
        problem_name=problem_name, scenario_name=scenario_name)


# ========== プロット ==========

def mean_curve_by_time(histories, field, n_points=300):
    valid = [h for h in histories if h is not None]
    if not valid:
        return [], []
    t_max = min(hist[-1]['cpu_time'] for hist in valid)
    if t_max <= 0:
        return [], []
    t_grid = np.linspace(0, t_max, n_points)
    interpolated = []
    for hist in valid:
        times = np.array([h['cpu_time'] for h in hist])
        values = np.array([h[field] for h in hist])
        interp_vals = np.interp(t_grid, times, values)
        interpolated.append(interp_vals)
    return t_grid, np.mean(interpolated, axis=0)


def plot_cpu_time_comparison(all_results, methods, w_label, out_dir, prob_label):
    """CPU時間軸で全手法を重ねる (Score/MS/Stability)"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for mk in methods:
        histories = all_results.get(f'{mk}_histories', [])
        color = METHOD_COLORS[mk]
        valid = [h for h in histories if h is not None]

        for history in valid:
            times = [h['cpu_time'] for h in history]
            axes[0].plot(times, [h['best_score']     for h in history],
                         color=color, alpha=0.15, lw=0.7)
            axes[1].plot(times, [h['best_makespan']  for h in history],
                         color=color, alpha=0.15, lw=0.7)
            axes[2].plot(times, [h['best_stability'] for h in history],
                         color=color, alpha=0.15, lw=0.7)

        if valid:
            for ax_i, field in enumerate(['best_score', 'best_makespan', 'best_stability']):
                xs, ys = mean_curve_by_time(valid, field)
                if len(xs) > 0:
                    axes[ax_i].plot(xs, ys, color=color, lw=2.0, alpha=0.9,
                                     label=f"{METHOD_LABELS[mk]} (mean)")

    for ax, ylabel, title in zip(
            axes,
            ['Weighted Score', 'Makespan', 'Stability'],
            ['Score vs CPU Time', 'Makespan vs CPU Time', 'Stability vs CPU Time']):
        ax.set_xlabel('CPU Time (s)')
        ax.set_ylabel(ylabel)
        ax.set_title(f'{prob_label}: {title}')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    fig.suptitle(f'{prob_label}: repair キック効果比較 ({w_label})', fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"cpu_overview_{w_label}.png"), dpi=150)
    plt.close(fig)


def plot_pair_comparison(all_results, w_label, out_dir, prob_label):
    """base vs base+repair ペア比較: 2x2パネル (swap/insert × MS/Stability)"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    for row_i, (base_key, repair_key, label) in enumerate(REPAIR_PAIRS):
        for col_i, (field, ylabel) in enumerate(
                [('best_makespan', 'Makespan'), ('best_stability', 'Stability')]):
            ax = axes[row_i, col_i]
            for mk in [base_key, repair_key]:
                histories = all_results.get(f'{mk}_histories', [])
                valid = [h for h in histories if h is not None]
                color = METHOD_COLORS[mk]
                for history in valid:
                    times = [h['cpu_time'] for h in history]
                    ax.plot(times, [h[field] for h in history],
                             color=color, alpha=0.2, lw=0.7)
                if valid:
                    xs, ys = mean_curve_by_time(valid, field)
                    if len(xs) > 0:
                        ax.plot(xs, ys, color=color, lw=2.0, alpha=0.9,
                                 label=f"{METHOD_LABELS[mk]} (mean)")
            ax.set_xlabel('CPU Time (s)')
            ax.set_ylabel(f'Best {ylabel}')
            ax.set_title(f'{prob_label} {label}: Best {ylabel}')
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)

    fig.suptitle(f'{prob_label}: base vs base+repair ペア比較 ({w_label})', fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"pair_compare_{w_label}.png"), dpi=150)
    plt.close(fig)


def plot_final_distribution(all_results, methods, w_label, out_dir, prob_label):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    def collect(key):
        return [[d[key] for d in all_results[mk]
                 if d is not None and 'error' not in d]
                for mk in methods]

    ms_data = collect('makespan')
    st_data = collect('stability')

    labels = [METHOD_LABELS[mk] for mk in methods]
    colors = [METHOD_COLORS[mk] for mk in methods]

    for ax, data, ylabel, title in zip(
            axes[:2], [ms_data, st_data],
            ['Final Makespan', 'Final Stability'],
            ['Makespan 分布', 'Stability 分布']):
        bp = ax.boxplot(data, labels=labels, patch_artist=True)
        for patch, c in zip(bp['boxes'], colors):
            patch.set_facecolor(c); patch.set_alpha(0.6)
        ax.set_ylabel(ylabel)
        ax.set_title(f'{prob_label}: {title}')
        ax.grid(True, axis='y', alpha=0.3)
        ax.tick_params(axis='x', rotation=15)

    ips_data = []
    for mk in methods:
        valid = [d for d in all_results[mk] if d is not None and 'error' not in d]
        ips = [ILS_MAX_ITER / d['convergence']['total_cpu_time'] for d in valid]
        ips_data.append(ips)
    bp = axes[2].boxplot(ips_data, labels=labels, patch_artist=True)
    for patch, c in zip(bp['boxes'], colors):
        patch.set_facecolor(c); patch.set_alpha(0.6)
    axes[2].set_ylabel('Iterations / sec')
    axes[2].set_title(f'{prob_label}: 反復速度')
    axes[2].grid(True, axis='y', alpha=0.3)
    axes[2].tick_params(axis='x', rotation=15)

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"final_distribution_{w_label}.png"), dpi=150)
    plt.close(fig)


# ========== 統計 ==========

def compute_comparison_stats(all_results, methods, w_label, out_dir, prob_label):
    lines = [f"\n=== repair キック 比較統計 ({prob_label}, weights={w_label}) ==="]

    col_w = (18, 14)
    header = (f"  {'指標':<{col_w[0]}} "
              + " ".join(f"{METHOD_LABELS[mk]:>{col_w[1]}}" for mk in methods))
    lines.append(header)
    lines.append("  " + "-" * (col_w[0] + (col_w[1] + 1) * len(methods)))

    def row(label, values, fmt='.1f'):
        cells = [f"{v:{fmt}}" for v in values]
        return (f"  {label:<{col_w[0]}} "
                + " ".join(f"{c:>{col_w[1]}}" for c in cells))

    def collect(key):
        out = []
        for mk in methods:
            valid = [d for d in all_results[mk] if d is not None and 'error' not in d]
            vals = [d[key] for d in valid]
            out.append(vals)
        return out

    ms_lists = collect('makespan')
    st_lists = collect('stability')

    lines.append(row('Makespan 平均',  [float(np.mean(v)) if v else 0 for v in ms_lists], '.1f'))
    lines.append(row('Makespan 最良',  [float(min(v))    if v else 0 for v in ms_lists], '.0f'))
    lines.append(row('Makespan std',   [float(np.std(v)) if v else 0 for v in ms_lists], '.1f'))
    lines.append(row('Stability 平均', [float(np.mean(v)) if v else 0 for v in st_lists], '.3f'))
    lines.append(row('Stability 最良', [float(min(v))    if v else 0 for v in st_lists], '.3f'))
    lines.append(row('Stability std',  [float(np.std(v)) if v else 0 for v in st_lists], '.3f'))

    ips_means = []
    for mk in methods:
        valid = [d for d in all_results[mk] if d is not None and 'error' not in d]
        ips = [ILS_MAX_ITER / d['convergence']['total_cpu_time'] for d in valid]
        ips_means.append(float(np.mean(ips)) if ips else 0)
    lines.append(row('iters/sec 平均', ips_means, '.1f'))

    # base vs base+repair ペア比較（相対変化）
    lines.append("")
    lines.append("  --- base vs base+repair 差分（repair - base）---")
    lines.append(f"  {'ペア':<14} {'ΔMS平均':>12} {'ΔStab平均':>12} {'Δiters/s':>10}")
    for base_key, repair_key, pair_label in REPAIR_PAIRS:
        if base_key not in methods or repair_key not in methods:
            continue
        def mean_of(mk, key):
            valid = [d for d in all_results[mk] if d is not None and 'error' not in d]
            if not valid: return float('nan')
            return float(np.mean([d[key] for d in valid]))

        def mean_ips(mk):
            valid = [d for d in all_results[mk] if d is not None and 'error' not in d]
            if not valid: return float('nan')
            return float(np.mean([ILS_MAX_ITER / d['convergence']['total_cpu_time']
                                   for d in valid]))

        d_ms  = mean_of(repair_key, 'makespan') - mean_of(base_key, 'makespan')
        d_st  = mean_of(repair_key, 'stability') - mean_of(base_key, 'stability')
        d_ips = mean_ips(repair_key) - mean_ips(base_key)
        lines.append(f"  {pair_label:<14} {d_ms:>+12.2f} {d_st:>+12.3f} {d_ips:>+10.2f}")

    text = "\n".join(lines)
    print(text)

    stats_path = os.path.join(out_dir, f"stats_{w_label}.txt")
    with open(stats_path, 'w', encoding='utf-8') as f:
        f.write(text + "\n")

    return lines


# ========== ランナー ==========

def run_problem_experiment(problem_name, scenario_name, weights, methods, out_dir,
                           repair_trigger, repair_strength):
    prob_label = f"{problem_name}_{scenario_name}"
    w_label = f"eff={weights[0]}_stab={weights[1]}"
    print(f"\n{'='*70}")
    print(f"問題: {prob_label}, weights={weights}")
    print(f"  repair_trigger={repair_trigger}, repair_strength={repair_strength}")
    print(f"{'='*70}")

    norm_params = compute_shared_norm_params(problem_name, scenario_name)
    init_ms = get_initial_makespan(problem_name, scenario_name)
    print(f"  初期解メイクスパン: {init_ms}")

    futures = {}
    with ProcessPoolExecutor() as executor:
        for trial in range(N_TRIALS):
            seed = trial * 100 + 7
            for mk in methods:
                f = executor.submit(
                    _run_method, mk, weights, seed, norm_params,
                    problem_name, scenario_name,
                    repair_trigger, repair_strength)
                futures[f] = (mk, trial, seed)

        all_results = {
            'problem': problem_name,
            'scenario': scenario_name,
            'weights': weights,
            'init_makespan': init_ms,
            'repair_trigger': repair_trigger,
            'repair_strength': repair_strength,
        }
        for mk in methods:
            all_results[mk] = [None] * N_TRIALS
            all_results[f'{mk}_histories'] = [None] * N_TRIALS

        for future in as_completed(futures):
            mk, trial, seed = futures[future]
            try:
                r = future.result()
                all_results[mk][trial] = {
                    'trial': trial, 'seed': seed,
                    'makespan': r['makespan'],
                    'stability': r['stability'],
                    'convergence': r['convergence'],
                }
                all_results[f'{mk}_histories'][trial] = r['history']
                print(f"  Trial {trial:2d} {METHOD_LABELS[mk]:16s}: "
                      f"MS={r['makespan']}, Stab={r['stability']:.2f}, "
                      f"CPU={r['convergence']['total_cpu_time']:.2f}s")
            except Exception as e:
                import traceback
                print(f"  Trial {trial:2d} {METHOD_LABELS[mk]:16s}: ERROR - {e}")
                traceback.print_exc()
                all_results[mk][trial] = {'trial': trial, 'seed': seed, 'error': str(e)}

    prob_dir = os.path.join(out_dir, prob_label)
    os.makedirs(prob_dir, exist_ok=True)

    # 履歴は per-iteration の (ls_ms, ls_st, accepted) だけ抽出して保存。
    # 多目的評価（Pareto front / EAF）の分析に使う。
    save_results = {}
    for k, v in all_results.items():
        if k.endswith('_histories'):
            save_results[k] = [
                None if hist is None else
                [[h['ls_makespan'], h['ls_stability'], h['accepted']]
                 for h in hist]
                for hist in v
            ]
        else:
            save_results[k] = v
    with open(os.path.join(prob_dir, f"results_{w_label}.json"), 'w') as f:
        json.dump(save_results, f, ensure_ascii=False)

    summary_lines = [f"\n問題: {prob_label}, weights={weights}"]
    for mk in methods:
        valid = [d for d in all_results[mk] if d is not None and 'error' not in d]
        if valid:
            text = print_method_summary(METHOD_LABELS[mk], valid, init_ms)
            summary_lines.append(text)

    plot_cpu_time_comparison(all_results, methods, w_label, prob_dir, prob_label)
    plot_pair_comparison(all_results, w_label, prob_dir, prob_label)
    plot_final_distribution(all_results, methods, w_label, prob_dir, prob_label)

    stats_lines = compute_comparison_stats(
        all_results, methods, w_label, prob_dir, prob_label)
    summary_lines.extend(stats_lines)

    return all_results, summary_lines


def write_cross_summary(all_summaries, out_dir,
                        repair_trigger, repair_strength):
    path = os.path.join(out_dir, "cross_problem_summary.txt")
    with open(path, 'w', encoding='utf-8') as f:
        f.write("repair キック (P-1) 検証実験 横断サマリー\n")
        f.write("=" * 70 + "\n")
        f.write(f"repair_trigger={repair_trigger}, "
                f"repair_strength={repair_strength}\n\n")
        f.write("判断基準:\n")
        f.write("  stab={0.1, 0.2} レンジで base vs base+repair を比較。\n")
        f.write("  (1) Stability 改善が主効果（base+repair が base の Stab を下回る）\n")
        f.write("  (2) MS 悪化は weighted score の改善と相殺する範囲に収まること\n")
        f.write("  (3) 合成スコア平均で base+repair が base と同等以上なら成功\n")
        f.write("=" * 70 + "\n\n")
        for lines in all_summaries:
            for line in lines:
                f.write(str(line) + "\n")
            f.write("\n")
    print(f"\n横断サマリー: {path}")


# ========== エントリポイント ==========

def main():
    global N_TRIALS
    parser = argparse.ArgumentParser(description="P-1 repair キック検証実験")
    parser.add_argument(
        '--problems', nargs='+', type=str, default=None,
        help='問題セット (例: mt10:mt10_delay60 la36:la36_delay148)')
    parser.add_argument(
        '--weights', nargs='+', type=str, default=['0.9,0.1', '0.8,0.2'],
        help='比較する重み設定 (例: 0.9,0.1 0.8,0.2)')
    parser.add_argument(
        '--methods', nargs='+', default=list(ILS_METHODS.keys()),
        choices=list(ILS_METHODS.keys()),
        help='実行する手法')
    parser.add_argument(
        '--trials', type=int, default=N_TRIALS,
        help='試行回数 (デフォルト: 10)')
    parser.add_argument(
        '--repair-trigger', type=int, default=REPAIR_TRIGGER_DEFAULT,
        help='repair キック発動までの無改善反復数 (デフォルト: 30)')
    parser.add_argument(
        '--repair-strength', type=int, default=REPAIR_STRENGTH_DEFAULT,
        help='repair 1回あたりの direct swap 回数 (デフォルト: 2)')
    args = parser.parse_args()

    N_TRIALS = args.trials

    if args.problems:
        problem_sets = [tuple(p.split(':')) for p in args.problems]
    else:
        problem_sets = PROBLEM_SETS

    weight_list = [[float(x) for x in w.split(',')] for w in args.weights]
    methods = args.methods

    out_dir = setup_output_dir("repair_perturb", base_dir=os.path.dirname(__file__))
    print(f"出力先: {out_dir}")
    print(f"問題セット: {problem_sets}")
    print(f"重み: {weight_list}")
    print(f"手法: {methods}")
    print(f"試行回数: {N_TRIALS}")
    print(f"ILS最大反復数: {ILS_MAX_ITER}")
    print(f"repair_trigger={args.repair_trigger}, "
          f"repair_strength={args.repair_strength}")

    all_summaries = []
    for problem_name, scenario_name in problem_sets:
        for weights in weight_list:
            try:
                _, summary_lines = run_problem_experiment(
                    problem_name, scenario_name, weights, methods, out_dir,
                    args.repair_trigger, args.repair_strength)
                all_summaries.append(summary_lines)
            except Exception as e:
                import traceback
                print(f"\nERROR: {problem_name}/{scenario_name} weights={weights}: {e}")
                traceback.print_exc()
                all_summaries.append([f"ERROR: {problem_name}/{scenario_name}: {e}"])

    write_cross_summary(all_summaries, out_dir,
                        args.repair_trigger, args.repair_strength)
    print(f"\n全実験完了。結果は {out_dir} に保存されました。")


if __name__ == "__main__":
    main()
