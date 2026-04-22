#!/usr/bin/env python3
"""
FI vs BI 比較実験: First-Improvement vs Best-Improvement 局所探索戦略の比較

=== 実験の判断基準 ===

(1) 同一CPU時間での解品質 [最重要]
    BIの中央値CPU時間を基準として、その時間内にFIが達成できる品質を比較。
    FIの速度優位がBIの反復品質優位を上回るかどうかの直接的な判断基準。

(2) 反復速度 (iters/sec)
    FIはBI比較で何倍速いか。N5近傍サイズが大きいほど差が出やすい（la36, la40）。

(3) 同一反復数（800iter）での最終解品質
    両者の「1反復あたりの品質」を比較。BIは1ステップで最良隣接を選ぶ分、
    反復あたりの改善量が大きい可能性がある。

=== 結論の解釈ガイド ===
    - FI優位: 同じ時間内に多くの反復 → より広い探索 → 最終解が良い
    - BI優位: 1反復あたりの品質が高く、速度差を補う
    - 問題サイズ依存性: 大規模問題（la36, la40）ほどFIの速度優位が出やすい
    - 摂動依存性: swap と insert で傾向が異なる可能性あり
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

PROBLEM_SETS = [
    ('mt10', 'mt10_delay60'),
    ('la21', 'la21_delay147'),
    ('la36', 'la36_delay148'),
    ('la40', 'la40_delay148'),
]

WEIGHTS_LIST = [
    [1.0, 0.0],
    [0.9, 0.1],
]

ILS_METHODS = {
    'ILS_swap_best':    ('swap',   'best'),
    'ILS_swap_first':   ('swap',   'first'),
    'ILS_insert_best':  ('insert', 'best'),
    'ILS_insert_first': ('insert', 'first'),
}

METHOD_LABELS = {
    'ILS_swap_best':    'swap/BI',
    'ILS_swap_first':   'swap/FI',
    'ILS_insert_best':  'insert/BI',
    'ILS_insert_first': 'insert/FI',
}

METHOD_COLORS = {
    'ILS_swap_best':    'tab:blue',
    'ILS_swap_first':   'tab:cyan',
    'ILS_insert_best':  'tab:orange',
    'ILS_insert_first': 'tab:green',
}

METHOD_LS_STYLE = {
    'ILS_swap_best':    '-',
    'ILS_swap_first':   '--',
    'ILS_insert_best':  '-',
    'ILS_insert_first': '--',
}

# BI-FI 対応ペア
BI_FI_PAIRS = [
    ('ILS_swap_best',   'ILS_swap_first',   'swap'),
    ('ILS_insert_best', 'ILS_insert_first', 'insert'),
]


# ========== ユーティリティ ==========

def _run_method(method_key, weights, seed, norm_params, problem_name, scenario_name):
    perturb, strategy = ILS_METHODS[method_key]
    return run_ils(weights, seed, perturb, ILS_MAX_ITER, norm_params,
                   strategy=strategy,
                   problem_name=problem_name, scenario_name=scenario_name)


def get_quality_at_time(history, t_cutoff):
    """history から t_cutoff 秒以内に達成された最良状態を返す。
    history の best_* フィールドは累積最良値なので、t_cutoff 以内の
    最後のエントリを参照すれば良い。
    Returns: (best_score, best_makespan, best_stability)
    """
    result = history[0]
    for h in history:
        if h['cpu_time'] <= t_cutoff:
            result = h
        else:
            break
    return result['best_score'], result['best_makespan'], result['best_stability']


def mean_curve_by_iteration(histories, field):
    """反復軸での平均曲線。全試行が同じ反復数なので element-wise mean。"""
    arrays = np.array([
        [h[field] for h in hist]
        for hist in histories if hist is not None
    ])
    if arrays.size == 0:
        return [], []
    iters = list(range(arrays.shape[1]))
    return iters, np.mean(arrays, axis=0)


def mean_curve_by_time(histories, field, n_points=300):
    """CPU時間軸での平均曲線。各試行を共通時間グリッドに線形補間。
    グリッド範囲は 0 〜 各試行の最小終了時刻（共通部分のみ）。
    """
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


# ========== プロット関数 ==========

def plot_cpu_time_overview(all_results, methods, w_label, out_dir, prob_label):
    """全4手法のCPU時間軸オーバーレイ（スコア・MS・安定性の3パネル）"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for mk in methods:
        histories = all_results.get(f'{mk}_histories', [])
        color = METHOD_COLORS[mk]
        ls = METHOD_LS_STYLE[mk]
        first_drawn = False

        for history in histories:
            if history is None:
                continue
            times = [h['cpu_time'] for h in history]
            scores = [h['best_score'] for h in history]
            ms_vals = [h['best_makespan'] for h in history]
            st_vals = [h['best_stability'] for h in history]
            alpha = 0.8 if not first_drawn else 0.15
            lw = 1.5 if not first_drawn else 0.7
            lbl = METHOD_LABELS[mk] if not first_drawn else None
            axes[0].plot(times, scores,  color=color, ls=ls, alpha=alpha, lw=lw, label=lbl)
            axes[1].plot(times, ms_vals, color=color, ls=ls, alpha=alpha, lw=lw, label=lbl)
            axes[2].plot(times, st_vals, color=color, ls=ls, alpha=alpha, lw=lw, label=lbl)
            first_drawn = True

    for ax, ylabel, title in zip(
            axes,
            ['Weighted Score', 'Makespan', 'Stability'],
            ['Score vs CPU Time', 'Makespan vs CPU Time', 'Stability vs CPU Time']):
        ax.set_xlabel('CPU Time (s)')
        ax.set_ylabel(ylabel)
        ax.set_title(f'{prob_label}: {title}')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle(f'{prob_label}: FI vs BI 全手法比較 ({w_label})', fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"cpu_overview_{w_label}.png"), dpi=150)
    plt.close(fig)


def plot_fi_vs_bi_detail(all_results, w_label, out_dir, prob_label, init_ms):
    """BI と FI の詳細比較: 4パネル (swap/insert × 反復軸/CPU時間軸)"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    panel_specs = [
        (BI_FI_PAIRS[0], 'iteration', axes[0, 0]),   # swap × 反復
        (BI_FI_PAIRS[0], 'cpu_time',  axes[0, 1]),   # swap × 時間
        (BI_FI_PAIRS[1], 'iteration', axes[1, 0]),   # insert × 反復
        (BI_FI_PAIRS[1], 'cpu_time',  axes[1, 1]),   # insert × 時間
    ]

    for (bi_key, fi_key, perturb), x_axis, ax in panel_specs:
        for mk in [bi_key, fi_key]:
            histories = all_results.get(f'{mk}_histories', [])
            color = METHOD_COLORS[mk]
            ls = METHOD_LS_STYLE[mk]

            # 個別試行（薄い）
            for history in histories:
                if history is None:
                    continue
                xs = [h[x_axis] for h in history]
                ms = [h['best_makespan'] for h in history]
                ax.plot(xs, ms, color=color, ls=ls, alpha=0.15, lw=0.7)

            # 平均曲線（太い）
            valid_hists = [h for h in histories if h is not None]
            if valid_hists:
                if x_axis == 'iteration':
                    xs_mean, ms_mean = mean_curve_by_iteration(valid_hists, 'best_makespan')
                else:
                    xs_mean, ms_mean = mean_curve_by_time(valid_hists, 'best_makespan')
                if len(xs_mean) > 0:
                    ax.plot(xs_mean, ms_mean, color=color, ls=ls, alpha=0.9,
                            lw=2.0, label=f'{METHOD_LABELS[mk]} (mean)')

        ax.axhline(y=init_ms, color='gray', ls=':', alpha=0.5, label='Initial MS')
        xlabel = 'Iteration' if x_axis == 'iteration' else 'CPU Time (s)'
        ax.set_xlabel(xlabel)
        ax.set_ylabel('Best Makespan')
        ax.set_title(f'{prob_label} {perturb}: BI vs FI ({xlabel})')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle(f'{prob_label}: BI vs FI 詳細比較 ({w_label})', fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"fi_vs_bi_detail_{w_label}.png"), dpi=150)
    plt.close(fig)


def plot_iters_per_sec(all_results, methods, w_label, out_dir, prob_label):
    """手法別の反復速度（iters/sec）棒グラフ"""
    method_ips_mean = []
    method_ips_std = []
    labels = []

    for mk in methods:
        valid = [d for d in all_results[mk] if d is not None and 'error' not in d]
        if not valid:
            method_ips_mean.append(0)
            method_ips_std.append(0)
        else:
            ips = [ILS_MAX_ITER / d['convergence']['total_cpu_time'] for d in valid]
            method_ips_mean.append(np.mean(ips))
            method_ips_std.append(np.std(ips))
        labels.append(METHOD_LABELS[mk])

    colors = [METHOD_COLORS[mk] for mk in methods]
    x = np.arange(len(methods))

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(x, method_ips_mean, yerr=method_ips_std, color=colors,
                  capsize=5, alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel('Iterations per Second')
    ax.set_title(f'{prob_label}: 反復速度 iters/sec ({w_label})')
    ax.grid(True, axis='y', alpha=0.3)

    for bar, mean_val in zip(bars, method_ips_mean):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f'{mean_val:.1f}', ha='center', va='bottom', fontsize=9)

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"iters_per_sec_{w_label}.png"), dpi=150)
    plt.close(fig)


# ========== 数値統計 ==========

def compute_fi_vs_bi_stats(all_results, w_label, out_dir, prob_label):
    """FI vs BI の数値比較統計を計算してテキストと行リストを返す"""
    lines = [f"\n=== FI vs BI 比較統計 ({prob_label}, weights={w_label}) ==="]
    init_ms_val = all_results['init_makespan']

    for bi_key, fi_key, perturb in BI_FI_PAIRS:
        lines.append(f"\n[{perturb}摂動]")
        bi_valid = [d for d in all_results[bi_key] if d is not None and 'error' not in d]
        fi_valid = [d for d in all_results[fi_key] if d is not None and 'error' not in d]
        bi_hists = [h for h in all_results[f'{bi_key}_histories'] if h is not None]
        fi_hists = [h for h in all_results[f'{fi_key}_histories'] if h is not None]

        if not bi_valid or not fi_valid:
            lines.append("  データ不足")
            continue

        col_w = (36, 14, 14, 9)
        header = (f"  {'指標':<{col_w[0]}} {'BI(best)':>{col_w[1]}} "
                  f"{'FI(first)':>{col_w[2]}} {'FI/BI比':>{col_w[3]}}")
        lines.append(header)
        lines.append("  " + "-" * (sum(col_w) + 3))

        def row(label, bi_val, fi_val, fmt='.1f', ratio=True):
            bi_str = f"{bi_val:{fmt}}"
            fi_str = f"{fi_val:{fmt}}"
            if ratio and bi_val != 0:
                ratio_str = f"{fi_val / bi_val:.3f}x"
            else:
                ratio_str = "-"
            return (f"  {label:<{col_w[0]}} {bi_str:>{col_w[1]}} "
                    f"{fi_str:>{col_w[2]}} {ratio_str:>{col_w[3]}}")

        # (1) 反復速度
        bi_ips = [ILS_MAX_ITER / d['convergence']['total_cpu_time'] for d in bi_valid]
        fi_ips = [ILS_MAX_ITER / d['convergence']['total_cpu_time'] for d in fi_valid]
        lines.append(row('反復速度 (iters/sec) 平均', np.mean(bi_ips), np.mean(fi_ips), fmt='.1f'))

        # (2) 全体CPU時間 (800iter)
        bi_cpu_total = [d['convergence']['total_cpu_time'] for d in bi_valid]
        fi_cpu_total = [d['convergence']['total_cpu_time'] for d in fi_valid]
        lines.append(row('全体CPU時間 800iter (s) 平均', np.mean(bi_cpu_total), np.mean(fi_cpu_total), fmt='.2f'))

        # (3) 同一反復数での最終 Makespan
        bi_ms = [d['makespan'] for d in bi_valid]
        fi_ms = [d['makespan'] for d in fi_valid]
        lines.append(row('Makespan (800iter) 平均', np.mean(bi_ms), np.mean(fi_ms), fmt='.1f'))
        lines.append(row('Makespan (800iter) 最良',
                         float(min(bi_ms)), float(min(fi_ms)), fmt='.0f'))
        lines.append(row('Makespan (800iter) std', np.std(bi_ms), np.std(fi_ms), fmt='.1f', ratio=False))

        # (4) 同一CPU時間での解品質（BI中央値を基準）
        bi_ref_time = float(np.median(bi_cpu_total))
        bi_ms_at_ref = []
        fi_ms_at_ref = []
        for hist in bi_hists:
            _, ms_at_t, _ = get_quality_at_time(hist, bi_ref_time)
            bi_ms_at_ref.append(ms_at_t)
        for hist in fi_hists:
            _, ms_at_t, _ = get_quality_at_time(hist, bi_ref_time)
            fi_ms_at_ref.append(ms_at_t)

        lines.append(f"\n  * BI中央値CPU時間 = {bi_ref_time:.2f}s を共通タイムバジェットとして比較")
        if bi_ms_at_ref and fi_ms_at_ref:
            lines.append(row(f'  MS @{bi_ref_time:.1f}s 平均',
                             np.mean(bi_ms_at_ref), np.mean(fi_ms_at_ref), fmt='.1f'))
            lines.append(row(f'  MS @{bi_ref_time:.1f}s 最良',
                             float(min(bi_ms_at_ref)), float(min(fi_ms_at_ref)), fmt='.0f'))

        # (5) 安定性
        bi_st = [d['stability'] for d in bi_valid]
        fi_st = [d['stability'] for d in fi_valid]
        lines.append(row('\n  安定性 (800iter) 平均',
                         np.mean(bi_st), np.mean(fi_st), fmt='.3f', ratio=False))

        # (6) 改善成功率
        bi_impr = sum(1 for d in bi_valid if d['makespan'] < init_ms_val)
        fi_impr = sum(1 for d in fi_valid if d['makespan'] < init_ms_val)
        lines.append(f"  {'改善成功率':<{col_w[0]}} "
                     f"{bi_impr}/{len(bi_valid):>{col_w[1]-2}} "
                     f"{fi_impr}/{len(fi_valid):>{col_w[2]-2}}")

        # (7) 最良解到達CPU時間
        bi_conv_cpu = [d['convergence']['cpu_time'] for d in bi_valid]
        fi_conv_cpu = [d['convergence']['cpu_time'] for d in fi_valid]
        lines.append(row('最良解到達CPU時間 (s) 平均',
                         np.mean(bi_conv_cpu), np.mean(fi_conv_cpu), fmt='.2f'))

    text = "\n".join(str(l) for l in lines)
    print(text)

    stats_path = os.path.join(out_dir, f"fi_vs_bi_stats_{w_label}.txt")
    with open(stats_path, 'w', encoding='utf-8') as f:
        f.write(text + "\n")

    return lines


# ========== 実験ランナー ==========

def run_problem_experiment(problem_name, scenario_name, weights, methods, out_dir):
    """1問題 × 1重み設定の実験を実行し結果を保存する"""
    prob_label = f"{problem_name}_{scenario_name}"
    w_label = f"eff={weights[0]}_stab={weights[1]}"
    print(f"\n{'='*70}")
    print(f"問題: {prob_label}, weights={weights}")
    print(f"{'='*70}")

    norm_params = compute_shared_norm_params(problem_name, scenario_name)
    init_ms = get_initial_makespan(problem_name, scenario_name)
    print(f"  初期解メイクスパン: {init_ms}")

    # 並列実行
    futures = {}
    with ProcessPoolExecutor() as executor:
        for trial in range(N_TRIALS):
            seed = trial * 100 + 7
            for mk in methods:
                f = executor.submit(
                    _run_method, mk, weights, seed, norm_params,
                    problem_name, scenario_name)
                futures[f] = (mk, trial, seed)

    all_results = {
        'problem': problem_name,
        'scenario': scenario_name,
        'weights': weights,
        'init_makespan': init_ms,
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
            print(f"  Trial {trial:2d} {METHOD_LABELS[mk]:15s}: "
                  f"MS={r['makespan']}, Stab={r['stability']:.2f}, "
                  f"CPU={r['convergence']['total_cpu_time']:.2f}s")
        except Exception as e:
            import traceback
            print(f"  Trial {trial:2d} {METHOD_LABELS[mk]:15s}: ERROR - {e}")
            traceback.print_exc()
            all_results[mk][trial] = {'trial': trial, 'seed': seed, 'error': str(e)}

    # 出力ディレクトリ
    prob_dir = os.path.join(out_dir, prob_label)
    os.makedirs(prob_dir, exist_ok=True)

    # JSON 保存（history は除外）
    save_results = {k: v for k, v in all_results.items()
                    if not k.endswith('_histories')}
    with open(os.path.join(prob_dir, f"results_{w_label}.json"), 'w') as f:
        json.dump(save_results, f, indent=2, ensure_ascii=False)

    # 通常サマリー
    summary_lines = [f"\n問題: {prob_label}, weights={weights}"]
    for mk in methods:
        valid = [d for d in all_results[mk] if d is not None and 'error' not in d]
        if valid:
            text = print_method_summary(METHOD_LABELS[mk], valid, init_ms)
            summary_lines.append(text)

    # プロット
    plot_cpu_time_overview(all_results, methods, w_label, prob_dir, prob_label)
    plot_fi_vs_bi_detail(all_results, w_label, prob_dir, prob_label, init_ms)
    plot_iters_per_sec(all_results, methods, w_label, prob_dir, prob_label)

    # FI vs BI 統計
    fi_vs_bi_lines = compute_fi_vs_bi_stats(all_results, w_label, prob_dir, prob_label)
    summary_lines.extend(fi_vs_bi_lines)

    return all_results, summary_lines


# ========== 横断サマリー ==========

def write_cross_problem_summary(all_summaries, out_dir):
    """全問題・全重みの横断サマリーをファイルに出力"""
    path = os.path.join(out_dir, "cross_problem_summary.txt")
    with open(path, 'w', encoding='utf-8') as f:
        f.write("FI vs BI 実験 横断サマリー\n")
        f.write("=" * 70 + "\n\n")
        f.write("判断基準:\n")
        f.write("  (1) 同一CPU時間 (BI中央値) での MS 比較  ← 主指標\n")
        f.write("  (2) 反復速度 (iters/sec) の比較\n")
        f.write("  (3) 同一反復数 (800iter) での MS 比較\n\n")
        f.write("FI/BI比 < 1.0 : FI が優位 (小さいほど良い)\n")
        f.write("FI/BI比 > 1.0 : BI が優位\n")
        f.write("=" * 70 + "\n\n")
        for lines in all_summaries:
            for line in lines:
                f.write(str(line) + "\n")
            f.write("\n")
    print(f"\n横断サマリー: {path}")


# ========== エントリポイント ==========

def main():
    global N_TRIALS
    parser = argparse.ArgumentParser(description="FI vs BI 局所探索戦略比較実験")
    parser.add_argument(
        '--problems', nargs='+', type=str, default=None,
        help='問題セット (例: mt10:mt10_delay60 la36:la36_delay148)')
    parser.add_argument(
        '--weights', nargs='+', type=str, default=['1.0,0.0', '0.9,0.1'],
        help='比較する重み設定 (例: 1.0,0.0 0.9,0.1)')
    parser.add_argument(
        '--methods', nargs='+', default=list(ILS_METHODS.keys()),
        choices=list(ILS_METHODS.keys()),
        help='実行する手法')
    parser.add_argument(
        '--trials', type=int, default=N_TRIALS,
        help='試行回数 (デフォルト: 10)')
    args = parser.parse_args()

    N_TRIALS = args.trials

    if args.problems:
        problem_sets = [tuple(p.split(':')) for p in args.problems]
    else:
        problem_sets = PROBLEM_SETS

    weight_list = [[float(x) for x in w.split(',')] for w in args.weights]
    methods = args.methods

    out_dir = setup_output_dir("fi_vs_bi", base_dir=os.path.dirname(__file__))
    print(f"出力先: {out_dir}")
    print(f"問題セット: {problem_sets}")
    print(f"重み: {weight_list}")
    print(f"手法: {methods}")
    print(f"試行回数: {N_TRIALS}")
    print(f"ILS最大反復数: {ILS_MAX_ITER}")

    all_summaries = []
    for problem_name, scenario_name in problem_sets:
        for weights in weight_list:
            try:
                _, summary_lines = run_problem_experiment(
                    problem_name, scenario_name, weights, methods, out_dir)
                all_summaries.append(summary_lines)
            except Exception as e:
                import traceback
                print(f"\nERROR: {problem_name}/{scenario_name} weights={weights}: {e}")
                traceback.print_exc()
                all_summaries.append([f"ERROR: {problem_name}/{scenario_name}: {e}"])

    write_cross_problem_summary(all_summaries, out_dir)
    print(f"\n全実験完了。結果は {out_dir} に保存されました。")


if __name__ == "__main__":
    main()
