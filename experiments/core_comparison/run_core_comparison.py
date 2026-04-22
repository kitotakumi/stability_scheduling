#!/usr/bin/env python3
"""
実験1: コア比較（GA vs ILS (+repair)）の実行パイプライン

=== 位置づけ ===
evaluation_design.md §6 の実験1「コア比較」を回すための実行スクリプト。
主張 (A) 速度 (B) Pareto 覆域 (C) repair 摂動の貢献 (D) 重み頑健性 の
同時検証に必要な raw データ（履歴・最終値）を収集する。

=== 予算設計 ===
CPU 時間での打ち切りはしない。ILS は max_iterations、GA は ngen まで
自然収束まで走らせる。速度比較は anytime curve で任意時刻を事後抽出。
（evaluation_design.md §4.1 参照）

=== 出力 ===
results/core_<timestamp>/
├── config.json                               # 実行設定
├── <problem>_<scenario>/
│   ├── results_<w_label>.json                # 全手法×全trial の履歴・最終値
│   └── summary_<w_label>.txt                 # 数値サマリ
└── cross_summary.txt                         # 横断サマリ

分析（anytime HV curve, snapshot Pareto, conditional HV, attainment surface,
差分 EAF, degeneracy heatmap, C-metric 表）は別スクリプト analyze_core.py に
切り出す予定。

=== 使い方 ===
  # デフォルト: 4 問題 × weights=[0.85, 0.15] × 3 手法 × 10 trial
  python run_core_comparison.py

  # 問題・手法・重み・試行回数を指定
  python run_core_comparison.py \
      --problems mt10:mt10_delay60 la36:la36_delay148 \
      --methods ga ils_insert ils_insert_repair \
      --weights 0.85,0.15 0.8,0.2 \
      --trials 10
"""

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np

from experiment_utils import (
    setup_output_dir, compute_shared_norm_params,
    get_initial_makespan, run_ils, run_ga,
    print_method_summary,
    ILS_MAX_ITER, GA_NGEN,
)


# ========== 手法定義 ==========

METHODS = {
    'ga': {
        'kind': 'ga',
        'label': 'GA',
    },
    'ils_insert': {
        'kind': 'ils',
        'perturb': 'insert',
        'repair_mode': False,
        'label': 'ILS-insert',
    },
    'ils_insert_repair': {
        'kind': 'ils',
        'perturb': 'insert',
        'repair_mode': True,
        'label': 'ILS-insert+repair',
    },
    'ils_swap': {
        'kind': 'ils',
        'perturb': 'swap',
        'repair_mode': False,
        'label': 'ILS-swap',
    },
    'ils_swap_repair': {
        'kind': 'ils',
        'perturb': 'swap',
        'repair_mode': True,
        'label': 'ILS-swap+repair',
    },
}

DEFAULT_METHODS = ['ga', 'ils_insert', 'ils_insert_repair']

DEFAULT_WEIGHTS = [[0.85, 0.15]]

DEFAULT_TRIALS = 10

DEFAULT_PROBLEM_SETS = [
    ('mt10', 'mt10_delay60'),
    ('la21', 'la21_delay147'),
    ('la36', 'la36_delay148'),
    ('la40', 'la40_delay148'),
]

# repair / 停滞受理のデフォルトパラメータ（実験2で確定したら差し替える）
STAGNATION_THRESHOLD_DEFAULT = 30
REPAIR_STRENGTH_DEFAULT = 2


# ========== 個別実行 dispatcher ==========

def _run_method(method_key, weights, seed, norm_params, problem_name, scenario_name,
                ils_max_iter, ga_ngen, stagnation_threshold, repair_strength):
    cfg = METHODS[method_key]
    kind = cfg['kind']
    if kind == 'ga':
        return run_ga(weights, seed, ga_ngen, norm_params,
                      problem_name=problem_name, scenario_name=scenario_name,
                      track_population=True)
    elif kind == 'ils':
        return run_ils(
            weights, seed, cfg['perturb'], ils_max_iter, norm_params,
            strategy='best',
            repair_mode=cfg['repair_mode'],
            repair_strength=repair_strength,
            stagnation_threshold=stagnation_threshold,
            problem_name=problem_name, scenario_name=scenario_name)
    else:
        raise ValueError(f"Unknown method kind: {kind}")


# ========== 履歴のスリム化（JSON 保存用） ==========

def _slim_anytime(history, kind):
    """履歴から anytime curve に必要な情報だけを抜き出す。
    GA / ILS 共通スキーマ: [{cpu_time, best_ms, best_st, best_score}, ...]"""
    if history is None:
        return None
    out = []
    for h in history:
        out.append({
            'cpu_time': h.get('cpu_time'),
            'best_ms': h.get('best_makespan'),
            'best_st': h.get('best_stability'),
            'best_score': h.get('best_score'),
        })
    return out


def _slim_points(history, kind):
    """手法ごとの訪問点列を抜き出す（Pareto/EAF 用）。

    GA:  [[ms, st], ...]  (全世代の全個体、flat)
    ILS: [[ls_ms, ls_st, accepted], ...]  (各反復の LS 結果)
    """
    if history is None:
        return None
    if kind == 'ga':
        pts = []
        for h in history:
            if 'pop_points' in h:
                pts.extend(h['pop_points'])
        return pts
    elif kind == 'ils':
        return [[h['ls_makespan'], h['ls_stability'], h['accepted']]
                for h in history]
    return None


# ========== 1 問題×1 weights の実行 ==========

def run_problem_experiment(problem_name, scenario_name, weights, methods, n_trials,
                           out_dir, ils_max_iter, ga_ngen,
                           stagnation_threshold, repair_strength):
    prob_label = f"{problem_name}_{scenario_name}"
    w_label = f"eff={weights[0]}_stab={weights[1]}"
    print(f"\n{'='*70}")
    print(f"問題: {prob_label}, weights={weights}")
    print(f"  methods={methods}, trials={n_trials}")
    print(f"  ILS max_iter={ils_max_iter}, GA ngen={ga_ngen}")
    print(f"  stagnation_threshold={stagnation_threshold}, "
          f"repair_strength={repair_strength}")
    print(f"{'='*70}")

    norm_params = compute_shared_norm_params(problem_name, scenario_name)
    init_ms = get_initial_makespan(problem_name, scenario_name)
    print(f"  初期解メイクスパン: {init_ms}")

    # 並列投入
    futures = {}
    results_by_method = {mk: [None] * n_trials for mk in methods}
    anytime_by_method = {mk: [None] * n_trials for mk in methods}
    points_by_method = {mk: [None] * n_trials for mk in methods}

    with ProcessPoolExecutor() as executor:
        for trial in range(n_trials):
            seed = trial * 100 + 7
            for mk in methods:
                f = executor.submit(
                    _run_method, mk, weights, seed, norm_params,
                    problem_name, scenario_name,
                    ils_max_iter, ga_ngen,
                    stagnation_threshold, repair_strength)
                futures[f] = (mk, trial, seed)

        baselines_by_method = {mk: None for mk in methods}
        for future in as_completed(futures):
            mk, trial, seed = futures[future]
            cfg = METHODS[mk]
            try:
                r = future.result()
                results_by_method[mk][trial] = {
                    'trial': trial, 'seed': seed,
                    'makespan': r['makespan'],
                    'stability': r['stability'],
                    'convergence': r['convergence'],
                }
                anytime_by_method[mk][trial] = _slim_anytime(r['history'], cfg['kind'])
                points_by_method[mk][trial] = _slim_points(r['history'], cfg['kind'])
                # baseline は全 trial で共通（seed に依存しない決定的評価）。
                # 最初の有効 trial の値を保持。
                if baselines_by_method[mk] is None and r.get('baseline') is not None:
                    baselines_by_method[mk] = r['baseline']
                print(f"  Trial {trial:2d} {cfg['label']:22s}: "
                      f"MS={r['makespan']}, Stab={r['stability']:.2f}, "
                      f"CPU={r['convergence']['total_cpu_time']:.2f}s")
            except Exception as e:
                import traceback
                print(f"  Trial {trial:2d} {cfg['label']:22s}: ERROR - {e}")
                traceback.print_exc()
                results_by_method[mk][trial] = {
                    'trial': trial, 'seed': seed, 'error': str(e)}

    # 保存
    prob_dir = os.path.join(out_dir, prob_label)
    os.makedirs(prob_dir, exist_ok=True)

    save_data = {
        'problem': problem_name,
        'scenario': scenario_name,
        'weights': weights,
        'init_makespan': init_ms,
        'n_trials': n_trials,
        'ils_max_iter': ils_max_iter,
        'ga_ngen': ga_ngen,
        'stagnation_threshold': stagnation_threshold,
        'repair_strength': repair_strength,
        'methods': {},
    }
    for mk in methods:
        save_data['methods'][mk] = {
            'kind': METHODS[mk]['kind'],
            'label': METHODS[mk]['label'],
            'baseline': baselines_by_method[mk],  # [ms, stab] or None
            'finals': results_by_method[mk],
            'anytime': anytime_by_method[mk],
            'points': points_by_method[mk],
        }
    save_path = os.path.join(prob_dir, f"results_{w_label}.json")
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(save_data, f, ensure_ascii=False)
    print(f"  → 保存: {save_path}")

    # サマリ出力
    summary_lines = [f"\n問題: {prob_label}, weights={weights}",
                     f"  init_makespan={init_ms}"]
    for mk in methods:
        valid = [d for d in results_by_method[mk]
                 if d is not None and 'error' not in d]
        if valid:
            text = print_method_summary(METHODS[mk]['label'], valid, init_ms)
            summary_lines.append(text)

    stats_text = _compute_comparison_stats(
        results_by_method, methods, prob_label, weights)
    summary_lines.append(stats_text)

    summary_path = os.path.join(prob_dir, f"summary_{w_label}.txt")
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(summary_lines) + "\n")
    print(f"  → サマリ: {summary_path}")

    return summary_lines


# ========== 比較統計 ==========

def _compute_comparison_stats(results_by_method, methods, prob_label, weights):
    lines = [f"\n=== コア比較統計 ({prob_label}, weights={weights}) ==="]

    col_w = (18, 16)
    header = (f"  {'指標':<{col_w[0]}} "
              + " ".join(f"{METHODS[mk]['label']:>{col_w[1]}}" for mk in methods))
    lines.append(header)
    lines.append("  " + "-" * (col_w[0] + (col_w[1] + 1) * len(methods)))

    def row(label, values, fmt='.1f'):
        cells = [f"{v:{fmt}}" if v is not None else 'n/a' for v in values]
        return (f"  {label:<{col_w[0]}} "
                + " ".join(f"{c:>{col_w[1]}}" for c in cells))

    def collect(key):
        out = []
        for mk in methods:
            valid = [d for d in results_by_method[mk]
                     if d is not None and 'error' not in d]
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

    cpu_means = []
    for mk in methods:
        valid = [d for d in results_by_method[mk]
                 if d is not None and 'error' not in d]
        cpus = [d['convergence']['total_cpu_time'] for d in valid]
        cpu_means.append(float(np.mean(cpus)) if cpus else 0)
    lines.append(row('CPU 時間(s) 平均', cpu_means, '.2f'))

    text = "\n".join(lines)
    print(text)
    return text


# ========== 横断サマリ ==========

def write_cross_summary(all_summaries, out_dir, args):
    path = os.path.join(out_dir, "cross_summary.txt")
    with open(path, 'w', encoding='utf-8') as f:
        f.write("コア比較実験 (実験1) 横断サマリー\n")
        f.write("=" * 70 + "\n")
        f.write(f"methods={args.methods}\n")
        f.write(f"weights={args.weights}\n")
        f.write(f"trials={args.trials}\n")
        f.write(f"ILS max_iter={args.ils_max_iter}, GA ngen={args.ga_ngen}\n")
        f.write(f"stagnation_threshold={args.stagnation_threshold}, "
                f"repair_strength={args.repair_strength}\n")
        f.write("=" * 70 + "\n\n")
        for lines in all_summaries:
            for line in lines:
                f.write(str(line) + "\n")
            f.write("\n")
    print(f"\n横断サマリー: {path}")


# ========== エントリポイント ==========

def main():
    parser = argparse.ArgumentParser(description="実験1: コア比較実行パイプライン")
    parser.add_argument(
        '--problems', nargs='+', type=str, default=None,
        help='問題セット (例: mt10:mt10_delay60 la36:la36_delay148)')
    parser.add_argument(
        '--methods', nargs='+', default=DEFAULT_METHODS,
        choices=list(METHODS.keys()),
        help=f'実行する手法 (デフォルト: {DEFAULT_METHODS})')
    parser.add_argument(
        '--weights', nargs='+', type=str,
        default=[f"{w[0]},{w[1]}" for w in DEFAULT_WEIGHTS],
        help='重み設定 (例: 0.85,0.15 0.8,0.2)')
    parser.add_argument(
        '--trials', type=int, default=DEFAULT_TRIALS,
        help=f'試行回数 (デフォルト: {DEFAULT_TRIALS})')
    parser.add_argument(
        '--ils-max-iter', type=int, default=ILS_MAX_ITER,
        help=f'ILS 最大反復数 (デフォルト: {ILS_MAX_ITER})')
    parser.add_argument(
        '--ga-ngen', type=int, default=GA_NGEN,
        help=f'GA 世代数 (デフォルト: {GA_NGEN})')
    parser.add_argument(
        '--stagnation-threshold', type=int, default=STAGNATION_THRESHOLD_DEFAULT,
        help=f'停滞判定の無改善反復数 (δ受理 / repair キックの共通ゲート, '
             f'デフォルト: {STAGNATION_THRESHOLD_DEFAULT})')
    parser.add_argument(
        '--repair-strength', type=int, default=REPAIR_STRENGTH_DEFAULT,
        help=f'repair 強度 (デフォルト: {REPAIR_STRENGTH_DEFAULT})')
    parser.add_argument(
        '--out-suffix', type=str, default='core',
        help='出力ディレクトリの prefix (デフォルト: core)')
    parser.add_argument(
        '--analyze', action='store_true',
        help='実行完了後に analyze_core.py を自動実行する')
    args = parser.parse_args()

    problem_sets = ([tuple(p.split(':')) for p in args.problems]
                    if args.problems else DEFAULT_PROBLEM_SETS)
    weights_list = [[float(x) for x in w.split(',')] for w in args.weights]

    out_dir = setup_output_dir(args.out_suffix, base_dir=os.path.dirname(__file__))
    print(f"出力先: {out_dir}")
    print(f"問題セット: {problem_sets}")
    print(f"手法: {args.methods}")
    print(f"重み: {weights_list}")
    print(f"試行回数: {args.trials}")
    print(f"ILS max_iter={args.ils_max_iter}, GA ngen={args.ga_ngen}")

    # config.json 保存
    config = {
        'problems': [list(p) for p in problem_sets],
        'methods': args.methods,
        'weights': weights_list,
        'trials': args.trials,
        'ils_max_iter': args.ils_max_iter,
        'ga_ngen': args.ga_ngen,
        'stagnation_threshold': args.stagnation_threshold,
        'repair_strength': args.repair_strength,
    }
    with open(os.path.join(out_dir, 'config.json'), 'w', encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    all_summaries = []
    for problem_name, scenario_name in problem_sets:
        for weights in weights_list:
            try:
                summary = run_problem_experiment(
                    problem_name, scenario_name, weights,
                    args.methods, args.trials, out_dir,
                    args.ils_max_iter, args.ga_ngen,
                    args.stagnation_threshold, args.repair_strength)
                all_summaries.append(summary)
            except Exception as e:
                import traceback
                print(f"\nERROR: {problem_name}/{scenario_name} weights={weights}: {e}")
                traceback.print_exc()
                all_summaries.append([f"ERROR: {problem_name}/{scenario_name}: {e}"])

    write_cross_summary(all_summaries, out_dir, args)
    print(f"\n全実験完了。結果は {out_dir} に保存されました。")

    if args.analyze:
        print(f"\n{'='*70}\n--analyze 指定されたので分析を実行します\n{'='*70}")
        import analyze_core
        sys.argv = ['analyze_core', out_dir]
        analyze_core.main()


if __name__ == "__main__":
    main()
