#!/usr/bin/env python3
"""
core_comparison_v3: コア比較実験 v3 実行スクリプト

4手法 (GA, ILS-baseline, ILS+repair, ILS+PR) × 11重み × 4問題 × n試行。
1ファイル=1run で保存し、途中停止・再開・部分実行に対応。

=== 使い方 ===
  # パイロット（10試行、デフォルト全問題・全重み・全手法）
  python run_v3.py --n-trials 10 --n-jobs 4

  # 重みを分割して同じ output-dir に統合
  python run_v3.py --weights "1.0,0 0.9,0.1 0.8,0.2 0.7,0.3 0.6,0.4" \\
      --output-dir results/main --n-jobs 4
  python run_v3.py --weights "0.5,0.5 0.4,0.6 0.3,0.7 0.2,0.8 0.1,0.9 0.0,1.0" \\
      --output-dir results/main --n-jobs 4

  # 本番（30試行）
  python run_v3.py --n-trials 30 --output-dir results/main --n-jobs 4

=== 並列戦略 ===
  (problem × weight × method × trial) の全組み合わせをフラットリスト化し
  ProcessPoolExecutor(max_workers=n_jobs) で並列実行。
  既存ファイルはスキップするので --output-dir を同じにすれば resume になる。

=== 出力構造 ===
  <output-dir>/
  ├── config.json
  └── <problem>_<scenario>/
      └── raw/
          └── <method>__<w_label>__t<trial:03d>.json
"""

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, '..', '..'))
sys.path.insert(0, os.path.join(_HERE, '..'))

import numpy as np

from experiment_utils import (
    compute_shared_norm_params,
    run_ga, run_ils, run_memetic,
    ILS_MAX_ITER, GA_NGEN, MEMETIC_NGEN,
)


# ========== 手法定義 ==========

METHODS = {
    'ga': {
        'kind': 'ga',
        'label': 'GA',
    },
    'ils_baseline': {
        'kind': 'ils',
        'perturb': 'insert',
        'repair_mode': False,
        'path_relink_mode': False,
        'label': 'ILS-baseline',
    },
    'ils_repair': {
        'kind': 'ils',
        'perturb': 'insert',
        'repair_mode': True,
        'path_relink_mode': False,
        'label': 'ILS+repair',
    },
    'ils_pr': {
        'kind': 'ils',
        'perturb': 'insert',
        'repair_mode': False,
        'path_relink_mode': True,
        'label': 'ILS+PR',
    },
    'memetic_ls': {
        'kind': 'memetic',
        'kick_mode': 'none',
        'label': 'Memetic-LS',
    },
    'memetic_repair': {
        'kind': 'memetic',
        'kick_mode': 'repair',
        'kick_prob': 0.3,
        'repair_strength': 0,  # 0 = 経路長フル（[1, 経路長] でランダム）。>0 で天井
        'label': 'Memetic+repair',
    },
    'memetic_pr': {
        'kind': 'memetic',
        'kick_mode': 'pr',
        'kick_prob': 0.3,
        'label': 'Memetic+PR',
    },
}

DEFAULT_METHODS = ['ga', 'ils_baseline', 'ils_repair', 'ils_pr']

DEFAULT_WEIGHTS = [
    [1.0, 0.0], [0.9, 0.1], [0.8, 0.2], [0.7, 0.3], [0.6, 0.4],
    [0.5, 0.5], [0.4, 0.6], [0.3, 0.7], [0.2, 0.8], [0.1, 0.9], [0.0, 1.0],
]

DEFAULT_PROBLEM_SETS = [
    # 一般性（中間位置・非縮退）
    ('mt10', 'mt10_delay60'),
    ('la21', 'la21_delay147'),
    ('la40', 'la40_delay148'),
    # la36 位置軸（難易度 大小）
    ('la36', 'la36_large'),   # 大（遠い, pos0.17, headroom62）
    ('la36', 'la36_small'),   # 小（近傍, pos0.72, headroom29）
    # 大規模 showcase
    ('ta21', 'ta21_delay97'),
]

DEFAULT_N_TRIALS = 10
DEFAULT_N_JOBS = 8
# 適応トリガー: 初回キックは KICK_TRIGGER_FIRST（ILS を収束まで深ぼらせてから発動）、
# 以降は REPAIR_TRIGGER/RELINK_TRIGGER（収束後に密にキックして安定性側 front を掃く）。
KICK_TRIGGER_FIRST_DEFAULT = 400
REPAIR_TRIGGER_DEFAULT = 10
RELINK_TRIGGER_DEFAULT = 10
# repair 鋸歯エスカレーションの depth 天井。0 = 経路長（current→初期解の不一致数）をフルに
# 浅→深と掃く。>0 にすると min(経路長, この値) で深さを制限（深い repair の LS コスト抑制用）。
REPAIR_STRENGTH_DEFAULT = 0


# ========== ユーティリティ ==========

def _weight_label(w):
    """[0.9, 0.1] -> 'w09_01'"""
    return f"w{int(round(w[0] * 10)):02d}_{int(round(w[1] * 10)):02d}"


def _out_path(out_dir, problem, scenario, method, weight, trial):
    prob_label = f"{problem}_{scenario}"
    wl = _weight_label(weight)
    fn = f"{method}__{wl}__t{trial:03d}.json"
    return os.path.join(out_dir, prob_label, 'raw', fn)


def _extract_uea_points(history, kind):
    """UEA scenario の全訪問点と各点の訪問 cpu_time を抽出。
    GA: 全世代×全個体の (ms, st)
    ILS: 全 iter の LS 結果 (ls_makespan, ls_stability)

    各点には、それを生成した snapshot（GA: 世代 / ILS: 反復）の cpu_time を
    アラインして付与する。これにより anytime HV(t) や HV ベースの
    time-to-target を、点の順序や評価回数からの近似に頼らず正確に再構成できる。

    Returns:
        (points, times) — points: [[ms, st], ...], times: [cpu_time, ...]
        両者は同じ長さでインデックスがアラインしている。
    """
    if not history:
        return [], []
    pts = []
    times = []
    if kind == 'ga':
        for h in history:
            t = h.get('cpu_time')
            tv = round(float(t), 4) if t is not None else 0.0
            for pt in h.get('pop_points', []):
                if len(pt) >= 2:
                    ms, st = float(pt[0]), float(pt[1])
                    if np.isfinite(ms) and np.isfinite(st):
                        pts.append([ms, st]); times.append(tv)
            for pt in h.get('kick_points', []):
                if len(pt) >= 2:
                    ms, st = float(pt[0]), float(pt[1])
                    if np.isfinite(ms) and np.isfinite(st):
                        pts.append([ms, st]); times.append(tv)
    else:
        for h in history:
            t = h.get('cpu_time')
            tv = round(float(t), 4) if t is not None else 0.0
            ms = h.get('ls_makespan')
            st = h.get('ls_stability')
            if ms is not None and st is not None and np.isfinite(ms) and np.isfinite(st):
                pts.append([float(ms), float(st)]); times.append(tv)
            kick_ms = h.get('kick_makespan')
            kick_st = h.get('kick_stability')
            if kick_ms is not None and kick_st is not None and np.isfinite(kick_ms) and np.isfinite(kick_st):
                pts.append([float(kick_ms), float(kick_st)]); times.append(tv)
    return pts, times


def _slim_anytime(history, kind):
    """anytime curve 用: 共通スキーマ [{cpu_time, best_ms, best_st, best_score, evaluations}]"""
    if not history:
        return []
    out = []
    for h in history:
        t = h.get('cpu_time')
        if t is None:
            continue
        out.append({
            'cpu_time': float(t),
            'best_ms': h.get('best_makespan'),
            'best_st': h.get('best_stability'),
            'best_score': h.get('best_score'),
            'evaluations': h.get('evaluations'),
        })
    return out


# ========== 個別実行関数（並列用・モジュールレベル必須） ==========

def _run_one_task(task):
    """1 run を実行して JSON に保存。既存ファイルはスキップ。"""
    out_path = task['out_path']
    if os.path.exists(out_path):
        return {'status': 'skipped', 'path': out_path}

    import os as _os
    import sys as _sys
    import json as _json
    import traceback as _tb

    _here = _os.path.dirname(_os.path.abspath(__file__))
    _sys.path.insert(0, _os.path.join(_here, '..', '..'))
    _sys.path.insert(0, _os.path.join(_here, '..'))
    from experiment_utils import run_ga as _run_ga, run_ils as _run_ils, run_memetic as _run_memetic

    problem_name = task['problem']
    scenario_name = task['scenario']
    weights = task['weights']
    method_key = task['method']
    trial = task['trial']
    seed = task['seed']
    norm_params = task['norm_params']
    ils_max_iter = task['ils_max_iter']
    ga_ngen = task['ga_ngen']
    repair_trigger = task['repair_trigger']
    repair_strength = task['repair_strength']
    relink_trigger = task['relink_trigger']
    kick_trigger_first = task['kick_trigger_first']

    cfg = METHODS[method_key]
    try:
        if cfg['kind'] == 'ga':
            r = _run_ga(weights, seed, ga_ngen, norm_params,
                        problem_name=problem_name, scenario_name=scenario_name,
                        track_population=True)
        elif cfg['kind'] == 'memetic':
            r = _run_memetic(
                weights, seed, task['memetic_ngen'], norm_params,
                problem_name=problem_name, scenario_name=scenario_name,
                kick_mode=cfg.get('kick_mode', 'none'),
                kick_prob=cfg.get('kick_prob', 0.5),
                repair_strength=cfg.get('repair_strength', 0),  # 0 = 経路長フル（天井なし）
                track_population=True)
            kind = 'ga'  # pop_points 形式は GA と同じ
        else:
            r = _run_ils(
                weights, seed, cfg['perturb'], ils_max_iter, norm_params,
                strategy='best',
                repair_mode=cfg.get('repair_mode', False),
                repair_trigger=repair_trigger,
                repair_strength=repair_strength,
                path_relink_mode=cfg.get('path_relink_mode', False),
                relink_trigger=relink_trigger,
                kick_trigger_first=kick_trigger_first,
                problem_name=problem_name, scenario_name=scenario_name,
            )

        history = r['history']
        kind = 'ga' if cfg['kind'] == 'memetic' else cfg['kind']
        uea_points, uea_points_t = _extract_uea_points(history, kind)
        slim_history = _slim_anytime(history, kind)

        save_data = {
            'method': method_key,
            'problem': problem_name,
            'scenario': scenario_name,
            'weights': weights,
            'trial': trial,
            'seed': seed,
            'baseline': r.get('baseline'),
            'baseline_rsr': r.get('baseline_rsr'),
            'baseline_score': r.get('baseline_score'),
            'finals': {
                'makespan': r['makespan'],
                'stability': r['stability'],
            },
            'convergence': r['convergence'],
            'history': slim_history,
            'uea_points': uea_points,
            'uea_points_t': uea_points_t,  # uea_points と同長: 各点の訪問 cpu_time
        }

        _os.makedirs(_os.path.dirname(out_path), exist_ok=True)
        with open(out_path, 'w', encoding='utf-8') as f:
            _json.dump(save_data, f, ensure_ascii=False)

        return {
            'status': 'done',
            'path': out_path,
            'makespan': r['makespan'],
            'stability': r['stability'],
            'cpu': r['convergence'].get('total_cpu_time', 0.0),
        }

    except Exception as e:
        return {
            'status': 'error',
            'path': out_path,
            'error': str(e),
            'traceback': _tb.format_exc(),
        }


# ========== メイン ==========

def main():
    parser = argparse.ArgumentParser(
        description='core_comparison_v3: 4手法×11重み コア比較実験')
    parser.add_argument(
        '--problems', nargs='+', default=None,
        help='問題セット (例: mt10:mt10_delay60 la36:la36_delay148)')
    parser.add_argument(
        '--methods', nargs='+', default=DEFAULT_METHODS,
        choices=list(METHODS.keys()),
        help=f'手法 (デフォルト: {DEFAULT_METHODS})')
    parser.add_argument(
        '--weights', nargs='+', type=str, default=None,
        help='重み (例: "1.0,0 0.9,0.1"). デフォルト: 0.1刻み11点')
    parser.add_argument(
        '--n-trials', type=int, default=DEFAULT_N_TRIALS,
        help=f'試行回数 (デフォルト: {DEFAULT_N_TRIALS})')
    parser.add_argument(
        '--n-jobs', type=int, default=DEFAULT_N_JOBS,
        help=f'並列数 (デフォルト: {DEFAULT_N_JOBS})')
    parser.add_argument(
        '--ils-max-iter', type=int, default=ILS_MAX_ITER,
        help=f'ILS 最大反復数 (デフォルト: {ILS_MAX_ITER})')
    parser.add_argument(
        '--ga-ngen', type=int, default=GA_NGEN,
        help=f'GA 世代数 (デフォルト: {GA_NGEN})')
    parser.add_argument(
        '--repair-trigger', type=int, default=REPAIR_TRIGGER_DEFAULT,
        help=f'repair 無改善発動閾値 (デフォルト: {REPAIR_TRIGGER_DEFAULT})')
    parser.add_argument(
        '--repair-strength', type=int, default=REPAIR_STRENGTH_DEFAULT,
        help=f'repair 強度 (デフォルト: {REPAIR_STRENGTH_DEFAULT})')
    parser.add_argument(
        '--relink-trigger', type=int, default=RELINK_TRIGGER_DEFAULT,
        help=f'path relink 発動閾値（適応トリガーの「以降」値）(デフォルト: {RELINK_TRIGGER_DEFAULT})')
    parser.add_argument(
        '--kick-trigger-first', type=int, default=KICK_TRIGGER_FIRST_DEFAULT,
        help=f'適応トリガーの初回キック閾値 (デフォルト: {KICK_TRIGGER_FIRST_DEFAULT})')
    parser.add_argument(
        '--memetic-ngen', type=int, default=MEMETIC_NGEN,
        help=f'Memetic GA 世代数 (デフォルト: {MEMETIC_NGEN})')
    parser.add_argument(
        '--output-dir', type=str, default=None,
        help='出力先 (デフォルト: results/core_v3_<timestamp>). '
             '同じ dir を指定すれば resume になる。')
    args = parser.parse_args()

    # 問題セット
    if args.problems:
        problem_sets = [tuple(p.split(':')) for p in args.problems]
    else:
        problem_sets = DEFAULT_PROBLEM_SETS

    # 重み
    if args.weights:
        weights_list = [[float(x) for x in w.split(',')] for ws in args.weights for w in ws.split()]
    else:
        weights_list = DEFAULT_WEIGHTS

    # 出力先
    if args.output_dir:
        out_dir = args.output_dir
    else:
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        out_dir = os.path.join(_HERE, 'results', f'core_v3_{ts}')
    os.makedirs(out_dir, exist_ok=True)

    n_total_plan = (len(problem_sets) * len(weights_list) *
                    len(args.methods) * args.n_trials)
    print(f"出力先: {out_dir}")
    print(f"問題: {[p[0] for p in problem_sets]}")
    print(f"手法: {args.methods}")
    print(f"重み: {len(weights_list)} 点")
    print(f"試行回数: {args.n_trials}")
    print(f"総 run 数: {n_total_plan}")
    print(f"ILS max_iter={args.ils_max_iter}, GA ngen={args.ga_ngen}, Memetic ngen={args.memetic_ngen}")
    print(f"並列数: {args.n_jobs}")

    # config.json
    config = {
        'problems': [list(p) for p in problem_sets],
        'methods': args.methods,
        'weights': weights_list,
        'n_trials': args.n_trials,
        'ils_max_iter': args.ils_max_iter,
        'ga_ngen': args.ga_ngen,
        'memetic_ngen': args.memetic_ngen,
        'repair_trigger': args.repair_trigger,
        'repair_strength': args.repair_strength,
        'relink_trigger': args.relink_trigger,
        'kick_trigger_first': args.kick_trigger_first,
    }
    with open(os.path.join(out_dir, 'config.json'), 'w', encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    # 問題ごとに norm_params を取得（既存ファイルがあれば再利用、なければ計算して保存）
    norm_params_path = os.path.join(out_dir, 'norm_params.json')
    norm_params_cache = {}
    if os.path.exists(norm_params_path):
        with open(norm_params_path, 'r', encoding='utf-8') as f:
            raw = json.load(f)
        # キーは "problem/scenario" 文字列で保存
        for problem_name, scenario_name in problem_sets:
            k = f'{problem_name}/{scenario_name}'
            if k in raw:
                norm_params_cache[(problem_name, scenario_name)] = raw[k]
                print(f'[norm_params] {problem_name}/{scenario_name}: 既存ファイルから読み込み')
            else:
                print(f'[norm_params] {problem_name}/{scenario_name}: 新規計算')
                norm_params_cache[(problem_name, scenario_name)] = \
                    compute_shared_norm_params(problem_name, scenario_name)
    else:
        for problem_name, scenario_name in problem_sets:
            print(f'\n[norm_params] {problem_name}/{scenario_name}')
            norm_params_cache[(problem_name, scenario_name)] = \
                compute_shared_norm_params(problem_name, scenario_name)

    # norm_params を保存（新規キーをマージして上書き）
    save_data = {}
    if os.path.exists(norm_params_path):
        with open(norm_params_path, 'r', encoding='utf-8') as f:
            save_data = json.load(f)
    for (problem_name, scenario_name), params in norm_params_cache.items():
        save_data[f'{problem_name}/{scenario_name}'] = params
    with open(norm_params_path, 'w', encoding='utf-8') as f:
        json.dump(save_data, f, ensure_ascii=False, indent=2)
    print(f'\n[norm_params] {norm_params_path} に保存')

    # 全タスクをフラットリストに展開
    all_tasks = []
    for problem_name, scenario_name in problem_sets:
        key = (problem_name, scenario_name)
        np_params = norm_params_cache[key]
        for weight in weights_list:
            for method in args.methods:
                for trial in range(args.n_trials):
                    all_tasks.append({
                        'problem': problem_name,
                        'scenario': scenario_name,
                        'weights': weight,
                        'method': method,
                        'trial': trial,
                        'seed': trial * 100 + 7,
                        'out_path': _out_path(out_dir, problem_name, scenario_name,
                                              method, weight, trial),
                        'norm_params': np_params,
                        'ils_max_iter': args.ils_max_iter,
                        'ga_ngen': args.ga_ngen,
                        'memetic_ngen': args.memetic_ngen,
                        'repair_trigger': args.repair_trigger,
                        'repair_strength': args.repair_strength,
                        'relink_trigger': args.relink_trigger,
                        'kick_trigger_first': args.kick_trigger_first,
                    })

    pending = [t for t in all_tasks if not os.path.exists(t['out_path'])]
    n_skip = len(all_tasks) - len(pending)
    print(f"\n総タスク: {len(all_tasks)}  スキップ(済): {n_skip}  実行: {len(pending)}")

    if not pending:
        print("全タスク完了済み。分析は analyze_v3.py を実行してください。")
        return

    # 並列実行
    done_count = 0
    error_count = 0
    with ProcessPoolExecutor(max_workers=args.n_jobs) as executor:
        futures = {executor.submit(_run_one_task, t): t for t in pending}
        for future in as_completed(futures):
            task = futures[future]
            label = (f"{task['problem']}/{task['method']}/"
                     f"{_weight_label(task['weights'])}/t{task['trial']:03d}")
            try:
                r = future.result()
                if r['status'] == 'done':
                    done_count += 1
                    total_done = done_count + n_skip
                    print(f"  [{total_done}/{len(all_tasks)}] {label} "
                          f"MS={r.get('makespan', '?')} "
                          f"Stab={r.get('stability', 0.0):.3f} "
                          f"CPU={r.get('cpu', 0.0):.1f}s")
                elif r['status'] == 'error':
                    error_count += 1
                    print(f"  [ERROR] {label}: {r.get('error')}")
                    if r.get('traceback'):
                        for line in r['traceback'].splitlines()[-5:]:
                            print(f"    {line}")
            except Exception as e:
                error_count += 1
                print(f"  [FATAL] {label}: {e}")

    print(f"\n実行完了: done={done_count}  error={error_count}")
    print(f"結果: {out_dir}")
    print(f"分析: python analyze_v3.py --input-dir {out_dir}")


if __name__ == '__main__':
    main()
