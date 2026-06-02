#!/usr/bin/env python3
"""
kick_rtb_ablation: PR / repair キック後の current 更新方針の ablation

問い: PR / repair キック後、current を「移し続ける(displacement)」のと
      「best に戻す(return-to-best, RTB)」のとどちらが良いか？

手法 (すべて perturb='insert', weight 既定 [0.8, 0.2]):
  ils_baseline      : キックなし (参照)
  ils_repair_disp   : repair キック, current 移し続け (現行動作)
  ils_repair_rtb    : repair キック, キック失敗時 current←best
  ils_pr_disp       : PR キック, current 移し続け (現行動作)
  ils_pr_rtb        : PR キック, キック失敗時 current←best

保存形式は core_comparison_v2/run_v2.py と同一 (analyze_v2 のメトリクスを流用するため)。
1ファイル=1run。既存ファイルはスキップ (resume 可)。

使い方:
  python run_ablation.py                      # 既定: la36+la40, 15試行, w=[0.8,0.2]
  python run_ablation.py --n-trials 20 --n-jobs 8
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

from experiment_utils import compute_shared_norm_params, ILS_MAX_ITER


# ========== 手法定義 ==========

METHODS = {
    'ils_baseline': dict(
        repair_mode=False, path_relink_mode=False, kick_return_to_best=False,
        label='ILS-baseline'),
    'ils_repair_disp': dict(
        repair_mode=True, path_relink_mode=False, kick_return_to_best=False,
        label='ILS+repair(displace)'),
    'ils_repair_rtb': dict(
        repair_mode=True, path_relink_mode=False, kick_return_to_best=True,
        label='ILS+repair(RTB)'),
    'ils_pr_disp': dict(
        repair_mode=False, path_relink_mode=True, kick_return_to_best=False,
        pr_return_intermediate=False, label='ILS+PR(displace)'),
    'ils_pr_rtb': dict(
        repair_mode=False, path_relink_mode=True, kick_return_to_best=True,
        pr_return_intermediate=False, label='ILS+PR(RTB)'),
    # --- 修正版 PR: 始点・終点を除外した中間解を返す（確実に解が入れ替わる） ---
    'ils_prfix_disp': dict(
        repair_mode=False, path_relink_mode=True, kick_return_to_best=False,
        pr_return_intermediate=True, label='ILS+PRfix(displace)'),
    'ils_prfix_rtb': dict(
        repair_mode=False, path_relink_mode=True, kick_return_to_best=True,
        pr_return_intermediate=True, label='ILS+PRfix(RTB)'),
}

DEFAULT_METHODS = list(METHODS.keys())

# mt10 は縮退ベンチ (headroom ゼロ) のため除外。headroom のある 15x15 を使用。
DEFAULT_PROBLEM_SETS = [
    ('la36', 'la36_delay148'),
    ('la40', 'la40_delay148'),
]

DEFAULT_WEIGHT = [0.8, 0.2]
DEFAULT_N_TRIALS = 15
DEFAULT_N_JOBS = 8
REPAIR_TRIGGER = 50
REPAIR_STRENGTH = 1
RELINK_TRIGGER = 50


def _weight_label(w):
    return f"w{int(round(w[0] * 10)):02d}_{int(round(w[1] * 10)):02d}"


def _out_path(out_dir, problem, scenario, method, weight, trial):
    prob_label = f"{problem}_{scenario}"
    fn = f"{method}__{_weight_label(weight)}__t{trial:03d}.json"
    return os.path.join(out_dir, prob_label, 'raw', fn)


def _extract_uea_points(history):
    """ILS history から全訪問点 (ls + kick) を抽出。run_v2 と同形式。"""
    pts = []
    for h in history:
        ms, st = h.get('ls_makespan'), h.get('ls_stability')
        if ms is not None and st is not None and np.isfinite(ms) and np.isfinite(st):
            pts.append([float(ms), float(st)])
        kms, kst = h.get('kick_makespan'), h.get('kick_stability')
        if kms is not None and kst is not None and np.isfinite(kms) and np.isfinite(kst):
            pts.append([float(kms), float(kst)])
    return pts


def _slim_anytime(history):
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


def _run_one_task(task):
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
    from experiment_utils import run_ils as _run_ils

    cfg = METHODS[task['method']]
    try:
        r = _run_ils(
            task['weights'], task['seed'], 'insert', task['ils_max_iter'],
            task['norm_params'], strategy='best',
            repair_mode=cfg['repair_mode'], repair_trigger=task['repair_trigger'],
            repair_strength=task['repair_strength'],
            path_relink_mode=cfg['path_relink_mode'], relink_trigger=task['relink_trigger'],
            kick_return_to_best=cfg['kick_return_to_best'],
            pr_return_intermediate=cfg.get('pr_return_intermediate', False),
            problem_name=task['problem'], scenario_name=task['scenario'],
        )
        history = r['history']
        save_data = {
            'method': task['method'],
            'problem': task['problem'],
            'scenario': task['scenario'],
            'weights': task['weights'],
            'trial': task['trial'],
            'seed': task['seed'],
            'baseline': r.get('baseline'),
            'baseline_score': r.get('baseline_score'),
            'finals': {'makespan': r['makespan'], 'stability': r['stability']},
            'convergence': r['convergence'],
            'history': _slim_anytime(history),
            'uea_points': _extract_uea_points(history),
        }
        _os.makedirs(_os.path.dirname(out_path), exist_ok=True)
        with open(out_path, 'w', encoding='utf-8') as f:
            _json.dump(save_data, f, ensure_ascii=False)
        return {'status': 'done', 'path': out_path,
                'makespan': r['makespan'], 'stability': r['stability'],
                'cpu': r['convergence'].get('total_cpu_time', 0.0)}
    except Exception as e:
        return {'status': 'error', 'path': out_path,
                'error': str(e), 'traceback': _tb.format_exc()}


def main():
    parser = argparse.ArgumentParser(description='kick RTB ablation')
    parser.add_argument('--methods', nargs='+', default=DEFAULT_METHODS,
                        choices=list(METHODS.keys()))
    parser.add_argument('--problems', nargs='+', default=None,
                        help='例: la36:la36_delay148')
    parser.add_argument('--weight', type=str, default=None, help='例: 0.8,0.2')
    parser.add_argument('--n-trials', type=int, default=DEFAULT_N_TRIALS)
    parser.add_argument('--n-jobs', type=int, default=DEFAULT_N_JOBS)
    parser.add_argument('--ils-max-iter', type=int, default=ILS_MAX_ITER)
    parser.add_argument('--output-dir', type=str, default=None)
    args = parser.parse_args()

    if args.problems:
        problem_sets = [tuple(p.split(':')) for p in args.problems]
    else:
        problem_sets = DEFAULT_PROBLEM_SETS
    weight = [float(x) for x in args.weight.split(',')] if args.weight else DEFAULT_WEIGHT

    if args.output_dir:
        out_dir = args.output_dir
    else:
        out_dir = os.path.join(_HERE, 'results', 'ablation')
    os.makedirs(out_dir, exist_ok=True)

    print(f"出力先: {out_dir}")
    print(f"問題: {[p[0] for p in problem_sets]}  手法: {args.methods}")
    print(f"重み: {weight}  試行: {args.n_trials}  ILS max_iter={args.ils_max_iter}")

    config = {
        'problems': [list(p) for p in problem_sets], 'methods': args.methods,
        'weight': weight, 'n_trials': args.n_trials,
        'ils_max_iter': args.ils_max_iter,
        'repair_trigger': REPAIR_TRIGGER, 'repair_strength': REPAIR_STRENGTH,
        'relink_trigger': RELINK_TRIGGER,
    }
    with open(os.path.join(out_dir, 'config.json'), 'w', encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    # norm_params (問題ごと、キャッシュ)
    norm_path = os.path.join(out_dir, 'norm_params.json')
    norm_cache = {}
    raw = {}
    if os.path.exists(norm_path):
        with open(norm_path, 'r', encoding='utf-8') as f:
            raw = json.load(f)
    for prob, scen in problem_sets:
        k = f'{prob}/{scen}'
        if k in raw:
            norm_cache[(prob, scen)] = raw[k]
            print(f'[norm_params] {k}: 既存読み込み')
        else:
            print(f'[norm_params] {k}: 計算中...')
            norm_cache[(prob, scen)] = compute_shared_norm_params(prob, scen)
            raw[k] = norm_cache[(prob, scen)]
    with open(norm_path, 'w', encoding='utf-8') as f:
        json.dump(raw, f, ensure_ascii=False, indent=2)

    all_tasks = []
    for prob, scen in problem_sets:
        for method in args.methods:
            for trial in range(args.n_trials):
                all_tasks.append({
                    'problem': prob, 'scenario': scen, 'weights': weight,
                    'method': method, 'trial': trial, 'seed': trial * 100 + 7,
                    'out_path': _out_path(out_dir, prob, scen, method, weight, trial),
                    'norm_params': norm_cache[(prob, scen)],
                    'ils_max_iter': args.ils_max_iter,
                    'repair_trigger': REPAIR_TRIGGER,
                    'repair_strength': REPAIR_STRENGTH,
                    'relink_trigger': RELINK_TRIGGER,
                })

    pending = [t for t in all_tasks if not os.path.exists(t['out_path'])]
    n_skip = len(all_tasks) - len(pending)
    print(f"\n総タスク: {len(all_tasks)}  スキップ: {n_skip}  実行: {len(pending)}")
    if not pending:
        print("完了済み。analyze_ablation.py を実行してください。")
        return

    done = err = 0
    with ProcessPoolExecutor(max_workers=args.n_jobs) as ex:
        futures = {ex.submit(_run_one_task, t): t for t in pending}
        for fut in as_completed(futures):
            t = futures[fut]
            label = f"{t['problem']}/{t['method']}/t{t['trial']:03d}"
            try:
                r = fut.result()
                if r['status'] == 'done':
                    done += 1
                    print(f"  [{done + n_skip}/{len(all_tasks)}] {label} "
                          f"MS={r.get('makespan')} St={r.get('stability', 0):.2f} "
                          f"CPU={r.get('cpu', 0):.1f}s")
                elif r['status'] == 'error':
                    err += 1
                    print(f"  [ERROR] {label}: {r.get('error')}")
                    for line in (r.get('traceback') or '').splitlines()[-4:]:
                        print(f"    {line}")
            except Exception as e:
                err += 1
                print(f"  [FATAL] {label}: {e}")

    print(f"\n完了: done={done} error={err}  結果: {out_dir}")
    print(f"分析: python analyze_ablation.py --input-dir {out_dir}")


if __name__ == '__main__':
    main()
