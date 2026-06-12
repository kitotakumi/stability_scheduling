#!/usr/bin/env python3
"""
memetic 版 kick ablation: memetic（kick なし）と memetic+PR を比較する。

手法 (kick_prob=0.3, weight 既定 [0.8,0.2]):
  memetic_ls : kick なし (memetic 単体 = baseline)
  memetic_pr : PR kick (memetic では return_intermediate=False が適切)

※ 始点・終点除外PR (return_intermediate=True) は memetic では効果なし＋大幅減速で
   不採用と確定済み（2026-06-02）。本スクリプトからは prfix を除外した。
   （ILS では逆に return_intermediate=True が採用 → run_ablation.py 側を参照）

保存形式は run_ablation.py と同一 (analyze_memetic_ablation.py で流用)。
"""

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, '..', '..'))
sys.path.insert(0, os.path.join(_HERE, '..'))

import numpy as np

from experiment_utils import compute_shared_norm_params, MEMETIC_NGEN


METHODS = {
    'memetic_ls': dict(kick_mode='none', pr_step_strategy='best', label='Memetic-LS'),
    'memetic_pr': dict(kick_mode='pr', pr_step_strategy='best', label='Memetic+PR(BI)'),
    'memetic_pr_fi': dict(kick_mode='pr', pr_step_strategy='first', label='Memetic+PR(FI)'),
    'memetic_pr_rand': dict(kick_mode='pr', pr_step_strategy='random', label='Memetic+PR(rand)'),
}
DEFAULT_METHODS = list(METHODS.keys())

DEFAULT_PROBLEM_SETS = [
    ('la36', 'la36_delay148'),
    ('la40', 'la40_delay148'),
]
DEFAULT_WEIGHT = [0.8, 0.2]
DEFAULT_N_TRIALS = 10
DEFAULT_N_JOBS = 8
KICK_PROB = 0.3


def _weight_label(w):
    return f"w{int(round(w[0] * 10)):02d}_{int(round(w[1] * 10)):02d}"


def _out_path(out_dir, problem, scenario, method, weight, trial):
    fn = f"{method}__{_weight_label(weight)}__t{trial:03d}.json"
    return os.path.join(out_dir, f"{problem}_{scenario}", 'raw', fn)


def _extract_uea_points_memetic(history):
    """memetic history (track_population) から pop_points + kick_points を抽出。"""
    pts = []
    for h in history:
        for pt in h.get('pop_points', []):
            if len(pt) >= 2 and np.isfinite(pt[0]) and np.isfinite(pt[1]):
                pts.append([float(pt[0]), float(pt[1])])
        for pt in h.get('kick_points', []):
            if len(pt) >= 2 and np.isfinite(pt[0]) and np.isfinite(pt[1]):
                pts.append([float(pt[0]), float(pt[1])])
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
    from experiment_utils import run_memetic as _run_memetic

    cfg = METHODS[task['method']]
    try:
        r = _run_memetic(
            task['weights'], task['seed'], task['ngen'], task['norm_params'],
            problem_name=task['problem'], scenario_name=task['scenario'],
            kick_mode=cfg['kick_mode'], kick_prob=KICK_PROB,
            pr_step_strategy=cfg['pr_step_strategy'],
            track_population=True)
        history = r['history']
        save_data = {
            'method': task['method'], 'problem': task['problem'],
            'scenario': task['scenario'], 'weights': task['weights'],
            'trial': task['trial'], 'seed': task['seed'],
            'baseline': r.get('baseline'), 'baseline_rsr': r.get('baseline_rsr'),
            'baseline_score': r.get('baseline_score'),
            'finals': {'makespan': r['makespan'], 'stability': r['stability']},
            'convergence': r['convergence'],
            'history': _slim_anytime(history),
            'uea_points': _extract_uea_points_memetic(history),
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
    parser = argparse.ArgumentParser(description='memetic kick PR ablation')
    parser.add_argument('--methods', nargs='+', default=DEFAULT_METHODS,
                        choices=list(METHODS.keys()))
    parser.add_argument('--problems', nargs='+', default=None)
    parser.add_argument('--weight', type=str, default=None)
    parser.add_argument('--n-trials', type=int, default=DEFAULT_N_TRIALS)
    parser.add_argument('--n-jobs', type=int, default=DEFAULT_N_JOBS)
    parser.add_argument('--ngen', type=int, default=MEMETIC_NGEN)
    parser.add_argument('--output-dir', type=str,
                        default=os.path.join(_HERE, 'results', 'memetic_w08_02'))
    args = parser.parse_args()

    if args.problems:
        problem_sets = [tuple(p.split(':')) for p in args.problems]
    else:
        problem_sets = DEFAULT_PROBLEM_SETS
    weight = [float(x) for x in args.weight.split(',')] if args.weight else DEFAULT_WEIGHT
    out_dir = args.output_dir
    os.makedirs(out_dir, exist_ok=True)

    print(f"出力先: {out_dir}")
    print(f"問題: {[p[0] for p in problem_sets]}  手法: {args.methods}")
    print(f"重み: {weight}  試行: {args.n_trials}  ngen={args.ngen}")

    # norm_params (既存 ILS ablation のものを再利用可)
    norm_path = os.path.join(out_dir, 'norm_params.json')
    raw = {}
    if os.path.exists(norm_path):
        with open(norm_path, 'r', encoding='utf-8') as f:
            raw = json.load(f)
    norm_cache = {}
    for prob, scen in problem_sets:
        k = f'{prob}/{scen}'
        if k in raw:
            norm_cache[(prob, scen)] = raw[k]
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
                    'norm_params': norm_cache[(prob, scen)], 'ngen': args.ngen,
                })
    pending = [t for t in all_tasks if not os.path.exists(t['out_path'])]
    n_skip = len(all_tasks) - len(pending)
    print(f"\n総タスク: {len(all_tasks)}  スキップ: {n_skip}  実行: {len(pending)}")
    if not pending:
        print("完了済み。analyze_memetic_ablation.py を実行してください。")
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


if __name__ == '__main__':
    main()
