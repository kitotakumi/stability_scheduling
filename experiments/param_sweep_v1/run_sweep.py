#!/usr/bin/env python3
"""param_sweep_v1: パラメータ感度分析（OAT）実行スクリプト。

center 設定を基準に1パラメータずつ振る。各軸は対象手法だけで実行し、center は
全軸で共有して1回だけ計算する（tag による自然 dedup）。記録形式は run_v3 と同一
（uea_points + uea_points_t）なので analyze_v3 の指標をそのまま再利用できる。

詳細は DESIGN.md を参照。

使い方:
  python run_sweep.py --n-trials 10 --n-jobs 8 --output-dir results/main
  python run_sweep.py --axes kick_prob pr_ls_top_k --n-trials 10   # 一部の軸だけ
"""

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime

_HERE = os.path.dirname(os.path.abspath(__file__))
_CORE_V3 = os.path.join(_HERE, '..', 'core_comparison_v3')
sys.path.insert(0, os.path.join(_HERE, '..', '..'))
sys.path.insert(0, os.path.join(_HERE, '..'))
sys.path.insert(0, _CORE_V3)

from experiment_utils import (
    compute_shared_norm_params, ILS_MAX_ITER, GA_NGEN, MEMETIC_NGEN,
)
from run_v3 import _extract_uea_points, _slim_anytime, _weight_label


# ========== center（基準設定） ==========

CENTER = {
    'perturb': 'insert', 'initial_strength': 2, 'max_strength': 5,
    'relink_trigger': 10, 'repair_trigger': 10, 'kick_trigger_first': 400,
    'pr_step_strategy': 'random', 'pr_ls_top_k': 1,
    'kick_prob': 0.3, 'repair_strength': 0,
    'pop_size': 50, 'cxpb': 0.85, 'mutpb': 0.1,
}

METHOD_KIND = {
    'ga': 'ga',
    'ils_baseline': 'ils', 'ils_pr': 'ils', 'ils_repair': 'ils',
    'memetic_ls': 'memetic', 'memetic_pr': 'memetic', 'memetic_repair': 'memetic',
}

# ========== 掃引軸（OAT） ==========
# 各軸: param(=CENTER のキー or 'trigger' 特殊), methods, values, center_val
AXES = {
    'pr_step_strategy': dict(param='pr_step_strategy', methods=['ils_pr', 'memetic_pr'],
                             values=['random', 'best'], center='random'),
    'pr_ls_top_k':      dict(param='pr_ls_top_k', methods=['ils_pr', 'memetic_pr'],
                             values=[1, 3, 5], center=1),
    'repair_strength':  dict(param='repair_strength', methods=['ils_repair', 'memetic_repair'],
                             values=[0, 2, 4, 8], center=0),
    'kick_prob':        dict(param='kick_prob', methods=['memetic_pr', 'memetic_repair'],
                             values=[0.1, 0.2, 0.3, 0.5, 0.7], center=0.3),
    'kick_trigger_first': dict(param='kick_trigger_first', methods=['ils_pr', 'ils_repair'],
                               values=[100, 200, 400, 600], center=400),
    'trigger':          dict(param='trigger', methods=['ils_pr', 'ils_repair'],
                             values=[5, 10, 20, 40], center=10),
    'pop_size':         dict(param='pop_size', methods=['ga', 'memetic_ls'],
                             values=[30, 50, 80], center=50),
    'cxpb':             dict(param='cxpb', methods=['ga'],
                             values=[0.6, 0.85, 0.95], center=0.85),
    'mutpb':            dict(param='mutpb', methods=['ga'],
                             values=[0.05, 0.1, 0.2], center=0.1),
    'perturb':          dict(param='perturb', methods=['ils_baseline'],
                             values=['insert', 'swap'], center='insert'),
    'max_strength':     dict(param='max_strength', methods=['ils_baseline'],
                             values=[3, 5, 8], center=5),
}

DEFAULT_PROBLEM_SETS = [
    ('la21', 'la21_delay147'),
    ('la36', 'la36_large'),
]
DEFAULT_WEIGHTS = [
    [1.0, 0.0], [0.9, 0.1], [0.8, 0.2], [0.6, 0.4], [0.4, 0.6], [0.2, 0.8],
]
DEFAULT_N_TRIALS = 10
DEFAULT_N_JOBS = 8


def _axis_override(axis_name, param, value, method):
    """軸の (param, value) を、その手法の実 run パラメータ override に変換。"""
    if param == 'trigger':
        # PR 系は relink_trigger、repair 系は repair_trigger を振る
        key = 'relink_trigger' if 'pr' in method else 'repair_trigger'
        return {key: value}
    return {param: value}


def _tag(axis_name, value):
    sval = str(value).replace('.', 'p')
    return f'{axis_name}-{sval}'


def build_configs(axes_sel, methods_filter=None, exclude_tags=None):
    """全 (method, config) を dedup して構築。

    Returns: list of dict(method, kind, tag, axis, value, overrides)
      tag='center' は手法ごとに1個（全軸共有）。各軸の非 center 値が追加 config。

    methods_filter: 指定時、各軸の methods をこの集合との積に絞る（center も連動）。
    exclude_tags: 指定 tag（例 'pr_ls_top_k-5'）の config を生成しない。
    """
    methods_filter = set(methods_filter) if methods_filter else None
    exclude_tags = set(exclude_tags) if exclude_tags else set()
    seen = set()        # (method, tag)
    configs = []

    def _ax_methods(ax):
        ms = ax['methods']
        return [m for m in ms if m in methods_filter] if methods_filter else ms

    def _add(method, tag, axis, value, overrides):
        key = (method, tag)
        if key in seen:
            return
        seen.add(key)
        configs.append(dict(method=method, kind=METHOD_KIND[method], tag=tag,
                            axis=axis, value=value, overrides=overrides))

    # center（各軸が触る全手法ぶん）
    methods_used = set()
    for an in axes_sel:
        methods_used.update(_ax_methods(AXES[an]))
    for m in methods_used:
        _add(m, 'center', 'center', None, {})

    # 非 center 値
    for an in axes_sel:
        ax = AXES[an]
        for v in ax['values']:
            if v == ax['center']:
                continue  # center に dedup
            tag = _tag(an, v)
            if tag in exclude_tags:
                continue
            for m in _ax_methods(ax):
                _add(m, tag, an, v, _axis_override(an, ax['param'], v, m))
    return configs


def _out_path(out_dir, problem, scenario, method, tag, weight, trial):
    prob_label = f'{problem}_{scenario}'
    fn = f'{method}__{tag}__{_weight_label(weight)}__t{trial:03d}.json'
    return os.path.join(out_dir, prob_label, 'raw', fn)


# ========== 個別実行（並列・モジュールレベル必須） ==========

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
    _sys.path.insert(0, _os.path.join(_here, '..', 'core_comparison_v3'))
    from experiment_utils import run_ga, run_ils, run_memetic
    from run_v3 import _extract_uea_points as _ext, _slim_anytime as _slim

    method = task['method']
    kind = task['kind']
    p = dict(CENTER)
    p.update(task['overrides'])
    weights = task['weights']
    seed = task['seed']
    problem = task['problem']
    scenario = task['scenario']
    norm_params = task['norm_params']

    try:
        if kind == 'ga':
            r = run_ga(weights, seed, task['ga_ngen'], norm_params,
                       problem_name=problem, scenario_name=scenario,
                       track_population=True,
                       pop_size=p['pop_size'], cxpb=p['cxpb'], mutpb=p['mutpb'])
            ekind = 'ga'
        elif kind == 'memetic':
            kick_mode = {'memetic_ls': 'none', 'memetic_pr': 'pr',
                         'memetic_repair': 'repair'}[method]
            r = run_memetic(weights, seed, task['memetic_ngen'], norm_params,
                            problem_name=problem, scenario_name=scenario,
                            kick_mode=kick_mode, kick_prob=p['kick_prob'],
                            repair_strength=p['repair_strength'], track_population=True,
                            pr_step_strategy=p['pr_step_strategy'], pr_ls_top_k=p['pr_ls_top_k'],
                            pop_size=p['pop_size'], cxpb=p['cxpb'], mutpb=p['mutpb'])
            ekind = 'ga'  # pop_points 形式は GA と同じ
        else:  # ils
            path_relink_mode = (method == 'ils_pr')
            repair_mode = (method == 'ils_repair')
            r = run_ils(weights, seed, p['perturb'], task['ils_max_iter'], norm_params,
                        strategy='best',
                        repair_mode=repair_mode, repair_trigger=p['repair_trigger'],
                        repair_strength=p['repair_strength'],
                        path_relink_mode=path_relink_mode, relink_trigger=p['relink_trigger'],
                        kick_trigger_first=p['kick_trigger_first'],
                        pr_step_strategy=p['pr_step_strategy'], pr_ls_top_k=p['pr_ls_top_k'],
                        initial_strength=p['initial_strength'], max_strength=p['max_strength'],
                        problem_name=problem, scenario_name=scenario)
            ekind = 'ils'

        history = r['history']
        uea_points, uea_points_t, _d_hist = _ext(history, ekind)  # d_hist は sweep では未使用
        slim_history = _slim(history, ekind)

        save_data = {
            'method': method, 'tag': task['tag'], 'axis': task['axis'], 'value': task['value'],
            'overrides': task['overrides'],
            'problem': problem, 'scenario': scenario,
            'weights': weights, 'trial': task['trial'], 'seed': seed,
            'baseline': r.get('baseline'), 'baseline_rsr': r.get('baseline_rsr'),
            'baseline_score': r.get('baseline_score'),
            'finals': {'makespan': r['makespan'], 'stability': r['stability']},
            'convergence': r['convergence'],
            'history': slim_history,
            'uea_points': uea_points, 'uea_points_t': uea_points_t,
        }
        _os.makedirs(_os.path.dirname(out_path), exist_ok=True)
        with open(out_path, 'w', encoding='utf-8') as f:
            _json.dump(save_data, f, ensure_ascii=False)
        return {'status': 'done', 'path': out_path,
                'makespan': r['makespan'], 'stability': r['stability'],
                'cpu': r['convergence'].get('total_cpu_time', 0.0)}
    except Exception as e:
        return {'status': 'error', 'path': out_path, 'error': str(e),
                'traceback': _tb.format_exc()}


# ========== メイン ==========

def main():
    parser = argparse.ArgumentParser(description='param_sweep_v1: OAT パラメータ感度分析')
    parser.add_argument('--axes', nargs='+', default=list(AXES.keys()),
                        choices=list(AXES.keys()), help='掃引する軸（デフォルト全軸）')
    parser.add_argument('--methods', nargs='+', default=None,
                        choices=list(METHOD_KIND.keys()),
                        help='対象手法を絞る（各軸の methods との積。center も連動）')
    parser.add_argument('--exclude-tags', nargs='+', default=None,
                        help="除外する config tag（例: pr_ls_top_k-5）")
    parser.add_argument('--problems', nargs='+', default=None,
                        help='問題 (例: la21:la21_delay147)')
    parser.add_argument('--n-trials', type=int, default=DEFAULT_N_TRIALS)
    parser.add_argument('--n-jobs', type=int, default=DEFAULT_N_JOBS)
    parser.add_argument('--ils-max-iter', type=int, default=ILS_MAX_ITER)
    parser.add_argument('--ga-ngen', type=int, default=GA_NGEN)
    parser.add_argument('--memetic-ngen', type=int, default=MEMETIC_NGEN)
    parser.add_argument('--output-dir', type=str, default=None)
    args = parser.parse_args()

    if args.problems:
        problem_sets = [tuple(p.split(':')) for p in args.problems]
    else:
        problem_sets = DEFAULT_PROBLEM_SETS
    weights_list = DEFAULT_WEIGHTS

    out_dir = args.output_dir or os.path.join(
        _HERE, 'results', 'sweep_' + datetime.now().strftime('%Y%m%d_%H%M%S'))
    os.makedirs(out_dir, exist_ok=True)

    configs = build_configs(args.axes, methods_filter=args.methods,
                            exclude_tags=args.exclude_tags)
    n_plan = len(configs) * len(problem_sets) * len(weights_list) * args.n_trials
    print(f'出力先: {out_dir}')
    print(f'軸: {args.axes}  手法絞り: {args.methods or "なし"}  除外tag: {args.exclude_tags or "なし"}')
    print(f'問題: {[p[0] for p in problem_sets]}  重み: {len(weights_list)}点  試行: {args.n_trials}')
    print(f'ユニーク config: {len(configs)}（center 含む・dedup 済）')
    print(f'総 run 数(計画): {n_plan}  並列: {args.n_jobs}')

    # config.json（同一 output-dir への複数回実行に対応: axes の methods は union、
    # configs は (method, tag) で dedup マージ。analyze_sweep が軸→手法の対応に使う）
    cfg_path = os.path.join(out_dir, 'config.json')
    prev_cfg = {}
    if os.path.exists(cfg_path):
        with open(cfg_path, encoding='utf-8') as f:
            prev_cfg = json.load(f)
    axes_out = prev_cfg.get('axes', {})
    for a in args.axes:
        ax = dict(AXES[a])
        if args.methods:
            ax['methods'] = [m for m in ax['methods'] if m in set(args.methods)]
        if a in axes_out:
            ax['methods'] = sorted(set(axes_out[a].get('methods', [])) | set(ax['methods']))
        axes_out[a] = ax
    cfg_merged = {(c['method'], c['tag']): c for c in prev_cfg.get('configs', [])}
    for c in configs:
        cfg_merged[(c['method'], c['tag'])] = c
    with open(cfg_path, 'w', encoding='utf-8') as f:
        json.dump({'center': CENTER, 'axes': axes_out,
                   'configs': list(cfg_merged.values()),
                   'problems': [list(p) for p in problem_sets],
                   'weights': weights_list, 'n_trials': args.n_trials,
                   'ils_max_iter': args.ils_max_iter, 'ga_ngen': args.ga_ngen,
                   'memetic_ngen': args.memetic_ngen}, f, ensure_ascii=False, indent=2)

    # norm_params（問題ごと、既存なら再利用）
    np_path = os.path.join(out_dir, 'norm_params.json')
    np_cache = {}
    existing = {}
    if os.path.exists(np_path):
        with open(np_path, encoding='utf-8') as f:
            existing = json.load(f)
    for pn, sn in problem_sets:
        k = f'{pn}/{sn}'
        if k in existing:
            np_cache[(pn, sn)] = existing[k]
            print(f'[norm_params] {k}: 既存再利用')
        else:
            print(f'[norm_params] {k}: 計算')
            np_cache[(pn, sn)] = compute_shared_norm_params(pn, sn)
            existing[k] = np_cache[(pn, sn)]
    with open(np_path, 'w', encoding='utf-8') as f:
        json.dump(existing, f, ensure_ascii=False, indent=2)

    # タスク展開
    all_tasks = []
    for pn, sn in problem_sets:
        npr = np_cache[(pn, sn)]
        for cfg in configs:
            for w in weights_list:
                for t in range(args.n_trials):
                    all_tasks.append(dict(
                        method=cfg['method'], kind=cfg['kind'], tag=cfg['tag'],
                        axis=cfg['axis'], value=cfg['value'], overrides=cfg['overrides'],
                        problem=pn, scenario=sn, weights=w, trial=t, seed=t * 100 + 7,
                        norm_params=npr,
                        out_path=_out_path(out_dir, pn, sn, cfg['method'], cfg['tag'], w, t),
                        ils_max_iter=args.ils_max_iter, ga_ngen=args.ga_ngen,
                        memetic_ngen=args.memetic_ngen))

    pending = [t for t in all_tasks if not os.path.exists(t['out_path'])]
    print(f'\n総タスク: {len(all_tasks)}  スキップ(済): {len(all_tasks)-len(pending)}  実行: {len(pending)}')
    if not pending:
        print('全タスク完了済み。分析は analyze_sweep.py を実行。')
        return

    done = err = 0
    with ProcessPoolExecutor(max_workers=args.n_jobs) as ex:
        futs = {ex.submit(_run_one_task, t): t for t in pending}
        for fut in as_completed(futs):
            t = futs[fut]
            label = f"{t['problem']}/{t['method']}/{t['tag']}/{_weight_label(t['weights'])}/t{t['trial']:03d}"
            try:
                r = fut.result()
                if r['status'] == 'done':
                    done += 1
                    print(f"  [{done+len(all_tasks)-len(pending)}/{len(all_tasks)}] {label} "
                          f"MS={r.get('makespan','?')} Stab={r.get('stability',0):.2f} CPU={r.get('cpu',0):.1f}s")
                elif r['status'] == 'error':
                    err += 1
                    print(f"  [ERROR] {label}: {r.get('error')}")
                    for ln in (r.get('traceback') or '').splitlines()[-4:]:
                        print(f'    {ln}')
            except Exception as e:
                err += 1
                print(f'  [FATAL] {label}: {e}')

    print(f'\n実行完了: done={done} error={err}\n結果: {out_dir}')
    print(f'分析: python analyze_sweep.py --input-dir {out_dir}')


if __name__ == '__main__':
    main()
