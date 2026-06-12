#!/usr/bin/env python3
"""
Stage 1 / Stage 2-A: ILS パラメータ掃引

=== 位置づけ ===
doc/ils_parameter_sweep.md の Stage 1 / Stage 2 を実行する。

  Stage 1: ILS 本体（perturb / strategy / initial_strength / strength_delta）
  Stage 2-A: 拡張機構の repair（trigger × strength）

=== Stage 構成 ===
  Stage 1-A: perturb_method × strategy の 2D グリッド (主因子)
             (2 × 2 = 4 cells, trial=10)
             initial_strength=2, strength_delta=3 で固定
  Stage 1-B: OFAT (Stage 1-A 確定設定を base に initial_strength と
             strength_delta を 1 軸ずつ振る)
             (baseline + 差分のみ, trial=5)
  Stage 2-A: repair_trigger × repair_strength × ILS variant
             (4 × 4 grid + 1 baseline) × 2 variants = 34 configs, trial=5
             Stage 1 確定 ILS variants (swap+best, insert+best) を base にする

=== 使い方 ===
  # Stage 1
  python run_ils_sweep.py --stage 1a
  python run_ils_sweep.py --stage 1b

  # Stage 2-A (la21 除外、la36/mt10 のみ)
  python run_ils_sweep.py --stage 2a

  # variant 限定 (例: ILS-swap だけ)
  python run_ils_sweep.py --stage 2a --variant swap

  # 問題・重みの上書き
  python run_ils_sweep.py --stage 2a \
      --problems la36:la36_delay148 la36:la36_multi3_x15 \
      --weights 0.8,0.2 --trials 10

=== 出力 ===
  results/<stage>_<timestamp>/
  ├── config.json                       # 掃引設定
  ├── <problem>_<scenario>/
  │   ├── results.json                  # 全 config × 全 trial の履歴・最終値
  │   └── summary.txt                   # 数値サマリ
  └── cross_summary.txt
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
    get_initial_makespan, run_ils, ILS_MAX_ITER,
)


# ========== デフォルト設定 ==========

# la40 系は ILS が単一最適解に収束する saturation 問題があり対象外。
# 詳細は doc/ils_parameter_sweep.md §5.2 参照。
DEFAULT_PROBLEM_SETS = [
    ('mt10', 'mt10_delay60'),
    ('la21', 'la21_delay147'),
    ('la36', 'la36_delay148'),
    ('la36', 'la36_multi3_x15'),
]
DEFAULT_WEIGHTS = [0.8, 0.2]

# Stage 1-A: 2D グリッド (主因子: perturb × strategy)
# initial_strength と strength_delta は Stage 1-A では固定し、Stage 1-B で OFAT する。
STAGE_1A_PERTURBS = ['swap', 'insert']
STAGE_1A_STRATEGIES = ['best', 'first']
STAGE_1A_INITIAL_STRENGTH = 2  # Stage 1-A 中の固定値
STAGE_1A_STRENGTH_DELTA = 3    # Stage 1-A 中の固定値 (max_strength = initial + delta)
STAGE_1A_TRIALS = 10

# Stage 1-B: OFAT (initial_strength と strength_delta を基準点から 1 軸ずつ振る)
# 基準点 (perturb, strategy) は Stage 1-A 結果に基づき --base CLI で上書き可能。
# デフォルトは insert + best。
STAGE_1B_BASE = {
    'perturb': 'insert',
    'strategy': 'best',
    'initial_strength': 2,
    'strength_delta': 3,
}
# 各軸で振る水準。基準値と重複する水準は自動スキップ。
STAGE_1B_AXES = {
    'initial_strength': [1, 2, 3, 4],
    'strength_delta':   [1, 3, 6],
}
STAGE_1B_TRIALS = 5

# Stage 2-A: repair_trigger × repair_strength × ILS variant
# Stage 1 確定の ILS-swap (swap+best) と ILS-insert (insert+best) で
# repair の効果を比較。各 variant に baseline（repair_mode=False）も含む。
# la21 は Stage 1 で saturated だったため Stage 2-A の問題セットからは除外。
STAGE_2A_TRIGGERS = [10, 30, 50, 100]
STAGE_2A_STRENGTHS = [1, 2, 3, 4]
STAGE_2A_VARIANTS = [
    {'perturb': 'swap',   'strategy': 'best'},
    {'perturb': 'insert', 'strategy': 'best'},
]
STAGE_2A_INITIAL_STRENGTH = 2  # Stage 1 確定値
STAGE_2A_STRENGTH_DELTA = 3    # Stage 1 確定値
STAGE_2A_TRIALS = 5
STAGE_2A_DEFAULT_PROBLEM_SETS = [
    ('mt10', 'mt10_delay60'),
    ('la36', 'la36_delay148'),
    ('la36', 'la36_multi3_x15'),
]


# ========== 設定列挙 ==========

def stage_1a_configs():
    """Stage 1-A: perturb × strategy の全セル (initial_strength, delta 固定)"""
    configs = []
    init_s = STAGE_1A_INITIAL_STRENGTH
    delta = STAGE_1A_STRENGTH_DELTA
    for perturb in STAGE_1A_PERTURBS:
        for strategy in STAGE_1A_STRATEGIES:
            configs.append({
                'config_id': f'{perturb}_{strategy}',
                'perturb': perturb,
                'strategy': strategy,
                'initial_strength': init_s,
                'max_strength': init_s + delta,
                'axis': 'grid',  # 分析用ラベル
            })
    return configs


def stage_1b_configs(base_override=None):
    """Stage 1-B: 基準点 + 各軸 1 ステップずらし。
    strength_delta 軸は max_strength = initial + delta として展開。

    base_override: dict で perturb/strategy 等を上書き（CLI から渡される）。
    """
    def _expand(cfg):
        """strength_delta → max_strength 展開"""
        out = dict(cfg)
        out['max_strength'] = out['initial_strength'] + out.pop('strength_delta')
        return out

    base_dict = dict(STAGE_1B_BASE)
    if base_override:
        base_dict.update(base_override)

    base = dict(base_dict)
    base['config_id'] = 'base'
    base['axis'] = 'baseline'
    configs = [_expand(base)]

    for axis, values in STAGE_1B_AXES.items():
        for v in values:
            if v == base_dict[axis]:
                continue  # 基準点と重複
            cfg = dict(base_dict)
            cfg[axis] = v
            cfg['config_id'] = f'{axis}={v}'
            cfg['axis'] = axis
            configs.append(_expand(cfg))
    return configs


def stage_2a_configs(variants_filter=None):
    """Stage 2-A: ILS variant × repair_trigger × repair_strength グリッド + baseline

    各 variant ごとに:
      - 1 baseline (repair_mode=False)
      - 4 × 4 = 16 grid cells (trigger × strength, repair_mode=True)
    合計 17 configs / variant、デフォルトの 2 variants で 34 configs。

    variants_filter: list of perturb names to include (例: ['swap'] で ILS-swap のみ)
    """
    init_s = STAGE_2A_INITIAL_STRENGTH
    max_s  = init_s + STAGE_2A_STRENGTH_DELTA
    configs = []
    for v in STAGE_2A_VARIANTS:
        if variants_filter and v['perturb'] not in variants_filter:
            continue
        # baseline (no repair) — trigger/strength は記録のみで実体に影響なし
        configs.append({
            'config_id': f"{v['perturb']}_baseline",
            'perturb': v['perturb'],
            'strategy': v['strategy'],
            'initial_strength': init_s,
            'max_strength': max_s,
            'repair_mode': False,
            'repair_trigger': 30,
            'repair_strength': 2,
            'axis': 'baseline',
            'variant': v['perturb'],
        })
        # repair grid
        for trig in STAGE_2A_TRIGGERS:
            for s in STAGE_2A_STRENGTHS:
                configs.append({
                    'config_id': f"{v['perturb']}_t{trig}_s{s}",
                    'perturb': v['perturb'],
                    'strategy': v['strategy'],
                    'initial_strength': init_s,
                    'max_strength': max_s,
                    'repair_mode': True,
                    'repair_trigger': trig,
                    'repair_strength': s,
                    'axis': 'grid',
                    'variant': v['perturb'],
                })
    return configs


# ========== 個別実行（並列ワーカー） ==========

def _run_config(cfg, weights, seed, norm_params, problem_name, scenario_name,
                max_iter):
    return run_ils(
        weights, seed, cfg['perturb'], max_iter, norm_params,
        strategy=cfg.get('strategy', 'best'),
        initial_strength=cfg.get('initial_strength', 2),
        max_strength=cfg.get('max_strength', 5),
        repair_mode=cfg.get('repair_mode', False),
        repair_trigger=cfg.get('repair_trigger', 30),
        repair_strength=cfg.get('repair_strength', 2),
        path_relink_mode=False,
        problem_name=problem_name, scenario_name=scenario_name,
    )


# ========== 履歴スリム化（JSON 保存用） ==========

def _slim_history(history):
    """分析に必要な項目だけ残す。strength 追加で少し太くなったので整理。"""
    out = []
    for h in history:
        out.append({
            'iteration':      h['iteration'],
            'cpu_time':       h.get('cpu_time'),
            'evaluations':    h.get('evaluations'),
            'best_makespan':  h.get('best_makespan'),
            'best_stability': h.get('best_stability'),
            'best_score':     h.get('best_score'),
            'ls_makespan':    h.get('ls_makespan'),
            'ls_stability':   h.get('ls_stability'),
            'ls_score':       h.get('ls_score'),
            'accepted':       h.get('accepted'),
            'perturb_used':   h.get('perturb_used'),
            'strength':       h.get('strength'),
        })
    return out


# ========== 1 Stage × 1 問題 の実行 ==========

def run_problem(stage, configs, problem_name, scenario_name, weights, n_trials,
                max_iter, out_dir):
    prob_label = f'{problem_name}_{scenario_name}'
    print(f"\n{'='*70}")
    print(f"Stage {stage} / {prob_label} / weights={weights}")
    print(f"  configs={len(configs)}, trials={n_trials}, max_iter={max_iter}")
    print(f"{'='*70}")

    norm_params = compute_shared_norm_params(problem_name, scenario_name)
    init_ms = get_initial_makespan(problem_name, scenario_name)
    print(f"  init_makespan={init_ms}")

    # 並列投入
    futures = {}
    results = {cfg['config_id']: [None] * n_trials for cfg in configs}
    baseline = None

    with ProcessPoolExecutor() as executor:
        for trial in range(n_trials):
            seed = trial * 100 + 7
            for cfg in configs:
                f = executor.submit(
                    _run_config, cfg, weights, seed, norm_params,
                    problem_name, scenario_name, max_iter)
                futures[f] = (cfg['config_id'], trial, seed)

        for future in as_completed(futures):
            cid, trial, seed = futures[future]
            try:
                r = future.result()
                results[cid][trial] = {
                    'trial':       trial,
                    'seed':        seed,
                    'makespan':    r['makespan'],
                    'stability':   r['stability'],
                    'convergence': r['convergence'],
                    'history':     _slim_history(r['history']),
                }
                if baseline is None and r.get('baseline') is not None:
                    baseline = r['baseline']
                print(f"  [{cid:20s}] trial={trial:2d}: "
                      f"MS={r['makespan']}, St={r['stability']:.2f}, "
                      f"CPU={r['convergence']['total_cpu_time']:.2f}s")
            except Exception as e:
                import traceback
                print(f"  [{cid:20s}] trial={trial:2d}: ERROR - {e}")
                traceback.print_exc()
                results[cid][trial] = {'trial': trial, 'seed': seed,
                                        'error': str(e)}

    # 保存
    prob_dir = os.path.join(out_dir, prob_label)
    os.makedirs(prob_dir, exist_ok=True)

    save_data = {
        'stage':          stage,
        'problem':        problem_name,
        'scenario':       scenario_name,
        'weights':        weights,
        'init_makespan':  init_ms,
        'baseline':       baseline,
        'n_trials':       n_trials,
        'max_iterations': max_iter,
        'configs':        {cfg['config_id']: cfg for cfg in configs},
        'results':        results,
    }
    save_path = os.path.join(prob_dir, 'results.json')
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(save_data, f, ensure_ascii=False)
    print(f"  → 保存: {save_path}")

    # テキストサマリ
    summary_lines = _make_summary(save_data)
    summary_path = os.path.join(prob_dir, 'summary.txt')
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(summary_lines) + "\n")
    print(f"  → サマリ: {summary_path}")

    return summary_lines


def _make_summary(data):
    lines = [f"Stage {data['stage']} / {data['problem']}_{data['scenario']}",
             f"  weights={data['weights']}, init_ms={data['init_makespan']}",
             f"  trials={data['n_trials']}, max_iter={data['max_iterations']}",
             ""]
    header = (f"  {'config':<22} {'MS_mean':>10} {'MS_best':>8} "
              f"{'MS_std':>8} {'St_mean':>10} {'St_std':>8} "
              f"{'Score_mean':>12} {'CPU_mean':>10}")
    lines.append(header)
    lines.append("  " + "-" * (len(header) - 2))

    for cid, cfg in data['configs'].items():
        valid = [d for d in data['results'][cid]
                 if d is not None and 'error' not in d]
        if not valid:
            lines.append(f"  {cid:<22} {'no valid trials':>60}")
            continue
        ms = [d['makespan'] for d in valid]
        st = [d['stability'] for d in valid]
        sc = [d['history'][-1]['best_score'] for d in valid]
        cpu = [d['convergence']['total_cpu_time'] for d in valid]
        lines.append(
            f"  {cid:<22} {np.mean(ms):>10.1f} {int(min(ms)):>8d} "
            f"{np.std(ms):>8.2f} {np.mean(st):>10.2f} {np.std(st):>8.2f} "
            f"{np.mean(sc):>12.4f} {np.mean(cpu):>10.2f}"
        )
    return lines


# ========== エントリポイント ==========

def run_stage(stage, args, out_dir):
    if stage == '1a':
        configs = stage_1a_configs()
        n_trials = args.trials if args.trials else STAGE_1A_TRIALS
    elif stage == '1b':
        # --base CLI で base を上書き可能 (例: "swap,first" / "insert,best")
        base_override = None
        if args.base:
            parts = [p.strip() for p in args.base.split(',')]
            if len(parts) != 2:
                raise ValueError(f"--base は 'perturb,strategy' 形式: {args.base}")
            perturb, strategy = parts
            if perturb not in ('swap', 'insert'):
                raise ValueError(f"perturb は swap/insert: {perturb}")
            if strategy not in ('best', 'first'):
                raise ValueError(f"strategy は best/first: {strategy}")
            base_override = {'perturb': perturb, 'strategy': strategy}
        configs = stage_1b_configs(base_override)
        n_trials = args.trials if args.trials else STAGE_1B_TRIALS
    elif stage == '2a':
        # --variant で perturb を絞り込み可能 (swap / insert / both)
        variants_filter = None
        if args.variant and args.variant != 'both':
            if args.variant not in ('swap', 'insert'):
                raise ValueError(f"--variant は swap/insert/both: {args.variant}")
            variants_filter = [args.variant]
        configs = stage_2a_configs(variants_filter)
        n_trials = args.trials if args.trials else STAGE_2A_TRIALS
    else:
        raise ValueError(f"Unknown stage: {stage}")

    # config.json
    config_meta = {
        'stage':          stage,
        'problems':       [list(p) for p in args.problems],
        'weights':        args.weights,
        'max_iterations': args.max_iter,
        'n_trials':       n_trials,
        'configs':        configs,
    }
    with open(os.path.join(out_dir, 'config.json'), 'w', encoding='utf-8') as f:
        json.dump(config_meta, f, ensure_ascii=False, indent=2)

    all_summaries = []
    for problem_name, scenario_name in args.problems:
        try:
            s = run_problem(stage, configs, problem_name, scenario_name,
                            args.weights, n_trials, args.max_iter, out_dir)
            all_summaries.append(s)
        except Exception as e:
            import traceback
            print(f"\nERROR: {problem_name}/{scenario_name}: {e}")
            traceback.print_exc()
            all_summaries.append([f"ERROR: {problem_name}/{scenario_name}: {e}"])

    cross_path = os.path.join(out_dir, 'cross_summary.txt')
    with open(cross_path, 'w', encoding='utf-8') as f:
        f.write(f"Stage {stage} 横断サマリー\n")
        f.write("=" * 70 + "\n\n")
        for s in all_summaries:
            for line in s:
                f.write(str(line) + "\n")
            f.write("\n")
    print(f"\n横断サマリ: {cross_path}")


def main():
    parser = argparse.ArgumentParser(description="ILS パラメータ掃引 (Stage 1 / Stage 2-A)")
    parser.add_argument('--stage', choices=['1a', '1b', '2a', 'all'], default='1a',
                        help='実行する Stage（all は 1a, 1b のみ）')
    parser.add_argument('--problems', nargs='+', type=str, default=None,
                        help='問題セット (例: la36:la36_delay148)。省略時は Stage 別 default')
    parser.add_argument('--weights', type=str, default='0.8,0.2',
                        help='重み (例: 0.8,0.2)')
    parser.add_argument('--trials', type=int, default=None,
                        help='試行回数 (default: 1a=10, 1b=5, 2a=5)')
    parser.add_argument('--max-iter', type=int, default=ILS_MAX_ITER,
                        help=f'ILS 最大反復数 (default: {ILS_MAX_ITER})')
    parser.add_argument('--base', type=str, default=None,
                        help='Stage 1-B の base を "perturb,strategy" で上書き '
                             '(例: "swap,first")。省略時は STAGE_1B_BASE を使用')
    parser.add_argument('--variant', type=str, default='both',
                        choices=['swap', 'insert', 'both'],
                        help='Stage 2-A の ILS variant 絞り込み (default: both)')
    parser.add_argument('--out-prefix', type=str, default=None,
                        help='出力ディレクトリ prefix')
    parser.add_argument('--analyze', action='store_true',
                        help='実行後に analyze_ils_sweep.py を自動実行')
    args = parser.parse_args()

    # 引数加工: 問題セットは Stage 別の default あり
    if args.problems:
        args.problems = [tuple(p.split(':')) for p in args.problems]
    elif args.stage == '2a':
        args.problems = STAGE_2A_DEFAULT_PROBLEM_SETS
    else:
        args.problems = DEFAULT_PROBLEM_SETS
    args.weights = [float(x) for x in args.weights.split(',')]

    stages = ['1a', '1b'] if args.stage == 'all' else [args.stage]
    created_dirs = []
    for stage in stages:
        prefix = args.out_prefix or f'stage{stage}'
        out_dir = setup_output_dir(prefix, base_dir=os.path.dirname(__file__))
        print(f"\n出力先: {out_dir}")
        print(f"問題: {args.problems}")
        print(f"重み: {args.weights}")
        run_stage(stage, args, out_dir)
        created_dirs.append(out_dir)

    if args.analyze:
        print(f"\n{'='*70}\n--analyze 指定されたので分析を実行します\n{'='*70}")
        import analyze_ils_sweep
        for d in created_dirs:
            sys.argv = ['analyze_ils_sweep', d]
            analyze_ils_sweep.main()


if __name__ == "__main__":
    main()
