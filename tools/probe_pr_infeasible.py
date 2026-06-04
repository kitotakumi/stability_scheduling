#!/usr/bin/env python3
"""PR の経路打ち切り頻度プローブ

memetic の PR が経路途中で `end_all_infeasible`（全 direct-swap が実行不可で
打ち切り）になる頻度と、その打ち切り地点（S_p からどれだけ手前か）を計測する。
本番コードは変更せず path_relinking(trace=True) の trace_log を集計するだけ。

代表的な S_cur（memetic の局所最適個体の代理）は、初期解 S_p を insert 摂動で
散らして N5 local_search で局所最適化したものを使う。

判断材料:
  - end_all_infeasible の発火率が低く、打ち切り地点が S_p 近傍なら現状維持で十分。
  - 頻発 or 中盤で打ち切るなら、2手先読み / 挿入move などの導入価値あり。

使い方:
  python tools/probe_pr_infeasible.py --problem la36 --scenario la36_large --n 50
"""

import os
import sys
import json
import random
import argparse
import statistics as st

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, '..'))

import job_shop_scheduling
import gantt_chart_operation
import ils_scheduling

WEIGHTS = [0.8, 0.2]
STRENGTH_RANGE = (3, 20)
NORM_JSON = os.path.join(
    _HERE, '..', 'experiments', 'kick_rtb_ablation', 'results',
    'seqdev_pilot_la36_w08_02', 'norm_params.json')


def _med(xs):
    return st.median(xs) if xs else float('nan')


def main(problem, scenario, n_trials):
    jm_table = job_shop_scheduling.get_jm_table(problem, scenario)
    init_gantt = jm_table.initial_gantt()
    delayed_gantt = jm_table.delayed_gantt()
    fixed_gantt, reschedule_gantt, reschedule_time, msg = \
        gantt_chart_operation.check_disturbance(init_gantt, delayed_gantt)

    solver = ils_scheduling.ILSSolver(
        jm_table, fixed_gantt, reschedule_gantt, reschedule_time, WEIGHTS,
        active_schedule=False, taillard_acceleration=True)

    norm = None
    if os.path.exists(NORM_JSON):
        with open(NORM_JSON, 'r', encoding='utf-8') as f:
            raw = json.load(f)
        norm = raw.get(f'{problem}/{scenario}')
    solver.estimate_normalization_params(n_samples=100, norm_params=norm)

    S_p = solver.initial_machine_orders
    print(f"問題: {problem}/{scenario}  N={n_trials}  reschedule_time={reschedule_time}")
    print(f"S_cur 生成: insert 摂動 strength={STRENGTH_RANGE} → N5 local_search\n")

    stats = {s: {
        'n': 0, 'infeasible_end': 0,
        'init_diffs': [], 'total_steps': [],
        'inf_remaining': [], 'inf_remaining_frac': [],
        'reached_frac': [],
    } for s in ('best', 'first')}

    for trial in range(n_trials):
        random.seed(1000 + trial)
        strength = random.randint(*STRENGTH_RANGE)
        s_cur = solver.perturb(S_p, 'insert', strength)
        s_cur, _, _ = solver.local_search(s_cur, strategy='best')

        for strat in ('best', 'first'):
            _, _, tlog = solver.path_relinking(
                s_cur, S_p, trace=True, step_strategy=strat)
            types = [e['type'] for e in tlog]
            result = next(e for e in tlog if e['type'] == 'result')
            init_d = result['initial_diffs']
            steps = result['total_steps']
            d = stats[strat]
            d['n'] += 1
            d['init_diffs'].append(init_d)
            d['total_steps'].append(steps)
            d['reached_frac'].append(steps / init_d if init_d else 1.0)
            if 'end_all_infeasible' in types:
                d['infeasible_end'] += 1
                ent = next(e for e in tlog if e['type'] == 'end_all_infeasible')
                rem = ent.get('diffs_to_ref', init_d - steps)
                d['inf_remaining'].append(rem)
                d['inf_remaining_frac'].append(rem / init_d if init_d else 0.0)

        if (trial + 1) % 10 == 0:
            print(f"  ...{trial + 1}/{n_trials}")

    print(f"\n{'='*70}")
    for strat in ('best', 'first'):
        d = stats[strat]
        rate = d['infeasible_end'] / d['n'] * 100 if d['n'] else 0
        print(f"\n[step_strategy={strat}]  PR実行={d['n']}")
        print(f"  initial_diffs  : median={_med(d['init_diffs']):.1f}  "
              f"min={min(d['init_diffs'])} max={max(d['init_diffs'])}")
        print(f"  total_steps    : median={_med(d['total_steps']):.1f}")
        print(f"  到達率(steps/init_diffs) median={_med(d['reached_frac']):.2%}")
        print(f"  >> end_all_infeasible 発火率: {d['infeasible_end']}/{d['n']} ({rate:.1f}%)")
        if d['inf_remaining']:
            print(f"     打ち切り時の残りdiffs   : median={_med(d['inf_remaining']):.1f}")
            print(f"     残りdiffs割合(S_p手前)  : median={_med(d['inf_remaining_frac']):.1%} "
                  f"(大きいほど早く打ち切り=痛い)")
        else:
            print(f"     （infeasible 打ち切りなし → 全件 S_p まで到達）")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--problem', default='la36')
    ap.add_argument('--scenario', default='la36_large')
    ap.add_argument('--n', type=int, default=50)
    args = ap.parse_args()
    main(args.problem, args.scenario, args.n)
