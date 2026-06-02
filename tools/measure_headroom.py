"""
headroom 診断ツール

「headroom」= RSR（右シフトのみの修復）の makespan − 最適化後の makespan。
すなわち「順序を並べ替えることで右シフトよりどれだけ makespan を縮められるか」
＝ アルゴリズム（ILS / Memetic / PR）が差を出せる余地の大きさ。

  headroom = 0  : 右シフト（元順序のまま）が既に makespan 最適。並べ替える価値なし。
                  → どの手法も同じ解に収束し区別不能（縮退シナリオ）。
  headroom 大   : 並べ替えに価値があり、良い手法ほど多く縮める → 手法間の差が出る。

シナリオ作成時に「そのシナリオがそもそもアルゴリズムを区別できるか」を事前診断するために使う。
詳細な背景は doc/problem_setup_design.md を参照。

=== 使い方 ===
  # 既存シナリオ1個の headroom を測る
  python tools/measure_headroom.py --scenario la36_delay148

  # あるシナリオの initial_gantt を使い、全クリティカルopを遅延ターゲットに
  # 振って headroom の分布を見る（= 非縮退ターゲットの探索）
  python tools/measure_headroom.py --scan la36_delay148 --ils-iters 1500

  # 重み・反復回数・遅延倍率の指定
  python tools/measure_headroom.py --scan la21_delay147 --delay-ratio 1.0 \
      --ils-iters 2000 --weights 1,0
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.dirname(__file__))

import job_shop_scheduling
import gantt_chart_operation
import evaluation
from ils_scheduling import ILSSolver

from generate_scenario import find_critical_path_tasks, inject_delay


def measure_headroom(jm_table, init_gantt, delayed_gantt, ils_iters=1500,
                     weights=(1.0, 0.0)):
    """1つの (initial_gantt, delayed_gantt) ペアの headroom を測る。

    Returns:
        dict or None（外乱がリスケを誘発しない場合は None）
        {init_ms, rsr_ms, ils_ms, headroom, n_reschedule, n_fixed, reschedule_time}
    """
    init_ms = evaluation.compute_makespan_from_gantt(init_gantt)
    fixed, resched, rt, _msg = gantt_chart_operation.check_disturbance(
        init_gantt, delayed_gantt)
    n_res = sum(len(m) for m in resched)
    if n_res == 0:
        return None

    rsr, _ = gantt_chart_operation.create_rsr_gantt(fixed, resched)
    rsr_ms = evaluation.compute_makespan_from_gantt(rsr)

    solver = ILSSolver(jm_table, fixed, resched, rt, list(weights))
    best, _score, _conv, _hist = solver.run(
        max_iterations=ils_iters, verbose=False)
    ils_ms, _ils_st = solver.evaluate_pareto(best)

    return {
        'init_ms': init_ms,
        'rsr_ms': rsr_ms,
        'ils_ms': ils_ms,
        'headroom': rsr_ms - ils_ms,
        'n_reschedule': n_res,
        'n_fixed': sum(len(m) for m in fixed),
        'reschedule_time': rt,
    }


def load_scenario(name):
    """scenarios/{name}.json を読み込む。"""
    path = os.path.join('scenarios', f'{name}.json')
    with open(path, encoding='utf-8') as f:
        d = json.load(f)
    return d


def cmd_scenario(name, ils_iters, weights):
    """既存シナリオの headroom を測る。"""
    d = load_scenario(name)
    jm = job_shop_scheduling.get_jm_table(d['problem'], name)
    r = measure_headroom(jm, d['initial_gantt'], d['delayed_gantt'],
                         ils_iters=ils_iters, weights=weights)
    print(f"=== {name}  ({d['problem']}) ===")
    if r is None:
        print("  外乱がリスケを誘発しません（headroom 測定不可）")
        return
    print(f"  init_MS        = {r['init_ms']}")
    print(f"  RSR_MS         = {r['rsr_ms']}  (右シフトのみ)")
    print(f"  ILS_MS         = {r['ils_ms']}  (並べ替え最適化, iters={ils_iters})")
    print(f"  headroom       = {r['headroom']}   <<< これがアルゴリズム差の土俵")
    print(f"  reschedule_ops = {r['n_reschedule']}  (fixed={r['n_fixed']}, "
          f"reschedule_time={r['reschedule_time']})")
    if r['headroom'] == 0:
        print("  ⚠ headroom=0 → 縮退シナリオ。どの手法も区別不能。ターゲット選び直し推奨。")


def cmd_scan(name, delay_ratio, ils_iters, weights):
    """シナリオの initial_gantt を使い、全クリティカルopを遅延ターゲットに
    振って headroom 分布を出す。"""
    d = load_scenario(name)
    jm = job_shop_scheduling.get_jm_table(d['problem'], name)
    init = d['initial_gantt']
    init_ms = evaluation.compute_makespan_from_gantt(init)
    crit = find_critical_path_tasks(init, jm)
    print(f"=== scan: {name} ({d['problem']})  init_MS={init_ms}  "
          f"critical_tasks={len(crit)}  delay_ratio={delay_ratio} ===")

    rows = []
    for (m_idx, t_idx, st, et, job, pt) in crit:
        delay = int(pt * delay_ratio)
        if delay <= 0:
            continue
        delayed = inject_delay(init, m_idx, t_idx, delay)
        r = measure_headroom(jm, init, delayed, ils_iters=ils_iters,
                             weights=weights)
        if r is None:
            continue
        rows.append((r['headroom'], m_idx, job, st, pt,
                     r['n_reschedule'], r['rsr_ms'], r['ils_ms']))

    rows.sort(reverse=True)
    print(f"{'head':>5} {'M':>2} {'job':>3} {'start':>5} {'pt':>3} "
          f"{'nres':>4} {'RSR':>5} {'ILS':>5}")
    for hr, m_idx, job, st, pt, nres, rsr_ms, ils_ms in rows:
        print(f"{hr:>5} {m_idx:>2} {job:>3} {st:>5} {pt:>3} "
              f"{nres:>4} {rsr_ms:>5} {ils_ms:>5}")

    if rows:
        hs = sorted(r[0] for r in rows)
        print(f"\nheadroom: min={hs[0]} max={hs[-1]} median={hs[len(hs)//2]}")
        print("注意: 早い外乱(start小)は reschedule_ops が大きく headroom も大きいが、"
              "S_p が準最適だとその分が水増しされる。S_p を near-optimal にして再測定推奨。")


def main():
    parser = argparse.ArgumentParser(description='headroom 診断ツール')
    g = parser.add_mutually_exclusive_group(required=True)
    g.add_argument('--scenario', help='既存シナリオ1個の headroom を測る')
    g.add_argument('--scan', help='シナリオの initial_gantt で全クリティカルopを'
                                   'ターゲットに振り headroom 分布を出す')
    parser.add_argument('--delay-ratio', type=float, default=1.0,
                        help='scan時の遅延量 = PT × この倍率 (default: 1.0)')
    parser.add_argument('--ils-iters', type=int, default=1500,
                        help='ILS反復回数 (default: 1500)')
    parser.add_argument('--weights', default='1,0',
                        help='[効率,安定] の重み (default: "1,0" = makespanのみ)')
    args = parser.parse_args()

    weights = tuple(float(x) for x in args.weights.split(','))

    if args.scenario:
        cmd_scenario(args.scenario, args.ils_iters, weights)
    else:
        cmd_scan(args.scan, args.delay_ratio, args.ils_iters, weights)


if __name__ == '__main__':
    main()
