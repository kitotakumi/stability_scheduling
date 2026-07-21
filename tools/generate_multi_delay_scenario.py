"""
複数遅延シナリオ生成

既存の単一遅延シナリオから initial_gantt を再利用し、
クリティカルパス上の N 箇所に PT × ratio の遅延を注入した新シナリオを作成する。

=== 選定アルゴリズム ===
  1. 元シナリオの initial_gantt のクリティカルパスを抽出
  2. 開始時刻を MS の position_range で絞る (デフォルト: [0.15, 0.85])
  3. 範囲を N 等分し、各 bin で最大 PT のクリティカルタスクを 1 個選ぶ
  4. 空 bin が出た場合は未選択の最大 PT 候補にフォールバック (実用上 la40 では発生しない)

これにより:
  - 時間軸上に均等に分布する N 個の遅延箇所
  - 各箇所はクリティカルパス上 (= 遅延が必ず影響を与える)
  - PT が大きいタスクを優先 (= 影響度が大きい)
  - 決定論的 (= seed フリーで再現可能)

=== 使い方 ===
  python tools/generate_multi_delay_scenario.py \
      --source-scenario la40_delay148 \
      --num-delays 3 --delay-ratio 1.5 \
      --scenario-name la40_multi3_x15

  python tools/generate_multi_delay_scenario.py \
      --source-scenario la40_delay148 \
      --num-delays 5 --delay-ratio 1.5 \
      --scenario-name la40_multi5_x15
"""

import argparse
import copy
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.dirname(__file__))

import job_shop_scheduling
import gantt_chart_operation
import evaluation

from generate_scenario import find_critical_path_tasks, inject_delay


def select_multi_delay_targets(gantt, jm_table, num_delays,
                                position_range=(0.15, 0.85)):
    """時間軸 N 等分で各 bin の最大 PT クリティカルタスクを選定する。

    Returns:
        (selected_targets, makespan)
        selected_targets: list of (m_idx, t_idx, start, end, job, pt)
    """
    makespan = evaluation.compute_makespan_from_gantt(gantt)
    critical = find_critical_path_tasks(gantt, jm_table)

    lo = makespan * position_range[0]
    hi = makespan * position_range[1]

    # 位置範囲でフィルタ
    in_range = [c for c in critical if lo <= c[2] <= hi]
    if len(in_range) < num_delays:
        print(f"WARN: range [{lo:.0f}, {hi:.0f}] に候補 {len(in_range)} 個 "
              f"(< N={num_delays}). 範囲を全 CP に広げます。")
        in_range = critical

    # N 等分の bin
    bin_width = (hi - lo) / num_delays
    selected = []
    selected_keys = set()

    for i in range(num_delays):
        bin_lo = lo + i * bin_width
        bin_hi = lo + (i + 1) * bin_width
        # bin 内、未選択の候補
        bin_cands = [c for c in in_range
                     if bin_lo <= c[2] < bin_hi
                     and (c[0], c[1]) not in selected_keys]
        if not bin_cands:
            # フォールバック: 範囲全体から未選択の最大 PT
            bin_cands = [c for c in in_range
                         if (c[0], c[1]) not in selected_keys]
            if not bin_cands:
                # 想定外: 候補が枯渇
                raise RuntimeError(f"bin {i+1} に選定可能な候補がありません")
        # PT 最大を選ぶ
        bin_cands.sort(key=lambda x: x[5], reverse=True)
        picked = bin_cands[0]
        selected.append(picked)
        selected_keys.add((picked[0], picked[1]))

    return selected, makespan


def main():
    parser = argparse.ArgumentParser(description='複数遅延シナリオ生成')
    parser.add_argument('--source-scenario', required=True,
                        help='元シナリオ名 (initial_gantt を流用)')
    parser.add_argument('--num-delays', type=int, required=True,
                        help='遅延箇所の数 N')
    parser.add_argument('--delay-ratio', type=float, default=1.5,
                        help='delay_amount = 元 PT × この倍率 (既存 generate_scenario.py と同規約)。'
                             'default 1.5 なら delay_amount = 1.5 * PT, new_PT = 2.5 * PT')
    parser.add_argument('--position-min', type=float, default=0.15,
                        help='対象選定の位置下限 (MS に対する割合, default: 0.15)')
    parser.add_argument('--position-max', type=float, default=0.85,
                        help='対象選定の位置上限 (MS に対する割合, default: 0.85)')
    parser.add_argument('--scenario-name', required=True,
                        help='出力シナリオ名')
    args = parser.parse_args()

    # 元シナリオから initial_gantt を取得
    src_path = os.path.join('scenarios', f'{args.source_scenario}.json')
    print(f"読込: {src_path}")
    with open(src_path, encoding='utf-8') as f:
        src = json.load(f)
    initial = src['initial_gantt']
    problem = src['problem']

    # jm_table（PT, machine table 取得用; シナリオ自体はソースを流用）
    jm_table = job_shop_scheduling.get_jm_table(problem, args.source_scenario)
    init_ms = evaluation.compute_makespan_from_gantt(initial)
    print(f"問題: {problem} ({jm_table.get_job_count()} jobs × "
          f"{jm_table.get_machine_count()} machines)")
    print(f"初期 MS = {init_ms}")

    # 対象選定
    print(f"\n--- 遅延対象選定 (N={args.num_delays}, "
          f"range=[{args.position_min}, {args.position_max}], "
          f"ratio=×{args.delay_ratio}) ---")
    targets, makespan = select_multi_delay_targets(
        initial, jm_table, args.num_delays,
        (args.position_min, args.position_max))

    # 遅延注入
    delayed = copy.deepcopy(initial)
    delays_info = []
    for i, t in enumerate(targets):
        m_idx, t_idx, st, et, job, pt = t
        # 既存 generate_scenario.py と同じ規約: delay_amount = PT × ratio
        delay_amount = int(pt * args.delay_ratio)
        delayed = inject_delay(delayed, m_idx, t_idx, delay_amount)
        delays_info.append({
            'machine': m_idx, 'job': job, 'start': st, 'end': et,
            'pt': pt, 'delay_amount': delay_amount,
            'new_pt': pt + delay_amount, 'pos': st / makespan,
        })
        print(f"  #{i+1}: M{m_idx:>2} J{job:>2}  start={st:>4}  PT={pt:>3}"
              f" → {pt + delay_amount:>3} (+{delay_amount})  pos={st/makespan:.2f}")

    # 検証
    print(f"\n--- check_disturbance 検証 ---")
    fixed_gantt, reschedule_gantt, reschedule_time, msg = \
        gantt_chart_operation.check_disturbance(initial, delayed)
    print(f"  {msg}")
    print(f"  reschedule_time = {reschedule_time}")
    print(f"  確定済タスク数 = {sum(len(m) for m in fixed_gantt)}")
    print(f"  リスケ対象数 = {sum(len(m) for m in reschedule_gantt)}")

    rsr_gantt, _ = gantt_chart_operation.create_rsr_gantt(fixed_gantt, reschedule_gantt)
    rsr_ms = evaluation.compute_makespan_from_gantt(rsr_gantt)
    print(f"  right-shift 後 MS = {rsr_ms} "
          f"(init={makespan}, 悪化 +{rsr_ms - makespan})")

    # description
    delay_summary = ", ".join(
        f"M{d['machine']}J{d['job']}(+{d['delay_amount']})"
        for d in delays_info
    )
    description = (f"複数遅延シナリオ: {args.num_delays} 箇所, PT×{args.delay_ratio}. "
                   f"遅延: [{delay_summary}]. "
                   f"初期MS={makespan}, right-shift後MS={rsr_ms}.")

    # 保存
    job_shop_scheduling.save_scenario(
        args.scenario_name, problem, description, initial, delayed)
    print(f"\n=== 保存完了: scenarios/{args.scenario_name}.json ===")


if __name__ == '__main__':
    main()
