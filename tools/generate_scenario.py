"""
シナリオ生成スクリプト

GAで初期スケジュールを生成し、クリティカルパス上のタスクに遅延を注入して
シナリオJSONを保存する。

使い方:
    python tools/generate_scenario.py --problem la21 --delay-ratio 1.5 --ga-ngen 500
    python tools/generate_scenario.py --problem la36 --delay-ratio 1.5 --ga-ngen 800
    python tools/generate_scenario.py --problem la40 --delay-ratio 1.5 --ga-ngen 800
"""

import argparse
import copy
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import job_shop_scheduling
import gantt_chart_operation
import evaluation


# ========== GA for initial scheduling (makespan minimization) ==========

def create_random_individual(num_jobs, num_machines):
    """ランダムな遺伝子（ジョブ番号の順列表現）を生成"""
    gene = []
    for j in range(num_jobs):
        gene.extend([j] * num_machines)
    random.shuffle(gene)
    return gene


def order_crossover(p1, p2):
    """順序交叉 (OX): JSPのジョブ番号繰り返し表現に対応"""
    size = len(p1)
    cx1, cx2 = sorted(random.sample(range(size), 2))

    def ox(parent1, parent2):
        child = [None] * size
        child[cx1:cx2] = parent1[cx1:cx2]
        # parent2の残りを順番に埋める
        fill = [g for g in parent2 if child.count(g) < parent1.count(g) or g not in child[cx1:cx2]]
        # 正確にカウントベースで埋める
        from collections import Counter
        needed = Counter(parent1)
        placed = Counter(child[cx1:cx2])
        remaining = []
        for g in parent2:
            if placed[g] < needed[g]:
                remaining.append(g)
                placed[g] += 1
        pos = 0
        for i in range(size):
            if child[i] is None:
                child[i] = remaining[pos]
                pos += 1
        return child

    return ox(p1, p2), ox(p2, p1)


def inversion_mutation(ind):
    """逆位突然変異"""
    size = len(ind)
    i, j = sorted(random.sample(range(size), 2))
    ind[i:j+1] = ind[i:j+1][::-1]
    return ind


def run_initial_ga(jm_table, ngen=500, pop_size=100, seed=42):
    """初期スケジューリング用GA（メイクスパン最小化）

    Returns:
        (best_individual, best_gantt, best_makespan)
    """
    random.seed(seed)
    num_jobs = jm_table.get_job_count()
    num_machines = jm_table.get_machine_count()

    # 初期集団
    population = []
    for _ in range(pop_size):
        ind = create_random_individual(num_jobs, num_machines)
        population.append(ind)

    def evaluate(ind):
        gantt = gantt_chart_operation.get_gantt(jm_table, ind)
        return evaluation.compute_makespan_from_gantt(gantt)

    # 評価
    fitnesses = [(ind, evaluate(ind)) for ind in population]
    fitnesses.sort(key=lambda x: x[1])
    best_ind, best_ms = fitnesses[0]
    best_gantt = gantt_chart_operation.get_gantt(jm_table, best_ind)

    print(f"  Gen 0: best MS = {best_ms}")

    for gen in range(1, ngen + 1):
        # トーナメント選択 + 交叉 + 突然変異
        offspring = []
        # エリート保存
        offspring.append(fitnesses[0][0][:])

        while len(offspring) < pop_size:
            # トーナメント選択
            candidates = random.sample(fitnesses, min(4, len(fitnesses)))
            p1 = min(candidates, key=lambda x: x[1])[0]
            candidates = random.sample(fitnesses, min(4, len(fitnesses)))
            p2 = min(candidates, key=lambda x: x[1])[0]

            # 交叉
            if random.random() < 0.9:
                c1, c2 = order_crossover(p1, p2)
            else:
                c1, c2 = p1[:], p2[:]

            # 突然変異
            if random.random() < 0.1:
                inversion_mutation(c1)
            if random.random() < 0.1:
                inversion_mutation(c2)

            offspring.append(c1)
            if len(offspring) < pop_size:
                offspring.append(c2)

        population = offspring[:pop_size]
        fitnesses = [(ind, evaluate(ind)) for ind in population]
        fitnesses.sort(key=lambda x: x[1])

        if fitnesses[0][1] < best_ms:
            best_ind, best_ms = fitnesses[0][0][:], fitnesses[0][1]
            best_gantt = gantt_chart_operation.get_gantt(jm_table, best_ind)

        if gen % 100 == 0 or gen == ngen:
            print(f"  Gen {gen}: best MS = {best_ms}")

    return best_ind, best_gantt, best_ms


# ========== Critical path analysis ==========

def find_critical_path_tasks(gantt, jm_table):
    """クリティカルパス上のタスクを特定する。

    Returns:
        list of (machine_idx, task_idx, start, end, job, processing_time)
        sorted by start time
    """
    makespan = evaluation.compute_makespan_from_gantt(gantt)
    num_machines = len(gantt)

    # 全タスクをリスト化
    all_tasks = []
    for m_idx, machine in enumerate(gantt):
        for t_idx, task in enumerate(machine):
            st, et, job = task
            all_tasks.append((m_idx, t_idx, st, et, job, et - st))

    # 各タスクの後続制約を構築してバックワードパスを計算
    # タスクを (job, op_index) で索引
    job_ops = {}  # (job_id) -> [(m_idx, t_idx, st, et, pt), ...] in order
    for m_idx, machine in enumerate(gantt):
        for t_idx, task in enumerate(machine):
            st, et, job = task
            if job not in job_ops:
                job_ops[job] = []
            job_ops[job].append((m_idx, t_idx, st, et, et - st))

    # ジョブ内の工程順序（開始時刻でソート）
    for job in job_ops:
        job_ops[job].sort(key=lambda x: x[2])

    # クリティカルパス: 開始時刻 + 処理時間 == 終了時刻 で、
    # さらに後続タスクの開始時刻 == このタスクの終了時刻（隙間なし）
    critical = []
    for m_idx, t_idx, st, et, job, pt in all_tasks:
        # 機械上の後続タスク
        machine_slack = True
        if t_idx + 1 < len(gantt[m_idx]):
            next_st = gantt[m_idx][t_idx + 1][0]
            if next_st == et:
                machine_slack = False

        # ジョブ内の後続タスク
        ops = job_ops[job]
        job_slack = True
        for i, (om, ot, ost, oet, opt) in enumerate(ops):
            if om == m_idx and ot == t_idx:
                if i + 1 < len(ops):
                    next_job_st = ops[i + 1][2]
                    if next_job_st == et:
                        job_slack = False
                break

        # 末端タスク（メイクスパンを構成）
        is_terminal = (et == makespan)

        # スラックなし = クリティカル
        if is_terminal or not machine_slack or not job_slack:
            critical.append((m_idx, t_idx, st, et, job, pt))

    critical.sort(key=lambda x: x[2])
    return critical


def select_disruption_target(critical_tasks, makespan, position_range=(0.3, 0.6)):
    """クリティカルパス上の中盤タスクから遅延対象を選択する。

    position_range: メイクスパンに対する開始時刻の相対位置の範囲
    """
    min_pos = makespan * position_range[0]
    max_pos = makespan * position_range[1]

    candidates = [t for t in critical_tasks if min_pos <= t[2] <= max_pos]

    if not candidates:
        # 範囲を広げる
        candidates = [t for t in critical_tasks if t[2] > makespan * 0.2]
        if not candidates:
            candidates = critical_tasks

    # 加工時間が大きいものを優先（遅延の影響が大きい）
    candidates.sort(key=lambda x: x[5], reverse=True)
    return candidates[0]


# ========== Disruption injection ==========

def inject_delay(gantt, machine_idx, task_idx, delay_amount):
    """指定タスクの終了時刻を遅延させたdelayed_ganttを生成する。

    遅延タスクの終了時刻のみ変更し、他のタスクはそのまま（right-shift repairは
    check_disturbance + create_rsr_gantt で行う）。
    """
    delayed = copy.deepcopy(gantt)
    task = delayed[machine_idx][task_idx]
    original_end = task[1]
    task[1] = original_end + delay_amount
    return delayed


# ========== Main ==========

def main():
    parser = argparse.ArgumentParser(description='シナリオ生成: GAで初期スケジュール作成 + 外乱注入')
    parser.add_argument('--problem', required=True, help='問題名 (例: la21, la36, la40)')
    parser.add_argument('--delay-ratio', type=float, default=1.5,
                        help='遅延量 = 元の加工時間 × この倍率 (default: 1.5)')
    parser.add_argument('--ga-ngen', type=int, default=500, help='GA世代数 (default: 500)')
    parser.add_argument('--ga-pop', type=int, default=100, help='GA集団サイズ (default: 100)')
    parser.add_argument('--seed', type=int, default=42, help='乱数シード (default: 42)')
    parser.add_argument('--scenario-name', default=None,
                        help='シナリオ名 (default: {problem}_delay{amount})')
    parser.add_argument('--position-min', type=float, default=0.3,
                        help='遅延対象の位置下限 (MSに対する割合, default: 0.3)')
    parser.add_argument('--position-max', type=float, default=0.6,
                        help='遅延対象の位置上限 (MSに対する割合, default: 0.6)')
    args = parser.parse_args()

    print(f"=== シナリオ生成: {args.problem} ===")
    print(f"遅延倍率: {args.delay_ratio}x, GA世代数: {args.ga_ngen}, seed: {args.seed}")

    # 問題読み込み
    jm_table = job_shop_scheduling.get_jm_table(args.problem)
    print(f"問題サイズ: {jm_table.get_job_count()} jobs × {jm_table.get_machine_count()} machines")

    # GA実行
    print("\n--- GA初期スケジューリング ---")
    best_ind, best_gantt, best_ms = run_initial_ga(
        jm_table, ngen=args.ga_ngen, pop_size=args.ga_pop, seed=args.seed)
    print(f"初期スケジュール: MS = {best_ms}")

    # クリティカルパス分析
    print("\n--- クリティカルパス分析 ---")
    critical = find_critical_path_tasks(best_gantt, jm_table)
    print(f"クリティカルパス上のタスク数: {len(critical)}")

    # 遅延対象選択
    target = select_disruption_target(
        critical, best_ms, (args.position_min, args.position_max))
    m_idx, t_idx, st, et, job, pt = target
    delay_amount = int(pt * args.delay_ratio)
    print(f"遅延対象: Machine {m_idx}, Task[{t_idx}] (Job {job})")
    print(f"  開始={st}, 終了={et}, 加工時間={pt}")
    print(f"  遅延量={delay_amount} (PT×{args.delay_ratio})")
    print(f"  遅延後終了時刻: {et} → {et + delay_amount}")

    # 遅延注入
    delayed_gantt = inject_delay(best_gantt, m_idx, t_idx, delay_amount)

    # check_disturbanceで検証
    fixed_gantt, reschedule_gantt, reschedule_time, msg = \
        gantt_chart_operation.check_disturbance(best_gantt, delayed_gantt)
    print(f"\n--- 外乱検証 ---")
    print(f"  {msg}")
    print(f"  reschedule_time: {reschedule_time}")
    print(f"  確定済みタスク数: {sum(len(m) for m in fixed_gantt)}")
    print(f"  リスケ対象タスク数: {sum(len(m) for m in reschedule_gantt)}")

    # right-shift後のメイクスパン
    rsr_gantt, _ = gantt_chart_operation.create_rsr_gantt(fixed_gantt, reschedule_gantt)
    rsr_ms = evaluation.compute_makespan_from_gantt(rsr_gantt)
    print(f"  right-shift後のMS: {rsr_ms} (元: {best_ms}, 悪化: +{rsr_ms - best_ms})")

    # シナリオ保存
    scenario_name = args.scenario_name
    if scenario_name is None:
        scenario_name = f"{args.problem}_delay{delay_amount}"

    description = (f"Machine{m_idx}のJob{job}の終了時刻が{et}→{et + delay_amount}に遅延 "
                   f"({delay_amount}分遅延, PT×{args.delay_ratio}). "
                   f"初期MS={best_ms}, right-shift後MS={rsr_ms}")

    job_shop_scheduling.save_scenario(
        scenario_name, args.problem, description, best_gantt, delayed_gantt)

    print(f"\n=== シナリオ保存完了 ===")
    print(f"  ファイル: scenarios/{scenario_name}.json")
    print(f"  使い方: get_jm_table('{args.problem}', '{scenario_name}')")


if __name__ == "__main__":
    main()
