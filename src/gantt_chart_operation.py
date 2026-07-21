"""
ガントチャートを操作する関数を格納するモジュール
def get_gantt:遺伝子をガントチャートに展開する関数
def get_gantt_reactive:リスケジューリング後のガントチャートを取得する関数
def get_gene:リスケジューリング対象のガントチャートを遺伝子に変換
"""

import sys
import copy


# 遺伝子をガントチャートに展開。リスケではない通常のデコーディング。
def get_gantt(jm_table, individual):
    MAX_MACHINES = jm_table.get_machine_count()
    # gantt [ MACHINE NUMBER ] = [ [0,0,None], [start, end, job_num], ...]
    # startの昇順に並ぶ、初期値にダミーの作業をセットしておく
    gantt = [[[0, 0, None], [sys.maxsize, sys.maxsize, None]] for _ in range(MAX_MACHINES)]  # fmt: skip
    jmChild = jm_table.get_child()
    for job_num in individual:
        # job_numジョブのこの工程の(Machine番号, 処理時間)を取得
        machine = jmChild.get_machine(job_num)
        process_time = jmChild.get_process_time(job_num)
        # このジョブの最も早い開始時刻を取得
        job_earliest = jmChild.get_earliest(job_num)
        # print ( job_num, machine, process_time )
        # 左シフトで挿入できる隙間をさがす
        for idx, ((st0, ed0, _), (st1, ed1, _)) in enumerate(
            zip(gantt[machine][:-1], gantt[machine][1:])):  # fmt: skip
            gap_st, gap_ed = ed0, st1
            # 隙間終了時刻でも最早時刻に満たない スキップ
            if gap_ed <= job_earliest:
                continue
            # 最早時刻が隙間の途中にあるとき 隙間の開始時刻を最早時刻にする
            gap_st = job_earliest if gap_st < job_earliest else ed0
            # 隙間にこの処理が入らない スキップ
            if (gap_ed - gap_st) < process_time:
                continue
            # 隙間にこの処理が入る; スケジュールにこの工程を挿入
            job_end = gap_st + process_time
            gantt[machine].insert(idx + 1, [gap_st, job_end, job_num])
            break

        jmChild.set_next_earliest(job_num, job_end)
    # 最初と最後のダミー作業を削除
    gantt = [row[1:-1] for row in gantt]
    return gantt


# 左詰めありでリスケジューリング後のガントチャートを取得
def get_gantt_reactive(jm_table, individual, fixed_gantt, reschedule_time):
    MAX_MACHINES = jm_table.get_machine_count()
    gantt = copy.deepcopy(fixed_gantt)

    # 全ての機械に対して少なくとも空のリストがあることを確認
    for machine_index in range(MAX_MACHINES):
        if len(gantt[machine_index]) == 0:
            gantt[machine_index] = []

    jmChild = jm_table.get_child_reactive(gantt, reschedule_time)

    # 最後にダミー作業を追加
    for machine in gantt:
        machine.append([sys.maxsize, sys.maxsize, None])

    for job_num in individual:
        # job_numジョブのこの工程の(Machine番号, 処理時間)を取得
        machine = jmChild.get_machine(job_num)
        process_time = jmChild.get_process_time(job_num)

        # このジョブの最も早い開始時刻を取得
        job_earliest = jmChild.get_earliest(job_num)

        # 左シフトで挿入できる隙間を探す
        for idx, ((st0, ed0, _), (st1, ed1, _)) in enumerate(
            zip(gantt[machine][:-1], gantt[machine][1:])):  # fmt: skip
            gap_st, gap_ed = ed0, st1

            # 隙間終了時刻が最早開始時刻より早い場合はスキップ
            if gap_ed <= job_earliest:
                continue

            # 最早開始時刻が隙間の途中にある場合は、隙間の開始時刻を調整
            gap_st = job_earliest if gap_st < job_earliest else ed0

            # 隙間にこの処理が入らない場合はスキップ
            if (gap_ed - gap_st) < process_time:
                continue

            # 隙間に処理が入る場合、スケジュールに挿入
            job_end = gap_st + process_time
            gantt[machine].insert(idx + 1, [gap_st, job_end, job_num])
            break

        # 隙間が見つからない場合、最後にジョブを挿入
        if len(gantt[machine]) == 1 and gantt[machine][0] == [
            sys.maxsize,
            sys.maxsize,
            None,
        ]:
            job_end = job_earliest + process_time
            gantt[machine].insert(0, [job_earliest, job_end, job_num])

        jmChild.set_next_earliest(job_num, job_end)

    # 最後のダミー作業を削除
    for machine in gantt:
        machine.pop()

    return gantt


# 左詰めなしの自然な形でガントチャートを作成
def get_gantt_reactive_natural(jm_table, individual, fixed_gantt, reschedule_time):
    MAX_MACHINES = jm_table.get_machine_count()
    gantt = copy.deepcopy(fixed_gantt)

    # 全ての機械に対して少なくとも空のリストがあることを確認
    for machine_index in range(MAX_MACHINES):
        if len(gantt[machine_index]) == 0:
            gantt[machine_index] = []

    jmChild = jm_table.get_child_reactive(gantt, reschedule_time)

    for job_num in individual:
        # job_numジョブのこの工程の(Machine番号, 処理時間)を取得
        machine = jmChild.get_machine(job_num)
        process_time = jmChild.get_process_time(job_num)

        # このジョブの最も早い開始時刻を取得
        job_earliest = jmChild.get_earliest(job_num)

        # マシン上の最後のタスク終了時刻（なければ reschedule_time）
        last_end = gantt[machine][-1][1] if gantt[machine] else reschedule_time

        # 実際の開始時刻：ジョブの準備完了後か、マシンの開放後か遅いほう
        start = max(job_earliest, last_end)
        end = start + process_time

        gantt[machine].append([start, end, job_num])

        jmChild.set_next_earliest(job_num, end)

    return gantt


def convert_to_1d_gantt(gantt_chart):
    flattened_gantt = []
    # Iterate over each machine and its operations
    for machine_number, operations in enumerate(gantt_chart):
        for operation in operations:
            st, et, jobn = operation
            flattened_gantt.append([st, et, jobn, machine_number])
    # Sort by start time (st)
    sorted_gantt = sorted(flattened_gantt, key=lambda x: x[0])
    return sorted_gantt


def convert_to_2d_gantt(flattened_gantt, num_machines):
    gantt_chart_2d = [[] for _ in range(num_machines)]
    # Iterate over each entry in the flattened gantt chart
    for entry in flattened_gantt:
        st, et, jobn, machine_number = entry
        gantt_chart_2d[machine_number].append([st, et, jobn])
    return gantt_chart_2d


DELAY_DETECTION_THRESHOLD = 15  # この時刻以上ずれたら遅延として認識


def check_disturbance(init_gantt, delayed_gantt):
    """複数遅延に対応した外乱検知

    挙動:
      1. init_gantt と delayed_gantt を比較し、終了時刻が DELAY_DETECTION_THRESHOLD
         以上ずれたタスク（= 遅延タスク）を全て検出する。
      2. delayed_gantt 全体を right-shift して整合的なスケジュールを得る
         （inject_delay は単一タスクの end しか書き換えないため delayed_gantt は
          時刻不整合を含む。create_rsr_gantt を空の fixed_gantt 付きで使うことで
          全タスクを左詰め直し、伝播遅延を含む正しい right-shift 結果を得る）。
      3. right-shift 後の遅延タスクの end の最大値を reschedule_time とする
         （遅延が解消する時刻 = 再スケジューリング開始時刻）。
         +1 しないのが重要: 遅延opの直後工程は rs_start = 遅延op終了 = reschedule_time
         ちょうどに開始する。+1 すると下の < 判定でこの直後工程まで凍結され、外乱に直撃
         された「一番リスケすべき工程」を最適化対象から外してしまう（2026-06 修正）。
      4. right-shift 後のスケジュールを reschedule_time で分割:
         - 開始時刻 < reschedule_time → fixed_gantt（遅延解消までに実際に開始済み）
         - 開始時刻 >= reschedule_time → reschedule_gantt（最適化対象。ブロックされて
           まだ開始できていない直後工程もこちらに入る）

    Returns:
        (fixed_gantt, reschedule_gantt, reschedule_time, message)
    """
    n_machines = len(delayed_gantt)

    # ========== 1. 全遅延検出 ==========
    delayed_keys = set()  # (machine, job) のセット
    for m_idx, (machine_init, machine_delayed) in enumerate(
            zip(init_gantt, delayed_gantt)):
        for task_init, task_delayed in zip(machine_init, machine_delayed):
            if abs(task_delayed[1] - task_init[1]) > DELAY_DETECTION_THRESHOLD:
                delayed_keys.add((m_idx, task_delayed[2]))

    if not delayed_keys:
        return ([[] for _ in range(n_machines)],
                [list(m) for m in delayed_gantt],
                0,
                "リスケジューリングは行いません")

    # ========== 2. delayed_gantt 全体を right-shift ==========
    # 空の fixed_gantt で create_rsr_gantt を呼ぶと、全タスクが左詰めで
    # 再配置される（PT は et - st で保存されるので延長 PT も維持される）
    empty_fixed = [[] for _ in range(n_machines)]
    rs_gantt, _ = create_rsr_gantt(empty_fixed, delayed_gantt)

    # ========== 3. 遅延タスクの right-shift 後 end の最大値 ==========
    # +1 しない（直後工程を凍結しないため。docstring 参照）
    latest_end = 0
    for m_idx, machine in enumerate(rs_gantt):
        for task in machine:
            if (m_idx, task[2]) in delayed_keys:
                latest_end = max(latest_end, task[1])
    reschedule_time = latest_end

    # ========== 4. fixed / reschedule に分割 ==========
    # rs_gantt（時刻整合済）を基準に判定する
    fixed_gantt = [[] for _ in range(n_machines)]
    reschedule_gantt = [[] for _ in range(n_machines)]
    for m_idx, machine in enumerate(rs_gantt):
        for task in machine:
            if task[0] < reschedule_time:
                fixed_gantt[m_idx].append(list(task))
            else:
                reschedule_gantt[m_idx].append(list(task))

    msg = (f"遅延 {len(delayed_keys)} 箇所を検知。"
           f"リスケジューリングします（リスケ開始 t={reschedule_time}）")
    return fixed_gantt, reschedule_gantt, reschedule_time, msg


def create_rsr_gantt(fixed_gantt, rescheduled_gantt):
    # 深いコピーを作成し、fixed_ganttの内容をganttに格納
    rsr_gantt = [machine[:] for machine in fixed_gantt]
    rescheduled_rsr_gantt = [[] for machine in fixed_gantt]

    # 1Dガントチャートに変換して並び替える
    rescheduled_gantt_1d = convert_to_1d_gantt(rescheduled_gantt)

    # 各ジョブをrescheduled_gantt_1dに対して処理
    for st, et, jobn, machine_number in rescheduled_gantt_1d:
        # 1. 同じ機械の最後のジョブの終了時刻を見つける
        if rsr_gantt[machine_number]:
            last_operation_on_machine = rsr_gantt[machine_number][-1]
            last_end_time_machine = last_operation_on_machine[1]  # 終了時刻
        else:
            last_end_time_machine = 0

        # 2. 同じジョブの最後の終了時刻を見つける
        last_end_time_job = 0
        for machine in rsr_gantt:
            for operation in machine:
                if operation[2] == jobn:  # 同じジョブ番号
                    last_end_time_job = max(last_end_time_job, operation[1])

        # 開始時刻を決定（どちらか大きい方）
        start_time = max(last_end_time_machine, last_end_time_job)
        # 終了時刻を再計算
        end_time = start_time + (et - st)

        # 修正されたガントを追加
        rsr_gantt[machine_number].append([start_time, end_time, jobn])
        rescheduled_rsr_gantt[machine_number].append([start_time, end_time, jobn])

    return (rsr_gantt, rescheduled_rsr_gantt)


# リアクティブスケジューリングにおけるリスケ対象の作業のガントチャートを遺伝子で表現する
def get_gene(reschedule_gantt):
    gantt = copy.deepcopy(reschedule_gantt)
    src = []
    # 最も開始時間が早い作業のジョブ番号を遺伝子に追加してその作業をガントチャートから削除する
    while any(gantt):
        earliest_time = float("inf")  # 無限
        earliest_machine = -1
        earliest_index = -1
        earliest_job = -1
        for machine_index, machine in enumerate(gantt):
            for task_index, task in enumerate(machine):
                if task[0] < earliest_time:
                    earliest_time = task[0]
                    earliest_job = task[2]
                    earliest_machine = machine_index
                    earliest_index = task_index

        src.append(earliest_job)
        gantt[earliest_machine].pop(earliest_index)

    return src
