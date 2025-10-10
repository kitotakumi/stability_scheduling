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


# 終了時間が変わっている作業を認識する
def check_disturbance(init_gantt, delayed_gantt):
    # fixed_gantt = 既に始まっている作業  reschedule_gantt = リスケ対象の作業
    fixed_gantt = [[] for _ in range(len(delayed_gantt))]
    reschedule_gantt = []
    # 終了時間が変わっている作業を特定
    flag = False
    for machine_index, (machine_init, machine_delayed) in enumerate(
        zip(init_gantt, delayed_gantt)):  # fmt: skip
        for task_init, task_delayed in zip(machine_init, machine_delayed):
            # 作業終了時刻が15以上ずれていたら遅延として認識
            if (task_init[1] - 15 > task_delayed[1]
                or task_delayed[1] > task_init[1] + 15):  # fmt: skip
                # different_task = [開始時刻, 終了時刻, ジョブ番号, 機械番号] = 遅延した作業
                different_task = [task_delayed[0], task_delayed[1], task_delayed[2], machine_index]  # fmt: skip
                reschedule_time = task_delayed[1] + 1
                flag = True
                break
        if flag:
            break
    if flag:
        sorted_gantt = convert_to_1d_gantt(delayed_gantt)
        # 参照を切ってコピー
        gantt_copy = [row[:] for row in sorted_gantt]
        sorted_fixed_gantt = []
        for i, row in enumerate(sorted_gantt):
            st, et, job, machine = row
            # 自分より前に同じjobを持つ要素があるかどうかをチェック
            judge = True
            for c in gantt_copy:
                if c == row:
                    break
                if c[2] == job:
                    judge = False
            if judge == True:
                # stが379未満か確認
                if st < 379:
                    # diffの条件に該当しないか確認 (stがdiffのstより大きく、かつmachineが同じでないこと)
                    if not (st > different_task[0] and machine == different_task[3]):
                        # 条件を満たした場合、bに追加
                        sorted_fixed_gantt.append(row)
                        # その要素がdiffと等しくない限りa_copyから削除
                        if row != different_task:
                            gantt_copy.remove(row)
        fixed_gantt = convert_to_2d_gantt(sorted_fixed_gantt, len(delayed_gantt))
        # リスケジューリング対象の作業をreschedule_machineに格納
        for machine_idx, machine in enumerate(delayed_gantt):
            updated_machine = [task for task in machine if task not in fixed_gantt[machine_idx]]  # fmt: skip
            reschedule_gantt.append(updated_machine)

    if flag:
        return (fixed_gantt, reschedule_gantt, reschedule_time,
            "遅延作業を検知しました。リスケジューリングします",)  # fmt: skip
    else:
        reschedule_time = 0
        return (fixed_gantt, reschedule_gantt, reschedule_time,
            "リスケジューリングは行いません",)  # fmt: skip


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
