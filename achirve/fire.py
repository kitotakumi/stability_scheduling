from lib2to3.pytree import convert
import analysis


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


# fmt:off
init_gantt = [
    [[0, 76, 8], [76, 105, 0], [115, 201, 7], [201, 244, 1], [244, 250, 4], [250, 263, 9], [263, 334, 3], [364, 401, 6], [401, 448, 5], [455, 540, 2]], 
    [[0, 81, 3], [84, 86, 5], [86, 171, 9], [171, 249, 0], [249, 318, 8], [318, 364, 6], [364, 455, 2], [455, 501, 7], [501, 523, 4], [531, 559, 1]], 
    [[0, 84, 5], [84, 115, 7], [115, 210, 3], [210, 224, 4], [249, 258, 0], [258, 348, 1], [348, 409, 9], [445, 530, 8], [530, 543, 6], [579, 653, 2]], 
    [[138, 233, 5], [258, 294, 0], [318, 394, 8], [401, 462, 6], [462, 531, 1], [540, 579, 2], [636, 662, 4], [683, 735, 9], [755, 853, 3], [868, 947, 7]], 
    [[294, 343, 0], [348, 423, 1], [423, 522, 3], [522, 528, 5], [575, 607, 7], [662, 731, 4], [731, 757, 8], [757, 847, 9], [902, 957, 6], [971, 1004, 2]], 
    [[86, 138, 5], [343, 354, 0], [394, 445, 8], [501, 575, 7], [575, 636, 4], [636, 683, 9], [727, 748, 6], [748, 758, 2], [773, 819, 1], [926, 969, 3]], 
    [[354, 416, 0], [416, 423, 9], [448, 513, 5], [522, 531, 3], [541, 581, 8], [607, 695, 7], [695, 727, 6], [727, 773, 1], [773, 862, 2], [904, 957, 4]], 
    [[416, 472, 0], [528, 553, 5], [581, 670, 8], [670, 755, 3], [758, 770, 2], [783, 832, 4], [832, 868, 7], [872, 902, 6], [902, 974, 1], [974, 1019, 9]], 
    [[233, 281, 5], [423, 487, 9], [487, 531, 0], [531, 583, 3], [653, 743, 2], [743, 762, 7], [762, 783, 4], [783, 872, 6], [872, 946, 8], [974, 1004, 1]], 
    [[281, 353, 5], [423, 434, 1], [530, 541, 8], [541, 617, 9], [617, 638, 0], [748, 780, 6], [780, 828, 7], [832, 904, 4], [904, 926, 3], [926, 971, 2]]
]
# fmt:on

# fmt: off
delayed_gantt = [
    [[0, 76, 8], [76, 105, 0], [115, 201, 7], [201, 244, 1], [244, 250, 4], [250, 263, 9], [263, 334, 3], [364, 401, 6], [401, 448, 5], [455, 540, 2]], 
    [[0, 81, 3], [84, 86, 5], [86, 171, 9], [171, 249, 0], [249, 378, 8], [318, 364, 6], [364, 455, 2], [455, 501, 7], [501, 523, 4], [531, 559, 1]], 
    [[0, 84, 5], [84, 115, 7], [115, 210, 3], [210, 224, 4], [249, 258, 0], [258, 348, 1], [348, 409, 9], [445, 530, 8], [530, 543, 6], [579, 653, 2]], 
    [[138, 233, 5], [258, 294, 0], [318, 394, 8], [401, 462, 6], [462, 531, 1], [540, 579, 2], [636, 662, 4], [683, 735, 9], [755, 853, 3], [868, 947, 7]], 
    [[294, 343, 0], [348, 423, 1], [423, 522, 3], [522, 528, 5], [575, 607, 7], [662, 731, 4], [731, 757, 8], [757, 847, 9], [902, 957, 6], [971, 1004, 2]], 
    [[86, 138, 5], [343, 354, 0], [394, 445, 8], [501, 575, 7], [575, 636, 4], [636, 683, 9], [727, 748, 6], [748, 758, 2], [773, 819, 1], [926, 969, 3]], 
    [[354, 416, 0], [416, 423, 9], [448, 513, 5], [522, 531, 3], [541, 581, 8], [607, 695, 7], [695, 727, 6], [727, 773, 1], [773, 862, 2], [904, 957, 4]], 
    [[416, 472, 0], [528, 553, 5], [581, 670, 8], [670, 755, 3], [758, 770, 2], [783, 832, 4], [832, 868, 7], [872, 902, 6], [902, 974, 1], [974, 1019, 9]], 
    [[233, 281, 5], [423, 487, 9], [487, 531, 0], [531, 583, 3], [653, 743, 2], [743, 762, 7], [762, 783, 4], [783, 872, 6], [872, 946, 8], [974, 1004, 1]], 
    [[281, 353, 5], [423, 434, 1], [530, 541, 8], [541, 617, 9], [617, 638, 0], [748, 780, 6], [780, 828, 7], [832, 904, 4], [904, 926, 3], [926, 971, 2]]
]
# fmt: on

# diff = [249, 378, 8, 1]
# fixed_gantt, reschedule_gantt, reschedule_time, message =check_disturbance(init_gantt, delayed_gantt)  # fmt: skip
# sorted_gantt = convert_gantt_to_1D(delayed_gantt)

# # 参照を切ってコピー
# gantt_copy = [row[:] for row in sorted_gantt]
# sorted_fixed_gantt = []
# for i, row in enumerate(sorted_gantt):
#     st, et, job, machine = row
#     # 自分より前に同じjobを持つ要素があるかどうかをチェック
#     judge = True
#     for c in gantt_copy:
#         if c == row:
#             break
#         if c[2] == job:
#             judge = False
#     if judge == True:
#         # stが379未満か確認
#         if st < 379:
#             # diffの条件に該当しないか確認 (stがdiffのstより大きく、かつmachineが同じでないこと)
#             if not (st > diff[0] and machine == diff[3]):
#                 # 条件を満たした場合、bに追加
#                 sorted_fixed_gantt.append(row)
#                 # その要素がdiffと等しくない限りa_copyから削除
#                 if row != diff:
#                     gantt_copy.remove(row)
# fixed_gantt = convert_to_2d_gantt(sorted_fixed_gantt, len(delayed_gantt))
fixed_gantt, reschedule_gantt, reschedule_time, message =check_disturbance(init_gantt, delayed_gantt)  # fmt: skip
analysis.plot_gantt(fixed_gantt, 10, 10, "aa")  # fmt: skip
analysis.plot_gantt(reschedule_gantt, 10, 10, "aa")  # fmt: skip
