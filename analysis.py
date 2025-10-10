"""
グラフを描画する関数を格納するモジュール
def plot_scatter:評価関数と世代の散布図
def plot_gantt:ガントチャート
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d import Axes3D


# 評価関数と世代の散布図
def plot_scatter(ngen, pop_size, all_fitnesses):
    plt.figure(figsize=(10, 5))
    all_generations = []
    # 全個体の世代をall_generationsに格納
    for gen in range(ngen):
        all_generations.append([gen for ind in range(pop_size)])
    # concatenateで[[]]を一次元配列に変換
    plt.scatter(
        np.concatenate(all_generations), np.concatenate(all_fitnesses), alpha=0.5
    )
    plt.title("Evaluation Value per Generation")
    plt.xlabel("generation")
    plt.ylabel("makespan")
    plt.show()


# ガントチャートの表示
def plot_gantt(gannt, num_jobs, num_machines, jsp, reschedule_time=None):
    # ジョブに対応する色のリスト（ジョブの数に応じて動的に生成）
    colors = plt.cm.tab10(range(num_jobs))

    # プロットの作成
    fig, gnt = plt.subplots(figsize=(10, 5))
    gnt.set_ylim(0, num_machines)
    max_time = max(task[1] for machine in gannt for task in machine)
    # gnt.set_xlim(0, max_time)
    gnt.set_xlim(0, 1100)
    gnt.set_xlabel("Time")
    gnt.set_ylabel("Machines")

    # makespanを表示
    gnt.annotate(
        f"Makespan: {max_time}",
        xy=(max_time, 0),
        xytext=(max_time, 0.5),
        arrowprops=dict(facecolor="red", shrink=0.05),
    )

    # マシンごとのY軸ラベル
    gnt.set_yticks([i + 0.5 for i in range(num_machines)])
    gnt.set_yticklabels(["Machine {}".format(i) for i in range(num_machines)])
    gnt.grid(True)

    # ジョブごとにプロット
    for machine, tasks in enumerate(gannt):
        for task in tasks:
            start, end, job = task
            gnt.broken_barh(
                [(start, end - start)], (machine, 1), facecolors=(colors[job])
            )

    # reschedule_time地点で補助線を追加
    if reschedule_time is not None:
        gnt.axvline(x=reschedule_time, color="red", linestyle="--", linewidth=2)
        gnt.annotate(
            f"Reschedule Time: {reschedule_time}",
            xy=(reschedule_time, 0),
            xytext=(reschedule_time, 0.1),
        )

        # arrowprops=dict(facecolor='red', shrink=0.05))

    # 凡例の作成
    patches = [
        mpatches.Patch(color=colors[i], label="Job {}".format(i))
        for i in range(num_jobs)
    ]
    plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc="upper left")

    # plt.title(f"Gantt Chart for {jsp} Job Shop Scheduling")
    fig.tight_layout()
    plt.show()


def plot_cs_pe(CS: dict[int, list], PE: list, title: str = "CS & PE Scatter") -> None:
    """
    CS（各重みバケットごとの個体リスト）と
    PE（グローバル・アーカイブとして集めた個体リスト）の
    makespan vs stability を散布図で描画する。
    """
    plt.figure(figsize=(6,6))
    # PE: 黒×マーカー
    xs_pe = [ind.fitness.values[0] for ind in PE]
    ys_pe = [ind.fitness.values[1] for ind in PE]
    plt.scatter(xs_pe, ys_pe, c='k', marker='x', label='PE')

    # CS: バケットごとに色分け
    for bucket_idx, front in CS.items():
        xs_cs = [ind.fitness.values[0] for ind in front]
        ys_cs = [ind.fitness.values[1] for ind in front]
        plt.scatter(xs_cs, ys_cs, s=20, alpha=0.6, label=f'CS{bucket_idx}')

    plt.xlabel('Makespan')      # 効率性関数
    plt.ylabel('Stability')     # 安定性関数
    plt.title(title)
    plt.legend(fontsize='small', loc='best', ncol=2)
    plt.tight_layout()
    plt.show()

def plot_cs_pe_realtime(CS: dict[int, list], PE: list, gen: int, ngen: int):
    """
    毎世代のCSとPEの状態をリアルタイムで描画する関数
    """
    plt.clf()  # 現在のプロットをクリア

    # PE (Pareto Front) のプロット
    xs_pe = [ind.fitness.values[0] for ind in PE]
    ys_pe = [ind.fitness.values[1] for ind in PE]
    plt.scatter(xs_pe, ys_pe, c='k', marker='x', label='Pareto Front', s=100, zorder=2)

    # CS (Current Sets) のプロット
    # plt.cm.viridisは色のマップ。重みごとに異なる色を割り当てる
    colors = plt.cm.viridis(np.linspace(0, 1, len(CS)))
    for bucket_idx, front in CS.items():
        if not front: continue # バケットが空ならスキップ
        xs_cs = [ind.fitness.values[0] for ind in front]
        ys_cs = [ind.fitness.values[1] for ind in front]
        plt.scatter(xs_cs, ys_cs, s=20, alpha=0.7, color=colors[bucket_idx], label=f'CS {bucket_idx}', zorder=1)

    plt.xlabel('Makespan (Efficiency)')
    plt.ylabel('Stability')
    plt.title(f"Evolution of Solutions (Generation: {gen}/{ngen})")
    
    # 凡例が多くなりすぎるのを防ぐため、凡例は表示しないか、代表的なものだけ表示する
    # plt.legend() 
    
    plt.grid(True)
    plt.tight_layout()
    plt.pause(0.1)  # 0.1秒停止してグラフを表示