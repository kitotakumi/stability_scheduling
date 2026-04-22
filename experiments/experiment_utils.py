"""実験用共通ユーティリティ"""

import os
import random
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

import job_shop_scheduling
import gantt_chart_operation
import ga_scheduling
import genetic_operation
import ils_scheduling
import evaluation


# ========== 設定 ==========
PROBLEM_NAME = "mt10"
SCENARIO_NAME = "mt10_delay60"
GA_NGEN = 500
GA_POP_SIZE = 50
ILS_MAX_ITER = 800


def setup_output_dir(prefix="", base_dir=None):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = f"{prefix}_{timestamp}" if prefix else timestamp
    if base_dir is None:
        base_dir = os.path.dirname(__file__)
    out_dir = os.path.join(base_dir, "results", name)
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def get_problem(problem_name=None, scenario_name=None):
    if problem_name is None:
        problem_name = PROBLEM_NAME
    if scenario_name is None:
        scenario_name = SCENARIO_NAME
    jm_table = job_shop_scheduling.get_jm_table(problem_name, scenario_name)
    init_gantt = jm_table.initial_gantt()
    delayed_gantt = jm_table.delayed_gantt()
    fixed_gantt, reschedule_gantt, reschedule_time, msg = \
        gantt_chart_operation.check_disturbance(init_gantt, delayed_gantt)
    return jm_table, fixed_gantt, reschedule_gantt, reschedule_time


def compute_shared_norm_params(problem_name=None, scenario_name=None):
    """GT法ランダムサンプリングで共通正規化パラメータを推定"""
    jm_table, fixed_gantt, reschedule_gantt, reschedule_time = get_problem(
        problem_name, scenario_name)
    delayed_gantt = jm_table.delayed_gantt()
    _, rescheduled_rsr_gantt = gantt_chart_operation.create_rsr_gantt(
        fixed_gantt, reschedule_gantt)
    base_gene = gantt_chart_operation.get_gene(rescheduled_rsr_gantt)
    norm_params = evaluation.estimate_normalization_params(
        jm_table, fixed_gantt, reschedule_time,
        delayed_gantt, base_gene, n_samples=200)
    print(f"共通正規化パラメータ: {norm_params}")
    return norm_params


def get_initial_makespan(problem_name=None, scenario_name=None):
    jm_table, fixed_gantt, reschedule_gantt, reschedule_time = get_problem(
        problem_name, scenario_name)
    rsr_gantt, _ = gantt_chart_operation.create_rsr_gantt(fixed_gantt, reschedule_gantt)
    return evaluation.compute_makespan_from_gantt(rsr_gantt)


# ========== 個別実行関数 (並列用) ==========

def run_ga(weights, seed, ngen, norm_params=None, problem_name=None, scenario_name=None,
           track_population=False):
    jm_table, fixed_gantt, reschedule_gantt, reschedule_time = get_problem(
        problem_name, scenario_name)
    random.seed(seed)
    solver = ga_scheduling.GASolver(
        jm_table, fixed_gantt, reschedule_time, weights, pop_size=GA_POP_SIZE)
    _, ms, st, conv_info, history = solver.run(
        ngen=ngen, verbose=False, norm_params=norm_params,
        track_population=track_population)
    # GA の初期個体（original_individual）を active schedule でデコードした評価値。
    # 初期解相当として Pareto 分析で除外する目印。
    baseline = [solver.baseline_ms, solver.baseline_st]
    return {'makespan': ms, 'stability': st, 'convergence': conv_info,
            'history': history, 'baseline': baseline}


def run_ils(weights, seed, perturb_method, max_iterations, norm_params=None,
            active_schedule=False, taillard_acceleration=True,
            path_relink_mode=False, relink_trigger=200,
            pr_ls_strategy=None,
            repair_mode=False, repair_strength=2,
            stagnation_threshold=None, accept_delta=0.05,
            strategy='best',
            problem_name=None, scenario_name=None):
    jm_table, fixed_gantt, reschedule_gantt, reschedule_time = get_problem(
        problem_name, scenario_name)
    random.seed(seed)
    solver = ils_scheduling.ILSSolver(
        jm_table, fixed_gantt, reschedule_gantt, reschedule_time, weights,
        active_schedule=active_schedule,
        taillard_acceleration=taillard_acceleration)
    solver.estimate_normalization_params(n_samples=100, norm_params=norm_params)
    best_orders, _, conv_info, history = solver.run(
        max_iterations=max_iterations, perturb_method=perturb_method, verbose=False,
        path_relink_mode=path_relink_mode, relink_trigger=relink_trigger,
        pr_ls_strategy=pr_ls_strategy,
        repair_mode=repair_mode, repair_strength=repair_strength,
        stagnation_threshold=stagnation_threshold, accept_delta=accept_delta,
        strategy=strategy)
    ms, st = solver.evaluate_pareto(best_orders)
    # ILS は semi-active decoding なので initial_machine_orders の stability は定義上 0。
    # baseline = (init_ms, 0.0)
    init_ms, init_st = solver.evaluate_pareto(solver.initial_machine_orders)
    baseline = [init_ms, init_st]
    return {'makespan': ms, 'stability': st, 'convergence': conv_info,
            'history': history, 'baseline': baseline}


# ========== 可視化ユーティリティ ==========

def _trial_color(i, n):
    return f'C{i % 10}'


def plot_iteration_trace(histories, label, w_label, out_dir):
    """反復ごとの最良メイクスパンと安定性の推移"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    n = len(histories)
    for i, history in enumerate(histories):
        iters = [h['iteration'] for h in history]
        ms_vals = [h['best_makespan'] for h in history]
        st_vals = [h['best_stability'] for h in history]
        axes[0].plot(iters, ms_vals, color=_trial_color(i, n), alpha=0.7, linewidth=1.0,
                     label=f'Trial {i} (MS={ms_vals[-1]})')
        axes[1].plot(iters, st_vals, color=_trial_color(i, n), alpha=0.7, linewidth=1.0,
                     label=f'Trial {i}')

    axes[0].set_xlabel('Iteration'); axes[0].set_ylabel('Best Makespan')
    axes[0].set_title(f'{label}: Best Makespan per Iteration'); axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=7, loc='upper right')
    axes[1].set_xlabel('Iteration'); axes[1].set_ylabel('Best Stability')
    axes[1].set_title(f'{label}: Best Stability per Iteration'); axes[1].grid(True, alpha=0.3)

    fig.suptitle(f'{label} Iteration Trace ({w_label})', fontsize=12, y=1.02)
    fig.tight_layout()
    safe_label = label.replace('(', '').replace(')', '').replace('+', '_')
    fig.savefig(os.path.join(out_dir, f"trace_{safe_label}_{w_label}.png"), dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_trajectory(histories, label, w_label, out_dir, trial_idx=0):
    """ILS探索軌跡: 受理された局所最適解の遷移"""
    history = histories[trial_idx]
    fig, ax = plt.subplots(figsize=(10, 8))

    accepted_points = [(h['ls_makespan'], h['ls_stability']) for h in history if h['accepted']]
    rejected_points = [(h['ls_makespan'], h['ls_stability']) for h in history if not h['accepted']]

    if rejected_points:
        ax.scatter([p[0] for p in rejected_points], [p[1] for p in rejected_points],
                   c='lightgray', s=15, alpha=0.3, zorder=1, label='Rejected')
    ax.scatter([p[0] for p in accepted_points], [p[1] for p in accepted_points],
               c='blue', s=40, alpha=0.7, zorder=3, label='Accepted')

    for i in range(len(accepted_points) - 1):
        x1, y1 = accepted_points[i]
        x2, y2 = accepted_points[i + 1]
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                     arrowprops=dict(arrowstyle='->', color='blue', lw=1.0, alpha=0.5))

    if accepted_points:
        ax.scatter([accepted_points[0][0]], [accepted_points[0][1]],
                   c='green', s=120, marker='*', zorder=4, label='Start')
    best_h = min(history, key=lambda h: h['best_score'])
    ax.scatter([best_h['best_makespan']], [best_h['best_stability']],
               c='red', s=120, marker='*', zorder=4, label='Best')

    ax.set_xlabel('Makespan'); ax.set_ylabel('Stability')
    ax.set_title(f'{label} Search Trajectory (Trial {trial_idx}, {w_label})')
    ax.legend(loc='upper right'); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    safe_label = label.replace('(', '').replace(')', '').replace('+', '_')
    fig.savefig(os.path.join(out_dir, f"trajectory_{safe_label}_trial{trial_idx}_{w_label}.png"), dpi=150)
    plt.close(fig)


def print_method_summary(label, data, init_ms=1080):
    """1手法分のサマリーを出力して文字列を返す
    改善成功試行(makespan < init_ms)のみの統計と全体の改善成功率を表示"""
    ms_list = [d['makespan'] for d in data]
    n_total = len(ms_list)
    improved_data = [d for d in data if d['makespan'] < init_ms]
    n_improved = len(improved_data)

    lines = []
    lines.append(f"\n--- {label} ---")
    lines.append(f"  改善成功率: {n_improved}/{n_total} ({n_improved/n_total*100:.0f}%)")

    if n_improved > 0:
        imp_ms = [d['makespan'] for d in improved_data]
        imp_st = [d['stability'] for d in improved_data]
        imp_cpu = [d['convergence']['cpu_time'] for d in improved_data]
        imp_cpu_total = [d['convergence']['total_cpu_time'] for d in improved_data]
        imp_evals = [d['convergence']['evaluations'] for d in improved_data]
        lines.append(f"  [改善成功試行のみ]")
        lines.append(f"  Makespan:  平均={np.mean(imp_ms):.1f}, 最良={min(imp_ms)}, "
                     f"最悪={max(imp_ms)}, std={np.std(imp_ms):.1f}")
        lines.append(f"  Stability: 平均={np.mean(imp_st):.2f}, 最良={min(imp_st):.2f}, "
                     f"最悪={max(imp_st):.2f}")
        lines.append(f"  最良解到達CPU時間: 平均={np.mean(imp_cpu):.2f}s")
        lines.append(f"  全体CPU時間: 平均={np.mean(imp_cpu_total):.2f}s")
        lines.append(f"  最良解到達評価回数: 平均={np.mean(imp_evals):.0f}")
    else:
        lines.append(f"  改善成功試行なし")

    text = "\n".join(lines)
    print(text)
    return text
