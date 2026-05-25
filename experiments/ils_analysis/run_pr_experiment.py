#!/usr/bin/env python3
"""
PR 挙動確認実験

=== 目的 ===
Path Relinking (PR) が Pareto front の stability 軸端を拡張するかを検証する。

PR は current → initial_machine_orders 方向に経路をたどる。
- current: ILS 現在探索点 → 多様な出発点（best固定だと同一経路の再探索になるため）
- initial: 変更なし      → makespan 悪・stability 最大付近

この方向性から「stability 高い領域の Pareto front を補完する」効果が期待される。

=== 比較条件 ===
  (A) insert        : ベースライン
  (B) insert_repair : insert + repair
  (C) insert_PR     : insert + PR

=== 判断基準 ===
  - Pareto front overlay で stability 軸端の拡張を確認
  - 領域別 HV (low/mid/high stab) で PR の寄与領域を特定
  - HV 全体で net 効果を評価
"""

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from experiment_utils import (
    setup_output_dir, compute_shared_norm_params,
    get_initial_makespan, run_ils,
    ILS_MAX_ITER,
)


N_TRIALS = 10
RELINK_TRIGGER = 50
REPAIR_TRIGGER = 50
REPAIR_STRENGTH = 1   # Stage 2-A 確定値 (insert 用)

PROBLEM_SETS = [
    ('mt10', 'mt10_delay60'),
    ('la36', 'la36_delay148'),
    ('la36', 'la36_multi3_x15'),
]

WEIGHTS_LIST = [
    [0.9, 0.1],
    [0.8, 0.2],
]

ILS_METHODS = {
    'insert':        {'repair_mode': False, 'path_relink_mode': False},
    'insert_repair': {'repair_mode': True,  'path_relink_mode': False},
    'insert_PR':     {'repair_mode': False, 'path_relink_mode': True},
}

METHOD_LABELS = {
    'insert':        'insert',
    'insert_repair': 'insert+repair',
    'insert_PR':     'insert+PR',
}

METHOD_COLORS = {
    'insert':        'tab:gray',
    'insert_repair': 'tab:blue',
    'insert_PR':     'tab:orange',
}


# ========== 個別実行 ==========

def _run_method(method_key, weights, seed, norm_params, problem_name, scenario_name,
                relink_trigger):
    cfg = ILS_METHODS[method_key]
    return run_ils(
        weights, seed, 'insert', ILS_MAX_ITER, norm_params,
        strategy='best',
        initial_strength=2, max_strength=5,
        repair_mode=cfg['repair_mode'],
        repair_trigger=REPAIR_TRIGGER,
        repair_strength=REPAIR_STRENGTH,
        path_relink_mode=cfg['path_relink_mode'],
        relink_trigger=relink_trigger,
        problem_name=problem_name, scenario_name=scenario_name)


# ========== Pareto front ユーティリティ ==========

def pareto_front_2d(points):
    """(makespan, stability) の Pareto front（両軸最小化）"""
    if len(points) == 0:
        return np.empty((0, 2))
    pts = np.array(points)
    idx = np.argsort(pts[:, 0])
    pts = pts[idx]
    front = [pts[0]]
    for p in pts[1:]:
        if p[1] < front[-1][1]:
            front.append(p)
    return np.array(front)


def hv_2d(pareto_pts, ref):
    """2D ハイパーボリューム"""
    if len(pareto_pts) == 0:
        return 0.0
    pts = pareto_pts[pareto_pts[:, 0] < ref[0]]
    pts = pts[pts[:, 1] < ref[1]]
    if len(pts) == 0:
        return 0.0
    # ms 降順にソートして右→左スイープ（analyze_core.hypervolume_2d と同じ方式）
    pts = pts[np.argsort(pts[:, 0])[::-1]]
    hv = 0.0
    prev_ms = ref[0]
    for ms, st in pts:
        hv += (prev_ms - ms) * (ref[1] - st)
        prev_ms = ms
    return hv


# ========== プロット ==========

def plot_pareto_overlay(all_results, methods, w_label, out_dir, prob_label, init_ms):
    """全手法の per-trial Pareto front 重ね合わせ"""
    fig, ax = plt.subplots(figsize=(10, 7))

    for mk in methods:
        color = METHOD_COLORS[mk]
        # per-trial (薄線)
        all_accepted = []
        for hist in all_results.get(f'{mk}_histories', []):
            if hist is None:
                continue
            pts = [(h[0], h[1]) for h in hist if h[2]]
            all_accepted.extend(pts)
            if pts:
                front = pareto_front_2d(pts)
                ax.plot(front[:, 0], front[:, 1], color=color, alpha=0.2, lw=0.8)
        # union Pareto (太線)
        if all_accepted:
            union_front = pareto_front_2d(all_accepted)
            ax.plot(union_front[:, 0], union_front[:, 1],
                    color=color, lw=2.5, alpha=0.9, label=METHOD_LABELS[mk])

    ax.axvline(init_ms, color='gray', lw=1, ls='--', label=f'init MS={init_ms}')
    ax.set_xlabel('Makespan')
    ax.set_ylabel('Stability')
    ax.set_title(f'{prob_label}: Pareto front overlay ({w_label})')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f'pareto_overlay_{w_label}.png'), dpi=150)
    plt.close(fig)


def plot_region_hv(all_results, methods, w_label, out_dir, prob_label, init_ms):
    """stability を 3 領域に分割した領域別 HV 比較"""
    all_pts = []
    for mk in methods:
        for hist in all_results.get(f'{mk}_histories', []):
            if hist is None:
                continue
            all_pts.extend([(h[0], h[1]) for h in hist if h[2]])
    if not all_pts:
        return
    union_front = pareto_front_2d(all_pts)
    stab_min = union_front[:, 1].min()
    stab_max = union_front[:, 1].max()
    span = stab_max - stab_min if stab_max > stab_min else 1.0
    boundaries = [stab_min, stab_min + span / 3, stab_min + 2 * span / 3, stab_max]
    region_names = ['low_stab', 'mid_stab', 'high_stab']

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for r_i, (lo, hi, name) in enumerate(zip(boundaries[:-1], boundaries[1:], region_names)):
        ax = axes[r_i]
        ref = (init_ms * 1.05, hi + span * 0.02)

        hvs_per_method = []
        labels = []
        for mk in methods:
            trial_hvs = []
            for hist in all_results.get(f'{mk}_histories', []):
                if hist is None:
                    continue
                pts = [(h[0], h[1]) for h in hist if h[2] and lo <= h[1] <= hi]
                front = pareto_front_2d(pts) if pts else np.empty((0, 2))
                trial_hvs.append(hv_2d(front, ref))
            hvs_per_method.append(trial_hvs)
            labels.append(METHOD_LABELS[mk])

        bp = ax.boxplot(hvs_per_method, labels=labels, patch_artist=True)
        for patch, mk in zip(bp['boxes'], methods):
            patch.set_facecolor(METHOD_COLORS[mk])
            patch.set_alpha(0.6)
        ax.set_title(f'{name}\nstab ∈ [{lo:.2f}, {hi:.2f}]')
        ax.set_ylabel('HV')
        ax.grid(True, axis='y', alpha=0.3)
        ax.tick_params(axis='x', rotation=15)

    fig.suptitle(f'{prob_label}: 領域別 HV ({w_label})', fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f'region_hv_{w_label}.png'), dpi=150)
    plt.close(fig)


def plot_hv_distribution(all_results, methods, w_label, out_dir, prob_label, ref_point):
    """全体 HV の分布比較"""
    fig, ax = plt.subplots(figsize=(8, 5))
    hvs_per_method, labels = [], []
    for mk in methods:
        hvs = []
        for hist in all_results.get(f'{mk}_histories', []):
            if hist is None:
                continue
            pts = [(h[0], h[1]) for h in hist if h[2]]
            front = pareto_front_2d(pts) if pts else np.empty((0, 2))
            hvs.append(hv_2d(front, ref_point))
        hvs_per_method.append(hvs)
        labels.append(METHOD_LABELS[mk])

    bp = ax.boxplot(hvs_per_method, labels=labels, patch_artist=True)
    for patch, mk in zip(bp['boxes'], methods):
        patch.set_facecolor(METHOD_COLORS[mk])
        patch.set_alpha(0.6)
    ax.set_ylabel('HV (全体)')
    ax.set_title(f'{prob_label}: HV 分布 ({w_label})')
    ax.grid(True, axis='y', alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f'hv_dist_{w_label}.png'), dpi=150)
    plt.close(fig)


def plot_anytime(all_results, methods, w_label, out_dir, prob_label):
    """CPU 時間軸での score / makespan / stability 推移"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    # history format: [ls_ms, ls_st, accepted, cpu_time, perturb_used]
    field_indices = [6, 4, 5]  # best_score, best_ms, best_stab は history dict から取得
    ylabels = ['Weighted Score', 'Makespan', 'Stability']

    for mk in methods:
        color = METHOD_COLORS[mk]
        # history は dict 形式で保存されているので dict アクセス
        raw_hists = [h for h in all_results.get(f'{mk}_raw_histories', [])
                     if h is not None]
        if not raw_hists:
            continue
        t_max = min(h[-1]['cpu_time'] for h in raw_hists)
        t_grid = np.linspace(0, t_max, 300)

        for fi, (field, ylabel) in enumerate(
                zip(['best_score', 'best_makespan', 'best_stability'], ylabels)):
            curves = []
            for hist in raw_hists:
                times = np.array([h['cpu_time'] for h in hist])
                vals = np.array([h[field] for h in hist])
                curves.append(np.interp(t_grid, times, vals))
            axes[fi].plot(t_grid, np.mean(curves, axis=0), color=color, lw=2,
                          label=METHOD_LABELS[mk])
            for c in curves:
                axes[fi].plot(t_grid, c, color=color, alpha=0.12, lw=0.7)

    for ax, ylabel, title in zip(
            axes, ylabels,
            ['Score vs Time', 'Makespan vs Time', 'Stability vs Time']):
        ax.set_xlabel('CPU Time (s)')
        ax.set_ylabel(ylabel)
        ax.set_title(f'{prob_label}: {title} ({w_label})')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f'anytime_{w_label}.png'), dpi=150)
    plt.close(fig)


def _t_grid_from_histories(all_results, methods, n_points=60):
    """全手法で完走した cpu_time の最小値を t_max として等差グリッドを返す"""
    t_mins = []
    for mk in methods:
        hists = [h for h in all_results.get(f'{mk}_histories', [])
                 if h is not None and len(h) > 0]
        if hists:
            t_mins.append(min(h[-1][3] for h in hists))
    if not t_mins:
        return None
    t_max = min(t_mins)
    return np.linspace(t_max * 0.02, t_max, n_points)


def _anytime_hv_curves(hists, ref_point, t_grid):
    """compact history [[ls_ms, ls_st, accepted, cpu_time, ...], ...] から
    各 trial の anytime HV 曲線 (n_trials × n_points) を返す"""
    curves = []
    for hist in hists:
        if hist is None or len(hist) == 0:
            curves.append(np.zeros(len(t_grid)))
            continue
        curve = []
        for t in t_grid:
            pts = [(h[0], h[1]) for h in hist if h[3] <= t]
            front = pareto_front_2d(pts) if pts else np.empty((0, 2))
            curve.append(hv_2d(front, ref_point))
        curves.append(curve)
    return np.array(curves)


def _union_hv_curve(hists, ref_point, t_grid):
    """全 trial の union Pareto による anytime HV 曲線 (n_points,)"""
    curve = []
    for t in t_grid:
        all_pts = []
        for hist in hists:
            if hist is None:
                continue
            all_pts.extend((h[0], h[1]) for h in hist if h[3] <= t)
        front = pareto_front_2d(all_pts) if all_pts else np.empty((0, 2))
        curve.append(hv_2d(front, ref_point))
    return np.array(curve)


def plot_anytime_hv_pr(all_results, methods, w_label, out_dir, prob_label,
                       ref_point, n_points=60):
    """Anytime HV: per-trial median+IQR (left) と union HV (right)"""
    t_grid = _t_grid_from_histories(all_results, methods, n_points)
    if t_grid is None:
        return

    fig, (ax_pt, ax_un) = plt.subplots(1, 2, figsize=(16, 5), sharey=True)

    for mk in methods:
        color = METHOD_COLORS[mk]
        hists = [h for h in all_results.get(f'{mk}_histories', []) if h is not None]
        if not hists:
            continue

        arr = _anytime_hv_curves(hists, ref_point, t_grid)
        med = np.median(arr, axis=0)
        q25 = np.percentile(arr, 25, axis=0)
        q75 = np.percentile(arr, 75, axis=0)
        ax_pt.plot(t_grid, med, color=color, lw=2, label=METHOD_LABELS[mk])
        ax_pt.fill_between(t_grid, q25, q75, color=color, alpha=0.2)

        ax_un.plot(t_grid, _union_hv_curve(hists, ref_point, t_grid),
                   color=color, lw=2, label=METHOD_LABELS[mk])

    for ax, title in [(ax_pt, 'Per-trial HV (median±IQR)'), (ax_un, 'Union HV')]:
        ax.set_xlabel('CPU Time (s)')
        ax.set_ylabel('Hypervolume')
        ax.set_title(f'{prob_label}: {title} ({w_label})')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    fig.suptitle('Anytime HV', fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f'anytime_hv_{w_label}.png'), dpi=150)
    plt.close(fig)


def plot_anytime_region_hv_pr(all_results, methods, w_label, out_dir, prob_label,
                               init_ms, n_points=60):
    """安定性 3 領域別 anytime HV: 2×3 パネル
    上段: per-trial median, 下段: union
    """
    # 全手法の全 accepted 点から領域境界を決定
    all_pts = []
    for mk in methods:
        for hist in all_results.get(f'{mk}_histories', []):
            if hist is None:
                continue
            all_pts.extend((h[0], h[1]) for h in hist if h[2])
    if not all_pts:
        return
    union_front = pareto_front_2d(all_pts)
    stab_min = float(union_front[:, 1].min())
    stab_max = float(union_front[:, 1].max())
    span = max(stab_max - stab_min, 1.0)
    boundaries = [stab_min, stab_min + span / 3, stab_min + 2 * span / 3, stab_max]
    region_names = ['low_stab', 'mid_stab', 'high_stab']

    t_grid = _t_grid_from_histories(all_results, methods, n_points)
    if t_grid is None:
        return

    fig, axes = plt.subplots(2, 3, figsize=(18, 10), sharey='row')

    for r_i, (lo, hi, name) in enumerate(zip(boundaries[:-1], boundaries[1:], region_names)):
        stab_width = max(hi - lo, 1e-6)
        ref = (init_ms * 1.05, hi + stab_width * 0.02)

        for mk in methods:
            color = METHOD_COLORS[mk]
            hists = [h for h in all_results.get(f'{mk}_histories', []) if h is not None]
            if not hists:
                continue

            # per-trial
            per_trial = []
            for hist in hists:
                curve = []
                for t in t_grid:
                    pts = [(h[0], h[1]) for h in hist if h[3] <= t and lo <= h[1] <= hi]
                    front = pareto_front_2d(pts) if pts else np.empty((0, 2))
                    curve.append(hv_2d(front, ref))
                per_trial.append(curve)
            arr = np.array(per_trial)
            med = np.median(arr, axis=0)
            q25 = np.percentile(arr, 25, axis=0)
            q75 = np.percentile(arr, 75, axis=0)
            axes[0, r_i].plot(t_grid, med, color=color, lw=2, label=METHOD_LABELS[mk])
            axes[0, r_i].fill_between(t_grid, q25, q75, color=color, alpha=0.2)

            # union
            union_curve = []
            for t in t_grid:
                region_pts = []
                for hist in hists:
                    region_pts.extend(
                        (h[0], h[1]) for h in hist if h[3] <= t and lo <= h[1] <= hi)
                front = pareto_front_2d(region_pts) if region_pts else np.empty((0, 2))
                union_curve.append(hv_2d(front, ref))
            axes[1, r_i].plot(t_grid, union_curve, color=color, lw=2,
                              label=METHOD_LABELS[mk])

        for row_i, row_title in [(0, 'Per-trial median±IQR'), (1, 'Union')]:
            axes[row_i, r_i].set_xlabel('CPU Time (s)')
            axes[row_i, r_i].set_ylabel('HV')
            axes[row_i, r_i].set_title(
                f'{name} [{lo:.1f},{hi:.1f}]\n{row_title}')
            axes[row_i, r_i].legend(fontsize=8)
            axes[row_i, r_i].grid(True, alpha=0.3)

    fig.suptitle(f'{prob_label}: Anytime Region HV ({w_label})', fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f'anytime_region_hv_{w_label}.png'), dpi=150)
    plt.close(fig)


# ========== 統計 ==========

def compute_stats(all_results, methods, w_label, out_dir, prob_label, ref_point):
    lines = [f"\n=== PR 効果比較 ({prob_label}, {w_label}) ==="]
    col_w = (22, 14)
    lines.append(f"  {'指標':<{col_w[0]}} "
                 + " ".join(f"{METHOD_LABELS[mk]:>{col_w[1]}}" for mk in methods))
    lines.append("  " + "-" * (col_w[0] + (col_w[1] + 1) * len(methods)))

    def row(label, values, fmt='.2f'):
        cells = [f"{v:{fmt}}" for v in values]
        return (f"  {label:<{col_w[0]}} "
                + " ".join(f"{c:>{col_w[1]}}" for c in cells))

    def collect(key):
        return [[d[key] for d in all_results[mk]
                 if d is not None and 'error' not in d] for mk in methods]

    ms_lists = collect('makespan')
    st_lists = collect('stability')
    cpu_lists = [[d['convergence']['total_cpu_time'] for d in all_results[mk]
                  if d is not None and 'error' not in d] for mk in methods]

    lines.append(row('Makespan 平均',   [np.mean(v) if v else 0 for v in ms_lists], '.1f'))
    lines.append(row('Makespan 最良',   [min(v)     if v else 0 for v in ms_lists], '.0f'))
    lines.append(row('Stability 平均',  [np.mean(v) if v else 0 for v in st_lists], '.3f'))
    lines.append(row('Stability 最良',  [min(v)     if v else 0 for v in st_lists], '.3f'))
    lines.append(row('CPU time 平均(s)', [np.mean(v) if v else 0 for v in cpu_lists], '.1f'))
    lines.append(row('CPU time 最大(s)', [max(v)     if v else 0 for v in cpu_lists], '.1f'))

    hv_lists = []
    for mk in methods:
        hvs = []
        for hist in all_results.get(f'{mk}_histories', []):
            if hist is None:
                continue
            pts = [(h[0], h[1]) for h in hist if h[2]]
            front = pareto_front_2d(pts) if pts else np.empty((0, 2))
            hvs.append(hv_2d(front, ref_point))
        hv_lists.append(hvs)
    lines.append(row('HV median', [float(np.median(v)) if v else 0 for v in hv_lists], '.1f'))

    # PR / repair 発動回数と 1 回あたりの平均コスト
    lines.append("")
    lines.append("  --- PR / repair 発動 (trial 平均) ---")
    for mk in methods:
        hists = [h for h in all_results.get(f'{mk}_histories', []) if h is not None]
        if not hists:
            continue
        pr_counts     = [sum(1 for h in hist if h[4] == 'path_relink') for hist in hists]
        repair_counts = [sum(1 for h in hist if h[4] == 'repair')      for hist in hists]
        lines.append(f"  {METHOD_LABELS[mk]:<22}  PR={np.mean(pr_counts):.1f}回  "
                     f"repair={np.mean(repair_counts):.1f}回")

    text = "\n".join(lines)
    print(text)
    with open(os.path.join(out_dir, f'stats_{w_label}.txt'), 'w', encoding='utf-8') as f:
        f.write(text + "\n")
    return lines


# ========== ランナー ==========

def run_problem_experiment(problem_name, scenario_name, weights, methods, out_dir,
                           relink_trigger):
    prob_label = f"{problem_name}_{scenario_name}"
    w_label = f"eff={weights[0]}_stab={weights[1]}"
    print(f"\n{'='*70}")
    print(f"問題: {prob_label}, weights={weights}")
    print(f"{'='*70}")

    norm_params = compute_shared_norm_params(problem_name, scenario_name)
    init_ms = get_initial_makespan(problem_name, scenario_name)
    print(f"  初期解メイクスパン: {init_ms}")

    futures = {}
    with ProcessPoolExecutor() as executor:
        for trial in range(N_TRIALS):
            seed = trial * 100 + 7
            for mk in methods:
                f = executor.submit(_run_method, mk, weights, seed,
                                    norm_params, problem_name, scenario_name,
                                    relink_trigger)
                futures[f] = (mk, trial)

        all_results = {
            'problem': problem_name, 'scenario': scenario_name,
            'weights': weights, 'init_makespan': init_ms,
        }
        for mk in methods:
            all_results[mk] = [None] * N_TRIALS
            all_results[f'{mk}_histories'] = [None] * N_TRIALS
            all_results[f'{mk}_raw_histories'] = [None] * N_TRIALS

        for future in as_completed(futures):
            mk, trial = futures[future]
            try:
                r = future.result()
                all_results[mk][trial] = {
                    'trial': trial,
                    'makespan': r['makespan'],
                    'stability': r['stability'],
                    'convergence': r['convergence'],
                }
                # コンパクト形式: [ls_ms, ls_st, accepted, cpu_time, perturb_used]
                all_results[f'{mk}_histories'][trial] = [
                    [h['ls_makespan'], h['ls_stability'], h['accepted'],
                     h['cpu_time'], h['perturb_used']]
                    for h in r['history']
                ]
                # anytime プロット用にフル形式も保持（JSON 非保存）
                all_results[f'{mk}_raw_histories'][trial] = r['history']
                print(f"  Trial {trial:2d} {METHOD_LABELS[mk]:24s}: "
                      f"MS={r['makespan']}, Stab={r['stability']:.3f}")
            except Exception as e:
                import traceback
                print(f"  Trial {trial:2d} {METHOD_LABELS[mk]:24s}: ERROR - {e}")
                traceback.print_exc()
                all_results[mk][trial] = {'trial': trial, 'error': str(e)}

    prob_dir = os.path.join(out_dir, prob_label)
    os.makedirs(prob_dir, exist_ok=True)

    # JSON 保存（raw_histories は除外）
    save_data = {k: v for k, v in all_results.items()
                 if not k.endswith('_raw_histories')}
    with open(os.path.join(prob_dir, f'results_{w_label}.json'), 'w') as f:
        json.dump(save_data, f, ensure_ascii=False)

    all_stab = [d['stability'] for mk in methods for d in all_results[mk]
                if d is not None and 'error' not in d]
    stab_ref = (max(all_stab) * 1.05) if all_stab else 30.0
    ref_point = (init_ms * 1.05, stab_ref)

    plot_pareto_overlay(all_results, methods, w_label, prob_dir, prob_label, init_ms)
    plot_region_hv(all_results, methods, w_label, prob_dir, prob_label, init_ms)
    plot_hv_distribution(all_results, methods, w_label, prob_dir, prob_label, ref_point)
    plot_anytime(all_results, methods, w_label, prob_dir, prob_label)
    plot_anytime_hv_pr(all_results, methods, w_label, prob_dir, prob_label, ref_point)
    plot_anytime_region_hv_pr(all_results, methods, w_label, prob_dir, prob_label, init_ms)
    stats = compute_stats(all_results, methods, w_label, prob_dir, prob_label, ref_point)

    return all_results, stats


# ========== エントリポイント ==========

def main():
    global N_TRIALS
    parser = argparse.ArgumentParser(description='PR 挙動確認実験')
    parser.add_argument('--problems', nargs='+', default=None,
                        help='問題セット (例: mt10:mt10_delay60)')
    parser.add_argument('--weights', nargs='+', default=['0.9,0.1', '0.8,0.2'])
    parser.add_argument('--methods', nargs='+', default=list(ILS_METHODS.keys()),
                        choices=list(ILS_METHODS.keys()))
    parser.add_argument('--trials', type=int, default=N_TRIALS)
    parser.add_argument('--relink-trigger', type=int, default=RELINK_TRIGGER)
    args = parser.parse_args()

    N_TRIALS = args.trials
    problem_sets = ([tuple(p.split(':')) for p in args.problems]
                    if args.problems else PROBLEM_SETS)
    weight_list = [[float(x) for x in w.split(',')] for w in args.weights]

    out_dir = setup_output_dir('pr_experiment', base_dir=os.path.dirname(__file__))
    print(f"出力先: {out_dir}")
    print(f"問題: {problem_sets}")
    print(f"手法: {args.methods}")
    print(f"trials={N_TRIALS}, ILS_MAX_ITER={ILS_MAX_ITER}")
    print(f"relink_trigger={args.relink_trigger}, repair_trigger={REPAIR_TRIGGER}, "
          f"repair_strength={REPAIR_STRENGTH}")

    all_summaries = []
    for problem_name, scenario_name in problem_sets:
        for weights in weight_list:
            try:
                _, stats = run_problem_experiment(
                    problem_name, scenario_name, weights,
                    args.methods, out_dir, args.relink_trigger)
                all_summaries.extend(stats)
            except Exception as e:
                import traceback
                print(f"\nERROR: {problem_name}/{scenario_name}: {e}")
                traceback.print_exc()

    summary_path = os.path.join(out_dir, 'cross_summary.txt')
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("PR 挙動確認実験 横断サマリー\n")
        f.write(f"relink_trigger={args.relink_trigger}, "
                f"repair_trigger={REPAIR_TRIGGER}, repair_strength={REPAIR_STRENGTH}\n")
        f.write("=" * 70 + "\n\n")
        for line in all_summaries:
            f.write(str(line) + "\n")
    print(f"\n横断サマリー: {summary_path}")
    print(f"全実験完了。結果は {out_dir} に保存されました。")


if __name__ == '__main__':
    main()
