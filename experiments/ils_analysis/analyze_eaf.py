#!/usr/bin/env python3
"""
repair 摂動実験の多目的最適化評価分析（Pareto front ベース）

各 trial の履歴に含まれる全 LS 結果から per-trial Pareto front を抽出し、
trial ベースの EAF（Empirical Attainment Function）で手法間比較を行う。

EAF の定義:
  α(p) = (p を dominate する trial の割合)
  ここで「trial が p を dominate」とは、その trial の探索で訪れた
  いずれかの点が p を weakly dominate すること。

最終解1点だけの EAF では捉えられない「探索領域の到達確率分布」を定量化する。

使い方:
  python analyze_eaf.py <results_dir>
  例: python analyze_eaf.py experiments/ils_analysis/results/repair_perturb_20260417_220610
"""

import argparse
import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


METHODS = ['ILS_swap', 'ILS_insert', 'ILS_swap_repair', 'ILS_insert_repair']

METHOD_LABELS = {
    'ILS_swap':          'swap',
    'ILS_insert':        'insert',
    'ILS_swap_repair':   'swap+repair',
    'ILS_insert_repair': 'insert+repair',
}

METHOD_COLORS = {
    'ILS_swap':          'tab:blue',
    'ILS_insert':        'tab:orange',
    'ILS_swap_repair':   'tab:cyan',
    'ILS_insert_repair': 'tab:red',
}

PAIRS = [
    ('ILS_swap',   'ILS_swap_repair',   'swap'),
    ('ILS_insert', 'ILS_insert_repair', 'insert'),
]

# 補助比較: swap vs insert（どっちの主摂動が強いか）
SUPPLEMENTARY_PAIRS = [
    ('ILS_swap',        'ILS_insert',        'base_swap_vs_insert'),
    ('ILS_swap_repair', 'ILS_insert_repair', 'repair_swap_vs_insert'),
]

GRID_N = 150
ATTAIN_LEVELS = [0.25, 0.5, 0.75]
ATTAIN_STYLES = {0.25: ':', 0.5: '--', 0.75: '-'}


# ========== データ読み込み ==========

def load_trial_points(data, method, use_accepted_only=False):
    """各 trial の (ms, st) 点列を返す。履歴が無ければ final 1 点で代替。

    Returns: list[np.ndarray (n, 2)] の長さ n_trials。
    """
    hist_key = f'{method}_histories'
    if hist_key in data and data[hist_key] is not None:
        result = []
        for trial_hist in data[hist_key]:
            if trial_hist is None:
                result.append(np.zeros((0, 2)))
                continue
            pts = []
            for entry in trial_hist:
                # entry = [ls_ms, ls_st, accepted]
                if len(entry) >= 2:
                    ls_ms, ls_st = float(entry[0]), float(entry[1])
                    accepted = bool(entry[2]) if len(entry) >= 3 else True
                else:
                    continue
                if not np.isfinite(ls_ms) or not np.isfinite(ls_st):
                    continue
                if use_accepted_only and not accepted:
                    continue
                pts.append([ls_ms, ls_st])
            result.append(np.array(pts) if pts else np.zeros((0, 2)))
        return result

    # 履歴なしの古い JSON: final の 1 点だけ
    valid = data.get(method, [])
    result = []
    for d in valid:
        if d is None or 'error' in d:
            result.append(np.zeros((0, 2)))
        else:
            result.append(np.array([[d['makespan'], d['stability']]]))
    return result


# ========== Pareto / 指標計算 ==========

def pareto_front_2d(points):
    """2D minimization の Pareto front を抽出"""
    if len(points) == 0:
        return points
    points = np.asarray(points)
    idx = np.lexsort((points[:, 1], points[:, 0]))
    sorted_pts = points[idx]
    pareto = [sorted_pts[0]]
    for p in sorted_pts[1:]:
        if p[1] < pareto[-1][1]:
            pareto.append(p)
    return np.array(pareto)


def hypervolume_2d(points, ref):
    """2D hypervolume (minimization). ref = (MS_max, Stab_max) 参照点"""
    if len(points) == 0:
        return 0.0
    pareto = pareto_front_2d(points)
    pareto = pareto[np.argsort(pareto[:, 0])]
    hv = 0.0
    prev_x = ref[0]
    for p in pareto[::-1]:
        if p[0] >= ref[0] or p[1] >= ref[1]:
            continue
        hv += (prev_x - p[0]) * (ref[1] - p[1])
        prev_x = p[0]
    return hv


def c_metric(A_front, B_front):
    """Zitzler & Thiele の C-metric (coverage): A が弱く dominate する B の割合"""
    if len(B_front) == 0:
        return 0.0
    covered = 0
    for b in B_front:
        for a in A_front:
            if a[0] <= b[0] and a[1] <= b[1]:
                covered += 1
                break
    return covered / len(B_front)


def make_grid(all_points, pad_frac=0.05, include_ms=None):
    """全点をカバーする 2D グリッド

    include_ms を指定すると、その MS 値までグリッドを伸ばす
    （init_ms まで pcolormesh で描画させるため）。
    """
    if len(all_points) == 0:
        return np.array([0.0, 1.0]), np.array([0.0, 1.0])
    ms_min, ms_max = all_points[:, 0].min(), all_points[:, 0].max()
    if include_ms is not None:
        ms_max = max(ms_max, include_ms)
    st_min, st_max = all_points[:, 1].min(), all_points[:, 1].max()
    ms_pad = max((ms_max - ms_min) * pad_frac, 1.0)
    st_pad = max((st_max - st_min) * pad_frac, 0.1)
    grid_ms = np.linspace(ms_min - ms_pad, ms_max + ms_pad, GRID_N)
    grid_st = np.linspace(st_min - st_pad, st_max + st_pad, GRID_N)
    return grid_ms, grid_st


def attainment_function_trials(trial_points_list, grid_ms, grid_st):
    """Trial-based EAF

    各 trial について「その trial の Pareto front が dominate するグリッド領域」を
    計算し、trial 数で割って確率（attainment probability）にする。
    """
    n_trials = len(trial_points_list)
    if n_trials == 0:
        return np.zeros((len(grid_ms), len(grid_st)))
    MS, ST = np.meshgrid(grid_ms, grid_st, indexing='ij')
    attain_count = np.zeros_like(MS, dtype=float)
    for trial_pts in trial_points_list:
        if len(trial_pts) == 0:
            continue
        pf = pareto_front_2d(trial_pts)
        trial_mask = np.zeros_like(MS, dtype=bool)
        for p in pf:
            trial_mask |= ((p[0] <= MS) & (p[1] <= ST))
        attain_count += trial_mask.astype(float)
    return attain_count / n_trials


def attainment_front_line(attain, grid_ms, grid_st, level):
    """α >= level の領域の「左下境界」の step 化座標を返す"""
    mask = attain >= level
    xs, ys = [], []
    for j in range(len(grid_st)):
        col = mask[:, j]
        if col.any():
            i = np.argmax(col)
            xs.append(grid_ms[i])
            ys.append(grid_st[j])
    return np.array(xs), np.array(ys)


# ========== プロット ==========

def plot_attainment_surfaces(trial_pts_by_method, methods, title, outpath,
                              init_ms=None):
    """手法別の 25/50/75% attainment surface + union Pareto の散布"""
    all_pts = []
    for m in methods:
        for trial in trial_pts_by_method[m]:
            if len(trial) > 0:
                all_pts.append(trial)
    if not all_pts:
        return
    all_points = np.concatenate(all_pts)
    grid_ms, grid_st = make_grid(all_points, include_ms=init_ms)

    fig, ax = plt.subplots(figsize=(11, 8))
    for m in methods:
        trial_pts = trial_pts_by_method[m]
        if not any(len(t) > 0 for t in trial_pts):
            continue
        color = METHOD_COLORS[m]
        # Union Pareto front (trial を跨いだ非劣解)
        combined = np.concatenate([t for t in trial_pts if len(t) > 0])
        union_pf = pareto_front_2d(combined)
        if len(union_pf) > 0:
            ax.scatter(union_pf[:, 0], union_pf[:, 1], color=color,
                       s=30, alpha=0.6, edgecolors='black', linewidths=0.3,
                       label=f"{METHOD_LABELS[m]} (union Pareto, n={len(union_pf)})",
                       zorder=3)
        # Attainment surfaces
        attain = attainment_function_trials(trial_pts, grid_ms, grid_st)
        for lvl in ATTAIN_LEVELS:
            xs, ys = attainment_front_line(attain, grid_ms, grid_st, lvl)
            if len(xs) > 0:
                ax.step(xs, ys, color=color, alpha=0.6,
                        linestyle=ATTAIN_STYLES[lvl], linewidth=1.3,
                        where='pre', zorder=2)

    # 軸を Pareto 点ベースで引き締める
    union_pareto = _collect_union_pareto(trial_pts_by_method, methods)
    xlim, ylim = _compute_tight_axes(union_pareto, init_ms)
    if xlim: ax.set_xlim(xlim)
    if ylim: ax.set_ylim(ylim)
    _add_initial_marker(ax, init_ms)

    from matplotlib.lines import Line2D
    style_legend = [
        Line2D([0], [0], color='gray', linestyle=':',  label='25% attainment'),
        Line2D([0], [0], color='gray', linestyle='--', label='50% attainment'),
        Line2D([0], [0], color='gray', linestyle='-',  label='75% attainment'),
    ]
    leg1 = ax.legend(loc='upper right', fontsize=8, title='method')
    ax.add_artist(leg1)
    ax.legend(handles=style_legend, loc='lower left', fontsize=8, title='attainment')

    ax.set_xlabel('Makespan')
    ax.set_ylabel('Stability')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


def _split_pareto_by_overlap(pf_a, pf_b, tol_ms=0.5, tol_st=0.05):
    """2 つの Pareto set を (A のみ, B のみ, 共通) に分割する。

    浮動小数の誤差を許容するため座標毎にトレランスを設定。
    """
    overlap = []
    a_only = []
    b_only_mask = np.ones(len(pf_b), dtype=bool)
    for p in pf_a:
        matched = -1
        for j, q in enumerate(pf_b):
            if abs(p[0] - q[0]) <= tol_ms and abs(p[1] - q[1]) <= tol_st:
                matched = j
                break
        if matched >= 0:
            overlap.append(p)
            b_only_mask[matched] = False
        else:
            a_only.append(p)
    b_only = pf_b[b_only_mask] if len(pf_b) > 0 else pf_b
    return (np.array(a_only) if a_only else np.zeros((0, 2)),
            np.array(b_only),
            np.array(overlap) if overlap else np.zeros((0, 2)))


def plot_diff_eaf(trial_pts_a, trial_pts_b, label_a, label_b, title, outpath,
                   init_ms=None):
    """Differential EAF: EAF(A) - EAF(B) を heatmap 表示

    赤: A の方が高確率で dominate、青: B の方が高確率、白: 拮抗。
    散布点は union Pareto を (A のみ=赤, B のみ=青, 共通=白丸) に分けて描画。
    """
    all_pts_list = []
    for lst in [trial_pts_a, trial_pts_b]:
        for trial in lst:
            if len(trial) > 0:
                all_pts_list.append(trial)
    if not all_pts_list:
        return
    all_points = np.concatenate(all_pts_list)
    grid_ms, grid_st = make_grid(all_points, include_ms=init_ms)
    attain_a = attainment_function_trials(trial_pts_a, grid_ms, grid_st)
    attain_b = attainment_function_trials(trial_pts_b, grid_ms, grid_st)
    diff = attain_a - attain_b

    fig, ax = plt.subplots(figsize=(11, 8))
    im = ax.pcolormesh(grid_ms, grid_st, diff.T, cmap='RdBu_r',
                        vmin=-1, vmax=1, shading='auto')
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(f'EAF({label_a}) - EAF({label_b})')

    # Union Pareto 抽出
    def get_union_pareto(trial_pts):
        valid = [t for t in trial_pts if len(t) > 0]
        if not valid:
            return np.zeros((0, 2))
        return pareto_front_2d(np.concatenate(valid))

    pf_a = get_union_pareto(trial_pts_a)
    pf_b = get_union_pareto(trial_pts_b)
    a_only, b_only, overlap = _split_pareto_by_overlap(pf_a, pf_b)

    # 描画順: A のみ(赤) → B のみ(青) → 共通(白) で重なっても分別可能に
    if len(a_only) > 0:
        ax.scatter(a_only[:, 0], a_only[:, 1], color='darkred', s=55, alpha=0.9,
                    edgecolors='white', linewidths=0.6, marker='o',
                    label=f'{label_a} only (n={len(a_only)})', zorder=3)
    if len(b_only) > 0:
        ax.scatter(b_only[:, 0], b_only[:, 1], color='darkblue', s=55, alpha=0.9,
                    edgecolors='white', linewidths=0.6, marker='s',
                    label=f'{label_b} only (n={len(b_only)})', zorder=3)
    if len(overlap) > 0:
        ax.scatter(overlap[:, 0], overlap[:, 1], color='white', s=65,
                    edgecolors='black', linewidths=1.0, marker='o',
                    label=f'both (n={len(overlap)})', zorder=4)

    # 軸を Pareto 点ベースで引き締める
    union_pareto_ab = np.concatenate([pf for pf in [pf_a, pf_b] if len(pf) > 0]) \
        if (len(pf_a) > 0 or len(pf_b) > 0) else np.zeros((0, 2))
    xlim, ylim = _compute_tight_axes(union_pareto_ab, init_ms)
    if xlim: ax.set_xlim(xlim)
    if ylim: ax.set_ylim(ylim)
    _add_initial_marker(ax, init_ms)

    ax.set_xlabel('Makespan')
    ax.set_ylabel('Stability')
    ax.set_title(title)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


def _compute_tight_axes(pareto_points, init_ms,
                         margin_frac_x=0.03, margin_frac_y_top=0.10,
                         margin_frac_y_bot=0.03):
    """Pareto 点の範囲 + init_ms を含む最小限の軸範囲を返す

    - xlim: [min Pareto MS - margin, init_ms + margin]
    - ylim: [-bot_margin, max Pareto Stab + top_margin]
      Stab 上部は塗りつぶしが見切れないよう余裕を持たせる。
    """
    if len(pareto_points) == 0:
        return None, None
    ms_min = float(pareto_points[:, 0].min())
    stab_max = float(pareto_points[:, 1].max())

    xmax = init_ms if init_ms is not None else float(pareto_points[:, 0].max())
    range_ms = max(xmax - ms_min, 10.0)
    xlim_new = (ms_min - range_ms * margin_frac_x,
                xmax + range_ms * margin_frac_x)
    range_stab = max(stab_max, 1.0)
    ylim_new = (-range_stab * margin_frac_y_bot,
                stab_max + range_stab * margin_frac_y_top)
    return xlim_new, ylim_new


def _add_initial_marker(ax, init_ms):
    """初期解の基準線・グレー帯・緑星のみ追加。軸範囲は変更しない"""
    if init_ms is None:
        return
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    # 垂直線
    ax.axvline(init_ms, color='gray', linestyle='--', alpha=0.7, linewidth=1.3,
               zorder=1)
    # 右側グレー帯（init_ms が xlim 内のとき）
    if xlim[0] < init_ms < xlim[1]:
        ax.axvspan(init_ms, xlim[1], color='gray', alpha=0.08, zorder=0)
    # 緑星
    ax.scatter([init_ms], [0], marker='*', s=200, color='green',
                edgecolors='black', linewidths=0.8, zorder=5,
                label=f'initial (MS={init_ms}, Stab=0)')
    # 念のため軸を固定（axvspan が広げるのを防ぐ）
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)


def _collect_union_pareto(trial_pts_by_method, methods):
    """手法横断で全 trial の Pareto 点を集める。軸範囲決定に使う。"""
    all_pareto = []
    for m in methods:
        for trial in trial_pts_by_method.get(m, []):
            if len(trial) > 0:
                pf = pareto_front_2d(trial)
                all_pareto.append(pf)
    if not all_pareto:
        return np.zeros((0, 2))
    return np.concatenate(all_pareto)


def plot_individual_eaf(trial_pts_by_method, methods, title, outpath,
                         init_ms=None):
    """各手法の EAF を 0〜1 の絶対値として並べて表示

    diff EAF では 0-0 (誰も到達しない) と 1-1 (全員到達) が区別できないため、
    個別 EAF を並べて確認することで「両者が頻繁に到達する領域」と
    「誰も到達しない領域」の区別をつける。
    """
    all_pts = []
    for m in methods:
        for trial in trial_pts_by_method[m]:
            if len(trial) > 0:
                all_pts.append(trial)
    if not all_pts:
        return
    all_points = np.concatenate(all_pts)
    grid_ms, grid_st = make_grid(all_points, include_ms=init_ms)

    n = len(methods)
    fig, axes = plt.subplots(1, n, figsize=(5.5 * n, 6), sharex=True, sharey=True)
    if n == 1:
        axes = [axes]

    # 共通 tight axis
    union_pareto = _collect_union_pareto(trial_pts_by_method, methods)
    xlim, ylim = _compute_tight_axes(union_pareto, init_ms)

    for ax, m in zip(axes, methods):
        trial_pts = trial_pts_by_method[m]
        attain = attainment_function_trials(trial_pts, grid_ms, grid_st)
        im = ax.pcolormesh(grid_ms, grid_st, attain.T, cmap='viridis',
                            vmin=0, vmax=1, shading='auto')
        valid = [t for t in trial_pts if len(t) > 0]
        if valid:
            pf = pareto_front_2d(np.concatenate(valid))
            if len(pf) > 0:
                ax.scatter(pf[:, 0], pf[:, 1], color='white', s=35,
                            edgecolors='black', linewidths=0.5, zorder=3,
                            label=f'union Pareto (n={len(pf)})')
        if xlim: ax.set_xlim(xlim)
        if ylim: ax.set_ylim(ylim)
        _add_initial_marker(ax, init_ms)
        ax.set_xlabel('Makespan')
        ax.set_ylabel('Stability')
        ax.set_title(f'EAF: {METHOD_LABELS[m]}')
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, alpha=0.2)

    cbar = fig.colorbar(im, ax=axes, shrink=0.8)
    cbar.set_label('EAF (attainment probability)')
    fig.suptitle(title, fontsize=12)
    fig.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_all_visited_density(trial_pts_by_method, methods, title, outpath,
                              init_ms=None):
    """全 LS 訪問点の散布（薄く重ね描き、分布の概観用）"""
    fig, ax = plt.subplots(figsize=(11, 8))
    for m in methods:
        trial_pts = trial_pts_by_method[m]
        valid = [t for t in trial_pts if len(t) > 0]
        if not valid:
            continue
        combined = np.concatenate(valid)
        ax.scatter(combined[:, 0], combined[:, 1], s=6, alpha=0.08,
                    color=METHOD_COLORS[m], label=METHOD_LABELS[m])
    # 軸は Pareto 点ベースで引き締め（exploration の外れ値で広がらない）
    union_pareto = _collect_union_pareto(trial_pts_by_method, methods)
    xlim, ylim = _compute_tight_axes(union_pareto, init_ms)
    if xlim: ax.set_xlim(xlim)
    if ylim: ax.set_ylim(ylim)
    _add_initial_marker(ax, init_ms)
    ax.set_xlabel('Makespan')
    ax.set_ylabel('Stability')
    ax.set_title(title + ' (all visited points)')
    ax.legend(markerscale=3, fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


# ========== 統計 ==========

def compute_stats(trial_pts_by_method, methods):
    """HV（per-trial 平均 & union）と C-metric を計算"""
    all_pts = []
    for m in methods:
        for trial in trial_pts_by_method[m]:
            if len(trial) > 0:
                all_pts.append(trial)
    if not all_pts:
        return {}, {}, {}, None
    all_points = np.concatenate(all_pts)
    ref_ms = all_points[:, 0].max() + max(all_points[:, 0].max() * 0.01, 1.0)
    ref_st = all_points[:, 1].max() + max(all_points[:, 1].max() * 0.01, 0.1)
    ref = (ref_ms, ref_st)

    hv_per_trial = {}
    hv_union = {}
    union_pf = {}
    for m in methods:
        trials = trial_pts_by_method[m]
        per_t = []
        for trial in trials:
            if len(trial) > 0:
                pf = pareto_front_2d(trial)
                per_t.append(hypervolume_2d(pf, ref))
            else:
                per_t.append(0.0)
        hv_per_trial[m] = per_t
        valid = [t for t in trials if len(t) > 0]
        if valid:
            combined = np.concatenate(valid)
            pf = pareto_front_2d(combined)
            union_pf[m] = pf
            hv_union[m] = hypervolume_2d(pf, ref)
        else:
            union_pf[m] = np.zeros((0, 2))
            hv_union[m] = 0.0

    c_metrics = {}
    for a, b, _ in PAIRS + SUPPLEMENTARY_PAIRS:
        if a in methods and b in methods:
            pf_a = union_pf[a]
            pf_b = union_pf[b]
            c_metrics[(a, b)] = c_metric(pf_a, pf_b)  # a が b をどれだけカバー
            c_metrics[(b, a)] = c_metric(pf_b, pf_a)

    return hv_per_trial, hv_union, c_metrics, ref


def format_stats(hv_per_trial, hv_union, c_metrics, ref, label):
    lines = [f"\n=== {label} ==="]
    lines.append(f"参照点 (MS, Stab): ({ref[0]:.1f}, {ref[1]:.3f})")
    lines.append("")
    lines.append(f"  {'method':<18} {'HV (per-trial)':>22} {'HV (union)':>14}")
    lines.append("  " + "-" * 56)
    for m in hv_union:
        per = hv_per_trial[m]
        mean_hv = float(np.mean(per)) if per else 0.0
        std_hv = float(np.std(per)) if per else 0.0
        lines.append(f"  {METHOD_LABELS[m]:<18} "
                     f"{mean_hv:>10.2f} ± {std_hv:<8.2f} "
                     f"{hv_union[m]:>14.2f}")
    lines.append("")
    lines.append("  C-metric (union Pareto 基準、A が B をカバーする割合):")
    lines.append(f"    {'pair':<26} {'C(A,B)':>10} {'C(B,A)':>10}")
    for a, b, _ in PAIRS:
        if (a, b) not in c_metrics:
            continue
        label_pair = f"{METHOD_LABELS[a]} vs {METHOD_LABELS[b]}"
        lines.append(f"    {label_pair:<26} "
                     f"{c_metrics[(a, b)]:>10.3f} {c_metrics[(b, a)]:>10.3f}")
    return "\n".join(lines)


# ========== メイン処理 ==========

def analyze_problem_weight(problem_dir, weights_label, weights, out_subdir):
    json_path = os.path.join(problem_dir, f"results_{weights_label}.json")
    if not os.path.exists(json_path):
        return None
    with open(json_path) as f:
        data = json.load(f)

    methods = [m for m in METHODS if m in data]
    trial_pts_by_method = {m: load_trial_points(data, m) for m in methods}
    init_ms = data.get('init_makespan')

    n_pts_total = sum(sum(len(t) for t in v)
                       for v in trial_pts_by_method.values())
    print(f"  総訪問点: {n_pts_total}, 初期解 MS: {init_ms}")

    prob_label = f"{data['problem']}_{data['scenario']}"
    title_base = f"{prob_label}, weights=[{weights[0]}, {weights[1]}]"

    # (1) 全訪問点の密度散布
    plot_all_visited_density(
        trial_pts_by_method, methods,
        f"{title_base}: visited density",
        os.path.join(out_subdir, f"visited_density_{weights_label}.png"),
        init_ms=init_ms)

    # (2) Attainment surfaces + union Pareto - ペア別に分けて見やすく
    swap_methods = [m for m in ['ILS_swap', 'ILS_swap_repair'] if m in methods]
    if len(swap_methods) >= 1:
        plot_attainment_surfaces(
            trial_pts_by_method, swap_methods,
            f"{title_base}: swap family Attainment",
            os.path.join(out_subdir, f"attainment_swap_{weights_label}.png"),
            init_ms=init_ms)
    insert_methods = [m for m in ['ILS_insert', 'ILS_insert_repair'] if m in methods]
    if len(insert_methods) >= 1:
        plot_attainment_surfaces(
            trial_pts_by_method, insert_methods,
            f"{title_base}: insert family Attainment",
            os.path.join(out_subdir, f"attainment_insert_{weights_label}.png"),
            init_ms=init_ms)
    plot_attainment_surfaces(
        trial_pts_by_method, methods,
        f"{title_base}: All methods Attainment",
        os.path.join(out_subdir, f"attainment_all_{weights_label}.png"),
        init_ms=init_ms)

    # (2.5) 個別 EAF（diff の補完として 0-0 vs 1-1 を区別できる）
    if swap_methods:
        plot_individual_eaf(
            trial_pts_by_method, swap_methods,
            f"{title_base}: Individual EAF (swap family)",
            os.path.join(out_subdir, f"individual_eaf_swap_{weights_label}.png"),
            init_ms=init_ms)
    if insert_methods:
        plot_individual_eaf(
            trial_pts_by_method, insert_methods,
            f"{title_base}: Individual EAF (insert family)",
            os.path.join(out_subdir, f"individual_eaf_insert_{weights_label}.png"),
            init_ms=init_ms)

    # (3) Differential EAF（base vs +repair ペアごと）
    for a, b, pair_label in PAIRS:
        if a not in methods or b not in methods:
            continue
        plot_diff_eaf(
            trial_pts_by_method[b], trial_pts_by_method[a],
            f"{pair_label}+repair", pair_label,
            f"{title_base}: EAF diff ({pair_label}+repair vs {pair_label})",
            os.path.join(out_subdir, f"diff_eaf_{pair_label}_{weights_label}.png"),
            init_ms=init_ms)

    # (4) 補助比較: swap vs insert (base 同士 / +repair 同士)
    for a, b, pair_label in SUPPLEMENTARY_PAIRS:
        if a not in methods or b not in methods:
            continue
        plot_diff_eaf(
            trial_pts_by_method[a], trial_pts_by_method[b],
            METHOD_LABELS[a], METHOD_LABELS[b],
            f"{title_base}: EAF diff ({METHOD_LABELS[a]} vs {METHOD_LABELS[b]})",
            os.path.join(out_subdir, f"diff_eaf_{pair_label}_{weights_label}.png"),
            init_ms=init_ms)

    # 統計
    hv_per_trial, hv_union, c_metrics, ref = compute_stats(
        trial_pts_by_method, methods)
    stats_text = format_stats(hv_per_trial, hv_union, c_metrics, ref, title_base)
    stats_path = os.path.join(out_subdir, f"stats_eaf_{weights_label}.txt")
    with open(stats_path, 'w', encoding='utf-8') as f:
        f.write(stats_text + "\n")
    print(stats_text)

    return {
        'problem': prob_label,
        'weights': weights,
        'weights_label': weights_label,
        'hv_per_trial': hv_per_trial,
        'hv_union': hv_union,
        'c_metrics': c_metrics,
        'ref': ref,
    }


def write_cross_summary(records, out_dir):
    path = os.path.join(out_dir, "eaf_cross_summary.txt")
    lines = ["EAF 横断サマリー (Pareto front ベース)",
             "=" * 80, ""]

    # per-trial HV 平均
    lines.append("## HV per-trial 平均 ± 標準偏差")
    header = f"  {'問題':<28} {'重み':<12}"
    for m in METHODS:
        header += f" {METHOD_LABELS[m]:>18}"
    lines.append(header)
    lines.append("  " + "-" * 110)
    for r in records:
        if r is None:
            continue
        row = f"  {r['problem']:<28} {str(r['weights']):<12}"
        for m in METHODS:
            per = r['hv_per_trial'].get(m, [])
            if per:
                mean_v = float(np.mean(per))
                std_v = float(np.std(per))
                row += f" {mean_v:>9.2f}±{std_v:<7.2f}"
            else:
                row += f" {'-':>18}"
        lines.append(row)

    # Union HV
    lines.append("")
    lines.append("## Union Pareto の HV (全 trial の非劣解を合わせた front)")
    header = f"  {'問題':<28} {'重み':<12}"
    for m in METHODS:
        header += f" {METHOD_LABELS[m]:>14}"
    lines.append(header)
    lines.append("  " + "-" * 100)
    for r in records:
        if r is None:
            continue
        row = f"  {r['problem']:<28} {str(r['weights']):<12}"
        for m in METHODS:
            v = r['hv_union'].get(m, None)
            row += f" {v:>14.2f}" if v is not None else f" {'-':>14}"
        lines.append(row)

    # HV ratio
    lines.append("")
    lines.append("## Union HV 比 (base+repair / base) - 1 より大きいほど repair 優位")
    header = f"  {'問題':<28} {'重み':<12} {'swap':>14} {'insert':>14}"
    lines.append(header)
    lines.append("  " + "-" * 80)
    for r in records:
        if r is None:
            continue
        row = f"  {r['problem']:<28} {str(r['weights']):<12}"
        for a, b, _ in PAIRS:
            ha = r['hv_union'].get(a, 0)
            hb = r['hv_union'].get(b, 0)
            if ha > 0:
                row += f" {hb/ha:>14.4f}"
            else:
                row += f" {'-':>14}"
        lines.append(row)

    # C-metric (base vs +repair)
    lines.append("")
    lines.append("## C-metric on union Pareto (A が B をカバーする割合)")
    lines.append("   1.0 = A が B を完全に dominate、0.0 = A が B の誰も dominate しない")
    header = (f"  {'問題':<28} {'重み':<12}"
              f" {'swap→+rep':>12} {'+rep→swap':>12}"
              f" {'ins→+rep':>12} {'+rep→ins':>12}")
    lines.append(header)
    lines.append("  " + "-" * 100)
    for r in records:
        if r is None:
            continue
        row = f"  {r['problem']:<28} {str(r['weights']):<12}"
        for a, b, _ in PAIRS:
            c_ab = r['c_metrics'].get((a, b))
            c_ba = r['c_metrics'].get((b, a))
            row += f" {c_ab:>12.3f}" if c_ab is not None else f" {'-':>12}"
            row += f" {c_ba:>12.3f}" if c_ba is not None else f" {'-':>12}"
        lines.append(row)

    # swap vs insert 比較
    lines.append("")
    lines.append("## swap vs insert (主摂動の比較)")
    lines.append("   per-trial HV 比と C-metric。どちらの主摂動が有利か、repair ありなし別に比較。")
    header = (f"  {'問題':<28} {'重み':<12}"
              f" {'HV比(ins/swp)':>14} {'HV比(ins+r/swp+r)':>18}"
              f" {'swp→ins':>10} {'ins→swp':>10}"
              f" {'swp+r→ins+r':>12} {'ins+r→swp+r':>12}")
    lines.append(header)
    lines.append("  " + "-" * 120)
    for r in records:
        if r is None:
            continue
        row = f"  {r['problem']:<28} {str(r['weights']):<12}"
        # per-trial HV 平均の比
        hv_swap = np.mean(r['hv_per_trial'].get('ILS_swap', [0])) if r['hv_per_trial'].get('ILS_swap') else 0
        hv_insert = np.mean(r['hv_per_trial'].get('ILS_insert', [0])) if r['hv_per_trial'].get('ILS_insert') else 0
        hv_swap_r = np.mean(r['hv_per_trial'].get('ILS_swap_repair', [0])) if r['hv_per_trial'].get('ILS_swap_repair') else 0
        hv_insert_r = np.mean(r['hv_per_trial'].get('ILS_insert_repair', [0])) if r['hv_per_trial'].get('ILS_insert_repair') else 0
        row += f" {(hv_insert/hv_swap if hv_swap else 0):>14.4f}" if hv_swap > 0 else f" {'-':>14}"
        row += f" {(hv_insert_r/hv_swap_r if hv_swap_r else 0):>18.4f}" if hv_swap_r > 0 else f" {'-':>18}"
        # C-metric
        c_swp_ins = r['c_metrics'].get(('ILS_swap', 'ILS_insert'))
        c_ins_swp = r['c_metrics'].get(('ILS_insert', 'ILS_swap'))
        c_swpr_insr = r['c_metrics'].get(('ILS_swap_repair', 'ILS_insert_repair'))
        c_insr_swpr = r['c_metrics'].get(('ILS_insert_repair', 'ILS_swap_repair'))
        row += f" {c_swp_ins:>10.3f}" if c_swp_ins is not None else f" {'-':>10}"
        row += f" {c_ins_swp:>10.3f}" if c_ins_swp is not None else f" {'-':>10}"
        row += f" {c_swpr_insr:>12.3f}" if c_swpr_insr is not None else f" {'-':>12}"
        row += f" {c_insr_swpr:>12.3f}" if c_insr_swpr is not None else f" {'-':>12}"
        lines.append(row)

    # 読み方
    lines.append("")
    lines.append("## 読み方")
    lines.append("- HV per-trial 平均: 各 trial の Pareto front の HV を平均。")
    lines.append("  1つの trial がどれだけ広く良い領域をカバーしているかの平均。")
    lines.append("- HV union: 全 trial の非劣解を合わせた front の HV。")
    lines.append("  手法全体（10 trial の合算）がカバーできた領域の広さ。")
    lines.append("- Union HV 比 > 1: base+repair の集合が base 集合より広い領域をカバー。")
    lines.append("- C-metric: 'swap→+rep' = swap の union Pareto が swap+repair の")
    lines.append("  union Pareto をどれだけ dominate しているか。")
    lines.append("  小さい（<0.5）かつ逆の C が大きい、なら相手が強い。")
    lines.append("- diff_eaf_*.png: 赤=+repair 側が高確率で dominate、青=base 側が高確率")

    text = "\n".join(lines)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(text + "\n")
    print(f"\n横断サマリー: {path}")


def main():
    parser = argparse.ArgumentParser(description="repair 実験の EAF 分析 (Pareto front ベース)")
    parser.add_argument('result_dir', type=str, help='実験結果ディレクトリ')
    args = parser.parse_args()

    if not os.path.isdir(args.result_dir):
        print(f"ERROR: {args.result_dir} が存在しません")
        sys.exit(1)

    out_dir = os.path.join(args.result_dir, "eaf_analysis")
    os.makedirs(out_dir, exist_ok=True)

    records = []
    problem_dirs = sorted([
        d for d in os.listdir(args.result_dir)
        if os.path.isdir(os.path.join(args.result_dir, d))
        and d != "eaf_analysis"
    ])
    print(f"検出された問題ディレクトリ: {problem_dirs}")

    for pd in problem_dirs:
        full_path = os.path.join(args.result_dir, pd)
        json_files = sorted([f for f in os.listdir(full_path)
                              if f.startswith('results_') and f.endswith('.json')])
        out_subdir = os.path.join(out_dir, pd)
        os.makedirs(out_subdir, exist_ok=True)

        for jf in json_files:
            w_label = jf.replace('results_', '').replace('.json', '')
            try:
                parts = w_label.split('_')
                eff = float(parts[0].split('=')[1])
                stab = float(parts[1].split('=')[1])
                weights = [eff, stab]
            except (IndexError, ValueError):
                print(f"  スキップ: w_label パース失敗 {w_label}")
                continue

            print(f"\n--- 処理中: {pd} / {w_label} ---")
            rec = analyze_problem_weight(full_path, w_label, weights, out_subdir)
            records.append(rec)

    write_cross_summary(records, out_dir)
    print(f"\n全分析完了。出力: {out_dir}")


if __name__ == "__main__":
    main()
