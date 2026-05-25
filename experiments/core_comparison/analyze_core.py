#!/usr/bin/env python3
"""
実験1: コア比較の分析スクリプト

run_core_comparison.py が生成した raw データ（履歴・最終値）から
evaluation_design.md §8 の図表を生成する。

=== 主張との対応 ===
| 主張 | 指標 |
|---|---|
| (A) 速度 | anytime HV curve, anytime scalar curve, snapshot stats |
| (B) Pareto 覆域 | final Pareto, union HV, C-metric, 差分 EAF |
| (B') 安定性方向優位 | Region-restricted HV, attainment surface, 差分 EAF |
| (C) repair 貢献 | repair vs base の pair 比較（EAF/HV/conditional HV） |
| (D) 重み頑健性 | 改善成功率 heatmap |

=== 使い方 ===
  # run_core_comparison.py の出力ディレクトリを指定
  python analyze_core.py results/core_<timestamp>/

  # オプション
  python analyze_core.py results/core_<timestamp>/ \
      --snapshot-times 5 10 20 40 \
      --eaf-pairs ils_insert:ga ils_insert_repair:ils_insert

=== 設計 ===
- EAF は pairwise なので pair を CLI または自動選定（戦略的 pair）で指定
- 重み複数時は「全 weights 集約」で quality 指標を出す（§4.2 参照）
- Region-restricted HV は全手法の union Pareto の stab 軸 quartile で自動分割（§4.5）
"""

import argparse
import json
import os
import sys
from itertools import combinations

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


# ========== 定数 ==========

METHOD_COLOR_PALETTE = {
    'ga':                'tab:gray',
    'ils_insert':        'tab:orange',
    'ils_insert_repair': 'tab:red',
    'ils_swap':          'tab:blue',
    'ils_swap_repair':   'tab:cyan',
}
DEFAULT_COLORS = ['tab:purple', 'tab:green', 'tab:brown', 'tab:olive', 'tab:pink']

DEFAULT_SNAPSHOT_TIMES = [5.0, 10.0, 20.0, 40.0]

# Region-restricted HV の領域分割（stab 軸 quartile, 全手法共通）
REGION_NAMES = ['low_stab', 'mid_stab', 'high_stab']

# 差分 EAF のデフォルト pair 戦略（どちらも手法セットに含まれる場合に有効）
DEFAULT_EAF_PAIRS = [
    ('ils_insert', 'ga'),                    # 主張 A/B: ILS vs GA
    ('ils_insert_repair', 'ils_insert'),     # 主張 C: repair 貢献
    ('ils_insert_repair', 'ga'),             # 主張 A/B: 最強 ILS vs GA
    ('ils_swap_repair', 'ils_swap'),         # swap 系の repair 貢献
]


# ========== データローダ ==========

def load_result_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def extract_trial_points(method_data):
    """手法データから per-trial の (ms, st) 点列を抜き出す。

    GA:  points[trial] = [[ms, st], ...]  （全世代×全個体）
    ILS: points[trial] = [[ls_ms, ls_st, accepted], ...]

    Returns: list[np.ndarray (n,2)]（trial ごと）
    """
    kind = method_data['kind']
    points = method_data.get('points') or []
    result = []
    for trial_pts in points:
        if trial_pts is None or len(trial_pts) == 0:
            result.append(np.zeros((0, 2)))
            continue
        arr = []
        for entry in trial_pts:
            if kind == 'ils':
                ms, st = float(entry[0]), float(entry[1])
            else:
                ms, st = float(entry[0]), float(entry[1])
            if not (np.isfinite(ms) and np.isfinite(st)):
                continue
            arr.append([ms, st])
        result.append(np.array(arr) if arr else np.zeros((0, 2)))
    return result


def extract_anytime(method_data):
    """anytime 履歴を抜き出す。

    Returns: list[list[dict]]（trial ごと）
      各 dict: {cpu_time, best_ms, best_st, best_score}
    """
    any_list = method_data.get('anytime') or []
    return any_list


def extract_finals(method_data):
    """最終値（finals）を抜き出す。None/error は除外した valid list を返す。"""
    finals = method_data.get('finals') or []
    return [d for d in finals if d is not None and 'error' not in d]


def get_color(method_key, idx=0):
    return METHOD_COLOR_PALETTE.get(method_key,
                                     DEFAULT_COLORS[idx % len(DEFAULT_COLORS)])


def extract_baseline(method_data):
    """baseline = [ms, stab] or None。初期解相当の点を弱 dominance で除外するため。
    - ILS: (init_ms, 0.0) — semi-active decoding
    - GA: (ms_active, stab_active) — active schedule decoding （stab≠0 になる）
    """
    b = method_data.get('baseline')
    if b is None:
        return None
    return (float(b[0]), float(b[1]))


def filter_by_baseline(points, baseline):
    """baseline に弱く dominate される点（= 初期解相当、探索価値なし）を除外"""
    if baseline is None or len(points) == 0:
        return points
    points = np.asarray(points)
    b_ms, b_st = baseline
    # 数値誤差許容: 初期解ぴったりの点も確実に捕まえる
    eps_ms = max(abs(b_ms) * 1e-9, 1e-6)
    eps_st = max(abs(b_st) * 1e-9, 1e-6)
    mask = ~((points[:, 0] >= b_ms - eps_ms) & (points[:, 1] >= b_st - eps_st))
    return points[mask]


# ========== Pareto / HV / C-metric 基本関数 ==========

def pareto_front_2d(points, baseline=None):
    """2D minimization の Pareto front. baseline を指定すると
    弱 dominate される点を除外してから抽出する。"""
    if len(points) == 0:
        return np.zeros((0, 2))
    points = filter_by_baseline(points, baseline) if baseline is not None else np.asarray(points)
    if len(points) == 0:
        return np.zeros((0, 2))
    idx = np.lexsort((points[:, 1], points[:, 0]))
    sorted_pts = points[idx]
    pareto = [sorted_pts[0]]
    for p in sorted_pts[1:]:
        if p[1] < pareto[-1][1]:
            pareto.append(p)
    return np.array(pareto)


def hypervolume_2d(points, ref, baseline=None):
    """2D hypervolume (minimization). ref = (MS_ref, Stab_ref).
    baseline 指定時は pareto_front 抽出前に除外。"""
    if len(points) == 0:
        return 0.0
    pareto = pareto_front_2d(points, baseline=baseline)
    if len(pareto) == 0:
        return 0.0
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
    """Zitzler & Thiele C-metric: A が弱く dominate する B の割合"""
    if len(B_front) == 0:
        return 0.0
    covered = 0
    for b in B_front:
        for a in A_front:
            if a[0] <= b[0] and a[1] <= b[1]:
                covered += 1
                break
    return covered / len(B_front)


# ========== Region-restricted HV ==========

def compute_regions(union_pareto_all_methods):
    """全手法共通 union Pareto の stab_max を用いて [0, stab_max] を等 3 分割

    Returns: dict {region_name: (stab_lo, stab_hi)}
    MS 軸は呼び出し側で init_ms を上限に使う。
    """
    if len(union_pareto_all_methods) == 0:
        return None
    stab = union_pareto_all_methods[:, 1]
    stab_max = float(stab.max())
    if stab_max <= 0:
        return None
    t1 = stab_max / 3.0
    t2 = 2.0 * stab_max / 3.0
    eps = 1e-9
    return {
        'low_stab':  (0.0, t1),
        'mid_stab':  (t1 + eps, t2),
        'high_stab': (t2 + eps, stab_max),
        '_t1': t1, '_t2': t2, '_stab_max': stab_max,
    }


def region_restricted_hv(method_pareto, region_bounds, init_ms,
                         stab_margin_frac=0.02):
    """領域 R 内の HV. 参照点 = (init_ms, R_upper + margin)

    HV の参照点は全点を strict dominate する必要があるため、境界点
    （stab = stab_hi）で HV=0 にならないよう stab_hi に小マージンを足す。
    margin は領域全体の stab 幅に比例させる（手法間・領域間で比較可能）。
    """
    if len(method_pareto) == 0:
        return 0.0, 0
    stab_lo, stab_hi = region_bounds
    mask = (method_pareto[:, 1] >= stab_lo) & (method_pareto[:, 1] <= stab_hi)
    region_pts = method_pareto[mask]
    if len(region_pts) == 0:
        return 0.0, 0
    region_pts = region_pts[region_pts[:, 0] < init_ms]
    if len(region_pts) == 0:
        return 0.0, 0
    stab_width = max(stab_hi - stab_lo, 1e-6)
    ref_stab = stab_hi + stab_margin_frac * stab_width
    ref = (init_ms, ref_stab)
    return hypervolume_2d(region_pts, ref), int(len(region_pts))


# ========== EAF ==========

GRID_N = 150

def make_grid(all_points, pad_frac=0.05, include_ms=None):
    if len(all_points) == 0:
        return np.array([0.0, 1.0]), np.array([0.0, 1.0])
    ms_min, ms_max = all_points[:, 0].min(), all_points[:, 0].max()
    if include_ms is not None:
        ms_max = max(ms_max, include_ms)
    st_min, st_max = all_points[:, 1].min(), all_points[:, 1].max()
    ms_pad = max((ms_max - ms_min) * pad_frac, 1.0)
    st_pad = max((st_max - st_min) * pad_frac, 0.1)
    return (np.linspace(ms_min - ms_pad, ms_max + ms_pad, GRID_N),
            np.linspace(st_min - st_pad, st_max + st_pad, GRID_N))


def attainment_function_trials(trial_points_list, grid_ms, grid_st,
                                baseline=None):
    """Trial-based EAF. baseline 指定時は初期解相当の点を除外してから Pareto 構築。"""
    n_trials = len(trial_points_list)
    if n_trials == 0:
        return np.zeros((len(grid_ms), len(grid_st)))
    MS, ST = np.meshgrid(grid_ms, grid_st, indexing='ij')
    attain_count = np.zeros_like(MS, dtype=float)
    for trial_pts in trial_points_list:
        if len(trial_pts) == 0:
            continue
        pf = pareto_front_2d(trial_pts, baseline=baseline)
        if len(pf) == 0:
            continue
        trial_mask = np.zeros_like(MS, dtype=bool)
        for p in pf:
            trial_mask |= ((p[0] <= MS) & (p[1] <= ST))
        attain_count += trial_mask.astype(float)
    return attain_count / n_trials


def attainment_front_line(attain, grid_ms, grid_st, level):
    """α >= level の領域の左下境界 step 座標"""
    mask = attain >= level
    xs, ys = [], []
    for j in range(len(grid_st)):
        col = mask[:, j]
        if col.any():
            i = np.argmax(col)
            xs.append(grid_ms[i])
            ys.append(grid_st[j])
    return np.array(xs), np.array(ys)


# ========== anytime 系ユーティリティ ==========

def interpolate_best_at_times(anytime_list, t_grid, field='best_score'):
    """anytime 履歴を t_grid 上で各 trial 補間し、配列に積む

    Returns: (n_trials, len(t_grid)) 配列。データ欠損 trial は NaN 埋め。
    """
    interp = []
    for hist in anytime_list:
        if hist is None or len(hist) == 0:
            interp.append(np.full(len(t_grid), np.nan))
            continue
        times = np.array([h['cpu_time'] for h in hist])
        vals = np.array([h[field] for h in hist], dtype=float)
        # hist が単調に伸びていない可能性に備えて、各 t で t 以下の最新値を取る
        y = np.full(len(t_grid), np.nan)
        for i, t in enumerate(t_grid):
            idx = np.searchsorted(times, t, side='right') - 1
            if idx >= 0:
                y[i] = vals[idx]
        interp.append(y)
    return np.array(interp)


def pareto_points_until_time(trial_anytime_list, trial_points_list, kind, t,
                              baseline=None):
    """時刻 t 以下に訪問された全点から baseline 除外後の点を返す

    GA の points は世代番号と直接紐づかないため、anytime の cpu_time から
    「t 以下の世代数」を逆算して、points の先頭 n_gen * pop_size だけ採用する。
    ILS は 1 iter 1 点なので同様に「t 以下の iter 数」で切る。
    正確性より実装容易さを優先した近似。baseline 指定時は最後に除外。
    """
    pts_out = []
    for i, hist in enumerate(trial_anytime_list):
        if hist is None or len(hist) == 0:
            continue
        times = np.array([h['cpu_time'] for h in hist])
        idx = np.searchsorted(times, t, side='right')
        if idx <= 0:
            continue
        all_pts = trial_points_list[i] if i < len(trial_points_list) else np.zeros((0, 2))
        if len(all_pts) == 0:
            continue
        # kind に応じて切断点を決める
        if kind == 'ils':
            # ILS: 1 entry per iter, points は同じ順で per-iter 1 点（ただし hist は init も含むので len-1）
            # anytime 配列の先頭は init 用の 1 エントリ、以降が iter。points は iter 分のみ。
            # 近似: hist の iter 数 = idx - 1。points の先頭 min(idx, len(all_pts)) を採用
            n_take = min(idx, len(all_pts))
            pts_out.append(all_pts[:n_take])
        elif kind == 'ga':
            # GA: 新形式は entry['evaluations'] (累積評価数) を持つ。
            # 旧形式 (1 entry/gen) も entry['evaluations'] = (gen+1)*pop_size で同じく動く。
            last_entry = hist[idx - 1]
            ev = last_entry.get('evaluations')
            if ev is not None:
                n_take = min(int(ev), len(all_pts))
                pts_out.append(all_pts[:n_take])
            else:
                # 超旧 fallback: 世代単位で推定
                n_gens_total = len(hist)
                if n_gens_total > 0 and len(all_pts) % n_gens_total == 0:
                    pop_size = len(all_pts) // n_gens_total
                    n_take = idx * pop_size
                    pts_out.append(all_pts[:n_take])
                else:
                    ratio = idx / max(n_gens_total, 1)
                    n_take = int(len(all_pts) * ratio)
                    pts_out.append(all_pts[:n_take])
    if not pts_out:
        return np.zeros((0, 2))
    concat = np.concatenate(pts_out)
    if baseline is not None:
        concat = filter_by_baseline(concat, baseline)
    return concat


# ========== プロット: anytime curves ==========

def _trial_last_pareto_update_time(hist, all_pts, kind, baseline=None):
    """trial 内で最後に Pareto front が更新された cpu_time を返す.

    points の発見時刻を hist の cpu_time から逆算し、時系列順に Pareto に足し
    込んで、非被支配な新規点が入ったタイミングを追う。収束後の平坦部分を
    切り捨てるために使う。
    """
    if hist is None or len(hist) == 0 or len(all_pts) == 0:
        return None
    tp_pairs = []
    if kind == 'ga':
        # 各 entry の (累積評価数, cpu_time) を sort し、点 idx → cpu_time を割当
        ev_times = sorted(
            (int(e['evaluations']), float(e['cpu_time']))
            for e in hist if 'evaluations' in e and 'cpu_time' in e
        )
        if not ev_times:
            return hist[-1]['cpu_time']
        j = 0
        for i in range(len(all_pts)):
            target = i + 1
            while j < len(ev_times) and ev_times[j][0] < target:
                j += 1
            t_pt = ev_times[j][1] if j < len(ev_times) else ev_times[-1][1]
            tp_pairs.append((t_pt, tuple(all_pts[i])))
    elif kind == 'ils':
        # hist[0] は init, hist[1..] が iter。points は iter 分のみ。
        for i in range(len(all_pts)):
            h_idx = min(i + 1, len(hist) - 1)
            tp_pairs.append((hist[h_idx]['cpu_time'], tuple(all_pts[i])))
    else:
        return hist[-1]['cpu_time']

    if baseline is not None:
        tp_pairs = [(t, p) for (t, p) in tp_pairs
                    if not (p[0] >= baseline[0] and p[1] >= baseline[1])]

    tp_pairs.sort(key=lambda x: x[0])
    current = []
    last_update_t = None
    for t, p in tp_pairs:
        dominated = False
        new_pareto = []
        for q in current:
            if q[0] <= p[0] and q[1] <= p[1] and (q[0] < p[0] or q[1] < p[1]):
                dominated = True
                break
            if q[0] == p[0] and q[1] == p[1]:
                dominated = True
                break
            if p[0] <= q[0] and p[1] <= q[1] and (p[0] < q[0] or p[1] < q[1]):
                continue  # q dominated by p → drop
            new_pareto.append(q)
        if dominated:
            continue
        new_pareto.append(p)
        current = new_pareto
        last_update_t = t
    if last_update_t is None:
        return hist[-1]['cpu_time']
    return last_update_t


def _compute_t_grid(trial_pts_by_method, anytime_by_method, methods,
                    kind_by_method, baselines_by_method, n_points=60,
                    xscale='linear'):
    """各 trial の最終 Pareto 更新時刻 → trial 間 median → 手法間 max で t_max.

    収束後フラットになった部分は情報量ゼロなので切り捨てる。手法間で探索時間が
    大きく違う場合、max 採用で長い手法の後半も描画される（短い手法は end で
    フラット）。xscale='log' なら等比、'linear' なら等差サンプリング。
    """
    medians = []
    for m in methods:
        kind = kind_by_method[m]
        baseline = baselines_by_method.get(m)
        anytime_list = anytime_by_method[m]
        points_list = trial_pts_by_method[m]
        trial_lasts = []
        for i, hist in enumerate(anytime_list):
            pts = points_list[i] if i < len(points_list) else np.zeros((0, 2))
            t_last = _trial_last_pareto_update_time(hist, pts, kind,
                                                     baseline=baseline)
            if t_last is not None and t_last > 0:
                trial_lasts.append(t_last)
        if trial_lasts:
            medians.append(float(np.median(trial_lasts)))
    if not medians:
        return None
    t_max = max(medians)
    if xscale == 'log':
        t_min = max(0.02, t_max * 0.002)
        if t_min >= t_max:
            return None
        return np.geomspace(t_min, t_max, n_points)
    else:
        t_min = max(0.1, t_max * 0.02)
        if t_min >= t_max:
            return None
        return np.linspace(t_min, t_max, n_points)


def plot_anytime_hv(trial_pts_by_method, anytime_by_method,
                    methods, kind_by_method, baselines_by_method,
                    ref, init_ms,
                    title, outpath, n_points=60, xscale='linear'):
    """anytime HV curve: 1 画像 2 パネル (left=per-trial median+IQR, right=union).

    per-trial: 各 trial で Pareto-front を作って HV→trial 間で median+IQR
    union:     時刻 t までに全 trial が観測した点の合算 Pareto で HV（1 本）
    """
    t_grid = _compute_t_grid(trial_pts_by_method, anytime_by_method, methods,
                              kind_by_method, baselines_by_method, n_points,
                              xscale=xscale)
    if t_grid is None:
        return

    fig, (ax_pt, ax_un) = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
    for m in methods:
        kind = kind_by_method[m]
        baseline = baselines_by_method.get(m)
        anytime_list = anytime_by_method[m]
        points_list = trial_pts_by_method[m]
        color = get_color(m)

        # per-trial HV
        hv_curves = []
        for trial_idx in range(len(anytime_list)):
            hv_per_t = []
            for t in t_grid:
                pts = pareto_points_until_time(
                    [anytime_list[trial_idx]], [points_list[trial_idx]], kind, t,
                    baseline=baseline)
                if len(pts) > 0:
                    hv_per_t.append(hypervolume_2d(pts, ref))
                else:
                    hv_per_t.append(0.0)
            hv_curves.append(hv_per_t)
        if hv_curves:
            arr = np.array(hv_curves)
            median_hv = np.nanmedian(arr, axis=0)
            q25_hv = np.nanpercentile(arr, 25, axis=0)
            q75_hv = np.nanpercentile(arr, 75, axis=0)
            ax_pt.plot(t_grid, median_hv, color=color, lw=2.0, label=m)
            ax_pt.fill_between(t_grid, q25_hv, q75_hv, color=color, alpha=0.15)

        # union HV (全 trial 合算)
        union_hv = []
        for t in t_grid:
            pts = pareto_points_until_time(anytime_list, points_list, kind, t,
                                            baseline=baseline)
            if len(pts) > 0:
                union_hv.append(hypervolume_2d(pareto_front_2d(pts), ref))
            else:
                union_hv.append(0.0)
        ax_un.plot(t_grid, union_hv, color=color, lw=2.0, label=m)

    for ax, subtitle in [(ax_pt, 'per-trial (median, band=IQR 25-75%)'),
                          (ax_un, 'union (all trials combined)')]:
        ax.set_xlabel('CPU time (s)')
        if xscale == 'log':
            ax.set_xscale('log')
        ax.set_title(subtitle)
        ax.grid(True, alpha=0.3, which='both' if xscale == 'log' else 'major')
        ax.legend(loc='lower right', fontsize=10)
    ax_pt.set_ylabel('HV')
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


def plot_anytime_region_hv(trial_pts_by_method, anytime_by_method,
                           methods, kind_by_method, baselines_by_method,
                           regions, init_ms, title, outpath, n_points=60,
                           xscale='linear'):
    """anytime Region-restricted HV: 2×3 subplots (row=per-trial/union, col=region).

    領域境界は事前固定 (compute_regions: [0, stab_max] 等 3 分割)。
    top row: 各 trial で領域 HV → trial 間 median+IQR
    bottom row: 時刻 t までの全 trial 合算点の Pareto で領域 HV（1 本）
    """
    if regions is None:
        return
    t_grid = _compute_t_grid(trial_pts_by_method, anytime_by_method, methods,
                              kind_by_method, baselines_by_method, n_points,
                              xscale=xscale)
    if t_grid is None:
        return

    fig, axes = plt.subplots(2, 3, figsize=(18, 10), sharex=True, sharey='col')
    for ax_idx, region_name in enumerate(REGION_NAMES):
        stab_lo, stab_hi = regions[region_name]
        ax_pt = axes[0, ax_idx]
        ax_un = axes[1, ax_idx]
        for m in methods:
            kind = kind_by_method[m]
            baseline = baselines_by_method.get(m)
            anytime_list = anytime_by_method[m]
            points_list = trial_pts_by_method[m]
            color = get_color(m)

            # per-trial
            hv_curves = []
            for trial_idx in range(len(anytime_list)):
                hv_per_t = []
                for t in t_grid:
                    pts = pareto_points_until_time(
                        [anytime_list[trial_idx]], [points_list[trial_idx]],
                        kind, t, baseline=baseline)
                    if len(pts) == 0:
                        hv_per_t.append(0.0)
                        continue
                    pf = pareto_front_2d(pts)
                    hv, _ = region_restricted_hv(pf, (stab_lo, stab_hi), init_ms)
                    hv_per_t.append(hv)
                hv_curves.append(hv_per_t)
            if hv_curves:
                arr = np.array(hv_curves)
                median_hv = np.nanmedian(arr, axis=0)
                q25_hv = np.nanpercentile(arr, 25, axis=0)
                q75_hv = np.nanpercentile(arr, 75, axis=0)
                ax_pt.plot(t_grid, median_hv, color=color, lw=2.0, label=m)
                ax_pt.fill_between(t_grid, q25_hv, q75_hv, color=color, alpha=0.15)

            # union
            union_hv = []
            for t in t_grid:
                pts = pareto_points_until_time(anytime_list, points_list, kind, t,
                                                baseline=baseline)
                if len(pts) == 0:
                    union_hv.append(0.0)
                    continue
                pf = pareto_front_2d(pts)
                hv, _ = region_restricted_hv(pf, (stab_lo, stab_hi), init_ms)
                union_hv.append(hv)
            ax_un.plot(t_grid, union_hv, color=color, lw=2.0, label=m)

        for ax in (ax_pt, ax_un):
            if xscale == 'log':
                ax.set_xscale('log')
            ax.grid(True, alpha=0.3,
                    which='both' if xscale == 'log' else 'major')
            ax.legend(fontsize=9, loc='lower right')
        ax_pt.set_title(f'{region_name}  stab ∈ [{stab_lo:.2f}, {stab_hi:.2f}]',
                         fontsize=11)
        ax_un.set_xlabel('CPU time (s)')
        if ax_idx == 0:
            ax_pt.set_ylabel('HV  per-trial (median, band=IQR)')
            ax_un.set_ylabel('HV  union (all trials)')
    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


def plot_anytime_scalar(anytime_by_method, methods, title, outpath,
                         trial_pts_by_method=None, kind_by_method=None,
                         baselines_by_method=None, xscale='linear'):
    """anytime scalar curve: best_score の時系列（trial median+IQR）.

    時間軸は anytime HV / region HV と同じ「各 trial の最終 Pareto 更新時刻の
    trial 間 median、手法間 max」を採用。best_score の更新は Pareto 更新と
    同期的ではないが、大局的には同じ時間窓で見たいので揃える。
    """
    t_grid = None
    if (trial_pts_by_method is not None and kind_by_method is not None
            and baselines_by_method is not None):
        t_grid = _compute_t_grid(trial_pts_by_method, anytime_by_method, methods,
                                  kind_by_method, baselines_by_method,
                                  n_points=100, xscale=xscale)
    if t_grid is None:
        # fallback: 旧 ロジック相当
        max_t_per_method = []
        min_t_per_method = []
        for m in methods:
            ts, ts_first = [], []
            for hist in anytime_by_method[m]:
                if hist and len(hist) > 0:
                    ts.append(hist[-1]['cpu_time'])
                    if len(hist) >= 2:
                        ts_first.append(hist[1]['cpu_time'])
                    else:
                        ts_first.append(hist[0]['cpu_time'])
            if ts:
                max_t_per_method.append(max(ts))
            if ts_first:
                min_t_per_method.append(max(ts_first))
        if not max_t_per_method:
            return
        t_max = max(max_t_per_method)
        t_min = max(min_t_per_method) if min_t_per_method else max(0.01, t_max * 0.01)
        t_min = max(t_min, 0.01)
        if t_min >= t_max:
            return
        t_grid = np.linspace(t_min, t_max, 100)

    fig, ax = plt.subplots(figsize=(10, 6))
    for m in methods:
        anytime_list = anytime_by_method[m]
        interp = interpolate_best_at_times(anytime_list, t_grid, field='best_score')
        if interp.shape[0] == 0:
            continue
        with np.errstate(all='ignore'):
            median_v = np.nanmedian(interp, axis=0)
            q25_v = np.nanpercentile(interp, 25, axis=0)
            q75_v = np.nanpercentile(interp, 75, axis=0)
        valid = ~np.isnan(median_v)
        if not valid.any():
            continue
        color = get_color(m)
        ax.plot(t_grid[valid], median_v[valid], color=color, lw=2.0, label=m)
        ax.fill_between(t_grid[valid],
                        q25_v[valid], q75_v[valid],
                        color=color, alpha=0.15)

    ax.set_xlabel('CPU time (s)')
    if xscale == 'log':
        ax.set_xscale('log')
    ax.set_ylabel('Best weighted score (per-trial median, band=IQR 25-75%)')
    ax.set_title(title)
    ax.grid(True, alpha=0.3, which='both' if xscale == 'log' else 'major')
    ax.legend(loc='upper right', fontsize=10)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


# ========== プロット: Pareto / Attainment ==========

def _compute_tight_axes(pareto_points, init_ms,
                        margin_frac_x=0.03, margin_frac_y_top=0.10,
                        margin_frac_y_bot=0.03):
    if len(pareto_points) == 0:
        return None, None
    ms_min = float(pareto_points[:, 0].min())
    stab_max = float(pareto_points[:, 1].max())
    xmax = init_ms if init_ms is not None else float(pareto_points[:, 0].max())
    range_ms = max(xmax - ms_min, 10.0)
    xlim = (ms_min - range_ms * margin_frac_x, xmax + range_ms * margin_frac_x)
    range_stab = max(stab_max, 1.0)
    ylim = (-range_stab * margin_frac_y_bot,
            stab_max + range_stab * margin_frac_y_top)
    return xlim, ylim


def _add_initial_marker(ax, init_ms):
    if init_ms is None:
        return
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    ax.axvline(init_ms, color='gray', linestyle='--', alpha=0.7,
               linewidth=1.3, zorder=1)
    if xlim[0] < init_ms < xlim[1]:
        ax.axvspan(init_ms, xlim[1], color='gray', alpha=0.08, zorder=0)
    ax.scatter([init_ms], [0], marker='*', s=200, color='green',
               edgecolors='black', linewidths=0.8, zorder=5,
               label=f'initial (MS={init_ms}, Stab=0)')
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)


def plot_final_pareto_overlay(trial_pts_by_method, methods, baselines_by_method,
                              title, outpath, init_ms=None):
    """各手法の union Pareto を重ねて描画 (baseline で除外)"""
    fig, ax = plt.subplots(figsize=(11, 8))
    all_pareto = []
    for m in methods:
        trial_pts = trial_pts_by_method[m]
        baseline = baselines_by_method.get(m)
        valid = [t for t in trial_pts if len(t) > 0]
        if not valid:
            continue
        combined = np.concatenate(valid)
        pf = pareto_front_2d(combined, baseline=baseline)
        if len(pf) == 0:
            continue
        all_pareto.append(pf)
        color = get_color(m)
        # 散布 + step 線
        order = np.argsort(pf[:, 0])
        pf_s = pf[order]
        ax.scatter(pf_s[:, 0], pf_s[:, 1], color=color, s=45, alpha=0.75,
                   edgecolors='black', linewidths=0.4,
                   label=f"{m} (n={len(pf_s)})", zorder=3)
        ax.step(pf_s[:, 0], pf_s[:, 1], color=color, alpha=0.5,
                where='post', linewidth=1.2, zorder=2)

    if all_pareto:
        union = np.concatenate(all_pareto)
        xlim, ylim = _compute_tight_axes(union, init_ms)
        if xlim: ax.set_xlim(xlim)
        if ylim: ax.set_ylim(ylim)
    _add_initial_marker(ax, init_ms)

    ax.set_xlabel('Makespan')
    ax.set_ylabel('Stability')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=10)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


def plot_snapshot_pareto(anytime_by_method, trial_pts_by_method,
                         kind_by_method, baselines_by_method, methods,
                         snapshot_times, title_base, out_dir, w_label,
                         init_ms=None):
    """T 秒時点での union Pareto 散布 (baseline で除外)"""
    for t in snapshot_times:
        fig, ax = plt.subplots(figsize=(10, 7))
        all_pf = []
        for m in methods:
            anytime = anytime_by_method[m]
            points = trial_pts_by_method[m]
            kind = kind_by_method[m]
            baseline = baselines_by_method.get(m)
            # 全 trial の時刻 t までの点を集めて union Pareto
            all_pts = pareto_points_until_time(anytime, points, kind, t,
                                                baseline=baseline)
            if len(all_pts) == 0:
                continue
            pf = pareto_front_2d(all_pts)
            if len(pf) == 0:
                continue
            all_pf.append(pf)
            color = get_color(m)
            order = np.argsort(pf[:, 0])
            pf_s = pf[order]
            ax.scatter(pf_s[:, 0], pf_s[:, 1], color=color, s=40, alpha=0.7,
                       edgecolors='black', linewidths=0.4,
                       label=f"{m} (n={len(pf_s)})", zorder=3)
            ax.step(pf_s[:, 0], pf_s[:, 1], color=color, alpha=0.5,
                    where='post', linewidth=1.2, zorder=2)
        if all_pf:
            union = np.concatenate(all_pf)
            xlim, ylim = _compute_tight_axes(union, init_ms)
            if xlim: ax.set_xlim(xlim)
            if ylim: ax.set_ylim(ylim)
        _add_initial_marker(ax, init_ms)
        ax.set_xlabel('Makespan')
        ax.set_ylabel('Stability')
        ax.set_title(f"{title_base} - T={t:.0f}s snapshot")
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=9)
        fig.tight_layout()
        t_str = f"{int(t)}" if t == int(t) else f"{t}"
        fig.savefig(os.path.join(out_dir, f"snapshot_pareto_T{t_str}s_{w_label}.png"),
                    dpi=150)
        plt.close(fig)


ATTAIN_LEVELS = [0.25, 0.5, 0.75]
ATTAIN_STYLES = {0.25: ':', 0.5: '--', 0.75: '-'}


def plot_attainment_surfaces(trial_pts_by_method, methods, baselines_by_method,
                             title, outpath, init_ms=None):
    """手法別 25/50/75% attainment surface + union Pareto 散布 (baseline で除外)"""
    all_pts = []
    for m in methods:
        baseline = baselines_by_method.get(m)
        for trial in trial_pts_by_method[m]:
            if len(trial) > 0:
                filtered = filter_by_baseline(trial, baseline)
                if len(filtered) > 0:
                    all_pts.append(filtered)
    if not all_pts:
        return
    all_points = np.concatenate(all_pts)
    grid_ms, grid_st = make_grid(all_points, include_ms=init_ms)

    fig, ax = plt.subplots(figsize=(11, 8))
    all_pareto = []
    for m in methods:
        trial_pts = trial_pts_by_method[m]
        baseline = baselines_by_method.get(m)
        if not any(len(t) > 0 for t in trial_pts):
            continue
        color = get_color(m)
        combined = np.concatenate([t for t in trial_pts if len(t) > 0])
        union_pf = pareto_front_2d(combined, baseline=baseline)
        if len(union_pf) > 0:
            all_pareto.append(union_pf)
            ax.scatter(union_pf[:, 0], union_pf[:, 1], color=color, s=30,
                       alpha=0.6, edgecolors='black', linewidths=0.3,
                       label=f"{m} (n={len(union_pf)})", zorder=3)
        attain = attainment_function_trials(trial_pts, grid_ms, grid_st,
                                             baseline=baseline)
        for lvl in ATTAIN_LEVELS:
            xs, ys = attainment_front_line(attain, grid_ms, grid_st, lvl)
            if len(xs) > 0:
                ax.step(xs, ys, color=color, alpha=0.6,
                        linestyle=ATTAIN_STYLES[lvl], linewidth=1.3,
                        where='pre', zorder=2)

    if all_pareto:
        union = np.concatenate(all_pareto)
        xlim, ylim = _compute_tight_axes(union, init_ms)
        if xlim: ax.set_xlim(xlim)
        if ylim: ax.set_ylim(ylim)
    _add_initial_marker(ax, init_ms)

    from matplotlib.lines import Line2D
    style_legend = [
        Line2D([0], [0], color='gray', linestyle=':',  label='25%'),
        Line2D([0], [0], color='gray', linestyle='--', label='50%'),
        Line2D([0], [0], color='gray', linestyle='-',  label='75%'),
    ]
    leg1 = ax.legend(loc='upper right', fontsize=9, title='method')
    ax.add_artist(leg1)
    ax.legend(handles=style_legend, loc='lower left', fontsize=9,
              title='attainment level')

    ax.set_xlabel('Makespan')
    ax.set_ylabel('Stability')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


def plot_individual_eaf(trial_pts_by_method, methods, baselines_by_method,
                        title, outpath, init_ms=None):
    """各手法の EAF 絶対値を並べて表示 (baseline で除外)"""
    all_pts = []
    for m in methods:
        baseline = baselines_by_method.get(m)
        for trial in trial_pts_by_method[m]:
            if len(trial) > 0:
                filtered = filter_by_baseline(trial, baseline)
                if len(filtered) > 0:
                    all_pts.append(filtered)
    if not all_pts:
        return
    all_points = np.concatenate(all_pts)
    grid_ms, grid_st = make_grid(all_points, include_ms=init_ms)

    n = len(methods)
    fig, axes = plt.subplots(1, n, figsize=(5.2 * n, 6),
                             sharex=True, sharey=True)
    if n == 1:
        axes = [axes]
    # 共通 tight axis
    all_pareto = []
    for m in methods:
        baseline = baselines_by_method.get(m)
        for trial in trial_pts_by_method[m]:
            if len(trial) > 0:
                pf = pareto_front_2d(trial, baseline=baseline)
                if len(pf) > 0:
                    all_pareto.append(pf)
    union = np.concatenate(all_pareto) if all_pareto else np.zeros((0, 2))
    xlim, ylim = _compute_tight_axes(union, init_ms)

    im = None
    for ax, m in zip(axes, methods):
        trial_pts = trial_pts_by_method[m]
        baseline = baselines_by_method.get(m)
        attain = attainment_function_trials(trial_pts, grid_ms, grid_st,
                                             baseline=baseline)
        im = ax.pcolormesh(grid_ms, grid_st, attain.T, cmap='viridis',
                           vmin=0, vmax=1, shading='auto')
        valid = [t for t in trial_pts if len(t) > 0]
        if valid:
            pf = pareto_front_2d(np.concatenate(valid), baseline=baseline)
            if len(pf) > 0:
                ax.scatter(pf[:, 0], pf[:, 1], color='white', s=30,
                           edgecolors='black', linewidths=0.5, zorder=3,
                           label=f'union Pareto (n={len(pf)})')
        if xlim: ax.set_xlim(xlim)
        if ylim: ax.set_ylim(ylim)
        _add_initial_marker(ax, init_ms)
        ax.set_xlabel('Makespan')
        ax.set_ylabel('Stability')
        ax.set_title(f'EAF: {m}')
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, alpha=0.2)

    if im is not None:
        cbar = fig.colorbar(im, ax=axes, shrink=0.8)
        cbar.set_label('EAF (attainment probability)')
    fig.suptitle(title, fontsize=12)
    fig.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_diff_eaf(trial_pts_a, trial_pts_b, label_a, label_b,
                  baseline_a, baseline_b,
                  title, outpath, init_ms=None):
    """差分 EAF: EAF(A) - EAF(B) (baseline で除外)"""
    all_pts_list = []
    for lst, bl in [(trial_pts_a, baseline_a), (trial_pts_b, baseline_b)]:
        for trial in lst:
            if len(trial) > 0:
                filtered = filter_by_baseline(trial, bl)
                if len(filtered) > 0:
                    all_pts_list.append(filtered)
    if not all_pts_list:
        return
    all_points = np.concatenate(all_pts_list)
    grid_ms, grid_st = make_grid(all_points, include_ms=init_ms)
    attain_a = attainment_function_trials(trial_pts_a, grid_ms, grid_st,
                                           baseline=baseline_a)
    attain_b = attainment_function_trials(trial_pts_b, grid_ms, grid_st,
                                           baseline=baseline_b)
    diff = attain_a - attain_b

    fig, ax = plt.subplots(figsize=(11, 8))
    im = ax.pcolormesh(grid_ms, grid_st, diff.T, cmap='RdBu_r',
                       vmin=-1, vmax=1, shading='auto')
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(f'EAF({label_a}) - EAF({label_b})')

    def union_pf(lst, bl):
        valid = [t for t in lst if len(t) > 0]
        if not valid:
            return np.zeros((0, 2))
        return pareto_front_2d(np.concatenate(valid), baseline=bl)

    pf_a = union_pf(trial_pts_a, baseline_a)
    pf_b = union_pf(trial_pts_b, baseline_b)
    if len(pf_a) > 0:
        ax.scatter(pf_a[:, 0], pf_a[:, 1], color='darkred', s=40, alpha=0.8,
                   edgecolors='white', linewidths=0.5, marker='o',
                   label=f'{label_a} (n={len(pf_a)})', zorder=3)
    if len(pf_b) > 0:
        ax.scatter(pf_b[:, 0], pf_b[:, 1], color='darkblue', s=40, alpha=0.8,
                   edgecolors='white', linewidths=0.5, marker='s',
                   label=f'{label_b} (n={len(pf_b)})', zorder=3)

    all_pareto = []
    if len(pf_a) > 0: all_pareto.append(pf_a)
    if len(pf_b) > 0: all_pareto.append(pf_b)
    if all_pareto:
        union = np.concatenate(all_pareto)
        xlim, ylim = _compute_tight_axes(union, init_ms)
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


# ========== テーブル生成 ==========

def format_hv_c_metric_table(trial_pts_by_method, baselines_by_method,
                              methods, ref):
    """Union HV と C-metric（対称行列）のテキスト出力 (baseline で除外)"""
    lines = []
    lines.append(f"参照点 (MS, Stab): ({ref[0]:.1f}, {ref[1]:.3f})")
    lines.append("baseline 以下の点（初期解相当）は除外")
    lines.append("")
    lines.append(f"  {'method':<22} {'HV (per-trial mean±std)':>28} {'HV (union)':>14} {'|Pareto|':>10}")
    lines.append("  " + "-" * 78)

    union_pf_by_method = {}
    for m in methods:
        trials = trial_pts_by_method[m]
        baseline = baselines_by_method.get(m)
        per_t = []
        for trial in trials:
            if len(trial) > 0:
                pf = pareto_front_2d(trial, baseline=baseline)
                per_t.append(hypervolume_2d(pf, ref))
            else:
                per_t.append(0.0)
        mean_hv = float(np.mean(per_t)) if per_t else 0.0
        std_hv = float(np.std(per_t)) if per_t else 0.0
        valid = [t for t in trials if len(t) > 0]
        if valid:
            union_pf = pareto_front_2d(np.concatenate(valid), baseline=baseline)
        else:
            union_pf = np.zeros((0, 2))
        union_pf_by_method[m] = union_pf
        hv_union = hypervolume_2d(union_pf, ref)
        lines.append(f"  {m:<22} "
                     f"{mean_hv:>14.2f} ± {std_hv:<10.2f}"
                     f"{hv_union:>14.2f} {len(union_pf):>10}")

    lines.append("")
    lines.append(f"  C-metric matrix (C(row, col) = row が col をカバーする割合):")
    header = "    " + " " * 22 + "".join(f"{m:>18}" for m in methods)
    lines.append(header)
    for a in methods:
        row = f"    {a:<22}"
        for b in methods:
            if a == b:
                row += f"{'-':>18}"
            else:
                c = c_metric(union_pf_by_method[a], union_pf_by_method[b])
                row += f"{c:>18.3f}"
        lines.append(row)
    return "\n".join(lines), union_pf_by_method


def format_region_hv_table(union_pf_by_method, methods, regions, init_ms):
    """Region-restricted HV のテキスト出力"""
    if regions is None:
        return "(領域計算に必要なデータ不足)"
    lines = []
    lines.append("Region-restricted HV (stab 軸 [0, stab_max] 等 3 分割, stab_max は全手法共通 union Pareto の最大値)")
    lines.append(f"  境界: T1={regions['_t1']:.3f}, T2={regions['_t2']:.3f}, "
                 f"stab_max={regions['_stab_max']:.3f}")
    lines.append(f"  参照点: (init_ms={init_ms}, R_upper)")
    lines.append("")

    for region_name in REGION_NAMES:
        stab_lo, stab_hi = regions[region_name]
        lines.append(f"  [{region_name}]  stab ∈ [{stab_lo:.3f}, {stab_hi:.3f}]")
        lines.append(f"    {'method':<22} {'HV':>14} {'|points|':>10}")
        for m in methods:
            pf = union_pf_by_method.get(m, np.zeros((0, 2)))
            hv, n = region_restricted_hv(pf, (stab_lo, stab_hi), init_ms)
            lines.append(f"    {m:<22} {hv:>14.2f} {n:>10}")
        lines.append("")
    return "\n".join(lines)


def format_snapshot_stats(anytime_by_method, trial_pts_by_method,
                          kind_by_method, baselines_by_method,
                          methods, snapshot_times, ref):
    """T 秒時点での HV / best_score / best_ms / best_st を表形式で (baseline で除外)"""
    lines = []
    lines.append(f"Snapshot stats (各 T 時点、trial 平均)")
    lines.append(f"  参照点 (MS, Stab): ({ref[0]:.1f}, {ref[1]:.3f})")
    for t in snapshot_times:
        lines.append("")
        lines.append(f"  [T = {t:.1f} s]")
        lines.append(f"    {'method':<22} {'HV(union)':>12} {'best_score(avg)':>16} "
                     f"{'best_ms(avg)':>14} {'best_st(avg)':>14}")
        for m in methods:
            kind = kind_by_method[m]
            anytime = anytime_by_method[m]
            points = trial_pts_by_method[m]
            baseline = baselines_by_method.get(m)
            # union HV (baseline で除外した Pareto から)
            all_pts = pareto_points_until_time(anytime, points, kind, t,
                                                baseline=baseline)
            hv = hypervolume_2d(pareto_front_2d(all_pts), ref) if len(all_pts) > 0 else 0.0
            # per-trial best at time t
            bs_list, bm_list, bst_list = [], [], []
            for hist in anytime:
                if hist is None or len(hist) == 0:
                    continue
                times = np.array([h['cpu_time'] for h in hist])
                idx = np.searchsorted(times, t, side='right') - 1
                if idx < 0:
                    continue
                bs_list.append(hist[idx]['best_score'])
                bm_list.append(hist[idx]['best_ms'])
                bst_list.append(hist[idx]['best_st'])
            bs = float(np.mean(bs_list)) if bs_list else float('nan')
            bm = float(np.mean(bm_list)) if bm_list else float('nan')
            bst = float(np.mean(bst_list)) if bst_list else float('nan')
            lines.append(f"    {m:<22} {hv:>12.2f} {bs:>16.4f} "
                         f"{bm:>14.1f} {bst:>14.3f}")
    return "\n".join(lines)


# ========== pair 選定 ==========

def resolve_eaf_pairs(cli_pairs, available_methods):
    """CLI 指定 or デフォルト戦略から pair リストを決定"""
    if cli_pairs:
        pairs = []
        for spec in cli_pairs:
            a, b = spec.split(':')
            if a in available_methods and b in available_methods:
                pairs.append((a, b))
            else:
                print(f"  [WARN] EAF pair '{spec}' skipped: method not present")
        return pairs
    # デフォルト戦略的 pair
    pairs = [(a, b) for (a, b) in DEFAULT_EAF_PAIRS
             if a in available_methods and b in available_methods]
    # 何もマッチしないなら任意の最初の 2 手法を比較
    if not pairs and len(available_methods) >= 2:
        pairs = [(available_methods[1], available_methods[0])]
    return pairs


# ========== メイン処理（per problem × weight） ==========

def analyze_problem_weight(data, out_subdir, w_label, snapshot_times,
                           cli_pairs, xscale='linear'):
    methods = list(data['methods'].keys())
    kind_by_method = {m: data['methods'][m]['kind'] for m in methods}
    baselines_by_method = {m: extract_baseline(data['methods'][m])
                            for m in methods}
    trial_pts_by_method = {m: extract_trial_points(data['methods'][m])
                            for m in methods}
    anytime_by_method = {m: extract_anytime(data['methods'][m]) for m in methods}
    init_ms = data.get('init_makespan')
    prob_label = f"{data['problem']}_{data['scenario']}"
    weights = data['weights']
    title_base = f"{prob_label} w=[{weights[0]}, {weights[1]}]"

    # 参照点: 全手法全 trial の baseline-除外後訪問点の max + マージン
    all_pts_concat = []
    for m in methods:
        baseline = baselines_by_method.get(m)
        for t in trial_pts_by_method[m]:
            if len(t) > 0:
                filtered = filter_by_baseline(t, baseline)
                if len(filtered) > 0:
                    all_pts_concat.append(filtered)
    if not all_pts_concat:
        print(f"  [SKIP] {prob_label} {w_label}: baseline 除外後に訪問点なし")
        return None
    all_points = np.concatenate(all_pts_concat)
    ref_ms = float(all_points[:, 0].max()) + max(all_points[:, 0].max() * 0.01, 1.0)
    ref_st = float(all_points[:, 1].max()) + max(all_points[:, 1].max() * 0.01, 0.1)
    ref = (ref_ms, ref_st)

    print(f"  分析: {prob_label} {w_label}  (methods={methods}, init_ms={init_ms})")
    for m in methods:
        b = baselines_by_method.get(m)
        if b:
            print(f"    baseline[{m}] = (ms={b[0]:.1f}, stab={b[1]:.4f})")

    # 先に Region 境界を計算（全手法 baseline-除外後の union Pareto から自動決定）
    all_union_pareto = []
    for m in methods:
        baseline = baselines_by_method.get(m)
        for t in trial_pts_by_method[m]:
            if len(t) > 0:
                pf = pareto_front_2d(t, baseline=baseline)
                if len(pf) > 0:
                    all_union_pareto.append(pf)
    cross_method_union = (pareto_front_2d(np.concatenate(all_union_pareto))
                          if all_union_pareto else np.zeros((0, 2)))
    regions = compute_regions(cross_method_union)

    # 1) anytime HV curve (full)
    plot_anytime_hv(trial_pts_by_method, anytime_by_method, methods,
                    kind_by_method, baselines_by_method,
                    ref, init_ms,
                    f"{title_base}: anytime HV (full)",
                    os.path.join(out_subdir, f"anytime_hv_{w_label}.png"),
                    xscale=xscale)

    # 1b) anytime Region-restricted HV (3 regions × N methods)
    plot_anytime_region_hv(trial_pts_by_method, anytime_by_method, methods,
                           kind_by_method, baselines_by_method,
                           regions, init_ms,
                           f"{title_base}: anytime Region-restricted HV",
                           os.path.join(out_subdir,
                                         f"anytime_region_hv_{w_label}.png"),
                           xscale=xscale)

    # 2) anytime scalar curve
    plot_anytime_scalar(anytime_by_method, methods,
                         f"{title_base}: anytime best weighted score",
                         os.path.join(out_subdir, f"anytime_scalar_{w_label}.png"),
                         trial_pts_by_method=trial_pts_by_method,
                         kind_by_method=kind_by_method,
                         baselines_by_method=baselines_by_method,
                         xscale=xscale)

    # 3) final Pareto overlay
    plot_final_pareto_overlay(trial_pts_by_method, methods, baselines_by_method,
                              f"{title_base}: final union Pareto (all trials)",
                              os.path.join(out_subdir, f"final_pareto_{w_label}.png"),
                              init_ms=init_ms)

    # 4) snapshot Pareto
    plot_snapshot_pareto(anytime_by_method, trial_pts_by_method, kind_by_method,
                         baselines_by_method, methods, snapshot_times,
                         title_base, out_subdir, w_label, init_ms=init_ms)

    # 5) attainment surfaces
    plot_attainment_surfaces(trial_pts_by_method, methods, baselines_by_method,
                             f"{title_base}: Attainment Surfaces (25/50/75%)",
                             os.path.join(out_subdir, f"attainment_{w_label}.png"),
                             init_ms=init_ms)

    # 6) individual EAF
    plot_individual_eaf(trial_pts_by_method, methods, baselines_by_method,
                        f"{title_base}: Individual EAF",
                        os.path.join(out_subdir, f"individual_eaf_{w_label}.png"),
                        init_ms=init_ms)

    # 7) diff EAF (pairs)
    pairs = resolve_eaf_pairs(cli_pairs, methods)
    for a, b in pairs:
        safe = f"{a}_vs_{b}".replace(':', '_')
        plot_diff_eaf(trial_pts_by_method[a], trial_pts_by_method[b], a, b,
                      baselines_by_method.get(a), baselines_by_method.get(b),
                      f"{title_base}: diff EAF  {a} − {b}",
                      os.path.join(out_subdir, f"diff_eaf_{safe}_{w_label}.png"),
                      init_ms=init_ms)

    # 8) HV + C-metric テーブル
    hv_text, union_pf_by_method = format_hv_c_metric_table(
        trial_pts_by_method, baselines_by_method, methods, ref)
    with open(os.path.join(out_subdir, f"hv_cmetric_{w_label}.txt"),
              'w', encoding='utf-8') as f:
        f.write(f"=== {title_base} ===\n\n")
        f.write(hv_text + "\n")

    # 9) Region-restricted HV テーブル
    region_text = format_region_hv_table(union_pf_by_method, methods, regions,
                                          init_ms)
    with open(os.path.join(out_subdir, f"region_hv_{w_label}.txt"),
              'w', encoding='utf-8') as f:
        f.write(f"=== {title_base} ===\n\n")
        f.write(region_text + "\n")

    # 10) snapshot stats
    snap_text = format_snapshot_stats(anytime_by_method, trial_pts_by_method,
                                       kind_by_method, baselines_by_method,
                                       methods, snapshot_times, ref)
    with open(os.path.join(out_subdir, f"snapshot_stats_{w_label}.txt"),
              'w', encoding='utf-8') as f:
        f.write(f"=== {title_base} ===\n\n")
        f.write(snap_text + "\n")

    return {
        'prob_label': prob_label,
        'weights': weights,
        'methods': methods,
        'union_pf_by_method': {m: union_pf_by_method[m].tolist() for m in methods},
        'regions': {k: v for k, v in regions.items()} if regions else None,
        'ref': ref,
        'init_ms': init_ms,
        'trial_pts_by_method': trial_pts_by_method,
        'anytime_by_method': anytime_by_method,
        'kind_by_method': kind_by_method,
        'finals_by_method': {m: extract_finals(data['methods'][m]) for m in methods},
    }


# ========== cross-problem 集約 ==========

def plot_degeneracy_heatmap(all_results, out_path):
    """改善成功率 heatmap: (weights × methods) × problem サブプロット"""
    problems = sorted({r['prob_label'] for r in all_results})
    weights_set = sorted({tuple(r['weights']) for r in all_results})
    # 手法順は最初に見つけた順を保つ
    method_order = []
    for r in all_results:
        for m in r['methods']:
            if m not in method_order:
                method_order.append(m)
    n_prob = len(problems)
    if n_prob == 0 or not method_order or not weights_set:
        return
    fig, axes = plt.subplots(1, n_prob, figsize=(4.5 * n_prob, 3 + 0.3 * len(weights_set)),
                             squeeze=False)
    axes = axes[0]

    for ax, prob in zip(axes, problems):
        mat = np.full((len(weights_set), len(method_order)), np.nan)
        for i, w in enumerate(weights_set):
            for j, m in enumerate(method_order):
                for r in all_results:
                    if r['prob_label'] != prob or tuple(r['weights']) != w:
                        continue
                    if m not in r['methods']:
                        continue
                    finals = r['finals_by_method'].get(m, [])
                    init_ms = r['init_ms']
                    if init_ms is None or not finals:
                        continue
                    n_improved = sum(1 for d in finals if d['makespan'] < init_ms)
                    mat[i, j] = n_improved / len(finals)
        im = ax.imshow(mat, cmap='RdYlGn', vmin=0, vmax=1, aspect='auto')
        ax.set_xticks(range(len(method_order)))
        ax.set_xticklabels(method_order, rotation=30, ha='right', fontsize=9)
        ax.set_yticks(range(len(weights_set)))
        ax.set_yticklabels([f"[{w[0]},{w[1]}]" for w in weights_set], fontsize=9)
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                if np.isnan(mat[i, j]):
                    continue
                ax.text(j, i, f"{mat[i,j]*100:.0f}%",
                        ha='center', va='center', fontsize=8,
                        color='white' if mat[i, j] < 0.5 else 'black')
        ax.set_title(prob, fontsize=10)
    fig.colorbar(im, ax=axes, shrink=0.7, label='improve rate')
    fig.suptitle('Improvement success rate (makespan < init_ms)', fontsize=12)
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def write_cross_summary(all_results, out_dir):
    """cross-problem の HV/ Region-restricted HV / C-metric サマリ"""
    path = os.path.join(out_dir, "cross_summary.txt")
    with open(path, 'w', encoding='utf-8') as f:
        f.write("コア比較分析 横断サマリー\n")
        f.write("=" * 70 + "\n\n")
        for r in all_results:
            f.write(f"--- {r['prob_label']}  weights={r['weights']} ---\n")
            f.write(f"  init_ms={r['init_ms']}, "
                    f"ref=({float(r['ref'][0]):.1f}, {float(r['ref'][1]):.3f})\n")
            f.write(f"  methods={r['methods']}\n")
            # 各 method の union HV
            f.write(f"  union HV:\n")
            for m in r['methods']:
                pf = np.array(r['union_pf_by_method'][m])
                hv = hypervolume_2d(pf, r['ref']) if len(pf) > 0 else 0.0
                f.write(f"    {m:<22} HV={hv:.2f}  |Pareto|={len(pf)}\n")
            # region HV
            if r['regions'] is not None:
                reg = r['regions']
                f.write(f"  Region-restricted HV "
                        f"(T1={reg['_t1']:.3f}, T2={reg['_t2']:.3f}):\n")
                for region_name in REGION_NAMES:
                    stab_lo, stab_hi = reg[region_name]
                    f.write(f"    [{region_name} stab∈[{stab_lo:.3f},{stab_hi:.3f}]]\n")
                    for m in r['methods']:
                        pf = np.array(r['union_pf_by_method'][m])
                        hv, n = region_restricted_hv(
                            pf, (stab_lo, stab_hi), r['init_ms'])
                        f.write(f"      {m:<22} HV={hv:.2f}  n={n}\n")
            f.write("\n")
    print(f"  横断サマリ: {path}")


# ========== エントリポイント ==========

def main():
    parser = argparse.ArgumentParser(description='実験1 コア比較の分析')
    parser.add_argument('results_dir', type=str,
                        help='run_core_comparison.py の出力ディレクトリ')
    parser.add_argument('--out-dir', type=str, default=None,
                        help='分析出力先 (デフォルト: <results_dir>/analysis)')
    parser.add_argument('--snapshot-times', nargs='+', type=float,
                        default=DEFAULT_SNAPSHOT_TIMES,
                        help=f'snapshot する CPU 時刻 (s). デフォルト {DEFAULT_SNAPSHOT_TIMES}')
    parser.add_argument('--eaf-pairs', nargs='+', type=str, default=None,
                        help='差分 EAF を描く pair. 形式: A:B (例: ils_insert:ga). '
                             '指定なしなら戦略 pair を自動選択')
    parser.add_argument('--xscale', type=str, default='linear',
                        choices=['linear', 'log'],
                        help='anytime curve の横軸スケール (デフォルト: linear)')
    args = parser.parse_args()

    results_dir = args.results_dir
    out_dir = args.out_dir or os.path.join(results_dir, 'analysis')
    os.makedirs(out_dir, exist_ok=True)

    per_problem_dir = os.path.join(out_dir, 'per_problem')
    os.makedirs(per_problem_dir, exist_ok=True)
    cross_dir = os.path.join(out_dir, 'cross_problem')
    os.makedirs(cross_dir, exist_ok=True)

    # 各 problem_scenario / weights の JSON をスキャン
    all_results = []
    for entry in sorted(os.listdir(results_dir)):
        prob_path = os.path.join(results_dir, entry)
        if not os.path.isdir(prob_path) or entry in ('analysis',):
            continue
        prob_out = os.path.join(per_problem_dir, entry)
        os.makedirs(prob_out, exist_ok=True)
        for fn in sorted(os.listdir(prob_path)):
            if not fn.startswith('results_') or not fn.endswith('.json'):
                continue
            w_label = fn[len('results_'):-len('.json')]
            w_out = os.path.join(prob_out, w_label)
            os.makedirs(w_out, exist_ok=True)
            data = load_result_json(os.path.join(prob_path, fn))
            r = analyze_problem_weight(data, w_out, w_label,
                                        args.snapshot_times, args.eaf_pairs,
                                        xscale=args.xscale)
            if r is not None:
                all_results.append(r)

    if not all_results:
        print('分析対象のデータが見つかりませんでした。')
        return

    # cross-problem
    plot_degeneracy_heatmap(
        all_results,
        os.path.join(cross_dir, 'degeneracy_heatmap.png'))
    write_cross_summary(all_results, cross_dir)

    print(f'\n分析完了: {out_dir}')


if __name__ == '__main__':
    main()
