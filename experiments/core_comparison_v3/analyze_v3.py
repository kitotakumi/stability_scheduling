#!/usr/bin/env python3
"""
core_comparison_v3: コア比較実験 v3 分析スクリプト

run_v3.py が生成した raw データから evaluation_design.md §6 の指標・図表を生成する。

=== 主な出力 ===
  <input-dir>/analysis/
  └── <problem>_<scenario>/
      ├── thresholds.json            # P33/P67/stab_max（B-2b 境界）
      ├── b1_scalar_stats.txt        # per-weight scalar Wilcoxon (B-1)
      ├── b1_uea_hv_stats.txt        # per-weight UEA HV Wilcoxon (B-1)
      ├── b1_improvement_heatmap.png # 改善成功率 heatmap (B-1 degeneracy)
      ├── b2a_union_hv_boxplot.png   # per-trial union UEA HV (B-2a 主筋)
      ├── b2a_union_hv_stats.txt     # Wilcoxon + Cliff's delta (B-2a)
      ├── b2a_c_metric.txt           # C-metric (B-2a)
      ├── b2b_coverage.txt           # カバー率 (B-2b Step 1)
      ├── b2b_region_hv.png          # 領域別 HV (B-2b)
      ├── b2b_cond_ms_wilcoxon.txt   # 条件付き MS Wilcoxon (B-2b Step 2)
      ├── b2b_diff_eaf.png           # 差分 EAF (B-2b 視覚証拠)
      ├── n_sensitivity.txt          # N=3/6/11 union HV (lucky punch 対策)
      ├── anytime_scalar_<w>.png     # anytime scalar curve (代表重み, A)
      └── anytime_uea_hv_<w>.png    # anytime per-weight UEA HV curve (A)

=== 使い方 ===
  python analyze_v3.py --input-dir results/core_v3_<timestamp>/
  python analyze_v3.py --input-dir results/main/ --problems la36 la40
  python -u experiments/core_comparison_v3/analyze_v3.py --input-dir experiments/core_comparison_v3/results/pilot_v3 --repr-weights w08_02 --n-jobs 4

"""

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from itertools import combinations

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, '..', '..'))
sys.path.insert(0, os.path.join(_HERE, '..'))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from experiment_utils import compute_shared_norm_params, get_initial_makespan

try:
    from scipy.stats import wilcoxon as scipy_wilcoxon, fisher_exact
    SCIPY_OK = True
except ImportError:
    SCIPY_OK = False
    print('[WARN] scipy not found. 統計検定はスキップされます。')


# ========== 定数 ==========

METHOD_COLORS = {
    'ga':             'tab:gray',
    'ils_baseline':   'tab:orange',
    'ils_repair':     'tab:red',
    'ils_pr':         'tab:blue',
    'memetic_ls':     'tab:green',
    'memetic_repair': 'tab:purple',
    'memetic_pr':     'tab:brown',
}
METHOD_LABELS = {
    'ga':             'GA',
    'ils_baseline':   'ILS-baseline',
    'ils_repair':     'ILS+repair',
    'ils_pr':         'ILS+PR',
    'memetic_ls':     'Memetic-LS',
    'memetic_repair': 'Memetic+repair',
    'memetic_pr':     'Memetic+PR',
}
DEFAULT_COLORS = ['tab:cyan', 'tab:olive', 'tab:pink', 'deepskyblue']

# anytime 曲線を描く代表重み (w_label 形式) — CLI の --repr-weights で上書き可
REPR_WEIGHTS_DEFAULT = ['w08_02']

# N sensitivity check で使う重みセット (w_label リスト)
N_SENSITIVITY = {
    3:  ['w10_00', 'w05_05', 'w00_10'],
    6:  ['w10_00', 'w08_02', 'w06_04', 'w04_06', 'w02_08', 'w00_10'],
    11: None,  # None = 全重み
}

SNAPSHOT_TIMES = [5.0, 10.0, 20.0, 40.0]

# 比較ペア (main, secondary)
COMPARE_PAIRS = [
    ('ils_baseline',   'ga'),
    ('ils_repair',     'ga'),
    ('ils_pr',         'ga'),
    ('ils_repair',     'ils_baseline'),
    ('ils_pr',         'ils_baseline'),
    ('memetic_ls',     'ga'),
    ('memetic_repair', 'ga'),
    ('memetic_pr',     'ga'),
    ('memetic_repair', 'memetic_ls'),
    ('memetic_pr',     'memetic_ls'),
]


# ========== データ読み込み ==========

def load_all_runs(input_dir):
    """raw ディレクトリを再帰スキャンして全 run を読み込む。
    Returns: {(problem, scenario): {method: {w_label: {trial_idx: data_dict}}}}
    """
    grouped = {}
    for prob_dir_name in sorted(os.listdir(input_dir)):
        raw_dir = os.path.join(input_dir, prob_dir_name, 'raw')
        if not os.path.isdir(raw_dir):
            continue
        for fn in sorted(os.listdir(raw_dir)):
            if not fn.endswith('.json'):
                continue
            fpath = os.path.join(raw_dir, fn)
            try:
                with open(fpath, 'r', encoding='utf-8') as f:
                    d = json.load(f)
            except Exception as e:
                print(f'  [WARN] 読み込み失敗 {fpath}: {e}')
                continue
            prob = d.get('problem', '')
            scen = d.get('scenario', '')
            method = d.get('method', '')
            weights = d.get('weights', [])
            trial = d.get('trial', 0)
            if not (prob and scen and method and weights):
                continue
            w_label = _weight_label(weights)
            key = (prob, scen)
            grouped.setdefault(key, {})
            grouped[key].setdefault(method, {})
            grouped[key][method].setdefault(w_label, {})
            grouped[key][method][w_label][trial] = d
    return grouped


def _weight_label(w):
    if isinstance(w, str):
        return w
    return f"w{int(round(w[0] * 10)):02d}_{int(round(w[1] * 10)):02d}"


def _improved_over_baseline(data):
    """重みスカラー値が baseline より改善したか（小さいほど良い）。

    baseline_score が保存されていない旧データの場合は False を返す。
    """
    b_score = data.get('baseline_score')
    if b_score is None:
        return False
    history = data.get('history', [])
    final_scores = [h.get('best_score') for h in history if h.get('best_score') is not None]
    if not final_scores:
        return False
    f_score = final_scores[-1]
    eps = 1e-6
    return float(f_score) < float(b_score) - eps


def get_uea_points(data_dict, trial_idx):
    """run data から trial の UEA 点列を np.array (N,2) で返す。"""
    pts = data_dict.get('uea_points', [])
    if not pts:
        return np.zeros((0, 2))
    arr = np.array(pts, dtype=float)
    if arr.ndim != 2 or arr.shape[1] < 2:
        return np.zeros((0, 2))
    return arr[:, :2]


def get_uea_points_xyt(data_dict, trial_idx):
    """anytime 用に UEA 点を np.array (N,3)=[ms, st, cpu_time] で返す。

    run_v3 が記録した uea_points_t（各点の正確な訪問時刻）があれば 3 列目に付与する。
    古い結果（uea_points_t なし）は (N,2) を返し、_point_times が hist から近似する。
    """
    pts = data_dict.get('uea_points', [])
    if not pts:
        return np.zeros((0, 2))
    arr = np.array(pts, dtype=float)
    if arr.ndim != 2 or arr.shape[1] < 2:
        return np.zeros((0, 2))
    arr = arr[:, :2]
    ts = data_dict.get('uea_points_t')
    if ts is not None and len(ts) == len(arr):
        return np.hstack([arr, np.asarray(ts, dtype=float).reshape(-1, 1)])
    return arr


def get_anytime(data_dict):
    """anytime history を [{cpu_time, best_ms, best_st, best_score, evaluations}] で返す。"""
    return data_dict.get('history', [])



def get_method_color(m, idx=0):
    return METHOD_COLORS.get(m, DEFAULT_COLORS[idx % len(DEFAULT_COLORS)])


# ========== Pareto / HV / C-metric ==========

def filter_baseline(points, baseline):
    """baseline に弱く支配される点 (ms >= b_ms AND st >= b_st) を除外。"""
    if baseline is None or len(points) == 0:
        return points
    pts = np.asarray(points)
    b_ms, b_st = baseline
    eps = 1e-9
    mask = ~((pts[:, 0] >= b_ms - eps) & (pts[:, 1] >= b_st - eps))
    return pts[mask]


def filter_baselines(points, baselines):
    """複数 baseline すべてに対して filter_baseline を順次適用。
    baselines は [[ms,st], ...] のリスト、または単一の [ms,st]、または None。
    """
    if not baselines or len(points) == 0:
        return np.asarray(points)
    pts = np.asarray(points)
    bl_list = baselines if (isinstance(baselines, list) and
                            isinstance(baselines[0], (list, np.ndarray))) else [baselines]
    for bl in bl_list:
        pts = filter_baseline(pts, bl)
        if len(pts) == 0:
            break
    return pts


def pareto_front(points):
    """2D 最小化 Pareto front を返す。points: (N,2)"""
    if len(points) == 0:
        return np.zeros((0, 2))
    pts = np.asarray(points, dtype=float)
    idx = np.lexsort((pts[:, 1], pts[:, 0]))
    srt = pts[idx]
    if len(srt) == 1:
        return srt
    # x昇順ソート済み → y の累積最小値より小さい点がPareto front
    cummin_y = np.minimum.accumulate(srt[:, 1])
    on_front = np.empty(len(srt), dtype=bool)
    on_front[0] = True
    on_front[1:] = srt[1:, 1] < cummin_y[:-1]
    return srt[on_front]


def hypervolume(points, ref):
    """2D 最小化 HV。ref = (ms_ref, st_ref)"""
    if len(points) == 0:
        return 0.0
    pf = pareto_front(points)
    if len(pf) == 0:
        return 0.0
    pf = pf[np.argsort(pf[:, 0])]
    hv = 0.0
    prev_x = ref[0]
    for p in pf[::-1]:
        if p[0] >= ref[0] or p[1] >= ref[1]:
            continue
        hv += (prev_x - p[0]) * (ref[1] - p[1])
        prev_x = p[0]
    return hv


def c_metric(A, B):
    """C(A,B): A が弱く支配する B の割合。"""
    if len(B) == 0:
        return 0.0
    count = 0
    for b in B:
        for a in A:
            if a[0] <= b[0] and a[1] <= b[1]:
                count += 1
                break
    return count / len(B)


def region_hv(points, stab_lo, stab_hi, ref_ms, ref_stab=None,
              hi_inclusive=False, stab_margin=0.02):
    """領域内の点だけで HV を計算。

    区間: hi_inclusive=False (デフォルト) → [stab_lo, stab_hi)
           hi_inclusive=True              → [stab_lo, stab_hi]
    ref_stab: stab 側参照点。None の場合はローカル参照点
              (stab_hi + stab_margin * width) を使用する。
              low/mid 領域では各ゾーン上限を参照点にすることで
              ゾーン内品質を独立に評価できる。
    """
    if len(points) == 0:
        return 0.0, 0
    pts = np.asarray(points)
    if hi_inclusive:
        mask = (pts[:, 1] >= stab_lo) & (pts[:, 1] <= stab_hi)
    else:
        mask = (pts[:, 1] >= stab_lo) & (pts[:, 1] < stab_hi)
    reg = pts[mask]
    if len(reg) == 0:
        return 0.0, 0
    if ref_stab is None:
        width = max(stab_hi - stab_lo, 1e-6)
        ref_stab = stab_hi + stab_margin * width
    ref = (ref_ms, ref_stab)
    return hypervolume(reg, ref), int(len(reg))


# ========== P33/P67 閾値計算（新方式） ==========

def compute_p33_p67(method_data_by_method_weight_trial, baselines_by_method):
    """各手法×各trial の個別 Pareto front 解を全プール → P33/P67 を計算。

    手順:
      1. 各 (method, w_label, trial) について UEA 点から trial 個別 PF を構築
      2. 全 PF 解を cross-method dominance フィルタなしで一括プール（集合 S）
      3. S の stab 値の P33・P67 を返す（実験全体で固定）
    """
    all_stab = []
    for method, by_weight in method_data_by_method_weight_trial.items():
        baseline = baselines_by_method.get(method)
        for w_label, by_trial in by_weight.items():
            for trial_idx, data in by_trial.items():
                pts = get_uea_points(data, trial_idx)
                if baseline is not None:
                    pts = filter_baselines(pts, baseline)
                if len(pts) == 0:
                    continue
                pf = pareto_front(pts)
                if len(pf) > 0:
                    all_stab.extend(pf[:, 1].tolist())

    if not all_stab:
        return None
    all_stab = np.array(all_stab)
    p33 = float(np.percentile(all_stab, 33))
    p50 = float(np.percentile(all_stab, 50))
    p67 = float(np.percentile(all_stab, 67))
    stab_max = float(all_stab.max())
    return {'P33': p33, 'P50': p50, 'P67': p67, 'stab_max': stab_max}


# ========== 統計検定 ==========

def wilcoxon_paired(x, y, alternative='less'):
    """paired Wilcoxon signed-rank. alternative='less' → x < y が主張。
    Returns (stat, p_value). n<10 or 全差 0 なら (nan, nan)。
    """
    if not SCIPY_OK:
        return float('nan'), float('nan')
    x, y = np.array(x, dtype=float), np.array(y, dtype=float)
    paired = [(xi, yi) for xi, yi in zip(x, y)
              if np.isfinite(xi) and np.isfinite(yi)]
    if len(paired) < 10:
        return float('nan'), float('nan')
    xp, yp = zip(*paired)
    diff = np.array(xp) - np.array(yp)
    if np.all(diff == 0):
        return float('nan'), float('nan')
    try:
        stat, p = scipy_wilcoxon(diff, alternative=alternative)
        return float(stat), float(p)
    except Exception:
        return float('nan'), float('nan')


def cliffs_delta(x, y):
    """Cliff's delta: (P(x<y) - P(x>y)). 負 = x が y より小さい傾向。"""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]
    if len(x) == 0 or len(y) == 0:
        return float('nan')
    total = len(x) * len(y)
    concordant = sum(1 for xi in x for yj in y if xi < yj)
    discordant = sum(1 for xi in x for yj in y if xi > yj)
    return (concordant - discordant) / total


def holm_bonferroni(p_values):
    """Holm 補正。Returns: corrected p-values (same order)。"""
    n = len(p_values)
    if n == 0:
        return []
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    corrected = [None] * n
    for rank, (orig_idx, p) in enumerate(indexed):
        corrected[orig_idx] = min(1.0, p * (n - rank))
    # monotonicity adjustment
    max_so_far = 0.0
    for _, (orig_idx, _) in enumerate(indexed):
        corrected[orig_idx] = max(corrected[orig_idx], max_so_far)
        max_so_far = corrected[orig_idx]
    return corrected


def effect_label(delta):
    """Cliff's delta → 効果量ラベル (|d| < 0.147: negligible, 0.33: small, 0.474: medium)"""
    a = abs(delta)
    if np.isnan(a):
        return '?'
    if a < 0.147:
        return 'neg'
    if a < 0.330:
        return 'small'
    if a < 0.474:
        return 'med'
    return 'large'


def _p_star(p):
    if np.isnan(p):
        return '  '
    if p < 0.001:
        return '***'
    if p < 0.01:
        return '** '
    if p < 0.05:
        return '*  '
    return '   '


# ========== Anytime ユーティリティ ==========

def _point_times(pts, hist, kind):
    """各 UEA 点に訪問 cpu_time を割り当てて (pts_xy, ptimes) を返す。

    pts が (N,3) の場合は 3 列目を「run_v3 が記録した正確な訪問時刻」として使う
    （uea_points_t、推奨経路）。(N,2) の旧データは hist の cpu_time/evaluations から
    index/eval ベースで近似再構成する（後方互換）。後者は ILS で kick 点が混ざると
    点数 > 履歴数になり対応がずれ得るため、新データでは必ず (N,3) を渡すこと。

    Returns:
        (pts_xy (N,2) float, ptimes (N,) float)
    """
    arr = np.asarray(pts, dtype=float)
    if arr.ndim != 2 or arr.shape[0] == 0 or arr.shape[1] < 2:
        return np.zeros((0, 2)), np.zeros(0)
    pts_xy = arr[:, :2]
    N = len(pts_xy)
    if arr.shape[1] >= 3:
        return pts_xy, arr[:, 2]
    # --- fallback: 旧 2 列データ。hist から近似 ---
    if not hist:
        return pts_xy, np.zeros(N)
    if kind == 'ga':
        ev_vals = np.array([int(e['evaluations']) if e.get('evaluations') else 0
                            for e in hist])
        cpu_vals = np.array([float(e['cpu_time']) if e.get('cpu_time') is not None else 0.0
                             for e in hist])
        valid = ev_vals > 0
        if not np.any(valid):
            return pts_xy, np.full(N, float(hist[-1].get('cpu_time') or 0.0))
        ev_v = ev_vals[valid]
        cpu_v = cpu_vals[valid]
        order = np.argsort(ev_v, kind='stable')
        ev_sorted = ev_v[order]
        cpu_sorted = cpu_v[order]
        idx = np.clip(np.searchsorted(ev_sorted, np.arange(1, N + 1), side='left'),
                      0, len(cpu_sorted) - 1)
        return pts_xy, cpu_sorted[idx]
    else:
        h_idx = np.minimum(np.arange(1, N + 1), len(hist) - 1)
        ptimes = np.array([float(hist[i].get('cpu_time') or 0.0) for i in h_idx])
        return pts_xy, ptimes


def _pareto_until_t(hist_list, pts_list, kind, t, baseline=None):
    """時刻 t 以前に訪問された全点（baseline 除外）を返す。"""
    collected = []
    for hist, pts in zip(hist_list, pts_list):
        if len(pts) == 0:
            continue
        pts_xy, ptimes = _point_times(pts, hist, kind)
        sub = pts_xy[ptimes <= t]
        if len(sub) > 0:
            collected.append(sub)
    if not collected:
        return np.zeros((0, 2))
    combined = np.concatenate(collected)
    if baseline is not None:
        combined = filter_baselines(combined, baseline)
    return combined


def _last_pf_update_time(hist, pts, kind, baseline=None):
    """trial 内で最後に Pareto front が更新された cpu_time を返す。"""
    if len(pts) == 0:
        return None
    # 各点の訪問時刻: 記録済み (N,3) なら正確、(N,2) 旧データは hist から近似
    pts_arr, t_arr = _point_times(pts, hist, kind)
    N = len(pts_arr)
    if N == 0:
        return None

    if baseline is not None:
        bl_list = (baseline if (isinstance(baseline, list) and
                   isinstance(baseline[0], (list, np.ndarray))) else [baseline])
        keep = np.ones(len(pts_arr), dtype=bool)
        eps = 1e-9
        for bl in bl_list:
            dominated = (pts_arr[:, 0] >= bl[0] - eps) & (pts_arr[:, 1] >= bl[1] - eps)
            keep &= ~dominated
        pts_arr = pts_arr[keep]
        t_arr = t_arr[keep]
        N = len(pts_arr)
    if N == 0:
        return None

    # グローバル Pareto front の点のうち最遅の訪問時刻を返す (O(N log N))
    # 大規模データ用近似: 最後に Pareto front を更新した時刻の近似として十分
    if N > 500:
        pf = pareto_front(pts_arr)
        if len(pf) == 0:
            return None
        # numpy 一括比較で Pareto front 点の訪問時刻を取得 (pf が小さい場合に高速)
        pf_match = np.zeros(N, dtype=bool)
        for pf_pt in pf:
            pf_match |= ((pts_arr[:, 0] == pf_pt[0]) & (pts_arr[:, 1] == pf_pt[1]))
        pf_times = t_arr[pf_match]
        return float(np.max(pf_times)) if len(pf_times) > 0 else float(t_arr[-1])

    # 小規模: 時系列 Pareto front 追跡
    sort_idx = np.argsort(t_arr, kind='stable')
    pts_s = pts_arr[sort_idx]
    t_s = t_arr[sort_idx]
    current = np.zeros((0, 2))
    last_t = None
    for i in range(N):
        p = pts_s[i]
        if len(current) > 0:
            dominated = np.any(
                (current[:, 0] <= p[0]) & (current[:, 1] <= p[1]) &
                ~((current[:, 0] == p[0]) & (current[:, 1] == p[1]))
            )
            if dominated:
                continue
            keep = ~((p[0] <= current[:, 0]) & (p[1] <= current[:, 1]) &
                     ~((current[:, 0] == p[0]) & (current[:, 1] == p[1])))
            current = current[keep]
        current = np.vstack([current, p]) if len(current) > 0 else p.reshape(1, 2)
        last_t = float(t_s[i])
    return last_t


def _build_t_grid(methods_info, n_pts=40, xscale='log'):
    """各手法 trial の最終 PF 更新時刻の median → 手法間 max → t_grid を構築。"""
    medians = []
    for m, hist_list, pts_list, kind, baseline in methods_info:
        lasts = []
        for hist, pts in zip(hist_list, pts_list):
            t = _last_pf_update_time(hist, pts, kind, baseline)
            if t and t > 0:
                lasts.append(t)
        if lasts:
            medians.append(float(np.median(lasts)))
    if not medians:
        # fallback: use max cpu_time from any trial's history
        for m, hist_list, pts_list, kind, baseline in methods_info:
            for hist in hist_list:
                if hist and hist[-1].get('cpu_time'):
                    medians.append(float(hist[-1]['cpu_time']))
    if not medians:
        return None
    t_max = max(medians)
    t_min = max(0.1, t_max * 0.02)
    if t_min >= t_max:
        return None
    if xscale == 'log':
        t_min = max(0.02, t_max * 0.002)
        return np.geomspace(t_min, t_max, n_pts)
    return np.linspace(t_min, t_max, n_pts)


# ========== EAF ==========

GRID_N = 100


def _make_grid(all_pts, include_ms=None, pad=0.05):
    if len(all_pts) == 0:
        return np.linspace(0, 1, GRID_N), np.linspace(0, 1, GRID_N)
    ms_min, ms_max = all_pts[:, 0].min(), all_pts[:, 0].max()
    if include_ms:
        ms_max = max(ms_max, include_ms)
    st_min, st_max = all_pts[:, 1].min(), all_pts[:, 1].max()
    ms_pad = max((ms_max - ms_min) * pad, 1.0)
    st_pad = max((st_max - st_min) * pad, 0.01)
    return (np.linspace(ms_min - ms_pad, ms_max + ms_pad, GRID_N),
            np.linspace(st_min - st_pad, st_max + st_pad, GRID_N))


def _eaf_grid(trial_pts_list, grid_ms, grid_st, baseline=None):
    """trial-based EAF: 各格子点を attainment する trial の割合。"""
    n = len(trial_pts_list)
    if n == 0:
        return np.zeros((len(grid_ms), len(grid_st)))
    MS, ST = np.meshgrid(grid_ms, grid_st, indexing='ij')
    count = np.zeros_like(MS, dtype=float)
    for pts in trial_pts_list:
        if len(pts) == 0:
            continue
        if baseline is not None:
            pts = filter_baselines(pts, baseline)
        pf = pareto_front(pts)
        if len(pf) == 0:
            continue
        mask = np.zeros_like(MS, dtype=bool)
        for p in pf:
            mask |= (p[0] <= MS) & (p[1] <= ST)
        count += mask.astype(float)
    return count / n


# ========== per-trial union UEA HV (B-2a 主筋) ==========

def compute_union_hv_per_trial(method_data_by_weight_trial, baselines_by_method,
                                ref, weights_subset=None):
    """各 (method, trial) で union UEA HV を計算して返す。

    method_data_by_weight_trial: {method: {w_label: {trial_idx: data}}}
    weights_subset: None = 全重み、list[str] = 指定 w_label のみ

    Returns: {method: list[float]}  (trial 順)
    """
    result = {}
    all_methods = list(method_data_by_weight_trial.keys())
    if not all_methods:
        return result

    all_w_labels = set()
    for m in all_methods:
        all_w_labels.update(method_data_by_weight_trial[m].keys())
    use_weights = (set(weights_subset) & all_w_labels
                   if weights_subset else all_w_labels)

    all_trials = set()
    for m in all_methods:
        for w in use_weights:
            all_trials.update(method_data_by_weight_trial[m].get(w, {}).keys())
    if not all_trials:
        return result
    n_trials = max(all_trials) + 1

    for method in all_methods:
        baseline = baselines_by_method.get(method)
        hv_list = []
        for t in range(n_trials):
            union_pts = []
            for w in use_weights:
                data = method_data_by_weight_trial[method].get(w, {}).get(t)
                if data is None:
                    continue
                pts = get_uea_points(data, t)
                if baseline is not None:
                    pts = filter_baselines(pts, baseline)
                if len(pts) > 0:
                    union_pts.append(pts)
            if union_pts:
                combined = np.concatenate(union_pts)
                pf = pareto_front(combined)
                hv_list.append(hypervolume(pf, ref))
            else:
                hv_list.append(0.0)
        result[method] = hv_list
    return result


# ========== プロット共通 ==========

def _tight_axes(all_pf_pts, init_ms):
    if len(all_pf_pts) == 0:
        return None, None
    ms_min = float(all_pf_pts[:, 0].min())
    st_max = float(all_pf_pts[:, 1].max())
    x_max = init_ms if init_ms else float(all_pf_pts[:, 0].max())
    rx = max(x_max - ms_min, 10.0)
    xlim = (ms_min - rx * 0.03, x_max + rx * 0.03)
    ry = max(st_max, 0.1)
    ylim = (-ry * 0.03, st_max + ry * 0.10)
    return xlim, ylim


# ========== Anytime 並列ワーカー（モジュールレベル必須） ==========

def _worker_trial_hv_curve(args):
    """1 trial 分の anytime HV curve を計算して返す。"""
    hist, pts, kind, t_grid_list, baseline, ref = args
    # 各点の訪問時刻: 記録済み (N,3) なら正確、(N,2) 旧データは hist から近似
    pts_xy, ptimes = _point_times(pts, hist, kind)
    curve = []
    for t in t_grid_list:
        sub = pts_xy[ptimes <= t]
        if len(sub) == 0:
            curve.append(0.0)
            continue
        if baseline is not None:
            sub = filter_baselines(sub, baseline)
        curve.append(float(hypervolume(sub, ref)) if len(sub) > 0 else 0.0)
    return curve


def _worker_union_hv_curve(args):
    """全 trial union の anytime HV curve を計算して返す。
    各 trial の Pareto front を計算してから合算することで大規模データを高速処理する。
    """
    hist_list, pts_list, kind, t_grid_list, baseline, ref = args
    # 各 trial の (pts_xy, 訪問時刻) を事前構築
    prepared = [_point_times(pts, hist, kind)
                for hist, pts in zip(hist_list, pts_list)]
    curve = []
    for t in t_grid_list:
        pf_parts = []
        for pts_xy, ptimes in prepared:
            if len(pts_xy) == 0:
                continue
            sub = pts_xy[ptimes <= t]
            if baseline is not None:
                sub = filter_baselines(sub, baseline)
            if len(sub) > 0:
                pf_parts.append(pareto_front(sub))  # per-trial Pareto front に縮約
        if not pf_parts:
            curve.append(0.0)
            continue
        combined = np.concatenate(pf_parts)
        curve.append(float(hypervolume(combined, ref)))
    return curve


# ========== Anytime テキスト出力 ==========

# 前半密・後半疎な相対時刻チェックポイント (詳細テキスト用)
_ANYTIME_FRACS = [0.001, 0.002, 0.003, 0.005, 0.007, 0.01, 0.02, 0.03, 0.05, 0.07, 0.10, 0.15, 0.20, 0.30, 0.50, 1.00]

# summary.md 収束速度セクション用チェックポイント
_SUMMARY_CONV_FRACS = [0.03, 0.05, 0.10, 0.30, 1.00]


def _compute_convergence_summary(method_info_list, ref, n_jobs=1):
    """各手法の HV 中央値を代表時刻 (10%/30%/50%/100% t_max) で返す。
    Returns {method: {'t_max': float, 'fracs': {frac: hv_med}}}
    """
    t_grid_tmp = _build_t_grid(method_info_list, n_pts=20, xscale='linear')
    if t_grid_tmp is None:
        return {}
    t_max = float(t_grid_tmp[-1])
    ck_times = [f * t_max for f in _SUMMARY_CONV_FRACS]

    executor = ProcessPoolExecutor(max_workers=n_jobs) if n_jobs > 1 else None
    result = {}
    try:
        for m, hist_list, pts_list, kind, baseline in method_info_list:
            valid_pairs = [(h, p) for h, p in zip(hist_list, pts_list) if h and len(p) > 0]
            invalid_count = len(hist_list) - len(valid_pairs)
            if executor is not None and valid_pairs:
                hv_curves = list(executor.map(
                    _worker_trial_hv_curve,
                    [(h, p, kind, ck_times, baseline, ref) for h, p in valid_pairs]))
            else:
                hv_curves = [_worker_trial_hv_curve((h, p, kind, ck_times, baseline, ref))
                             for h, p in valid_pairs]
            hv_curves += [[0.0] * len(ck_times)] * invalid_count
            hv_arr = np.array(hv_curves)
            fracs = {}
            for i, frac in enumerate(_SUMMARY_CONV_FRACS):
                with np.errstate(all='ignore'):
                    fracs[frac] = float(np.nanmedian(hv_arr[:, i]))
            result[m] = {'t_max': t_max, 'fracs': fracs}
    finally:
        if executor is not None:
            executor.shutdown(wait=False)
    return result


# time-to-target の閾値
_TTT_TAUS = [0.90, 0.95, 0.99]            # self-referenced: 各 trial 自身の最終 HV の τ%
_TTT_COMMON_NAMES = ['low', 'mid', 'high']  # common-target: 最弱/中堅/最強手法の最終 HV


def _worker_trial_ttt(args):
    """1 trial の time-to-target を計算する（self-referenced と common-target を同時に）。

    HV(t) は点を訪問時刻順に累積したときの単調非減少ステップ関数（領域制限版も同様に
    単調）なので、時刻ソート点の prefix 点数 k に対し HV(k) は単調 → 交差時刻を二分探索で
    正確に求める。

    - self-referenced: target = startHV + tau*(finalHV-startHV)。τ<=1 は必ず最終までに
      到達するので打ち切りなし（nan は退化ケースのみ）。
    - common-target: 絶対 HV ターゲット q（手法横断で固定）。finalHV < q の trial は
      未到達 = nan（打ち切り）。
    - region: None なら全フロント HV、(stab_lo, stab_hi, hi_inclusive) なら安定性バンドに
      制限した region HV（既存 region_hv と同一定義、ref_ms = ref[0]）。

    args = (hist, pts, kind, taus, abs_targets, baseline, ref, region)
    Returns: {'self': [t per tau], 'common': [t per abs_target], 'final': float, 'start': float}
    """
    hist, pts, kind, taus, abs_targets, baseline, ref, region = args
    nan_self = [float('nan')] * len(taus)
    nan_common = [float('nan')] * len(abs_targets)
    empty = {'self': nan_self, 'common': nan_common, 'final': 0.0, 'start': 0.0}

    pts_xy, ptimes = _point_times(pts, hist, kind)
    if len(pts_xy) == 0:
        return empty
    # baseline 支配点を除外（時刻も同期して落とす）
    if baseline is not None:
        bl_list = (baseline if (isinstance(baseline, list) and
                   isinstance(baseline[0], (list, np.ndarray))) else [baseline])
        keep = np.ones(len(pts_xy), dtype=bool)
        eps = 1e-9
        for bl in bl_list:
            dominated = (pts_xy[:, 0] >= bl[0] - eps) & (pts_xy[:, 1] >= bl[1] - eps)
            keep &= ~dominated
        pts_xy = pts_xy[keep]
        ptimes = ptimes[keep]
    if len(pts_xy) == 0:
        return empty

    # 高速化: (ms,st) で重複除去し最早時刻のみ残す。HV(t) は重複点に不変なので結果は
    # 厳密に同じだが、memetic（全世代×全個体で同一(ms,st)が大量）では点数が桁で減り、
    # 以降の prefix HV（二分探索 ~17 回）が劇的に速くなる。
    if len(pts_xy) > 1:
        o0 = np.argsort(ptimes, kind='stable')
        pts_xy, ptimes = pts_xy[o0], ptimes[o0]
        uvals, uidx = np.unique(pts_xy, axis=0, return_index=True)
        pts_xy = uvals               # value-sorted（下で時刻再ソートするので問題なし）
        ptimes = ptimes[uidx]        # 各ユニーク (ms,st) の最早時刻

    order = np.argsort(ptimes, kind='stable')
    pts_s = pts_xy[order]
    t_s = ptimes[order]
    N = len(pts_s)

    if region is None:
        def hv_prefix(k):
            if k <= 0:
                return 0.0
            return float(hypervolume(pts_s[:k], ref))  # hypervolume が内部で PF（二重PF排除）
    else:
        r_lo, r_hi, r_hi_inc = region

        def hv_prefix(k):
            if k <= 0:
                return 0.0
            # 領域 HV は「全 PF を band に制限」の意味なので PF を先に取る（既存定義と一致）
            return float(region_hv(pareto_front(pts_s[:k]), r_lo, r_hi, ref[0],
                                   hi_inclusive=r_hi_inc)[0])

    final_hv = hv_prefix(N)
    n0 = int(np.sum(t_s <= t_s[0]))
    start_hv = hv_prefix(n0)

    def time_to(target):
        """HV(t) >= target となる最小時刻。finalHV < target は nan（未到達）。"""
        if final_hv < target - 1e-12:
            return float('nan')
        if target <= start_hv + 1e-12:
            return float(t_s[0])
        lo, hi = 1, N
        while lo < hi:
            mid = (lo + hi) // 2
            if hv_prefix(mid) >= target:
                hi = mid
            else:
                lo = mid + 1
        return float(t_s[lo - 1])

    if final_hv > 0:
        self_t = [time_to(start_hv + tau * (final_hv - start_hv)) for tau in taus]
    else:
        self_t = nan_self
    common_t = [time_to(q) for q in abs_targets]
    return {'self': self_t, 'common': common_t,
            'final': float(final_hv), 'start': float(start_hv)}


def _agg_ttt_col(col):
    """TTT 列（trial 値, nan 含む）→ 中央値/IQR/到達 n。平均でなく中央値（右裾のため）。"""
    arr = np.asarray(col, dtype=float)
    fin = arr[np.isfinite(arr)]
    if len(fin) == 0:
        return {'median': float('nan'), 'q25': float('nan'),
                'q75': float('nan'), 'n': 0, 'n_total': len(arr)}
    return {'median': float(np.median(fin)),
            'q25': float(np.percentile(fin, 25)),
            'q75': float(np.percentile(fin, 75)),
            'n': int(len(fin)), 'n_total': len(arr)}


def _compute_ttt_block(method_info_list, ref, taus, region=None, n_jobs=1):
    """1 領域（region=None=全フロント / バンド）の TTT を集約して返す。

    self-referenced（τ ラダー）と common-target（最弱/中堅/最強手法の最終 HV を共通 q
    とする QRTD）を両方計算。q は (代表重み×領域) ごとに手法横断で固定した絶対スカラー
    なので、各 trial を 1 標本として手法間検定にかけられる。

    Returns: {
      'q':     {'low':float,'mid':float,'high':float},
      'q_src': {'low':worst_method, 'high':best_method},
      'self':   {method: {tau: {median,q25,q75,n,n_total}}},
      'common': {method: {qname: {median,q25,q75,n,n_total}}},
    }
    """
    executor = ProcessPoolExecutor(max_workers=n_jobs) if n_jobs > 1 else None
    prepared = {}   # method -> (valid_pairs, kind, baseline)
    finals_by_m = {}
    try:
        # --- Pass 1: 各 trial の final HV（2 HV 呼び出しのみ・安価）→ 手法別中央値 ---
        for m, hist_list, pts_list, kind, baseline in method_info_list:
            vps = [(h, p) for h, p in zip(hist_list, pts_list) if h and len(p) > 0]
            prepared[m] = (vps, kind, baseline)
            if not vps:
                finals_by_m[m] = []
                continue
            a1 = [(h, p, kind, (), (), baseline, ref, region) for h, p in vps]
            r1 = (list(executor.map(_worker_trial_ttt, a1)) if executor
                  else [_worker_trial_ttt(a) for a in a1])
            finals_by_m[m] = [r['final'] for r in r1]

        med_final = {m: float(np.median(v)) for m, v in finals_by_m.items() if v}
        if med_final:
            ordered = sorted(med_final.items(), key=lambda kv: kv[1])
            q_low_m, q_low = ordered[0]
            q_high_m, q_high = ordered[-1]
            q_mid = float(np.median([v for _, v in ordered]))
        else:
            q_low_m = q_high_m = None
            q_low = q_mid = q_high = float('nan')
        q_vals = [q_low, q_mid, q_high]

        # --- Pass 2: self τ ラダー + common q ラダー（ここで二分探索）---
        self_agg, common_agg = {}, {}
        for m, (vps, kind, baseline) in prepared.items():
            if not vps:
                self_agg[m] = {tau: _agg_ttt_col([]) for tau in taus}
                common_agg[m] = {nm: _agg_ttt_col([]) for nm in _TTT_COMMON_NAMES}
                continue
            a2 = [(h, p, kind, taus, q_vals, baseline, ref, region) for h, p in vps]
            r2 = (list(executor.map(_worker_trial_ttt, a2)) if executor
                  else [_worker_trial_ttt(a) for a in a2])
            self_rows = np.array([r['self'] for r in r2], dtype=float)      # (nt, ntau)
            common_rows = np.array([r['common'] for r in r2], dtype=float)  # (nt, 3)
            self_agg[m] = {tau: _agg_ttt_col(self_rows[:, i])
                           for i, tau in enumerate(taus)}
            common_agg[m] = {nm: _agg_ttt_col(common_rows[:, i])
                             for i, nm in enumerate(_TTT_COMMON_NAMES)}
    finally:
        if executor is not None:
            executor.shutdown(wait=False)

    return {'q': dict(zip(_TTT_COMMON_NAMES, q_vals)),
            'q_src': {'low': q_low_m, 'high': q_high_m},
            'self': self_agg, 'common': common_agg}


def _fmt_ttt_cell(d):
    """TTT 集約 dict → 'median [Q25, Q75]'（未到達/空は '—'）。"""
    if not d or d.get('n', 0) == 0 or not np.isfinite(d.get('median', float('nan'))):
        return '—'
    return f"{d['median']:.2f} [{d['q25']:.2f}, {d['q75']:.2f}]"


def _md_ttt_self_table(block, methods, taus):
    """self-referenced TTT 表（τ ラダー）を md 行リストで返す。"""
    self_d = block.get('self', {})
    lines = []
    hdr = ' | '.join(f't→{int(t*100)}%' for t in taus)
    lines.append(f'| 手法 | {hdr} | (n) |')
    lines.append('|------|' + ':---:|' * (len(taus) + 1))
    for m in methods:
        md = self_d.get(m)
        if not md:
            lines.append(f'| {METHOD_LABELS.get(m, m)} |' + ' — |' * (len(taus) + 1))
            continue
        cells, n_used = [], 0
        for t in taus:
            d = md.get(t, {})
            n_used = max(n_used, int(d.get('n', 0)))
            cells.append(_fmt_ttt_cell(d))
        lines.append(f"| {METHOD_LABELS.get(m, m)} | " + ' | '.join(cells) + f' | {n_used} |')
    return lines


def _md_ttt_common_table(block, methods):
    """common-target TTT 表（QRTD, 到達 n/N 付き）を md 行リストで返す。"""
    common_d = block.get('common', {})
    q = block.get('q', {})
    q_src = block.get('q_src', {})

    def _ql(name):
        v = q.get(name)
        return f'{v:.1f}' if (v is not None and np.isfinite(v)) else 'n/a'
    low_lbl = METHOD_LABELS.get(q_src.get('low'), q_src.get('low') or '?')
    high_lbl = METHOD_LABELS.get(q_src.get('high'), q_src.get('high') or '?')
    lines = []
    lines.append(f'共通ターゲット HV: q_low={_ql("low")} (最弱={low_lbl}) / '
                 f'q_mid={_ql("mid")} / q_high={_ql("high")} (最強={high_lbl})')
    lines.append('')
    lines.append('| 手法 | t→q_low (n/N) | t→q_mid (n/N) | t→q_high (n/N) |')
    lines.append('|------|:---:|:---:|:---:|')
    for m in methods:
        md = common_d.get(m)
        if not md:
            lines.append(f'| {METHOD_LABELS.get(m, m)} | — | — | — |')
            continue
        cells = []
        for nm in _TTT_COMMON_NAMES:
            d = md.get(nm, {})
            cell = _fmt_ttt_cell(d)
            nt, nn = int(d.get('n_total', 0)), int(d.get('n', 0))
            if nt:
                cell += f' ({nn}/{nt})'
            cells.append(cell)
        lines.append(f"| {METHOD_LABELS.get(m, m)} | " + ' | '.join(cells) + ' |')
    return lines


def write_anytime_txt(method_info_list, ref, w_label, outpath, n_jobs=1):
    """anytime curve の数値サマリ (scalar + HV) をテキストファイルに書き出す。"""
    # t_max 推定用に粗いグリッドを構築して t_max を取得
    t_grid_tmp = _build_t_grid(method_info_list, n_pts=20, xscale='linear')
    if t_grid_tmp is None:
        return
    t_max = float(t_grid_tmp[-1])
    ck_times = np.array([f * t_max for f in _ANYTIME_FRACS])
    t_grid_list = ck_times.tolist()

    lines = []
    lines.append(f'# Anytime Curve Detail  weight={w_label}  ref=({float(ref[0]):.2f}, {float(ref[1]):.4f})  t_max={t_max:.2f}s')
    lines.append(f'# trials per method: {len(method_info_list[0][1]) if method_info_list else 0}')
    lines.append('')

    executor = ProcessPoolExecutor(max_workers=n_jobs) if n_jobs > 1 else None
    try:
        for m, hist_list, pts_list, kind, baseline in method_info_list:
            label = METHOD_LABELS.get(m, m)
            lines.append(f'## {label}')
            hdr = f'{"t(s)":>8}  {"frac":>5}  {"scalar_med":>12}  {"scalar_[Q25,Q75]":<22}  {"HV_med":>10}  {"HV_[Q25,Q75]":<22}'
            lines.append(hdr)
            lines.append('-' * len(hdr))

            # scalar curves per trial (checkpointごと)
            sc_arr = []
            for hist in hist_list:
                if not hist:
                    sc_arr.append(np.full(len(ck_times), np.nan))
                    continue
                times = np.array([h['cpu_time'] for h in hist])
                vals = np.array([h.get('best_score', np.nan) for h in hist], dtype=float)
                y = np.full(len(ck_times), np.nan)
                for i, t in enumerate(ck_times):
                    idx = int(np.searchsorted(times, t, side='right')) - 1
                    if idx >= 0:
                        y[i] = vals[idx]
                sc_arr.append(y)
            sc_arr = np.array(sc_arr)

            # HV curves per trial (checkpointごと)
            valid_pairs = [(h, p) for h, p in zip(hist_list, pts_list) if h and len(p) > 0]
            invalid_count = len(hist_list) - len(valid_pairs)
            if executor is not None and valid_pairs:
                hv_curves = list(executor.map(
                    _worker_trial_hv_curve,
                    [(h, p, kind, t_grid_list, baseline, ref) for h, p in valid_pairs]))
            else:
                hv_curves = [_worker_trial_hv_curve((h, p, kind, t_grid_list, baseline, ref))
                             for h, p in valid_pairs]
            hv_curves += [np.zeros(len(ck_times)).tolist()] * invalid_count
            hv_arr = np.array(hv_curves)

            for i_ck, (t_act, frac) in enumerate(zip(ck_times, _ANYTIME_FRACS)):
                with np.errstate(all='ignore'):
                    sc_med = float(np.nanmedian(sc_arr[:, i_ck]))
                    sc_q25 = float(np.nanpercentile(sc_arr[:, i_ck], 25))
                    sc_q75 = float(np.nanpercentile(sc_arr[:, i_ck], 75))
                    hv_med = float(np.nanmedian(hv_arr[:, i_ck]))
                    hv_q25 = float(np.nanpercentile(hv_arr[:, i_ck], 25))
                    hv_q75 = float(np.nanpercentile(hv_arr[:, i_ck], 75))
                sc_iqr = f'[{sc_q25:.1f}, {sc_q75:.1f}]'
                hv_iqr = f'[{hv_q25:.4f}, {hv_q75:.4f}]'
                lines.append(
                    f'{t_act:8.2f}  {frac:5.0%}  {sc_med:12.1f}  {sc_iqr:<22}  {hv_med:10.4f}  {hv_iqr:<22}'
                )
            lines.append('')
    finally:
        if executor is not None:
            executor.shutdown(wait=False)

    with open(outpath, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))


# ========== プロット: anytime ==========

def plot_anytime_scalar(method_info_list, title, outpath, xscale='log'):
    """anytime best_score 曲線 (per-trial median + IQR)。"""
    t_grid = _build_t_grid(method_info_list, xscale=xscale)
    if t_grid is None:
        return
    fig, ax = plt.subplots(figsize=(10, 6))
    for m, hist_list, pts_list, kind, baseline in method_info_list:
        color = get_method_color(m)
        scores = []
        for hist in hist_list:
            if not hist:
                scores.append(np.full(len(t_grid), np.nan))
                continue
            times = np.array([h['cpu_time'] for h in hist])
            vals = np.array([h.get('best_score', np.nan) for h in hist], dtype=float)
            y = np.full(len(t_grid), np.nan)
            for i, t in enumerate(t_grid):
                idx = int(np.searchsorted(times, t, side='right')) - 1
                if idx >= 0:
                    y[i] = vals[idx]
            scores.append(y)
        arr = np.array(scores)
        with np.errstate(all='ignore'):
            med = np.nanmedian(arr, axis=0)
            q25 = np.nanpercentile(arr, 25, axis=0)
            q75 = np.nanpercentile(arr, 75, axis=0)
        valid = ~np.isnan(med)
        if not valid.any():
            continue
        label = METHOD_LABELS.get(m, m)
        ax.plot(t_grid[valid], med[valid], color=color, lw=2, label=label)
        ax.fill_between(t_grid[valid], q25[valid], q75[valid], color=color, alpha=0.15)
    ax.set_xlabel('CPU time (s)')
    ax.set_ylabel('Best weighted score (median ± IQR)')
    ax.set_title(title)
    if xscale == 'log':
        ax.set_xscale('log')
    ax.set_xlim(t_grid[0], t_grid[-1])
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


def plot_anytime_uea_hv(method_info_list, ref, title, outpath, xscale='log', n_jobs=1):
    """anytime per-weight UEA HV 曲線 (左: per-trial median+IQR, 右: union)。"""
    t_grid = _build_t_grid(method_info_list, xscale=xscale)
    if t_grid is None:
        return
    t_grid_list = t_grid.tolist()
    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    executor = ProcessPoolExecutor(max_workers=n_jobs) if n_jobs > 1 else None
    try:
        for m, hist_list, pts_list, kind, baseline in method_info_list:
            color = get_method_color(m)
            label = METHOD_LABELS.get(m, m)
            valid_pairs = [(h, p) for h, p in zip(hist_list, pts_list) if h and len(p) > 0]
            invalid_count = len(hist_list) - len(valid_pairs)

            # per-trial HV curves（並列化）
            if executor is not None and valid_pairs:
                args_list = [(h, p, kind, t_grid_list, baseline, ref) for h, p in valid_pairs]
                valid_curves = list(executor.map(_worker_trial_hv_curve, args_list))
            else:
                valid_curves = [_worker_trial_hv_curve((h, p, kind, t_grid_list, baseline, ref))
                                for h, p in valid_pairs]
            hv_curves = valid_curves + [np.zeros(len(t_grid)).tolist()] * invalid_count
            arr = np.array(hv_curves)
            med = np.nanmedian(arr, axis=0)
            q25 = np.nanpercentile(arr, 25, axis=0)
            q75 = np.nanpercentile(arr, 75, axis=0)
            ax_l.plot(t_grid, med, color=color, lw=2, label=label)
            ax_l.fill_between(t_grid, q25, q75, color=color, alpha=0.15)

            # union HV curve
            union_args = (hist_list, pts_list, kind, t_grid_list, baseline, ref)
            union_curve = _worker_union_hv_curve(union_args)
            ax_r.plot(t_grid, union_curve, color=color, lw=2, label=label)
    finally:
        if executor is not None:
            executor.shutdown(wait=False)
    for ax, sub in [(ax_l, 'per-trial (median ± IQR)'), (ax_r, 'union (all trials)')]:
        ax.set_xlabel('CPU time (s)')
        ax.set_title(sub)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)
        if xscale == 'log':
            ax.set_xscale('log')
        ax.set_xlim(t_grid[0], t_grid[-1])
    ax_l.set_ylabel('HV')
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


# ========== プロット: B-1 改善成功率ヒートマップ ==========

def plot_improvement_heatmap(
    method_data_by_weight_trial, init_ms, methods, w_labels, title, outpath
):
    """改善成功率 heatmap: 行=重み, 列=手法。"""
    mat = np.full((len(w_labels), len(methods)), np.nan)
    for i, wl in enumerate(w_labels):
        for j, m in enumerate(methods):
            by_trial = method_data_by_weight_trial.get(m, {}).get(wl, {})
            trials_data = [d for d in by_trial.values() if 'finals' in d]
            if not trials_data:
                continue
            n_imp = sum(1 for d in trials_data if _improved_over_baseline(d))
            mat[i, j] = n_imp / len(trials_data)

    fig, ax = plt.subplots(figsize=(max(5, len(methods) * 1.8), max(4, len(w_labels) * 0.4 + 1.5)))
    im = ax.imshow(mat, cmap='RdYlGn', vmin=0, vmax=1, aspect='auto')
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels([METHOD_LABELS.get(m, m) for m in methods], rotation=30, ha='right')
    ax.set_yticks(range(len(w_labels)))
    ax.set_yticklabels(w_labels)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if np.isnan(mat[i, j]):
                continue
            v = mat[i, j]
            ax.text(j, i, f'{v*100:.0f}%', ha='center', va='center',
                    fontsize=8, color='white' if v < 0.5 else 'black')
    fig.colorbar(im, ax=ax, label='improvement rate')
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.close(fig)


# ========== プロット: B-2a union UEA HV boxplot ==========

def plot_union_hv_boxplot(union_hv_by_method, methods, title, outpath):
    """per-trial union UEA HV の手法別 boxplot。"""
    data = [union_hv_by_method.get(m, []) for m in methods]
    labels = [METHOD_LABELS.get(m, m) for m in methods]
    colors = [get_method_color(m) for m in methods]
    fig, ax = plt.subplots(figsize=(max(6, len(methods) * 2), 6))
    bp = ax.boxplot(data, labels=labels, patch_artist=True, notch=False)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_ylabel('per-trial union UEA HV')
    ax.set_title(title)
    ax.grid(True, alpha=0.3, axis='y')
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


# ========== プロット: B-2b 領域別 HV bar ==========

def plot_region_hv_bars(region_hvs, methods, thresholds, title, outpath):
    """高/中/低安定性（D 小→大）の 3 分割領域別 HV bar chart。
    region_hvs: {method: {region_name: hv}}
    """
    regions = ['high_stability', 'mid_stability', 'low_stability']
    region_labels = [
        f'高安定性（D≤P33）\n[0, {thresholds["P33"]:.3f}]',
        f'中安定性\n({thresholds["P33"]:.3f}, {thresholds["P67"]:.3f}]',
        f'低安定性（D>P67）\n({thresholds["P67"]:.3f}, {thresholds["stab_max"]:.3f}]',
    ]
    n_groups = len(regions)
    n_methods = len(methods)
    x = np.arange(n_groups)
    width = 0.8 / n_methods
    fig, ax = plt.subplots(figsize=(10, 6))
    for j, m in enumerate(methods):
        vals = [region_hvs.get(m, {}).get(r, 0.0) for r in regions]
        color = get_method_color(m)
        offset = (j - n_methods / 2 + 0.5) * width
        ax.bar(x + offset, vals, width, label=METHOD_LABELS.get(m, m),
               color=color, alpha=0.8, edgecolor='black', linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(region_labels)
    ax.set_ylabel('Region-restricted HV')
    ax.set_title(title)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


def plot_region_hv_bars_2split(region_hvs_2split, methods, thresholds, title, outpath):
    """高安定性/低安定性 の 2分割 region HV bar chart（P50 境界）。"""
    regions = ['high_stability', 'low_stability']
    p50 = thresholds['P50']
    region_labels = [
        f'高安定性（D≤P50）\n[0, {p50:.3f}]',
        f'低安定性（D>P50）\n({p50:.3f}, max]',
    ]
    n_groups = len(regions)
    n_methods = len(methods)
    x = np.arange(n_groups)
    width = 0.8 / n_methods
    fig, ax = plt.subplots(figsize=(9, 6))
    for j, m in enumerate(methods):
        vals = [region_hvs_2split.get(m, {}).get(r, 0.0) for r in regions]
        color = get_method_color(m)
        offset = (j - n_methods / 2 + 0.5) * width
        ax.bar(x + offset, vals, width, label=METHOD_LABELS.get(m, m),
               color=color, alpha=0.8, edgecolor='black', linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(region_labels)
    ax.set_ylabel('Region-restricted HV')
    ax.set_title(title)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


def format_region_hv_table_2split(region_hvs_2split, region_hv_counts_2split, methods, thresholds):
    """2分割 領域別 HV の数値テキスト（P50 境界）。"""
    p50, st_max = thresholds['P50'], thresholds['stab_max']
    regions = [
        ('high_stability', 0.0,  p50,    f'[0,       {p50:.4f}]  高安定性'),
        ('low_stability',  p50,  st_max, f'({p50:.4f}, {st_max:.4f}]  低安定性'),
    ]
    lines = [
        'Region-restricted HV 2分割（P50 境界）',
        f'  P50={p50:.4f}, stab_max={st_max:.4f}',
        '',
    ]
    col_w = 20
    for rname, lo, hi, rng in regions:
        lines.append(f'  [{rname}]  stab ∈ {rng}')
        lines.append(f"    {'method':<{col_w}} {'HV':>12} {'n_points':>10}")
        lines.append('    ' + '-' * (col_w + 24))
        for m in methods:
            hv = region_hvs_2split.get(m, {}).get(rname, 0.0)
            n = region_hv_counts_2split.get(m, {}).get(rname, 0)
            lines.append(f"    {METHOD_LABELS.get(m,m):<{col_w}} {hv:>12.4f} {n:>10}")
        lines.append('')
    return '\n'.join(lines)


# ========== プロット: 差分 EAF ==========

def plot_diff_eaf(trial_pts_a, trial_pts_b, label_a, label_b,
                  baseline_a, baseline_b, title, outpath, init_ms=None):
    """EAF(A) - EAF(B) の差分ヒートマップ。"""
    all_pts = []
    for pts_list, bl in [(trial_pts_a, baseline_a), (trial_pts_b, baseline_b)]:
        for pts in pts_list:
            if len(pts) == 0:
                continue
            filtered = filter_baselines(pts, bl) if bl else pts
            if len(filtered) > 0:
                all_pts.append(filtered)
    if not all_pts:
        return
    combined = np.concatenate(all_pts)
    gms, gst = _make_grid(combined, include_ms=init_ms)
    eaf_a = _eaf_grid(trial_pts_a, gms, gst, baseline_a)
    eaf_b = _eaf_grid(trial_pts_b, gms, gst, baseline_b)
    diff = eaf_a - eaf_b

    fig, ax = plt.subplots(figsize=(10, 7))
    im = ax.pcolormesh(gms, gst, diff.T, cmap='RdBu_r', vmin=-1, vmax=1, shading='auto')
    plt.colorbar(im, ax=ax, label=f'EAF({label_a}) − EAF({label_b})')

    def _union_pf(pts_list, bl):
        valid = [p for p in pts_list if len(p) > 0]
        if not valid:
            return np.zeros((0, 2))
        combined = np.concatenate(valid)
        if bl:
            combined = filter_baselines(combined, bl)
        return pareto_front(combined)

    pf_a = _union_pf(trial_pts_a, baseline_a)
    pf_b = _union_pf(trial_pts_b, baseline_b)
    if len(pf_a) > 0:
        ax.scatter(pf_a[:, 0], pf_a[:, 1], c='darkred', s=30, alpha=0.8,
                   edgecolors='white', lw=0.5, marker='o', label=f'{label_a}', zorder=3)
    if len(pf_b) > 0:
        ax.scatter(pf_b[:, 0], pf_b[:, 1], c='darkblue', s=30, alpha=0.8,
                   edgecolors='white', lw=0.5, marker='s', label=f'{label_b}', zorder=3)
    if init_ms:
        ax.axvline(init_ms, color='gray', ls='--', alpha=0.5, lw=1.2)
    ax.set_xlabel('Makespan')
    ax.set_ylabel('Stability')
    ax.set_title(title)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


# ========== テーブル: B-1 ==========

def format_b1_scalar_stats(method_data_by_weight_trial, methods, w_labels,
                            baselines_by_method):
    """per-weight scalar 最終値の手法間 Wilcoxon + Cliff's delta + Holm 補正。"""
    lines = ['per-weight scalar 最終値比較 (Wilcoxon paired, alternative: A < B = A が良い)',
             '  ← 最終 makespan ではなく GA に渡した weighted score で比較',
             '']

    pairs = [(a, b) for (a, b) in COMPARE_PAIRS
             if a in methods and b in methods]
    if not pairs:
        return '\n'.join(lines)

    all_p = []
    pair_data = []
    for wl in w_labels:
        row_p = []
        row_d = []
        for (ma, mb) in pairs:
            vals_a, vals_b = [], []
            for t_idx, data in method_data_by_weight_trial.get(ma, {}).get(wl, {}).items():
                vals_a.append(data['finals'].get('makespan', np.nan))
            for t_idx, data in method_data_by_weight_trial.get(mb, {}).get(wl, {}).items():
                vals_b.append(data['finals'].get('makespan', np.nan))
            min_n = min(len(vals_a), len(vals_b))
            stat, p = wilcoxon_paired(vals_a[:min_n], vals_b[:min_n], alternative='less')
            d = cliffs_delta(vals_a, vals_b)
            row_p.append(p)
            row_d.append(d)
        pair_data.append((wl, row_p, row_d))
        all_p.extend(row_p)

    # Holm 補正（全 weight × pair をまとめて）
    finite_p = [(i, p) for i, p in enumerate(all_p) if not np.isnan(p)]
    corrected = [np.nan] * len(all_p)
    if finite_p:
        idxs, ps = zip(*finite_p)
        corr = holm_bonferroni(list(ps))
        for i, c in zip(idxs, corr):
            corrected[i] = c

    pair_header = '  '.join(f'{ma} < {mb}' for (ma, mb) in pairs)
    lines.append(f"  {'w_label':<12}  {pair_header}")
    lines.append('  ' + '-' * (14 + 20 * len(pairs)))

    p_idx = 0
    for wl, row_p, row_d in pair_data:
        cells = []
        for k, (p, d) in enumerate(zip(row_p, row_d)):
            p_corr = corrected[p_idx + k]
            cells.append(f'p={p:.3f}{_p_star(p_corr)} d={d:+.2f}({effect_label(d)})')
        lines.append(f"  {wl:<12}  {'  '.join(cells)}")
        p_idx += len(row_p)

    return '\n'.join(lines)


def format_b1_uea_hv_stats(method_data_by_weight_trial, methods, w_labels,
                             ref, baselines_by_method):
    """per-weight UEA HV の手法間 Wilcoxon + Cliff's delta。
    HV は大きい方が良いので alternative='greater' (A > B)。
    """
    lines = ['per-weight UEA HV 比較 (Wilcoxon paired, alternative: A > B = A の HV が大きい)',
             '']
    pairs = [(a, b) for (a, b) in COMPARE_PAIRS if a in methods and b in methods]
    if not pairs:
        return '\n'.join(lines)

    pair_header = '  '.join(f'{ma} > {mb}' for (ma, mb) in pairs)
    lines.append(f"  {'w_label':<12}  {pair_header}")
    lines.append('  ' + '-' * (14 + 22 * len(pairs)))

    for wl in w_labels:
        # per-weight の参照点: 全手法の全 trial UEA 点の max + margin
        all_w_pts = []
        for m in methods:
            bl = baselines_by_method.get(m)
            for data in method_data_by_weight_trial.get(m, {}).get(wl, {}).values():
                pts = get_uea_points(data, 0)
                if bl:
                    pts = filter_baselines(pts, bl)
                if len(pts) > 0:
                    all_w_pts.append(pts)
        if not all_w_pts:
            continue
        all_w_concat = np.concatenate(all_w_pts)
        w_ref = (float(all_w_concat[:, 0].max()) + max(all_w_concat[:, 0].max() * 0.01, 1.0),
                 float(all_w_concat[:, 1].max()) + max(all_w_concat[:, 1].max() * 0.01, 0.01))

        cells = []
        for (ma, mb) in pairs:
            hv_a, hv_b = [], []
            for t_idx in sorted(method_data_by_weight_trial.get(ma, {}).get(wl, {}).keys()):
                data = method_data_by_weight_trial[ma][wl][t_idx]
                pts = get_uea_points(data, t_idx)
                bl = baselines_by_method.get(ma)
                if bl:
                    pts = filter_baselines(pts, bl)
                hv_a.append(hypervolume(pts, w_ref))
            for t_idx in sorted(method_data_by_weight_trial.get(mb, {}).get(wl, {}).keys()):
                data = method_data_by_weight_trial[mb][wl][t_idx]
                pts = get_uea_points(data, t_idx)
                bl = baselines_by_method.get(mb)
                if bl:
                    pts = filter_baselines(pts, bl)
                hv_b.append(hypervolume(pts, w_ref))
            min_n = min(len(hv_a), len(hv_b))
            stat, p = wilcoxon_paired(hv_b[:min_n], hv_a[:min_n], alternative='less')
            d = cliffs_delta(hv_a, hv_b)
            cells.append(f'p={p:.3f}{_p_star(p)} d={d:+.2f}({effect_label(d)})')
        lines.append(f"  {wl:<12}  {'  '.join(cells)}")

    return '\n'.join(lines)


def format_b2a_union_hv_stats(union_hv_by_method, methods):
    """per-trial union UEA HV の手法間 Wilcoxon + Cliff's delta。"""
    lines = ['per-trial union UEA HV 統計 (B-2a 主筋)',
             '  Wilcoxon: alternative = A > B (A の HV が大きい)',
             '']
    # 各手法の median + IQR
    lines.append(f"  {'method':<18} {'median':>10} {'IQR':>12} {'n':>5}")
    lines.append('  ' + '-' * 50)
    for m in methods:
        hvs = [v for v in union_hv_by_method.get(m, []) if np.isfinite(v)]
        if not hvs:
            continue
        med = float(np.median(hvs))
        q25 = float(np.percentile(hvs, 25))
        q75 = float(np.percentile(hvs, 75))
        lines.append(f"  {METHOD_LABELS.get(m,m):<18} {med:>10.2f} [{q25:.2f},{q75:.2f}] {len(hvs):>5}")

    lines.append('')
    pairs = [(a, b) for (a, b) in COMPARE_PAIRS if a in methods and b in methods]
    all_p = []
    pair_rows = []
    for (ma, mb) in pairs:
        hv_a = union_hv_by_method.get(ma, [])
        hv_b = union_hv_by_method.get(mb, [])
        min_n = min(len(hv_a), len(hv_b))
        # A > B → test hv_b < hv_a → wilcoxon(hv_b - hv_a, alternative='less')
        stat, p = wilcoxon_paired(hv_b[:min_n], hv_a[:min_n], alternative='less')
        d = cliffs_delta(hv_a, hv_b)
        all_p.append(p)
        pair_rows.append((ma, mb, p, d))

    finite_p = [(i, p) for i, p in enumerate(all_p) if not np.isnan(p)]
    corrected = [np.nan] * len(all_p)
    if finite_p:
        idxs, ps = zip(*finite_p)
        corr = holm_bonferroni(list(ps))
        for i, c in zip(idxs, corr):
            corrected[i] = c

    _cd_hdr = "Cliff's d"
    lines.append(f"  {'A > B pair':<32} {'p (raw)':>10} {'p (Holm)':>10} {_cd_hdr:>12} {'effect':>8}")
    lines.append('  ' + '-' * 76)
    for k, (ma, mb, p, d) in enumerate(pair_rows):
        p_h = corrected[k]
        label = f'{METHOD_LABELS.get(ma,ma)} > {METHOD_LABELS.get(mb,mb)}'
        lines.append(f"  {label:<32} {p:>10.4f}{_p_star(p)} {p_h:>10.4f}{_p_star(p_h)} "
                     f"{d:>12.3f} {effect_label(d):>8}")
    return '\n'.join(lines)


def format_c_metric(union_pf_by_method, methods):
    """C-metric 行列テキスト。"""
    lines = ['C-metric  C(row, col) = row が col を弱く支配する割合', '']
    m_labels = [METHOD_LABELS.get(m, m) for m in methods]
    header = f"  {'':20}" + ''.join(f"{l:>16}" for l in m_labels)
    lines.append(header)
    for ma in methods:
        row = f"  {METHOD_LABELS.get(ma,ma):<20}"
        for mb in methods:
            if ma == mb:
                row += f"{'−':>16}"
            else:
                pf_a = union_pf_by_method.get(ma, np.zeros((0, 2)))
                pf_b = union_pf_by_method.get(mb, np.zeros((0, 2)))
                c = c_metric(pf_a, pf_b)
                row += f"{c:>16.3f}"
        lines.append(row)
    return '\n'.join(lines)


def format_b2b_coverage(method_data_by_weight_trial, methods, baselines_by_method,
                         thresholds):
    """B-2b Step 1: stab ≥ τ の解を持つ trial の割合 (τ = P67, P33)。
    trial ごとに全重みの UEA 点を合体して判定する。
    """
    lines = ['B-2b カバー率: stab ≥ τ の解を 1 個以上持つ trial の割合', '']
    for tau_name in ['P67', 'P33']:
        tau = thresholds[tau_name]
        lines.append(f'  閾値 τ = {tau_name} = {tau:.4f}')
        lines.append(f"    {'method':<20} {'cover_rate':>12} {'n_covered':>12} {'n_trials':>10}")
        lines.append('    ' + '-' * 58)
        cover_by_method = {}
        for m in methods:
            bl = baselines_by_method.get(m)
            # trial ごとに全重みを合体して判定
            all_trial_indices = set()
            for by_trial in method_data_by_weight_trial.get(m, {}).values():
                all_trial_indices.update(by_trial.keys())
            covered = 0
            total = len(all_trial_indices)
            for t_idx in all_trial_indices:
                union_pts = []
                for wl, by_trial in method_data_by_weight_trial.get(m, {}).items():
                    data = by_trial.get(t_idx)
                    if data is None:
                        continue
                    pts = get_uea_points(data, t_idx)
                    if bl:
                        pts = filter_baselines(pts, bl)
                    if len(pts) > 0:
                        union_pts.append(pts)
                if union_pts:
                    combined = np.concatenate(union_pts)
                    if combined[:, 1].max() >= tau:
                        covered += 1
            rate = covered / total if total > 0 else 0.0
            cover_by_method[m] = (covered, total, rate)
            lines.append(f"    {METHOD_LABELS.get(m,m):<20} {rate:>12.3f} "
                         f"{covered:>12} {total:>10}")
        lines.append('')
    return '\n'.join(lines)


def format_b2b_cond_ms_wilcoxon(method_data_by_weight_trial, methods,
                                  baselines_by_method, thresholds):
    """B-2b Step 2: stab ≥ τ の解の中の最小 MS → paired Wilcoxon。
    τ = P67, P33。
    """
    lines = ['B-2b 条件付き MS Wilcoxon (stab ≥ τ の解の中の最小 MS)',
             '  Wilcoxon: alternative = A < B (A の MS が小さい = 良い)',
             '']

    for tau_name in ['P67', 'P33']:
        tau = thresholds[tau_name]
        lines.append(f'  === 閾値 τ = {tau_name} = {tau:.4f} ===')

        # 各 (method, trial) で: 全重み全訪問点から stab >= tau を満たす解の最小 MS
        min_ms_by_method = {}
        for m in methods:
            bl = baselines_by_method.get(m)
            # trial_idx → min_ms
            trial_min_ms = {}
            for w_label, by_trial in method_data_by_weight_trial.get(m, {}).items():
                for t_idx, data in by_trial.items():
                    pts = get_uea_points(data, t_idx)
                    if bl:
                        pts = filter_baselines(pts, bl)
                    if len(pts) == 0:
                        continue
                    mask = pts[:, 1] >= tau
                    if not mask.any():
                        continue
                    ms_min = float(pts[mask, 0].min())
                    if t_idx not in trial_min_ms or ms_min < trial_min_ms[t_idx]:
                        trial_min_ms[t_idx] = ms_min
            min_ms_by_method[m] = trial_min_ms

        # カバー率 + 条件付き比較
        lines.append(f"  {'method':<20} {'n_reach':>8} {'n_total':>8} {'median_minMS':>14}")
        lines.append('  ' + '-' * 56)
        for m in methods:
            vals = list(min_ms_by_method[m].values())
            all_t = set()
            for by_trial in method_data_by_weight_trial.get(m, {}).values():
                all_t.update(by_trial.keys())
            n_total = len(all_t)
            n_reach = len(vals)
            med = float(np.median(vals)) if vals else float('nan')
            lines.append(f"  {METHOD_LABELS.get(m,m):<20} {n_reach:>8} "
                         f"{n_total:>8} {med:>14.1f}")
        lines.append('')

        pairs = [(a, b) for (a, b) in COMPARE_PAIRS if a in methods and b in methods]
        _cd_h = "Cliff's d"
        lines.append(f"  {'A < B pair':<32} {'p':>8} {_cd_h:>12} {'effect':>8} {'n_paired':>10}")
        lines.append('  ' + '-' * 76)
        for (ma, mb) in pairs:
            d_ma = min_ms_by_method.get(ma, {})
            d_mb = min_ms_by_method.get(mb, {})
            # 両方到達している trial のみ paired
            common = sorted(set(d_ma.keys()) & set(d_mb.keys()))
            if not common:
                lines.append(f"  {METHOD_LABELS.get(ma,ma)} < {METHOD_LABELS.get(mb,mb):<20} "
                              f"  (common trial なし)")
                continue
            vals_a = [d_ma[t] for t in common]
            vals_b = [d_mb[t] for t in common]
            stat, p = wilcoxon_paired(vals_a, vals_b, alternative='less')
            d = cliffs_delta(vals_a, vals_b)
            label = f'{METHOD_LABELS.get(ma,ma)} < {METHOD_LABELS.get(mb,mb)}'
            lines.append(f"  {label:<32} {p:>8.4f}{_p_star(p)} "
                         f"{d:>12.3f} {effect_label(d):>8} {len(common):>10}")
        lines.append('')

    return '\n'.join(lines)


def format_region_hv_table(region_hvs, region_hv_counts, methods, thresholds):
    """領域別 HV の数値テキスト。region_hv_counts: {method: {region: n_points}}"""
    p33, p67, st_max = thresholds['P33'], thresholds['P67'], thresholds['stab_max']
    regions = [
        ('high_stability', 0.0,  p33,    f'[0,       {p33:.4f}]  高安定性（D≤P33）'),
        ('mid_stability',  p33,  p67,    f'({p33:.4f}, {p67:.4f}]  中安定性'),
        ('low_stability',  p67,  st_max, f'({p67:.4f}, {st_max:.4f}]  低安定性（D>P67）'),
    ]
    lines = [
        'Region-restricted HV (各手法の全訪問点から領域内 Pareto → HV)',
        f'  P33={p33:.4f}, P67={p67:.4f}, stab_max={st_max:.4f}',
        f'  参照点 stab: R_upper + 2% margin',
        '',
    ]
    col_w = 20
    for rname, lo, hi, rng in regions:
        lines.append(f'  [{rname}]  stab ∈ {rng}')
        lines.append(f"    {'method':<{col_w}} {'HV':>12} {'n_points':>10}")
        lines.append('    ' + '-' * (col_w + 24))
        for m in methods:
            hv = region_hvs.get(m, {}).get(rname, 0.0)
            n = region_hv_counts.get(m, {}).get(rname, 0)
            lines.append(f"    {METHOD_LABELS.get(m,m):<{col_w}} {hv:>12.4f} {n:>10}")
        lines.append('')
    return '\n'.join(lines)


def format_improvement_table(method_data_by_weight_trial, init_ms, methods, w_labels):
    """改善成功率の数値テキスト（重み × 手法）。"""
    lines = [
        '改善成功率 (重みスカラー値が baseline より小さい trial 割合)',
        '',
    ]
    col_w = max(len(METHOD_LABELS.get(m, m)) for m in methods) + 2
    header = f"  {'w_label':<12}" + ''.join(f"{METHOD_LABELS.get(m,m):>{col_w}}" for m in methods)
    lines.append(header)
    lines.append('  ' + '-' * (12 + col_w * len(methods) + 2))
    for wl in w_labels:
        row = f"  {wl:<12}"
        for m in methods:
            by_trial = method_data_by_weight_trial.get(m, {}).get(wl, {})
            trials_data = [d for d in by_trial.values() if 'finals' in d]
            if not trials_data:
                row += f"{'N/A':>{col_w}}"
            else:
                n_imp = sum(1 for d in trials_data if _improved_over_baseline(d))
                row += f"{n_imp/len(trials_data):>{col_w}.2f}"
        lines.append(row)
    lines.append('')
    # 重みをまたいだ手法ごとの平均
    lines.append('  --- 全重みの平均 ---')
    for m in methods:
        rates = []
        for wl in w_labels:
            by_trial = method_data_by_weight_trial.get(m, {}).get(wl, {})
            trials_data = [d for d in by_trial.values() if 'finals' in d]
            if trials_data:
                n_imp = sum(1 for d in trials_data if _improved_over_baseline(d))
                rates.append(n_imp / len(trials_data))
        avg = float(np.mean(rates)) if rates else float('nan')
        lines.append(f"  {METHOD_LABELS.get(m,m):<20}: {avg:.3f}")
    return '\n'.join(lines)


def format_n_sensitivity(union_hv_by_n, methods):
    """N=3/6/11 の union UEA HV 比較テキスト。"""
    lines = ['N sensitivity check: union UEA HV の重み数依存性 (lucky punch 対策)', '']
    ns = sorted(union_hv_by_n.keys())
    for n in ns:
        lines.append(f'  N = {n} 重み:')
        hvs = union_hv_by_n[n]
        for m in methods:
            vals = [v for v in hvs.get(m, []) if np.isfinite(v)]
            if not vals:
                continue
            med = float(np.median(vals))
            q25 = float(np.percentile(vals, 25))
            q75 = float(np.percentile(vals, 75))
            lines.append(f"    {METHOD_LABELS.get(m,m):<20} median={med:.2f}  IQR=[{q25:.2f},{q75:.2f}]")
        lines.append('')
    return '\n'.join(lines)


# ========== プロット: 全重み union Pareto front 2D ==========

def plot_pf_weighted_sweep(union_pf_by_method, all_pts_by_method, methods,
                            init_ms, title, outpath):
    """全重み union Pareto front を MS×stability 2D にプロット。
    各手法の union PF をステップ線＋マーカーで描画。
    """
    fig, ax = plt.subplots(figsize=(10, 7))

    for m in methods:
        color = get_method_color(m)
        label = METHOD_LABELS.get(m, m)

        pf = union_pf_by_method.get(m, np.zeros((0, 2)))
        if len(pf) == 0:
            continue
        pf_s = pf[np.argsort(pf[:, 0])]
        ax.step(pf_s[:, 0], pf_s[:, 1], where='post',
                color=color, lw=1.5, alpha=0.85, zorder=2)
        ax.scatter(pf_s[:, 0], pf_s[:, 1], c=color, s=50,
                   marker='o', edgecolors='white', lw=0.5, zorder=3, label=label)

    if init_ms:
        ax.axvline(init_ms, color='gray', ls='--', alpha=0.5, lw=1.2, label='init MS')

    ax.set_xlabel('Makespan')
    ax.set_ylabel('Stability')
    ax.set_title(title)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


# ========== MD サマリ生成 ==========

def _fmt_v(v, fmt='.1f'):
    """数値フォーマット。None/nan は 'N/A'。fmt='d' は整数表示。"""
    if v is None:
        return 'N/A'
    try:
        if np.isnan(float(v)):
            return 'N/A'
    except (TypeError, ValueError):
        return 'N/A'
    if fmt == 'd':
        return str(int(v))
    return f'{v:{fmt}}'


def _compute_summary_for_md(method_data, methods, w_labels, baselines_by_method,
                              thresholds, union_hv_by_method, region_hvs,
                              region_hv_counts, region_hvs_2split,
                              region_hv_counts_2split, union_pf_by_method,
                              global_ref, union_hv_by_n, init_ms,
                              convergence_by_weight=None, ttt_by_weight=None):
    """MD サマリ用データを dict にまとめて返す。"""
    # --- per-weight × per-method ---
    per_weight = {}
    for wl in w_labels:
        per_weight[wl] = {}
        # per-weight 参照点
        all_w_pts = []
        for m in methods:
            bl = baselines_by_method.get(m)
            for t_idx, data in method_data.get(m, {}).get(wl, {}).items():
                pts = get_uea_points(data, t_idx)
                if bl:
                    pts = filter_baselines(pts, bl)
                if len(pts) > 0:
                    all_w_pts.append(pts)
        if all_w_pts:
            wc = np.concatenate(all_w_pts)
            w_ref = (float(wc[:, 0].max()) + max(wc[:, 0].max() * 0.01, 1.0),
                     float(wc[:, 1].max()) + max(wc[:, 1].max() * 0.01, 0.01))
        else:
            w_ref = global_ref

        p33 = thresholds['P33']
        p67 = thresholds['P67']
        st_max_thr = thresholds['stab_max']

        for m in methods:
            bl = baselines_by_method.get(m)
            by_trial = method_data.get(m, {}).get(wl, {})

            scalars = []
            ms_vals, st_vals, hv_vals = [], [], []
            rhv_low_vals, rhv_mid_vals, rhv_high_vals = [], [], []
            n_pf_vals, n_low_vals, n_mid_vals, n_high_vals = [], [], [], []
            for t_idx, data in by_trial.items():
                hist = data.get('history', [])
                if hist:
                    s = hist[-1].get('best_score')
                    if s is not None and np.isfinite(float(s)):
                        scalars.append(float(s))
                f = data.get('finals', {})
                ms = f.get('makespan')
                st = f.get('stability')
                if ms is not None:
                    ms_vals.append(float(ms))
                if st is not None:
                    st_vals.append(float(st))
                pts = get_uea_points(data, t_idx)
                if bl:
                    pts = filter_baselines(pts, bl)
                pf_t = pareto_front(pts) if len(pts) > 0 else np.zeros((0, 2))
                hv_vals.append(hypervolume(pf_t, w_ref) if len(pf_t) > 0 else 0.0)
                # per-trial 領域別 HV / n_PF（ローカル参照点・半開区間で二重カウント防止）
                rl, nl = region_hv(pf_t, 0.0, p33,           global_ref[0])
                rm, nm = region_hv(pf_t, p33, p67,           global_ref[0])
                rh, nh = region_hv(pf_t, p67, global_ref[1], global_ref[0],
                                   hi_inclusive=True)
                rhv_low_vals.append(rl);  n_low_vals.append(nl)
                rhv_mid_vals.append(rm);  n_mid_vals.append(nm)
                rhv_high_vals.append(rh); n_high_vals.append(nh)
                n_pf_vals.append(len(pf_t))

            trials_data = [d for d in by_trial.values() if 'finals' in d]
            if trials_data:
                n_imp = sum(1 for d in trials_data if _improved_over_baseline(d))
                imp_rate = n_imp / len(trials_data)
            else:
                imp_rate = float('nan')

            # per-trial 中央値に集約
            rhv_low  = float(np.median(rhv_low_vals))  if rhv_low_vals  else 0.0
            rhv_mid  = float(np.median(rhv_mid_vals))  if rhv_mid_vals  else 0.0
            rhv_high = float(np.median(rhv_high_vals)) if rhv_high_vals else 0.0
            n_pf     = float(np.median(n_pf_vals))     if n_pf_vals     else 0.0
            n_low    = float(np.median(n_low_vals))    if n_low_vals    else 0.0
            n_mid    = float(np.median(n_mid_vals))    if n_mid_vals    else 0.0
            n_high   = float(np.median(n_high_vals))   if n_high_vals   else 0.0

            per_weight[wl][m] = {
                'scalar_med': float(np.median(scalars)) if scalars else float('nan'),
                'ms_med':     float(np.median(ms_vals)) if ms_vals else float('nan'),
                'stab_med':   float(np.median(st_vals)) if st_vals else float('nan'),
                'hv_med':     float(np.median(hv_vals)) if hv_vals else float('nan'),
                'imp_rate':   imp_rate,
                'n_trials':   len(trials_data),
                'rhv_low':    rhv_low,
                'rhv_mid':    rhv_mid,
                'rhv_high':   rhv_high,
                'n_pf':       n_pf,
                'n_pf_low':   n_low,
                'n_pf_mid':   n_mid,
                'n_pf_high':  n_high,
            }

    # --- Union HV サマリ ---
    union_hv_summary = {}
    for m in methods:
        hvs = [v for v in union_hv_by_method.get(m, []) if np.isfinite(v)]
        if hvs:
            union_hv_summary[m] = {
                'median': float(np.median(hvs)),
                'q25':    float(np.percentile(hvs, 25)),
                'q75':    float(np.percentile(hvs, 75)),
                'n':      len(hvs),
            }

    # --- 全重み平均改善率 ---
    avg_imp = {}
    for m in methods:
        rates = []
        for wl in w_labels:
            by_trial = method_data.get(m, {}).get(wl, {})
            td = [d for d in by_trial.values() if 'finals' in d]
            if td:
                n_imp = sum(1 for d in td if _improved_over_baseline(d))
                rates.append(n_imp / len(td))
        avg_imp[m] = float(np.mean(rates)) if rates else float('nan')

    # --- C-metric ---
    c_mat = {}
    for ma in methods:
        c_mat[ma] = {}
        for mb in methods:
            if ma == mb:
                c_mat[ma][mb] = None
            else:
                pf_a = union_pf_by_method.get(ma, np.zeros((0, 2)))
                pf_b = union_pf_by_method.get(mb, np.zeros((0, 2)))
                c_mat[ma][mb] = c_metric(pf_a, pf_b)

    # --- カバー率 (stab >= P67, P33) ---
    coverage = {}
    for tau_name in ['P67', 'P33']:
        tau = thresholds[tau_name]
        coverage[tau_name] = {}
        for m in methods:
            bl = baselines_by_method.get(m)
            all_trial_indices = set()
            for wl in method_data.get(m, {}):
                all_trial_indices.update(method_data[m][wl].keys())
            covered = 0
            total = len(all_trial_indices)
            for t_idx in all_trial_indices:
                union_pts = []
                for wl in method_data.get(m, {}):
                    data = method_data[m][wl].get(t_idx)
                    if data is None:
                        continue
                    pts = get_uea_points(data, t_idx)
                    if bl:
                        pts = filter_baselines(pts, bl)
                    if len(pts) > 0:
                        union_pts.append(pts)
                if union_pts:
                    combined = np.concatenate(union_pts)
                    if combined[:, 1].max() >= tau:
                        covered += 1
            coverage[tau_name][m] = (covered, total, covered / total if total > 0 else 0.0)

    # --- N sensitivity ---
    n_sens = {}
    for n, hvs in union_hv_by_n.items():
        n_sens[n] = {}
        for m in methods:
            vals = [v for v in hvs.get(m, []) if np.isfinite(v)]
            if vals:
                n_sens[n][m] = {
                    'median': float(np.median(vals)),
                    'q25':    float(np.percentile(vals, 25)),
                    'q75':    float(np.percentile(vals, 75)),
                }

    return {
        'per_weight':           per_weight,
        'union_hv_summary':     union_hv_summary,
        'avg_imp':              avg_imp,
        'c_metric':             c_mat,
        'region_hvs':              region_hvs,
        'region_hv_counts':        region_hv_counts,
        'region_hvs_2split':       region_hvs_2split,
        'region_hv_counts_2split': region_hv_counts_2split,
        'thresholds':              thresholds,
        'coverage':             coverage,
        'n_sensitivity':        n_sens,
        'methods':              methods,
        'w_labels':             w_labels,
        'init_ms':              init_ms,
        'convergence_by_weight': convergence_by_weight or {},
        'ttt_by_weight':        ttt_by_weight or {},
    }


def generate_summary_md(all_summary, out_path, input_dir=''):
    """全問題のサマリ MD を生成して out_path に書き込む。

    all_summary: {prob_label: summary_data dict}
    """
    from datetime import date as _date
    lines = []
    lines.append('# 実験サマリ: core_comparison_v3')
    lines.append('')
    lines.append(f'生成: {_date.today().isoformat()}')
    if input_dir:
        lines.append(f'データ: `{input_dir}`')
    lines.append('')

    for prob_label in sorted(all_summary.keys()):
        sd = all_summary[prob_label]
        methods   = sd['methods']
        w_labels  = sd['w_labels']
        thr       = sd['thresholds']
        ml        = [METHOD_LABELS.get(m, m) for m in methods]

        lines.append('---')
        lines.append('')
        lines.append(f'## {prob_label}')
        lines.append('')

        # ---- 全体概要 ----
        lines.append('### 全体概要')
        lines.append('')
        lines.append('| 手法 | Union HV med | IQR | 改善率 avg |')
        lines.append('|------|:------------:|:---:|:----------:|')
        for m in methods:
            uhv = sd['union_hv_summary'].get(m, {})
            med = _fmt_v(uhv.get('median'), '.2f')
            q25 = _fmt_v(uhv.get('q25'), '.2f')
            q75 = _fmt_v(uhv.get('q75'), '.2f')
            imp = _fmt_v(sd['avg_imp'].get(m), '.3f')
            lines.append(f'| {METHOD_LABELS.get(m, m)} | {med} | [{q25}, {q75}] | {imp} |')
        lines.append('')

        # ---- 収束速度 (time-to-target) ----
        ttt_by_w = sd.get('ttt_by_weight', {})
        if ttt_by_w and any(ttt_by_w.values()):
            taus = _TTT_TAUS
            lines.append('### 収束速度 — time-to-target [s]（代表重み別, trial 間 中央値 [IQR]）')
            lines.append('')
            lines.append('各 trial が目標 HV に初到達する CPU 時間。小さいほど速い。'
                         '速いが品質の低い解への収束を区別するため、最終品質（上の Union HV）と'
                         '併読すること。quality@%t は `anytime_detail_<w>.txt`。')
            lines.append('')
            lines.append('- **self**: 各手法が自身の最終 HV の τ% に到達する時刻（自己ペース）。')
            lines.append('- **common**: 手法横断で固定した共通 HV ターゲット（最弱/中堅/最強手法の'
                         '最終 HV）への到達時刻。`(n/N)` の N 未満は未到達（打ち切り）。')
            lines.append('- 領域は安定性関数値 D で分割（**高安定性=D 小**, 低安定性=D 大）。')
            lines.append('')
            for wl in sorted(ttt_by_w.keys()):
                blocks = ttt_by_w[wl]
                if not blocks:
                    continue
                lines.append(f'#### {wl}')
                lines.append('')
                full = blocks.get('full', {})
                if full:
                    lines.append('**全フロント — self (τ ラダー)**')
                    lines.append('')
                    lines += _md_ttt_self_table(full, methods, taus)
                    lines.append('')
                    lines.append('**全フロント — common-target (QRTD)**')
                    lines.append('')
                    lines += _md_ttt_common_table(full, methods)
                    lines.append('')
                for rk, rlabel in [('lowD', '高安定性領域 (D 小)'),
                                   ('highD', '低安定性領域 (D 大)')]:
                    blk = blocks.get(rk)
                    if not blk:
                        continue
                    lines.append(f'**{rlabel} — common-target**')
                    lines.append('')
                    lines += _md_ttt_common_table(blk, methods)
                    lines.append('')

        # ---- カバー率 ----
        lines.append('### カバー率 (stab ≥ τ の解を 1 個以上持つ trial 割合)')
        lines.append('')
        lines.append(f'| 手法 | stab≥P67 ({thr["P67"]:.4f}) | stab≥P33 ({thr["P33"]:.4f}) |')
        lines.append('|------|:---:|:---:|')
        for m in methods:
            c67 = sd['coverage'].get('P67', {}).get(m, (0, 0, 0.0))
            c33 = sd['coverage'].get('P33', {}).get(m, (0, 0, 0.0))
            lines.append(f'| {METHOD_LABELS.get(m, m)} '
                         f'| {c67[2]:.3f} ({c67[0]}/{c67[1]}) '
                         f'| {c33[2]:.3f} ({c33[0]}/{c33[1]}) |')
        lines.append('')

        # ---- C-metric ----
        lines.append('### C-metric  `C(row, col)` = row が col を弱く支配する割合')
        lines.append('')
        lines.append('| |' + ''.join(f' {l} |' for l in ml))
        lines.append('|---|' + ''.join(':---:|' for _ in methods))
        for ma in methods:
            row = f'| {METHOD_LABELS.get(ma, ma)} |'
            for mb in methods:
                v = sd['c_metric'][ma][mb]
                row += f' {"−" if v is None else f"{v:.3f}"} |'
            lines.append(row)
        lines.append('')

        # ---- 領域別 HV (2分割・主筋) ----
        p50, st_max = thr['P50'], thr['stab_max']
        lines.append(f'### 領域別 HV 2分割  (P50={p50:.4f}, stab_max={st_max:.4f})')
        lines.append('')
        lines.append('| 領域 |' + ''.join(f' {l} |' for l in ml))
        lines.append('|------|' + ''.join(':---:|' for _ in methods))
        for rname, rlabel in [
            ('high_stability', f'高安定性 [0, {p50:.4f}]'),
            ('low_stability',  f'低安定性 ({p50:.4f}, max]'),
        ]:
            row = f'| {rlabel} |'
            for m in methods:
                hv = sd.get('region_hvs_2split', {}).get(m, {}).get(rname, 0.0)
                n  = sd.get('region_hv_counts_2split', {}).get(m, {}).get(rname, 0)
                row += f' {hv:.4f} (n={n}) |'
            lines.append(row)
        lines.append('')


        # ---- N sensitivity ----
        n_sens = sd.get('n_sensitivity', {})
        if n_sens:
            lines.append('### N sensitivity (Union HV, 重み数依存性チェック)')
            lines.append('')
            lines.append('| N |' + ''.join(f' {l} med [IQR] |' for l in ml))
            lines.append('|---|' + ''.join(':---:|' for _ in methods))
            for n in sorted(n_sens.keys()):
                row = f'| {n} |'
                for m in methods:
                    d = n_sens[n].get(m)
                    if d:
                        row += f' {d["median"]:.2f} [{d["q25"]:.2f}, {d["q75"]:.2f}] |'
                    else:
                        row += ' N/A |'
                lines.append(row)
            lines.append('')

        # ---- 重み別詳細テーブル ----
        lines.append('### 重み別詳細')
        lines.append('')
        lines.append('`scalar`: final weighted score 中央値 / `MS`: makespan 中央値 / '
                     '`stab`: stability 中央値 / `HV`: per-weight UEA HV 中央値 / '
                     '`imp`: 改善成功率')
        lines.append('')
        lines.append('| 重み | 指標 |' + ''.join(f' {l} |' for l in ml))
        lines.append('|------|------|' + ''.join(':---:|' for _ in methods))

        p33_v = sd['thresholds']['P33']
        p67_v = sd['thresholds']['P67']
        metrics = [
            ('scalar_med', 'scalar',                  '.4f'),
            ('ms_med',     'MS',                      '.1f'),
            ('stab_med',   'stab',                    '.4f'),
            ('hv_med',     'HV',                      '.2f'),
            ('n_pf',       'n_PF',                    'd'),
            ('rhv_low',    f'rHV-low(≤{p33_v:.3f})',  '.4f'),
            ('n_pf_low',   f'n_PF-low',               'd'),
            ('rhv_mid',    'rHV-mid',                 '.4f'),
            ('n_pf_mid',   'n_PF-mid',                'd'),
            ('rhv_high',   f'rHV-high(>{p67_v:.3f})', '.4f'),
            ('n_pf_high',  'n_PF-high',               'd'),
            ('imp_rate',   'imp',                     '.2f'),
        ]
        for wl in w_labels:
            first = True
            for key, label, fmt in metrics:
                w_disp = wl if first else ''
                first = False
                row = f'| {w_disp} | {label} |'
                for m in methods:
                    v = sd['per_weight'].get(wl, {}).get(m, {}).get(key)
                    row += f' {_fmt_v(v, fmt)} |'
                lines.append(row)

        lines.append('')

    lines.append('---')
    lines.append('')
    lines.append('*generated by analyze_v3.py*')

    with open(out_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    print(f'  → サマリ MD: {out_path}')


# ========== メイン分析: 問題ごと ==========

def analyze_problem(prob_key, method_data, out_dir, problems_filter=None,
                    repr_weights=None, n_jobs=1, xscale='log'):
    """1 問題分の全指標を計算して出力する。

    method_data: {method: {w_label: {trial_idx: data_dict}}}
    """
    problem_name, scenario_name = prob_key
    if problems_filter and problem_name not in problems_filter:
        return

    prob_label = f'{problem_name}_{scenario_name}'
    print(f'\n{"="*60}')
    print(f'分析: {prob_label}')
    print(f'{"="*60}')

    prob_out = os.path.join(out_dir, prob_label)
    os.makedirs(prob_out, exist_ok=True)

    methods = sorted(method_data.keys())
    if not methods:
        print('  データなし')
        return

    # 重みラベルの収集
    all_w_labels_set = set()
    for m in methods:
        all_w_labels_set.update(method_data[m].keys())
    w_labels = sorted(all_w_labels_set)

    # 試行数
    n_trials_by_method = {}
    for m in methods:
        all_trials = set()
        for wl in method_data[m]:
            all_trials.update(method_data[m][wl].keys())
        n_trials_by_method[m] = max(all_trials) + 1 if all_trials else 0
    print(f'  手法: {methods}')
    print(f'  重み: {w_labels}')
    print(f'  試行数: {n_trials_by_method}')

    # 初期解 makespan と norm_params
    try:
        init_ms = get_initial_makespan(problem_name, scenario_name)
    except Exception:
        init_ms = None
    print(f'  init_ms = {init_ms}')

    # baselines_by_method: 手法ごとに [[ms,st], ...] のリスト。
    # baseline (active decode) と baseline_rsr (RSR, st=0) の両方を格納。
    baselines_by_method = {}
    for m in methods:
        bls = []
        for wl in method_data[m]:
            for t_idx, data in method_data[m][wl].items():
                b1 = data.get('baseline')
                b2 = data.get('baseline_rsr')
                if b1 is not None:
                    bls.append(b1)
                if b2 is not None:
                    b2_entry = list(b2)
                    if b2_entry not in bls:
                        bls.append(b2_entry)
                break
            if bls:
                break
        baselines_by_method[m] = bls if bls else None

    # 全体参照点（全手法・全重み・全 trial の UEA 点 max + margin）
    all_pts_flat = []
    for m in methods:
        bl = baselines_by_method.get(m)
        for wl in method_data[m]:
            for t_idx, data in method_data[m][wl].items():
                pts = get_uea_points(data, t_idx)
                if bl:
                    pts = filter_baselines(pts, bl)
                if len(pts) > 0:
                    all_pts_flat.append(pts)
    if not all_pts_flat:
        print('  有効な訪問点なし。スキップ。')
        return
    all_pts_concat = np.concatenate(all_pts_flat)
    global_ref = (
        float(all_pts_concat[:, 0].max()) + max(all_pts_concat[:, 0].max() * 0.01, 1.0),
        float(all_pts_concat[:, 1].max()) + max(all_pts_concat[:, 1].max() * 0.01, 0.01),
    )

    # ===== P33/P67 閾値計算 =====
    thresholds = compute_p33_p67(method_data, baselines_by_method)
    if thresholds is None:
        print('  [WARN] P33/P67 計算失敗。閾値は固定値にフォールバック。')
        st_max = float(all_pts_concat[:, 1].max())
        thresholds = {'P33': st_max / 3, 'P67': 2 * st_max / 3, 'stab_max': st_max}
    print(f'  P33={thresholds["P33"]:.4f}, P67={thresholds["P67"]:.4f}, '
          f'stab_max={thresholds["stab_max"]:.4f}')
    with open(os.path.join(prob_out, 'thresholds.json'), 'w', encoding='utf-8') as f:
        json.dump(thresholds, f, indent=2)

    # ===== B-1: per-weight scalar 比較 =====
    print('  B-1: per-weight scalar 統計...')
    b1_scalar_text = format_b1_scalar_stats(method_data, methods, w_labels,
                                             baselines_by_method)
    with open(os.path.join(prob_out, 'b1_scalar_stats.txt'), 'w', encoding='utf-8') as f:
        f.write(f'=== {prob_label} ===\n\n{b1_scalar_text}\n')

    # ===== B-1: per-weight UEA HV 比較 =====
    print('  B-1: per-weight UEA HV 統計...')
    b1_uea_text = format_b1_uea_hv_stats(method_data, methods, w_labels,
                                          global_ref, baselines_by_method)
    with open(os.path.join(prob_out, 'b1_uea_hv_stats.txt'), 'w', encoding='utf-8') as f:
        f.write(f'=== {prob_label} ===\n\n{b1_uea_text}\n')

    # ===== B-1: 改善成功率ヒートマップ =====
    print('  B-1: 改善成功率ヒートマップ...')
    plot_improvement_heatmap(
        method_data, init_ms, methods, w_labels,
        f'{prob_label}: improvement success rate (weighted score < baseline)',
        os.path.join(prob_out, 'b1_improvement_heatmap.png'))
    imp_text = format_improvement_table(method_data, init_ms, methods, w_labels)
    with open(os.path.join(prob_out, 'b1_improvement_heatmap.txt'), 'w', encoding='utf-8') as f:
        f.write(f'=== {prob_label} ===\n\n{imp_text}\n')

    # ===== B-2a: per-trial union UEA HV =====
    print('  B-2a: per-trial union UEA HV...')
    union_hv_by_method = compute_union_hv_per_trial(
        method_data, baselines_by_method, global_ref, weights_subset=None)
    plot_union_hv_boxplot(
        union_hv_by_method, methods,
        f'{prob_label}: per-trial union UEA HV (N=all weights)',
        os.path.join(prob_out, 'b2a_union_hv_boxplot.png'))
    b2a_stats_text = format_b2a_union_hv_stats(union_hv_by_method, methods)
    with open(os.path.join(prob_out, 'b2a_union_hv_stats.txt'), 'w', encoding='utf-8') as f:
        f.write(f'=== {prob_label} ===\n\n{b2a_stats_text}\n')

    # ===== B-2a: C-metric =====
    print('  B-2a: C-metric...')
    union_pf_by_method = {}
    all_pts_by_method = {}
    for m in methods:
        bl = baselines_by_method.get(m)
        trial_pts_all = []
        for wl in method_data[m]:
            for t_idx, data in method_data[m][wl].items():
                pts = get_uea_points(data, t_idx)
                if bl:
                    pts = filter_baselines(pts, bl)
                if len(pts) > 0:
                    trial_pts_all.append(pts)
        if trial_pts_all:
            combined = np.concatenate(trial_pts_all)
            union_pf_by_method[m] = pareto_front(combined)
            all_pts_by_method[m] = combined
        else:
            union_pf_by_method[m] = np.zeros((0, 2))
            all_pts_by_method[m] = np.zeros((0, 2))
    c_metric_text = format_c_metric(union_pf_by_method, methods)
    with open(os.path.join(prob_out, 'b2a_c_metric.txt'), 'w', encoding='utf-8') as f:
        f.write(f'=== {prob_label} ===\n\n{c_metric_text}\n')

    # ===== Pareto front 2D プロット（全重み weighted sweep union）=====
    print('  PF 2D プロット (weighted sweep)...')
    plot_pf_weighted_sweep(
        union_pf_by_method, all_pts_by_method, methods,
        init_ms,
        f'{prob_label}: union Pareto front (weighted sum sweep)',
        os.path.join(prob_out, 'pf_weighted_sweep.png'))

    # ===== B-2b: カバー率 =====
    print('  B-2b: カバー率...')
    coverage_text = format_b2b_coverage(method_data, methods, baselines_by_method, thresholds)
    with open(os.path.join(prob_out, 'b2b_coverage.txt'), 'w', encoding='utf-8') as f:
        f.write(f'=== {prob_label} ===\n\n{coverage_text}\n')

    # ===== B-2b: 領域別 HV =====
    print('  B-2b: 領域別 HV...')
    p33, p50, p67, st_max = thresholds['P33'], thresholds['P50'], thresholds['P67'], thresholds['stab_max']
    # 命名は 2 分割と同じ安定性ベース（D 小 = 高安定性）。
    regions_3split = {
        'high_stability': (0.0,  p33),
        'mid_stability':  (p33,  p67),
        'low_stability':  (p67,  global_ref[1]),
    }
    regions_2split = {
        'high_stability': (0.0, p50),
        'low_stability':  (p50, global_ref[1]),
    }

    # per-trial union PF を一度だけ計算して両分割で共用する
    trial_pfs_by_method = {}
    for m in methods:
        bl = baselines_by_method.get(m)
        all_trial_indices = set()
        for wl in method_data[m]:
            all_trial_indices.update(method_data[m][wl].keys())
        trial_pfs_by_method[m] = {}
        for t_idx in sorted(all_trial_indices):
            union_pts = []
            for wl in method_data[m]:
                data = method_data[m][wl].get(t_idx)
                if data is None:
                    continue
                pts = get_uea_points(data, t_idx)
                if bl:
                    pts = filter_baselines(pts, bl)
                if len(pts) > 0:
                    union_pts.append(pts)
            trial_pfs_by_method[m][t_idx] = (
                pareto_front(np.concatenate(union_pts)) if union_pts else np.zeros((0, 2))
            )

    def _aggregate_region_hvs(regions_dict):
        hvs, counts = {}, {}
        for m in methods:
            hvs[m] = {}
            counts[m] = {}
            rn_hv_lists = {rn: [] for rn in regions_dict}
            rn_n_lists  = {rn: [] for rn in regions_dict}
            for t_idx, pf_t in trial_pfs_by_method[m].items():
                if len(pf_t) == 0:
                    for rn in regions_dict:
                        rn_hv_lists[rn].append(0.0)
                        rn_n_lists[rn].append(0)
                    continue
                for rn, (lo, hi) in regions_dict.items():
                    hi_inc = (hi == global_ref[1])
                    hv_val, n_val = region_hv(pf_t, lo, hi, global_ref[0],
                                              hi_inclusive=hi_inc)
                    rn_hv_lists[rn].append(hv_val)
                    rn_n_lists[rn].append(n_val)
            for rn in regions_dict:
                hvs[m][rn]    = float(np.median(rn_hv_lists[rn])) if rn_hv_lists[rn] else 0.0
                counts[m][rn] = float(np.median(rn_n_lists[rn]))  if rn_n_lists[rn]  else 0
        return hvs, counts

    # 3分割（参照用・per-problem 図表のみ）
    region_hvs, region_hv_counts = _aggregate_region_hvs(regions_3split)
    plot_region_hv_bars(
        region_hvs, methods, thresholds,
        f'{prob_label}: region-restricted HV (P33/P67 3分割)',
        os.path.join(prob_out, 'b2b_region_hv.png'))
    region_hv_text = format_region_hv_table(region_hvs, region_hv_counts, methods, thresholds)
    with open(os.path.join(prob_out, 'b2b_region_hv.txt'), 'w', encoding='utf-8') as f:
        f.write(f'=== {prob_label} ===\n\n{region_hv_text}\n')

    # 2分割（summary 主筋・per-problem 図表）
    region_hvs_2split, region_hv_counts_2split = _aggregate_region_hvs(regions_2split)
    plot_region_hv_bars_2split(
        region_hvs_2split, methods, thresholds,
        f'{prob_label}: region-restricted HV (P50 2分割)',
        os.path.join(prob_out, 'b2b_region_hv_2split.png'))
    region_hv_text_2split = format_region_hv_table_2split(
        region_hvs_2split, region_hv_counts_2split, methods, thresholds)
    with open(os.path.join(prob_out, 'b2b_region_hv_2split.txt'), 'w', encoding='utf-8') as f:
        f.write(f'=== {prob_label} ===\n\n{region_hv_text_2split}\n')

    # ===== B-2b: 条件付き MS Wilcoxon =====
    print('  B-2b: 条件付き MS Wilcoxon...')
    cond_text = format_b2b_cond_ms_wilcoxon(method_data, methods, baselines_by_method,
                                              thresholds)
    with open(os.path.join(prob_out, 'b2b_cond_ms_wilcoxon.txt'), 'w', encoding='utf-8') as f:
        f.write(f'=== {prob_label} ===\n\n{cond_text}\n')

    # ===== B-2b: 差分 EAF =====
    print('  B-2b: 差分 EAF...')
    _EAF_PAIRS = [('ils_baseline', 'ga'), ('ils_repair', 'ils_baseline'), ('ils_pr', 'ils_baseline')]
    eaf_pairs = [(a, b) for (a, b) in _EAF_PAIRS if a in methods and b in methods]
    for (ma, mb) in eaf_pairs:
        bl_a = baselines_by_method.get(ma)
        bl_b = baselines_by_method.get(mb)
        pts_a_list, pts_b_list = [], []
        for wl in method_data.get(ma, {}):
            for t_idx, data in method_data[ma][wl].items():
                pts = get_uea_points(data, t_idx)
                pts_a_list.append(pts)
        for wl in method_data.get(mb, {}):
            for t_idx, data in method_data[mb][wl].items():
                pts = get_uea_points(data, t_idx)
                pts_b_list.append(pts)
        safe = f'{ma}_vs_{mb}'
        plot_diff_eaf(
            pts_a_list, pts_b_list,
            METHOD_LABELS.get(ma, ma), METHOD_LABELS.get(mb, mb),
            bl_a, bl_b,
            f'{prob_label}: diff EAF  {METHOD_LABELS.get(ma,ma)} − {METHOD_LABELS.get(mb,mb)}',
            os.path.join(prob_out, f'b2b_diff_eaf_{safe}.png'),
            init_ms=init_ms)

    # ===== N sensitivity check =====
    print('  N sensitivity check...')
    union_hv_by_n = {}
    for n, w_subset in N_SENSITIVITY.items():
        use_w = w_subset if w_subset else None
        # w_subset が存在しない重みを含む場合は intersection
        if use_w:
            use_w = [wl for wl in use_w if wl in all_w_labels_set]
            if not use_w:
                continue
        union_hv_by_n[n] = compute_union_hv_per_trial(
            method_data, baselines_by_method, global_ref, weights_subset=use_w)
    n_sens_text = format_n_sensitivity(union_hv_by_n, methods)
    with open(os.path.join(prob_out, 'n_sensitivity.txt'), 'w', encoding='utf-8') as f:
        f.write(f'=== {prob_label} ===\n\n{n_sens_text}\n')

    # ===== Anytime 曲線（代表重み × 手法）=====
    print('  Anytime 曲線...')
    repr_w = [wl for wl in (repr_weights or REPR_WEIGHTS_DEFAULT) if wl in all_w_labels_set]
    convergence_by_weight = {}
    ttt_by_weight = {}
    for wl in repr_w:
        # method_info_list: [(m, hist_list, pts_list, kind, baseline)]
        # pts は anytime 用に (N,3)=[ms, st, cpu_time]。記録済み訪問時刻を使って
        # anytime HV(t) / HV ベース time-to-target を正確に再構成する。
        method_info = []
        for m in methods:
            kind = 'ga' if m == 'ga' else 'ils'
            bl = baselines_by_method.get(m)
            by_trial = method_data[m].get(wl, {})
            hist_list, pts_list = [], []
            for t_idx in sorted(by_trial.keys()):
                data = by_trial[t_idx]
                hist_list.append(get_anytime(data))
                pts_list.append(get_uea_points_xyt(data, t_idx))
            method_info.append((m, hist_list, pts_list, kind, bl))

        if not any(h for _, h, _, _, _ in method_info):
            continue

        # per-weight 参照点（ms, st の 2 列のみ使用。pts は (N,3) のこともあるので
        # 先頭 2 列に正規化してから集約する）
        w_pts = []
        for m, _, pts_list, _, bl in method_info:
            for pts in pts_list:
                if len(pts) == 0:
                    continue
                arr = np.asarray(pts, dtype=float)[:, :2]
                filtered = filter_baselines(arr, bl) if bl else arr
                if len(filtered) > 0:
                    w_pts.append(filtered)
        if not w_pts:
            continue
        w_concat = np.concatenate(w_pts)
        w_ref = (float(w_concat[:, 0].max()) + max(w_concat[:, 0].max() * 0.01, 1.0),
                 float(w_concat[:, 1].max()) + max(w_concat[:, 1].max() * 0.01, 0.01))

        plot_anytime_scalar(
            method_info,
            f'{prob_label} [{wl}]: anytime best weighted score',
            os.path.join(prob_out, f'anytime_scalar_{wl}.png'),
            xscale=xscale)

        plot_anytime_uea_hv(
            method_info, w_ref,
            f'{prob_label} [{wl}]: anytime per-weight UEA HV',
            os.path.join(prob_out, f'anytime_uea_hv_{wl}.png'),
            xscale=xscale,
            n_jobs=n_jobs)

        write_anytime_txt(
            method_info, w_ref, wl,
            os.path.join(prob_out, f'anytime_detail_{wl}.txt'),
            n_jobs=n_jobs)

        # 収束速度: time-to-target を主指標とする（quality@%t は anytime_detail_<w>.txt）。
        # 全フロント + 安定性バンド別（high_stability=D 小, low_stability=D 大）で、
        # self-referenced(τ) と common-target(QRTD) を計算する。
        p50_band = thresholds['P50']
        regions_ttt = {
            'full':  None,
            'lowD':  (0.0, p50_band, False),           # 高安定性（安定性関数値 D が小さい）
            'highD': (p50_band, global_ref[1], True),  # 低安定性（D が大きい）
        }
        ttt_by_weight[wl] = {
            rk: _compute_ttt_block(method_info, w_ref, _TTT_TAUS, region=rv, n_jobs=n_jobs)
            for rk, rv in regions_ttt.items()
        }

    print(f'  → {prob_out}')

    # ===== MD サマリ用データを収集して返す =====
    summary_data = _compute_summary_for_md(
        method_data, methods, w_labels, baselines_by_method,
        thresholds, union_hv_by_method, region_hvs, region_hv_counts,
        region_hvs_2split, region_hv_counts_2split,
        union_pf_by_method, global_ref, union_hv_by_n, init_ms,
        convergence_by_weight, ttt_by_weight)
    return prob_label, summary_data


# ========== エントリポイント ==========

def main():
    parser = argparse.ArgumentParser(description='core_comparison_v3 分析スクリプト')
    parser.add_argument(
        '--input-dir', required=True,
        help='run_v3.py の出力ディレクトリ')
    parser.add_argument(
        '--out-dir', default=None,
        help='分析出力先 (デフォルト: <input-dir>/analysis)')
    parser.add_argument(
        '--problems', nargs='+', default=None,
        help='分析する問題名 (例: la36 la40). デフォルト: 全問題')
    parser.add_argument(
        '--xscale', default='log', choices=['linear', 'log'],
        help='anytime 曲線の横軸スケール')
    parser.add_argument(
        '--repr-weights', nargs='+', default=None,
        help='anytime 曲線を描く重みラベル (例: w08_02 w05_05). '
             f'デフォルト: {REPR_WEIGHTS_DEFAULT}')
    parser.add_argument(
        '--n-jobs', type=int, default=4,
        help='anytime HV 曲線の並列計算ワーカー数 (デフォルト: 4)')
    args = parser.parse_args()

    input_dir = args.input_dir
    out_dir = args.out_dir or os.path.join(input_dir, 'analysis')
    os.makedirs(out_dir, exist_ok=True)

    print(f'データ読み込み: {input_dir}')
    grouped = load_all_runs(input_dir)
    if not grouped:
        print('データが見つかりませんでした。')
        return

    print(f'問題数: {len(grouped)}')
    for k, m in grouped.items():
        total_runs = sum(len(by_trial)
                         for method_data in m.values()
                         for by_trial in method_data.values())
        print(f'  {k[0]}/{k[1]}: {list(m.keys())} × {len(next(iter(m.values())))} weights'
              f' × ... = {total_runs} runs')

    repr_weights = args.repr_weights if args.repr_weights else REPR_WEIGHTS_DEFAULT

    all_summary = {}
    for prob_key in sorted(grouped.keys()):
        result = analyze_problem(
            prob_key, grouped[prob_key], out_dir,
            problems_filter=args.problems,
            repr_weights=repr_weights,
            n_jobs=args.n_jobs,
            xscale=args.xscale)
        if result is not None:
            prob_label, summary_data = result
            all_summary[prob_label] = summary_data

    if all_summary:
        md_path = os.path.join(out_dir, 'summary.md')
        generate_summary_md(all_summary, md_path, input_dir=input_dir)

    print(f'\n分析完了: {out_dir}')


if __name__ == '__main__':
    main()
