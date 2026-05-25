#!/usr/bin/env python3
"""
Stage 1 掃引結果の分析スクリプト

run_ils_sweep.py が出力した results.json を読み込み、以下を生成する。

=== 出力 ===
各問題 <problem>_<scenario>/analysis/ に以下:

[集約指標]
  summary_table.txt            # config × metric のテキスト表
  hv_heatmap.png               # (Stage 1-A) HV ヒートマップ
  tornado.png                  # (Stage 1-B) 基準点からの変動
  scalar_bar.png               # Score/HV の bar + IQR

[挙動指標]
  acceptance_breakdown.png     # perturb_used × accepted の stacked bar
  strength_trace.png           # strength の反復軸平均推移
  last_improve_iter_cdf.png    # best 最終更新 iteration の CDF
                                (= max_iterations の妥当性確認)

[反復軸 anytime]
  anytime_best_score.png       # best_score vs iteration (平均+IQR)
  anytime_best_ms.png
  anytime_best_stab.png
  anytime_hv.png               # per-trial cumulative Pareto HV vs iteration

[Pareto]
  pareto_overlay.png           # 全 config の per-trial union Pareto

=== 使い方 ===
  python analyze_ils_sweep.py results/stage1a_<timestamp>/
"""

import argparse
import json
import os
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import matplotlib
matplotlib.use('Agg')
# 日本語グリフ対応: Windows 標準フォントを優先、DejaVu Sans にフォールバック
matplotlib.rcParams['font.family'] = [
    'Meiryo', 'MS Gothic', 'Yu Gothic', 'DejaVu Sans'
]
matplotlib.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt
import numpy as np


# ========== ロード ==========

def load_problem_data(problem_dir):
    path = os.path.join(problem_dir, 'results.json')
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def iter_problem_dirs(root_dir):
    for name in sorted(os.listdir(root_dir)):
        p = os.path.join(root_dir, name)
        if os.path.isdir(p) and os.path.isfile(os.path.join(p, 'results.json')):
            yield p


# ========== Pareto / HV ==========

def pareto_front_2d(points, baseline=None):
    if len(points) == 0:
        return np.zeros((0, 2))
    pts = np.asarray(points, dtype=float)
    if baseline is not None:
        b_ms, b_st = baseline
        eps = 1e-9
        mask = ~((pts[:, 0] >= b_ms - eps) & (pts[:, 1] >= b_st - eps))
        pts = pts[mask]
    if len(pts) == 0:
        return np.zeros((0, 2))
    idx = np.lexsort((pts[:, 1], pts[:, 0]))
    sp = pts[idx]
    pf = [sp[0]]
    for p in sp[1:]:
        if p[1] < pf[-1][1]:
            pf.append(p)
    return np.array(pf)


def hv_2d(points, ref, baseline=None):
    if len(points) == 0:
        return 0.0
    pf = pareto_front_2d(points, baseline=baseline)
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


# ========== 履歴操作 ==========

def history_excluding_init(history):
    """iteration==0 (init) を除いた履歴"""
    return [h for h in history if h.get('iteration', 0) > 0]


def collect_all_ls_points(data):
    """全 config × 全 trial の (ls_ms, ls_st) 点を集める。
    HV 参照点の nadir 計算に使う。"""
    all_pts = []
    for cid, trial_list in data['results'].items():
        for trial_data in trial_list:
            if trial_data is None or 'error' in trial_data:
                continue
            for h in trial_data['history']:
                if h.get('ls_makespan') is None:
                    continue
                all_pts.append((h['ls_makespan'], h['ls_stability']))
    return np.array(all_pts) if all_pts else np.zeros((0, 2))


def make_reference_point(data):
    """HV 参照点 (MS_ref, Stab_ref) を決める。
    MS_ref = init_makespan
    Stab_ref = 全訪問点の max(stab) * 1.05
    """
    init_ms = float(data['init_makespan'])
    all_pts = collect_all_ls_points(data)
    if len(all_pts) == 0:
        return (init_ms, 1.0)
    stab_max = float(all_pts[:, 1].max())
    return (init_ms, stab_max * 1.05 + 1e-6)


# ========== Anytime (iteration-axis) ==========

def sample_iterations(max_iter, n_samples=60):
    """反復軸 anytime プロット用のサンプル iteration 列（密に取りつつログ寄り）"""
    # 前半は密、後半は疎
    a = np.unique(np.round(np.linspace(1, max_iter, n_samples)).astype(int))
    return a


def cumulative_hv_per_iteration(history, ref, baseline, sample_iters):
    """1 trial について、各サンプル iteration での cumulative Pareto HV を計算
    history: iteration>=1 の並び (iteration=0 は除外済み前提、含んでいても可)
    """
    # iteration ごとに (ls_ms, ls_st) を並べる
    iters = [h['iteration'] for h in history]
    pts   = [(h['ls_makespan'], h['ls_stability']) for h in history]
    if not pts:
        return np.full(len(sample_iters), np.nan)

    order = np.argsort(iters)
    iters = [iters[i] for i in order]
    pts   = [pts[i] for i in order]

    out = []
    j = 0
    current_pts = []
    for t in sample_iters:
        while j < len(iters) and iters[j] <= t:
            current_pts.append(pts[j])
            j += 1
        out.append(hv_2d(np.array(current_pts) if current_pts else np.zeros((0, 2)),
                         ref, baseline=baseline))
    return np.array(out)


def best_field_per_iteration(history, field, sample_iters):
    """best_* フィールドの反復軸値を補間"""
    iters = np.array([h['iteration'] for h in history])
    vals  = np.array([h[field] for h in history], dtype=float)
    if len(iters) == 0:
        return np.full(len(sample_iters), np.nan)
    order = np.argsort(iters)
    iters = iters[order]; vals = vals[order]
    # best_* は単調非増加。step 補間で最後の iter<=t の値を使う。
    out = []
    j = 0
    last = vals[0] if len(vals) > 0 else np.nan
    for t in sample_iters:
        while j < len(iters) and iters[j] <= t:
            last = vals[j]
            j += 1
        out.append(last)
    return np.array(out)


# ========== 可視化 ==========

def get_color(i, n):
    return plt.cm.tab10(i % 10)


def plot_anytime_curves(data, out_dir, sample_iters, ref, baseline):
    """best_score / best_ms / best_stab / cumulative HV vs iteration"""
    configs = data['configs']
    cids = list(configs.keys())

    # 各 config × 各 trial のカーブ計算
    curves = {
        'best_score': {},
        'best_makespan': {},
        'best_stability': {},
        'hv': {},
    }
    for cid in cids:
        per_trial_score = []
        per_trial_ms    = []
        per_trial_stab  = []
        per_trial_hv    = []
        for trial_data in data['results'][cid]:
            if trial_data is None or 'error' in trial_data:
                continue
            h = history_excluding_init(trial_data['history'])
            per_trial_score.append(best_field_per_iteration(h, 'best_score', sample_iters))
            per_trial_ms.append(best_field_per_iteration(h, 'best_makespan', sample_iters))
            per_trial_stab.append(best_field_per_iteration(h, 'best_stability', sample_iters))
            per_trial_hv.append(cumulative_hv_per_iteration(h, ref, baseline, sample_iters))
        curves['best_score'][cid] = np.array(per_trial_score) if per_trial_score else None
        curves['best_makespan'][cid] = np.array(per_trial_ms) if per_trial_ms else None
        curves['best_stability'][cid] = np.array(per_trial_stab) if per_trial_stab else None
        curves['hv'][cid] = np.array(per_trial_hv) if per_trial_hv else None

    plot_specs = [
        ('best_score',     'Best Score',      'anytime_best_score.png'),
        ('best_makespan',  'Best Makespan',   'anytime_best_ms.png'),
        ('best_stability', 'Best Stability',  'anytime_best_stab.png'),
        ('hv',             'Cumulative HV',   'anytime_hv.png'),
    ]
    for key, ylabel, fname in plot_specs:
        fig, ax = plt.subplots(figsize=(11, 6))
        for i, cid in enumerate(cids):
            arr = curves[key][cid]
            if arr is None or len(arr) == 0:
                continue
            med = np.nanmedian(arr, axis=0)
            q1  = np.nanpercentile(arr, 25, axis=0)
            q3  = np.nanpercentile(arr, 75, axis=0)
            color = get_color(i, len(cids))
            ax.plot(sample_iters, med, color=color, label=cid, linewidth=1.5)
            ax.fill_between(sample_iters, q1, q3, color=color, alpha=0.15)
        ax.set_xlabel('Iteration')
        ax.set_ylabel(ylabel)
        ax.set_title(f'{ylabel} vs Iteration (median + IQR, {data["problem"]}_{data["scenario"]})')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc='best', ncol=2)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, fname), dpi=150)
        plt.close(fig)


def plot_hv_heatmap_stage1a(data, out_dir, ref, baseline):
    """Stage 1-A: perturb × strategy の HV 2D ヒートマップ（trial 中央値）"""
    configs = data['configs']
    if any(c.get('axis') != 'grid' for c in configs.values()):
        return  # Stage 1-A 以外はスキップ

    perturbs = sorted({c['perturb'] for c in configs.values()})
    strategies = sorted({c['strategy'] for c in configs.values()})

    hv_mat    = np.full((len(perturbs), len(strategies)), np.nan)
    score_mat = np.full((len(perturbs), len(strategies)), np.nan)

    for cid, cfg in configs.items():
        i = perturbs.index(cfg['perturb'])
        j = strategies.index(cfg['strategy'])
        trials = [d for d in data['results'][cid]
                  if d is not None and 'error' not in d]
        if not trials:
            continue
        # 各 trial の最終 HV（全訪問点の Pareto）
        hvs = []
        scores = []
        for t in trials:
            pts = [(h['ls_makespan'], h['ls_stability'])
                   for h in history_excluding_init(t['history'])]
            hvs.append(hv_2d(np.array(pts), ref, baseline=baseline))
            scores.append(t['history'][-1]['best_score'])
        hv_mat[i, j]    = float(np.median(hvs))
        score_mat[i, j] = float(np.median(scores))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, mat, title, cmap in [
        (axes[0], hv_mat,    'HV (median over trials, higher = better)', 'viridis'),
        (axes[1], score_mat, 'Weighted Score (median, lower = better)',  'viridis_r'),
    ]:
        im = ax.imshow(mat, aspect='auto', cmap=cmap)
        ax.set_xticks(range(len(strategies)))
        ax.set_xticklabels(strategies)
        ax.set_yticks(range(len(perturbs)))
        ax.set_yticklabels(perturbs)
        ax.set_xlabel('strategy (LS 規則)')
        ax.set_ylabel('perturb_method')
        ax.set_title(title)
        for i in range(len(perturbs)):
            for j in range(len(strategies)):
                v = mat[i, j]
                if not np.isnan(v):
                    ax.text(j, i, f'{v:.3g}', ha='center', va='center',
                            color='white', fontsize=10)
        fig.colorbar(im, ax=ax)

    fig.suptitle(f'Stage 1-A HV heatmap ({data["problem"]}_{data["scenario"]})')
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'hv_heatmap.png'), dpi=150)
    plt.close(fig)


def plot_tornado_stage1b(data, out_dir, ref, baseline):
    """Stage 1-B: 基準点からの HV 変動量 tornado plot"""
    configs = data['configs']
    if 'base' not in configs:
        return  # Stage 1-B 以外はスキップ

    def config_median_hv(cid):
        trials = [d for d in data['results'][cid]
                  if d is not None and 'error' not in d]
        if not trials:
            return np.nan
        hvs = []
        for t in trials:
            pts = [(h['ls_makespan'], h['ls_stability'])
                   for h in history_excluding_init(t['history'])]
            hvs.append(hv_2d(np.array(pts), ref, baseline=baseline))
        return float(np.median(hvs))

    base_hv = config_median_hv('base')
    deltas = []  # [(label, delta, axis)]
    for cid, cfg in configs.items():
        if cid == 'base':
            continue
        d = config_median_hv(cid) - base_hv
        deltas.append((cid, d, cfg.get('axis', '?')))

    # axis ごとに並べる
    deltas.sort(key=lambda x: (x[2], x[0]))

    fig, ax = plt.subplots(figsize=(10, max(4, 0.4 * len(deltas) + 2)))
    labels = [f'{cid} [{ax_name}]' for cid, _, ax_name in deltas]
    values = [d for _, d, _ in deltas]
    colors = ['tab:green' if v > 0 else 'tab:red' for v in values]
    ax.barh(labels, values, color=colors, alpha=0.7)
    ax.axvline(0, color='black', linewidth=0.8)
    ax.set_xlabel(f'HV 変動量 (基準点 base の HV = {base_hv:.3g} を 0 とした差)')
    ax.set_title(f'Stage 1-B tornado ({data["problem"]}_{data["scenario"]})')
    ax.grid(True, alpha=0.3, axis='x')
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'tornado.png'), dpi=150)
    plt.close(fig)


def plot_repair_heatmap_stage2a(data, out_dir, ref, baseline):
    """Stage 2-A: ILS variant ごとに repair_trigger × repair_strength の HV ヒートマップ。

    各 variant について 2 枚のヒートマップ (HV, Score)、
    各セルは per-trial median。baseline (repair_mode=False) は別途表示。
    """
    configs = data['configs']
    # variant ラベルを持つ Stage 2-A 用 config か確認
    if not any('variant' in c for c in configs.values()):
        return

    # variant 別に config を分ける
    variants = sorted({c['variant'] for c in configs.values() if 'variant' in c})
    triggers = sorted({c['repair_trigger']
                       for c in configs.values()
                       if c.get('axis') == 'grid'})
    strengths = sorted({c['repair_strength']
                        for c in configs.values()
                        if c.get('axis') == 'grid'})

    if not triggers or not strengths:
        return

    def per_trial_hv_score(cid):
        """1 config の (HV per trial, score per trial) を返す"""
        trials = [d for d in data['results'][cid]
                  if d is not None and 'error' not in d]
        if not trials:
            return [], []
        hvs, scores = [], []
        for t in trials:
            pts = [(h['ls_makespan'], h['ls_stability'])
                   for h in history_excluding_init(t['history'])]
            hvs.append(hv_2d(np.array(pts), ref, baseline=baseline))
            scores.append(t['history'][-1]['best_score'])
        return hvs, scores

    for variant in variants:
        # baseline (no repair) の median 値
        baseline_cid = f"{variant}_baseline"
        if baseline_cid in configs:
            base_hvs, base_scores = per_trial_hv_score(baseline_cid)
            base_hv_med = float(np.median(base_hvs)) if base_hvs else np.nan
            base_score_med = float(np.median(base_scores)) if base_scores else np.nan
        else:
            base_hv_med = np.nan
            base_score_med = np.nan

        # grid cells の median
        hv_mat    = np.full((len(strengths), len(triggers)), np.nan)
        score_mat = np.full((len(strengths), len(triggers)), np.nan)
        for cid, cfg in configs.items():
            if cfg.get('axis') != 'grid' or cfg.get('variant') != variant:
                continue
            i = strengths.index(cfg['repair_strength'])
            j = triggers.index(cfg['repair_trigger'])
            hvs, scores = per_trial_hv_score(cid)
            if hvs:
                hv_mat[i, j]    = float(np.median(hvs))
                score_mat[i, j] = float(np.median(scores))

        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        for ax, mat, title, cmap, base_val in [
            (axes[0], hv_mat,    f'HV (median, higher = better)\n'
                                 f'baseline={base_hv_med:.2f}', 'viridis', base_hv_med),
            (axes[1], score_mat, f'Weighted Score (median, lower = better)\n'
                                 f'baseline={base_score_med:.4f}', 'viridis_r', base_score_med),
        ]:
            im = ax.imshow(mat, aspect='auto', cmap=cmap, origin='lower')
            ax.set_xticks(range(len(triggers)))
            ax.set_xticklabels(triggers)
            ax.set_yticks(range(len(strengths)))
            ax.set_yticklabels(strengths)
            ax.set_xlabel('repair_trigger')
            ax.set_ylabel('repair_strength')
            ax.set_title(title)
            for i in range(len(strengths)):
                for j in range(len(triggers)):
                    v = mat[i, j]
                    if not np.isnan(v):
                        # baseline 比で勝ってる cell に強調マーク
                        marker = ''
                        if not np.isnan(base_val):
                            if (cmap == 'viridis' and v > base_val) or \
                               (cmap == 'viridis_r' and v < base_val):
                                marker = '*'
                        ax.text(j, i, f'{v:.3g}{marker}', ha='center', va='center',
                                color='white', fontsize=9)
            fig.colorbar(im, ax=ax)

        fig.suptitle(
            f'Stage 2-A repair heatmap [{variant}] '
            f'({data["problem"]}_{data["scenario"]})\n'
            f'* = baseline 越え')
        fig.tight_layout()
        fig.savefig(
            os.path.join(out_dir, f'repair_heatmap_{variant}.png'), dpi=150)
        plt.close(fig)


def plot_repair_lift_stage2a(data, out_dir, baseline):
    """Stage 2-A: 各 variant の Region-restricted HV について baseline → 最良 grid cell
    の改善量（low_stab / mid_stab / high_stab 別）を表示。

    repair の主目的（low_stab 領域での Pareto 拡張）を可視化。
    """
    configs = data['configs']
    if not any('variant' in c for c in configs.values()):
        return

    variants = sorted({c['variant'] for c in configs.values() if 'variant' in c})
    init_ms = float(data['init_makespan'])

    # 領域境界決定（cross-method union）
    config_unions = {cid: _config_union_pareto(data, cid, baseline)
                     for cid in configs}
    all_pts = np.vstack([p for p in config_unions.values() if len(p) > 0]) \
              if any(len(p) > 0 for p in config_unions.values()) \
              else np.zeros((0, 2))
    if len(all_pts) == 0:
        return
    union_pareto = pareto_front_2d(all_pts)
    if len(union_pareto) == 0:
        return
    stab_max = float(union_pareto[:, 1].max())
    if stab_max <= 0:
        return
    eps = 1e-9
    t1, t2 = stab_max / 3.0, 2.0 * stab_max / 3.0
    regions = [
        ('low_stab',  (0.0,    t1)),
        ('mid_stab',  (t1+eps, t2)),
        ('high_stab', (t2+eps, stab_max)),
    ]

    def restricted_hv_per_trial_median(cid, region):
        lo, hi = region
        per_trial = []
        for pf in _config_per_trial_paretos(data, cid, baseline):
            per_trial.append(_restricted_hv(pf, lo, hi, init_ms))
        return float(np.median(per_trial)) if per_trial else 0.0

    # variant × region について baseline と best-grid を比較
    fig, ax = plt.subplots(figsize=(11, 5))
    bar_width = 0.18
    region_colors = {'low_stab': 'tab:blue', 'mid_stab': 'tab:orange',
                     'high_stab': 'tab:green'}
    x_centers = np.arange(len(variants))

    for jr, (rname, region) in enumerate(regions):
        baseline_vals = []
        best_grid_vals = []
        best_cell_labels = []
        for variant in variants:
            base_cid = f"{variant}_baseline"
            base_v = restricted_hv_per_trial_median(base_cid, region) \
                     if base_cid in configs else 0.0
            grid_cids = [cid for cid, c in configs.items()
                         if c.get('axis') == 'grid' and c.get('variant') == variant]
            grid_vals = [(cid, restricted_hv_per_trial_median(cid, region))
                         for cid in grid_cids]
            if grid_vals:
                best_cid, best_v = max(grid_vals, key=lambda x: x[1])
            else:
                best_cid, best_v = '?', 0.0
            baseline_vals.append(base_v)
            best_grid_vals.append(best_v)
            best_cell_labels.append(best_cid)

        offset = (jr - 1) * bar_width * 2
        bars_b = ax.bar(x_centers + offset, baseline_vals, bar_width,
                        label=f'{rname} baseline',
                        color=region_colors[rname], alpha=0.55, hatch='//')
        bars_g = ax.bar(x_centers + offset + bar_width, best_grid_vals,
                        bar_width, label=f'{rname} best-repair',
                        color=region_colors[rname], alpha=0.95)
        # best cell ラベル
        for xi, (b, lab) in enumerate(zip(best_grid_vals, best_cell_labels)):
            short = lab.replace(f'{variants[xi]}_', '')
            ax.text(x_centers[xi] + offset + bar_width, b + max(b * 0.02, 0.5),
                    short, ha='center', fontsize=7, alpha=0.7)

    ax.set_xticks(x_centers)
    ax.set_xticklabels([f'ILS-{v}' for v in variants])
    ax.set_ylabel('Region-restricted HV (per-trial median)')
    ax.set_title(
        f'Stage 2-A repair lift: baseline vs best-repair-cell '
        f'({data["problem"]}_{data["scenario"]})\n'
        f'各 variant × 各領域で「baseline (//) → 最良 grid cell」')
    ax.grid(True, alpha=0.3, axis='y')
    ax.legend(fontsize=8, loc='best', ncol=3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'repair_lift.png'), dpi=150)
    plt.close(fig)


def plot_acceptance_breakdown(data, out_dir):
    """各 config について perturb_used × accepted の stacked bar

    左: accepted iteration の perturb_used 分布
    右: 全 iteration の accepted / rejected 割合
    """
    configs = data['configs']
    cids = list(configs.keys())

    accepted_counts = []  # {perturb: count} per config
    reject_ratios = []    # rejected / total per config

    all_perturbs_seen = set()
    for cid in cids:
        acc = Counter()
        total = 0
        rejected = 0
        for trial_data in data['results'][cid]:
            if trial_data is None or 'error' in trial_data:
                continue
            for h in history_excluding_init(trial_data['history']):
                total += 1
                pu = h.get('perturb_used', '?')
                all_perturbs_seen.add(pu)
                if h.get('accepted'):
                    acc[pu] += 1
                else:
                    rejected += 1
        accepted_counts.append(acc)
        reject_ratios.append(rejected / total if total > 0 else 0.0)

    perturb_labels = sorted(all_perturbs_seen)
    color_map = {p: plt.cm.tab10(i % 10) for i, p in enumerate(perturb_labels)}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, max(4, 0.35 * len(cids) + 2)))

    # 左: 受理回数の stacked bar
    bottom = np.zeros(len(cids))
    y = np.arange(len(cids))
    for p in perturb_labels:
        vals = np.array([ac.get(p, 0) for ac in accepted_counts])
        ax1.barh(y, vals, left=bottom, label=p, color=color_map[p], alpha=0.85)
        bottom += vals
    ax1.set_yticks(y)
    ax1.set_yticklabels(cids)
    ax1.set_xlabel('受理回数の合計（全 trial 合算）')
    ax1.set_title('perturb_used 別・受理内訳')
    ax1.legend(fontsize=8, loc='lower right')
    ax1.grid(True, alpha=0.3, axis='x')

    # 右: 棄却率
    ax2.barh(y, reject_ratios, color='tab:red', alpha=0.7)
    ax2.set_yticks(y)
    ax2.set_yticklabels(cids)
    ax2.set_xlabel('棄却率 (rejected / total iterations)')
    ax2.set_xlim(0, 1)
    ax2.set_title('棄却率')
    ax2.grid(True, alpha=0.3, axis='x')
    for i, r in enumerate(reject_ratios):
        ax2.text(r + 0.01, i, f'{r:.2%}', va='center', fontsize=8)

    fig.suptitle(f'Acceptance breakdown ({data["problem"]}_{data["scenario"]})')
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'acceptance_breakdown.png'), dpi=150)
    plt.close(fig)


def plot_strength_trace(data, out_dir, sample_iters):
    """各 config の strength の反復軸推移（trial 平均）"""
    configs = data['configs']
    cids = list(configs.keys())

    fig, ax = plt.subplots(figsize=(11, 6))
    for i, cid in enumerate(cids):
        per_trial = []
        for trial_data in data['results'][cid]:
            if trial_data is None or 'error' in trial_data:
                continue
            h = history_excluding_init(trial_data['history'])
            if not h:
                continue
            iters = np.array([x['iteration'] for x in h])
            strs  = np.array([x['strength'] if x.get('strength') is not None else np.nan
                              for x in h], dtype=float)
            order = np.argsort(iters)
            iters = iters[order]; strs = strs[order]
            # step 補間
            interp = []
            last = strs[0] if not np.isnan(strs[0]) else np.nan
            j = 0
            for t in sample_iters:
                while j < len(iters) and iters[j] <= t:
                    if not np.isnan(strs[j]):
                        last = strs[j]
                    j += 1
                interp.append(last)
            per_trial.append(np.array(interp))
        if not per_trial:
            continue
        arr = np.array(per_trial)
        med = np.nanmedian(arr, axis=0)
        color = get_color(i, len(cids))
        ax.plot(sample_iters, med, color=color, label=cid, linewidth=1.5)

    ax.set_xlabel('Iteration')
    ax.set_ylabel('Strength (中央値)')
    ax.set_title(f'Strength adaptation trace ({data["problem"]}_{data["scenario"]})')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc='best', ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'strength_trace.png'), dpi=150)
    plt.close(fig)


def plot_last_improve_cdf(data, out_dir):
    """best_score が最後に改善した iteration の CDF（max_iter 妥当性確認）"""
    configs = data['configs']
    cids = list(configs.keys())

    fig, ax = plt.subplots(figsize=(11, 6))
    max_iter = data['max_iterations']
    for i, cid in enumerate(cids):
        last_iters = []
        for trial_data in data['results'][cid]:
            if trial_data is None or 'error' in trial_data:
                continue
            h = history_excluding_init(trial_data['history'])
            if not h:
                continue
            # best_score が減少した最後の iteration
            prev = h[0]['best_score']
            last = h[0]['iteration']
            for entry in h[1:]:
                if entry['best_score'] < prev - 1e-12:
                    last = entry['iteration']
                    prev = entry['best_score']
            last_iters.append(last)
        if not last_iters:
            continue
        sorted_iters = np.sort(last_iters)
        cdf = np.arange(1, len(sorted_iters) + 1) / len(sorted_iters)
        color = get_color(i, len(cids))
        ax.step(sorted_iters, cdf, where='post', color=color,
                label=f'{cid} (n={len(last_iters)})', linewidth=1.4)

    ax.axvline(max_iter, color='black', linestyle='--', alpha=0.5,
               label=f'max_iter={max_iter}')
    ax.axhline(0.95, color='gray', linestyle=':', alpha=0.6, label='95%')
    ax.set_xlabel('Last best improvement iteration')
    ax.set_ylabel('CDF (trials)')
    ax.set_title(f'Last improvement iteration CDF ({data["problem"]}_{data["scenario"]})\n'
                 f'95% が max_iter より十分手前なら現状の予算で OK')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc='lower right', ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'last_improve_iter_cdf.png'), dpi=150)
    plt.close(fig)


def plot_pareto_overlay(data, out_dir, baseline):
    """全 config の per-trial union Pareto を 1 枚に重ねる"""
    configs = data['configs']
    cids = list(configs.keys())

    fig, ax = plt.subplots(figsize=(11, 8))
    init_ms = data['init_makespan']
    for i, cid in enumerate(cids):
        all_pts = []
        for trial_data in data['results'][cid]:
            if trial_data is None or 'error' in trial_data:
                continue
            for h in history_excluding_init(trial_data['history']):
                all_pts.append((h['ls_makespan'], h['ls_stability']))
        if not all_pts:
            continue
        pf = pareto_front_2d(all_pts, baseline=baseline)
        if len(pf) == 0:
            continue
        pf = pf[np.argsort(pf[:, 0])]
        color = get_color(i, len(cids))
        ax.plot(pf[:, 0], pf[:, 1], color=color, linewidth=1.2,
                marker='o', markersize=5, alpha=0.85, label=cid)

    ax.axvline(init_ms, color='gray', linestyle=':', alpha=0.6, label=f'init_ms={init_ms}')
    ax.set_xlabel('Makespan')
    ax.set_ylabel('Stability')
    ax.set_title(f'Pareto overlay (union over trials, {data["problem"]}_{data["scenario"]})')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc='best', ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'pareto_overlay.png'), dpi=150)
    plt.close(fig)


def write_summary_table(data, out_dir, ref, baseline):
    """config × (MS, Stab, Score, HV) のテキスト表"""
    configs = data['configs']
    lines = [f"Stage {data['stage']} / {data['problem']}_{data['scenario']}",
             f"  weights={data['weights']}, init_ms={data['init_makespan']}, "
             f"HV_ref={ref}",
             ""]
    header = (f"  {'config':<22} {'MS_med':>8} {'MS_std':>8} "
              f"{'St_med':>8} {'St_std':>8} {'Score_med':>11} "
              f"{'HV_med':>12} {'HV_std':>10}")
    lines.append(header)
    lines.append("  " + "-" * (len(header) - 2))

    rows = []
    for cid, cfg in configs.items():
        trials = [d for d in data['results'][cid]
                  if d is not None and 'error' not in d]
        if not trials:
            lines.append(f"  {cid:<22} {'no valid trials':>60}")
            continue
        ms  = [t['makespan'] for t in trials]
        st  = [t['stability'] for t in trials]
        sc  = [t['history'][-1]['best_score'] for t in trials]
        hvs = []
        for t in trials:
            pts = [(h['ls_makespan'], h['ls_stability'])
                   for h in history_excluding_init(t['history'])]
            hvs.append(hv_2d(np.array(pts), ref, baseline=baseline))
        rows.append((cid, np.median(ms), np.std(ms), np.median(st), np.std(st),
                     np.median(sc), np.median(hvs), np.std(hvs)))

    # HV 降順で並べる
    rows.sort(key=lambda r: -r[6])
    for r in rows:
        cid, ms_m, ms_s, st_m, st_s, sc_m, hv_m, hv_s = r
        lines.append(
            f"  {cid:<22} {ms_m:>8.1f} {ms_s:>8.2f} {st_m:>8.2f} "
            f"{st_s:>8.2f} {sc_m:>11.4f} {hv_m:>12.4f} {hv_s:>10.4f}"
        )

    path = os.path.join(out_dir, 'summary_table.txt')
    with open(path, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines) + "\n")
    return lines


# ========== Region-restricted HV (per-trial) ==========

def _config_per_trial_paretos(data, cid, baseline):
    """1 config の各 trial の Pareto front (list of np.ndarray)。
    各 trial を独立に Pareto 抽出する。"""
    paretos = []
    for t in data['results'][cid]:
        if t is None or 'error' in t:
            continue
        pts = [(h['ls_makespan'], h['ls_stability'])
               for h in history_excluding_init(t['history'])]
        if not pts:
            paretos.append(np.zeros((0, 2)))
            continue
        paretos.append(pareto_front_2d(pts, baseline=baseline))
    return paretos


def _config_union_pareto(data, cid, baseline):
    """1 config の全 trial を集めた union Pareto（領域境界決定にのみ使用）"""
    pts = []
    for t in data['results'][cid]:
        if t is None or 'error' in t:
            continue
        for h in history_excluding_init(t['history']):
            pts.append((h['ls_makespan'], h['ls_stability']))
    if not pts:
        return np.zeros((0, 2))
    return pareto_front_2d(pts, baseline=baseline)


def _restricted_hv(pareto_pts, region_lo, region_hi, init_ms,
                   stab_margin_frac=0.02):
    """領域内の HV. 参照点 = (init_ms, region_hi + margin)"""
    if len(pareto_pts) == 0:
        return 0.0
    mask = (pareto_pts[:, 1] >= region_lo) & (pareto_pts[:, 1] <= region_hi)
    region_pts = pareto_pts[mask]
    if len(region_pts) == 0:
        return 0.0
    region_pts = region_pts[region_pts[:, 0] < init_ms]
    if len(region_pts) == 0:
        return 0.0
    width = max(region_hi - region_lo, 1e-6)
    ref_stab = region_hi + stab_margin_frac * width
    return hv_2d(region_pts, (init_ms, ref_stab))


def plot_region_restricted_hv(data, out_dir, baseline):
    """安定性 quartile (low/mid/high) ごとの HV を per-trial で計算 →
    グループ化棒グラフ (median + IQR) で表示

    1. 領域境界決定: 全 config × 全 trial の cross-method union Pareto の stab 軸を 3 等分
    2. HV 計算は per-trial: 各 trial の Pareto から領域内 HV を求める
    3. trial 間の median + IQR を集約値として表示
    """
    configs = data['configs']
    cids = list(configs.keys())
    init_ms = float(data['init_makespan'])

    # --- 領域境界の決定（cross-method union を使う）---
    config_unions = {cid: _config_union_pareto(data, cid, baseline) for cid in cids}
    all_pts = np.vstack([p for p in config_unions.values() if len(p) > 0]) \
              if any(len(p) > 0 for p in config_unions.values()) \
              else np.zeros((0, 2))
    if len(all_pts) == 0:
        return
    union_pareto = pareto_front_2d(all_pts)
    if len(union_pareto) == 0:
        return

    stab_max = float(union_pareto[:, 1].max())
    if stab_max <= 0:
        return
    eps = 1e-9
    t1, t2 = stab_max / 3.0, 2.0 * stab_max / 3.0
    regions = [
        ('low_stab',  (0.0,    t1)),
        ('mid_stab',  (t1+eps, t2)),
        ('high_stab', (t2+eps, stab_max)),
    ]

    # --- per-trial Region-restricted HV を集計 ---
    # hv_per_trial[cid][region_idx] = list of HV values across trials
    hv_per_trial = {cid: [[] for _ in regions] for cid in cids}
    for cid in cids:
        for pf in _config_per_trial_paretos(data, cid, baseline):
            for j, (_, (lo, hi)) in enumerate(regions):
                hv_per_trial[cid][j].append(_restricted_hv(pf, lo, hi, init_ms))

    # 各 (cid, region) で median と IQR
    median_mat = np.zeros((len(cids), len(regions)))
    q1_mat     = np.zeros((len(cids), len(regions)))
    q3_mat     = np.zeros((len(cids), len(regions)))
    for i, cid in enumerate(cids):
        for j in range(len(regions)):
            arr = np.array(hv_per_trial[cid][j])
            if len(arr) == 0:
                continue
            median_mat[i, j] = np.median(arr)
            q1_mat[i, j]     = np.percentile(arr, 25)
            q3_mat[i, j]     = np.percentile(arr, 75)

    # --- グループ化棒グラフ (median + IQR エラーバー) ---
    fig, ax = plt.subplots(figsize=(max(10, 0.7 * len(cids) + 4), 6))
    x = np.arange(len(cids))
    width = 0.27
    region_colors = {'low_stab': 'tab:blue', 'mid_stab': 'tab:orange',
                     'high_stab': 'tab:green'}
    for j, (rname, (lo, hi)) in enumerate(regions):
        med = median_mat[:, j]
        # 非対称エラーバー: [median - q1, q3 - median]
        err_lo = np.maximum(med - q1_mat[:, j], 0)
        err_hi = np.maximum(q3_mat[:, j] - med, 0)
        ax.bar(x + (j - 1) * width, med, width,
               label=f'{rname} (stab∈[{lo:.2f}, {hi:.2f}])',
               color=region_colors[rname], alpha=0.85,
               yerr=[err_lo, err_hi], capsize=3, ecolor='black')

    ax.set_xticks(x)
    ax.set_xticklabels(cids, rotation=30, ha='right')
    ax.set_ylabel('Region-restricted HV (per-trial median ± IQR)')
    ax.set_title(
        f'Region-restricted HV per stab region '
        f'({data["problem"]}_{data["scenario"]})\n'
        f'stab_max={stab_max:.2f}, 領域境界 t1={t1:.2f}, t2={t2:.2f}')
    ax.grid(True, alpha=0.3, axis='y')
    ax.legend(fontsize=9, loc='best')
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'region_restricted_hv.png'), dpi=150)
    plt.close(fig)

    # --- テキスト出力 (median + IQR) ---
    lines = [f"Region-restricted HV ({data['problem']}_{data['scenario']}) - per-trial",
             f"  stab_max={stab_max:.3f}, 領域境界 t1={t1:.3f}, t2={t2:.3f}",
             f"  (low_stab=[0, {t1:.2f}], mid_stab=({t1:.2f}, {t2:.2f}], high_stab=({t2:.2f}, {stab_max:.2f}])",
             "",
             "  数値は trial 間 median (Q1〜Q3 を [括弧] で併記)"]
    header = (f"  {'config':<22} "
              f"{'low_stab':>22} {'mid_stab':>22} {'high_stab':>22}")
    lines.append(header)
    lines.append("  " + "-" * (len(header) - 2))
    for i, cid in enumerate(cids):
        cells = []
        for j in range(len(regions)):
            m = median_mat[i, j]; q1 = q1_mat[i, j]; q3 = q3_mat[i, j]
            cells.append(f"{m:>10.3f} [{q1:>5.1f}-{q3:>5.1f}]")
        lines.append(f"  {cid:<22} " + " ".join(f"{c:>22}" for c in cells))
    with open(os.path.join(out_dir, 'region_restricted_hv.txt'), 'w',
              encoding='utf-8') as f:
        f.write("\n".join(lines) + "\n")


# ========== 反復数収束安全値 ==========

def write_convergence_safety(data, out_dir):
    """各 trial の最終 best 改善 iteration を集計し、max_iter の十分性を診断

    - 全 config × 全 trial の last_improvement_iter を集約
    - 中央値、95%, 99%, max を出力
    - max_iter の何 % で 95% が完了するかを表示
    - 推奨 max_iter（95% × 1.2 マージン）を表示
    """
    max_iter = data['max_iterations']
    last_iters = []
    for cid, trials in data['results'].items():
        for t in trials:
            if t is None or 'error' in t:
                continue
            h = history_excluding_init(t['history'])
            if not h:
                continue
            prev = h[0]['best_score']
            last = h[0]['iteration']
            for entry in h[1:]:
                if entry['best_score'] < prev - 1e-12:
                    last = entry['iteration']
                    prev = entry['best_score']
            last_iters.append(last)

    if not last_iters:
        return
    arr = np.array(last_iters)
    p50 = int(np.median(arr))
    p95 = int(np.percentile(arr, 95))
    p99 = int(np.percentile(arr, 99))
    p_max = int(arr.max())
    n = len(arr)

    recommended = int(np.ceil(p95 * 1.2 / 100.0) * 100)  # 100 単位で切り上げ
    sufficient = "✓ 十分" if p95 < max_iter * 0.85 else (
        "△ 余裕少ない" if p95 < max_iter else "✗ 不足"
    )

    lines = [
        f"max_iter 安全値診断 ({data['problem']}_{data['scenario']})",
        "",
        f"  対象: {n} trial × {len(data['configs'])} config の "
        f"last_improvement_iter (best 最終更新時点)",
        "",
        f"  current max_iter      = {max_iter}",
        f"  last_improve_iter 中央値 (p50)    = {p50}",
        f"  last_improve_iter 95 パーセンタイル = {p95}",
        f"  last_improve_iter 99 パーセンタイル = {p99}",
        f"  last_improve_iter 最大値 (p_max)   = {p_max}",
        "",
        f"  → max_iter={max_iter} は: {sufficient}",
        f"     (95% trial が iter ≤ {p95} で best 確定。"
        f"max_iter の {100*p95/max_iter:.0f}% 地点)",
        "",
        f"  推奨 max_iter (p95 × 1.2 マージン): {recommended}",
    ]
    with open(os.path.join(out_dir, 'convergence_safety.txt'), 'w',
              encoding='utf-8') as f:
        f.write("\n".join(lines) + "\n")


# ========== 1 問題分析 ==========

def analyze_problem(problem_dir):
    print(f"\n--- 分析: {problem_dir} ---")
    data = load_problem_data(problem_dir)
    analysis_dir = os.path.join(problem_dir, 'analysis')
    os.makedirs(analysis_dir, exist_ok=True)

    ref = make_reference_point(data)
    baseline = data.get('baseline')
    if baseline is not None:
        baseline = (float(baseline[0]), float(baseline[1]))

    max_iter = data['max_iterations']
    sample_iters = sample_iterations(max_iter, n_samples=60)

    # 集約指標
    write_summary_table(data, analysis_dir, ref, baseline)
    plot_hv_heatmap_stage1a(data, analysis_dir, ref, baseline)
    plot_tornado_stage1b(data, analysis_dir, ref, baseline)

    # Stage 2-A 専用: repair の trigger × strength ヒートマップ + lift 比較
    plot_repair_heatmap_stage2a(data, analysis_dir, ref, baseline)
    plot_repair_lift_stage2a(data, analysis_dir, baseline)

    # Region-restricted HV (低/中/高 stab 領域別の HV)
    plot_region_restricted_hv(data, analysis_dir, baseline)

    # 挙動指標
    plot_acceptance_breakdown(data, analysis_dir)
    plot_strength_trace(data, analysis_dir, sample_iters)
    plot_last_improve_cdf(data, analysis_dir)

    # 収束安全値診断 (max_iter の十分性)
    write_convergence_safety(data, analysis_dir)

    # 反復軸 anytime
    plot_anytime_curves(data, analysis_dir, sample_iters, ref, baseline)

    # Pareto
    plot_pareto_overlay(data, analysis_dir, baseline)

    print(f"  → {analysis_dir} に出力完了")


# ========== エントリポイント ==========

def main():
    parser = argparse.ArgumentParser(description='Stage 1 ILS 掃引の分析')
    parser.add_argument('root_dir', type=str,
                        help='run_ils_sweep.py の出力ディレクトリ')
    args = parser.parse_args()

    if not os.path.isdir(args.root_dir):
        print(f"ディレクトリが存在しません: {args.root_dir}")
        sys.exit(1)

    cross_safety = []  # [(problem, p50, p95, p99, p_max, max_iter)]
    for prob_dir in iter_problem_dirs(args.root_dir):
        try:
            analyze_problem(prob_dir)
            # 横断 safety 用に再計算
            data = load_problem_data(prob_dir)
            last_iters = []
            for cid, trials in data['results'].items():
                for t in trials:
                    if t is None or 'error' in t:
                        continue
                    h = history_excluding_init(t['history'])
                    if not h:
                        continue
                    prev = h[0]['best_score']; last = h[0]['iteration']
                    for entry in h[1:]:
                        if entry['best_score'] < prev - 1e-12:
                            last = entry['iteration']; prev = entry['best_score']
                    last_iters.append(last)
            if last_iters:
                arr = np.array(last_iters)
                cross_safety.append((
                    f"{data['problem']}_{data['scenario']}",
                    int(np.median(arr)),
                    int(np.percentile(arr, 95)),
                    int(np.percentile(arr, 99)),
                    int(arr.max()),
                    data['max_iterations'],
                ))
        except Exception as e:
            import traceback
            print(f"ERROR analyzing {prob_dir}: {e}")
            traceback.print_exc()

    # 横断 safety レポート
    if cross_safety:
        lines = ["max_iter 安全値診断 (横断)", "=" * 70, ""]
        header = f"  {'problem':<28} {'p50':>6} {'p95':>6} {'p99':>6} {'p_max':>6} {'max_iter':>10}"
        lines.append(header)
        lines.append("  " + "-" * (len(header) - 2))
        max_p95 = 0
        for prob, p50, p95, p99, pmax, mi in cross_safety:
            lines.append(f"  {prob:<28} {p50:>6} {p95:>6} {p99:>6} {pmax:>6} {mi:>10}")
            max_p95 = max(max_p95, p95)
        recommended = int(np.ceil(max_p95 * 1.2 / 100.0) * 100)
        lines.extend([
            "",
            f"  全問題横断 max(p95) = {max_p95}",
            f"  推奨 max_iter (p95 × 1.2): {recommended}",
        ])
        with open(os.path.join(args.root_dir, 'convergence_safety_cross.txt'),
                  'w', encoding='utf-8') as f:
            f.write("\n".join(lines) + "\n")
        print("\n" + "\n".join(lines))

    print("\n全問題の分析完了")


if __name__ == "__main__":
    main()
