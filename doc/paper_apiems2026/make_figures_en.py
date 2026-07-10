#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""APIEMS 2026 投稿用の英語ラベル図を一括生成する。

数値ソースは main_v1 の解析キャッシュ（_summary_data.pkl）と raw JSON で、
summary.md / 母艦の数値と完全一致する（同じ analyze_v3 ヘルパーを使用）。
出力は本スクリプトと同じディレクトリの figures/ 下。

usage: python make_figures_en.py
"""
import os
import sys
import glob
import json
import pickle

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
CORE = os.path.normpath(os.path.join(HERE, '..', '..', 'experiments', 'core_comparison_v3'))
sys.path.insert(0, CORE)
import analyze_v3 as A  # noqa: E402

RESULTS = os.path.join(CORE, 'results', 'main_v1')
OUT = os.path.join(HERE, 'figures')
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    'font.family': 'Times New Roman',
    'mathtext.fontset': 'stix',
    'axes.unicode_minus': False,
    'font.size': 7.5,
    'axes.titlesize': 8,
    'axes.labelsize': 7.5,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'legend.fontsize': 6.5,
})

FULLW = 6.93  # 本文幅 (in): A4 - 左右余白, 2段ぶち抜き

LBL = dict(A.METHOD_LABELS)


def load_pkl():
    with open(os.path.join(RESULTS, 'analysis', '_summary_data.pkl'), 'rb') as f:
        return pickle.load(f)


def ordered_probs(S):
    return A.order_prob_labels(S.keys())


def rho_pct(prob_label):
    return int(round(A.reschedule_rate(prob_label) * 100))


def med(v):
    v = [x for x in v if np.isfinite(x)]
    return float(np.median(v)) if v else 0.0


def star(p):
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return 'ns'
    if p < 0.001:
        return '***'
    if p < 0.01:
        return '**'
    if p < 0.05:
        return '*'
    return 'ns'


def _grouped_bars(ax, tags, bars, data_by_tag, ylabel, title, legend=True):
    ng = len(bars)
    x = np.arange(len(tags))
    w = 0.8 / ng
    for j, (mkey, label, color) in enumerate(bars):
        meds = [med(data_by_tag[t].get(mkey, [])) for t in tags]
        ax.bar(x + (j - (ng - 1) / 2) * w, meds, w, label=label, color=color)
    ax.set_xticks(x)
    ax.set_xticklabels(tags, rotation=30, ha='right')
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(axis='y', alpha=0.3)
    if legend:
        ax.legend(frameon=False, loc='upper right')


def _annotate_pair(ax, tags, data, a, b):
    x = np.arange(len(tags))
    for i, t in enumerate(tags):
        xa, xb = data[t].get(a, []), data[t].get(b, [])
        _, p = A.wilcoxon_paired(xa, xb, alternative='two-sided')
        ax.annotate(star(p), (x[i], max(med(xa), med(xb))), ha='center',
                    va='bottom', fontsize=6, color='dimgray')


def _annotate_vs_base(ax, tags, data, base, variants):
    x = np.arange(len(tags))
    w = 0.8 / 3
    for j, mk in enumerate(variants, start=1):
        for i, t in enumerate(tags):
            xa, xb = data[t].get(mk, []), data[t].get(base, [])
            _, p = A.wilcoxon_paired(xa, xb, alternative='two-sided')
            ax.annotate(star(p), (x[i] + (j - 1) * w, med(xa)), ha='center',
                        va='bottom', fontsize=5.5, color='dimgray')


# ---------- Fig: claim1 (H1) — union / high-stab / AOC, ILS-b vs Mem-LS ----------

def fig_claim1(S):
    probs = ordered_probs(S)
    tags = [A.problem_short_tag(p) for p in probs]
    union = {A.problem_short_tag(p): S[p]['union_hv_pt'] for p in probs}
    high = {A.problem_short_tag(p): S[p]['highstab_hv_pt'] for p in probs}
    aoc = {A.problem_short_tag(p): S[p].get('aoc_pt', {}) for p in probs}
    pair = [('ils_baseline', 'ILS-baseline', 'tab:orange'),
            ('memetic_ls', 'Memetic-LS', 'tab:green')]
    fig, axes = plt.subplots(1, 3, figsize=(FULLW, 1.42))
    _grouped_bars(axes[0], tags, pair, union, 'Normalized HV',
                  '(a) Union HV: comparable', legend=False)
    _annotate_pair(axes[0], tags, union, 'ils_baseline', 'memetic_ls')
    _grouped_bars(axes[1], tags, pair, high, 'Normalized HV',
                  '(b) High-stability HV: ILS dominates')
    _annotate_pair(axes[1], tags, high, 'ils_baseline', 'memetic_ls')
    _grouped_bars(axes[2], tags, pair, aoc, 'Normalized AOC',
                  '(c) AOC: ILS ahead (6/8)', legend=False)
    _annotate_pair(axes[2], tags, aoc, 'ils_baseline', 'memetic_ls')
    fig.tight_layout(pad=0.4)
    out = os.path.join(OUT, 'fig_claim1_en.png')
    fig.savefig(out, dpi=350)
    plt.close(fig)
    print(' ->', out)


# ---------- Fig: claim2 (H2) — high-stab HV, mechanism added per host ----------

def fig_claim2(S):
    probs = ordered_probs(S)
    tags = [A.problem_short_tag(p) for p in probs]
    high = {A.problem_short_tag(p): S[p]['highstab_hv_pt'] for p in probs}
    fig, axes = plt.subplots(1, 2, figsize=(FULLW, 1.47), sharey=True)
    _grouped_bars(axes[0], tags,
                  [('memetic_ls', 'Memetic-LS', 'tab:green'),
                   ('memetic_pr', 'Memetic+PR', 'tab:brown'),
                   ('memetic_repair', 'Memetic+repair', 'tab:purple')],
                  high, 'Normalized HV',
                  '(a) Population host: operators fill the gap')
    _annotate_vs_base(axes[0], tags, high, 'memetic_ls',
                      ['memetic_pr', 'memetic_repair'])
    _grouped_bars(axes[1], tags,
                  [('ils_baseline', 'ILS-baseline', 'tab:orange'),
                   ('ils_pr', 'ILS+PR', 'tab:blue'),
                   ('ils_repair', 'ILS+repair', 'tab:red')],
                  high, 'Normalized HV',
                  '(b) Trajectory host: already saturated')
    _annotate_vs_base(axes[1], tags, high, 'ils_baseline',
                      ['ils_pr', 'ils_repair'])
    fig.tight_layout(pad=0.4)
    out = os.path.join(OUT, 'fig_claim2_en.png')
    fig.savefig(out, dpi=350)
    plt.close(fig)
    print(' ->', out)


# ---------- Fig: density diff (H1 structural cause), 4 representative scenarios ----------

DENSITY_PROBS = ['la36_la36_small', 'la21_la21_delay147',
                 'la36_la36_middle', 'ta21_ta21_high']


def _collect_points_raw(prob, method):
    pts_all, bl = [], None
    for p in sorted(glob.glob(os.path.join(RESULTS, prob, 'raw', f'{method}__*.json'))):
        d = json.load(open(p, encoding='utf-8'))
        if bl is None:
            bl = []
            if d.get('baseline') is not None:
                bl.append(d['baseline'])
            if d.get('baseline_rsr') is not None and list(d['baseline_rsr']) not in bl:
                bl.append(list(d['baseline_rsr']))
            bl = bl or None
        pts = A.get_uea_points(d, 0)
        if len(pts):
            if bl:
                pts = A.filter_baselines(pts, bl)
            if len(pts):
                pts_all.append(np.asarray(pts, float))
    return np.concatenate(pts_all) if pts_all else np.zeros((0, 2))


def fig_density(nbins=44):
    fig, axes = plt.subplots(1, len(DENSITY_PROBS), figsize=(FULLW, 1.54))
    im = None
    for ax, prob in zip(axes, DENSITY_PROBS):
        Pi = _collect_points_raw(prob, 'ils_baseline')
        Pm = _collect_points_raw(prob, 'memetic_ls')
        if len(Pi) == 0 or len(Pm) == 0:
            ax.axis('off')
            continue
        allp = np.vstack([Pi, Pm])
        ms_e = np.linspace(allp[:, 0].min(), allp[:, 0].max(), nbins + 1)
        d_e = np.linspace(0, allp[:, 1].max(), nbins + 1)
        Hi, _, _ = np.histogram2d(Pi[:, 0], Pi[:, 1], bins=[ms_e, d_e])
        Hm, _, _ = np.histogram2d(Pm[:, 0], Pm[:, 1], bins=[ms_e, d_e])
        Hi = Hi / Hi.sum()
        Hm = Hm / Hm.sum()
        diff = Hi - Hm
        D = (np.sign(diff) * np.sqrt(np.abs(diff))).T
        vmax = np.nanmax(np.abs(D)) or 1.0
        im = ax.imshow(D, origin='lower', aspect='auto', cmap='RdBu_r',
                       vmin=-vmax, vmax=vmax,
                       extent=[ms_e[0], ms_e[-1], d_e[0], d_e[-1]])
        for P, color in [(Pi, 'darkorange'), (Pm, 'green')]:
            pf = A.pareto_front(np.unique(P, axis=0))
            pf = pf[np.argsort(pf[:, 0])]
            ax.step(pf[:, 0], pf[:, 1], where='post', color=color, lw=1.5, zorder=5)
        ax.set_title(f'{A.problem_short_tag(prob)} '
                     f'($\\rho$={rho_pct(prob)}%)')
        ax.set_xlabel('Makespan $MS$')
        ax.tick_params(labelsize=6)
        ax.xaxis.set_major_locator(plt.MaxNLocator(4))
    axes[0].set_ylabel('Sequence deviation $D$')
    fig.subplots_adjust(left=0.06, right=0.90, bottom=0.26, top=0.86, wspace=0.28)
    if im is not None:
        cax = fig.add_axes([0.915, 0.26, 0.013, 0.60])
        cb = fig.colorbar(im, cax=cax)
        cb.set_label('Density diff.\n(red: ILS, blue: Memetic)', fontsize=6)
        cb.ax.tick_params(labelsize=5.5)
    out = os.path.join(OUT, 'fig_density_en.png')
    fig.savefig(out, dpi=350)
    plt.close(fig)
    print(' ->', out)


# ---------- Fig: PR path statistics (H2 mechanism cause) ----------

def fig_mech_pr():
    agg = {}
    for prob_dir in sorted(glob.glob(os.path.join(RESULTS, '*', 'raw'))):
        prob = os.path.basename(os.path.dirname(prob_dir))
        acc = {}
        for mkey in ('memetic_pr', 'ils_pr'):
            a = {'pr_d0': [], 'pr_improved': []}
            for p in glob.glob(os.path.join(prob_dir, f'{mkey}__*.json')):
                ms = (json.load(open(p, encoding='utf-8')).get('mech_stats') or {})
                a['pr_d0'] += [int(x) for x in ms.get('pr_d0', [])]
                a['pr_improved'] += [int(x) for x in ms.get('pr_improved', [])]
            acc[mkey] = a
        if any(acc[m]['pr_d0'] for m in acc):
            agg[prob] = acc
    probs = A.order_prob_labels(agg.keys())
    tags = [A.problem_short_tag(p) for p in probs]
    methods = [('memetic_pr', 'Memetic+PR', 'tab:brown'),
               ('ils_pr', 'ILS+PR', 'tab:blue')]
    x = np.arange(len(probs))
    w = 0.38
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(FULLW, 1.47))
    for i, (mkey, mlabel, color) in enumerate(methods):
        d0_mean, imp_rate = [], []
        for prob in probs:
            a = agg[prob][mkey]
            d0_mean.append(float(np.mean(a['pr_d0'])) if a['pr_d0'] else 0.0)
            imp_rate.append(100.0 * np.sum(a['pr_improved']) / len(a['pr_improved'])
                            if a['pr_improved'] else 0.0)
        offs = (i - 0.5) * w
        b1 = axL.bar(x + offs, d0_mean, w, label=mlabel, color=color)
        b2 = axR.bar(x + offs, imp_rate, w, label=mlabel, color=color)
        for rect, v in zip(b1, d0_mean):
            axL.text(rect.get_x() + rect.get_width() / 2, v, f'{v:.0f}',
                     ha='center', va='bottom', fontsize=5.5)
        for rect, v in zip(b2, imp_rate):
            axR.text(rect.get_x() + rect.get_width() / 2, v, f'{v:.0f}',
                     ha='center', va='bottom', fontsize=5.5)
    axL.set_title('(a) Mean PR path length $d_0$')
    axR.set_title('(b) Improvement discovery rate (%)')
    axL.set_ylabel('Mean $d_0$ (disagreements to $S_p$)')
    axR.set_ylabel('Improved calls (%)')
    for ax in (axL, axR):
        ax.set_xticks(x)
        ax.set_xticklabels(tags, rotation=30, ha='right')
        ax.legend(frameon=False)
        ax.grid(axis='y', alpha=0.3)
    fig.tight_layout(pad=0.4)
    out = os.path.join(OUT, 'fig_mech_pr_en.png')
    fig.savefig(out, dpi=350)
    plt.close(fig)
    print(' ->', out)


# ---------- Fig: scoreboard — 3 stacked heatmap panels ----------

SB_TITLES = {'union': '(a) Union HV (overall quality)',
             'highstab': '(b) High-stability HV (filling near $S_p$)',
             'aoc': '(c) AOC (anytime performance)'}


def _per_problem_rpd(M):
    R = np.full_like(M, np.nan)
    for i in range(M.shape[0]):
        row = M[i]
        fin = row[np.isfinite(row)]
        if len(fin) == 0:
            continue
        best = float(np.max(fin))
        if best <= 0:
            continue
        R[i] = (1.0 - row / best) * 100.0
    return R


def fig_scoreboard(S):
    order_m = ['ga', 'memetic_ls', 'memetic_repair', 'memetic_pr',
               'ils_baseline', 'ils_repair', 'ils_pr']
    prob_labels = ordered_probs(S)
    present = set()
    for pl in prob_labels:
        present.update(S[pl].get('methods', []))
    methods = [m for m in order_m if m in present]
    tags = [A.problem_short_tag(pl) for pl in prob_labels]

    fig, axes = plt.subplots(3, 1, figsize=(FULLW, 3.45))
    for ax, (key, _jp, arr_key) in zip(axes, A.SCOREBOARD_METRICS):
        M = A._metric_matrix(S, prob_labels, methods, arr_key)
        valid = ~np.any(np.isnan(M), axis=1)
        Mv = M[valid]
        used_tags = [t for t, ok in zip(tags, valid) if ok]
        avg_rank, chi, p, W, ranks = A._friedman_avg_rank(Mv)
        arpd_mean, arpd_med = A._arpd_pct(Mv)
        RPD = _per_problem_rpd(Mv)
        order = list(np.argsort(avg_rank, kind='stable'))
        s_labels = [LBL.get(methods[j], methods[j]) for j in order]
        s_avg = [avg_rank[j] for j in order]
        s_am = [arpd_mean[j] for j in order]
        s_amed = [arpd_med[j] for j in order]
        s_RPD = RPD[:, order]
        nM, nP = len(order), len(used_tags)

        D = np.full((nM, nP + 2), np.nan)
        D[:, :nP] = s_RPD.T
        D[:, nP + 1] = np.array(s_am)
        vmax = np.nanmax(D) if np.any(np.isfinite(D)) else 1.0
        cmap = plt.get_cmap('RdYlGn_r').copy()
        cmap.set_bad('white')
        im = ax.imshow(D, aspect='auto', cmap=cmap, vmin=0, vmax=max(vmax, 1.0))
        ax.axvline(nP - 0.5, color='black', lw=1.2)
        ax.set_xticks(range(nP + 2))
        ax.set_xticklabels(used_tags + ['Avg. rank', 'ARPD%'], fontsize=7)
        ax.set_yticks(range(nM))
        ax.set_yticklabels(s_labels, fontsize=7)
        p_txt = f'$p$={p:.4f}' if p >= 0.0001 else '$p$<0.0001'
        ax.set_title(f"{SB_TITLES[key]}   Friedman {p_txt}, Kendall's $W$={W:.2f}",
                     fontsize=8)

        def _txt(v):
            return 'white' if (np.isfinite(v) and v > 60) else 'black'

        for i in range(nM):
            for j in range(nP):
                v = D[i, j]
                t = f'{v:.0f}' if np.isfinite(v) else '·'
                ax.text(j, i, t, ha='center', va='center', fontsize=6.5,
                        color=_txt(v))
            rk = int(np.sum(np.array(s_avg) < s_avg[i] - 1e-12) + 1)
            ax.text(nP, i, f'{s_avg[i]:.2f} ({rk})', ha='center', va='center',
                    fontsize=6.5, color='black')
            ax.text(nP + 1, i, f'{s_am[i]:.0f}/{s_amed[i]:.0f}', ha='center',
                    va='center', fontsize=6.5, color=_txt(s_am[i]))
        cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.015)
        cb.set_label('RPD%', fontsize=6)
        cb.ax.tick_params(labelsize=5.5)
    fig.tight_layout(pad=0.5)
    out = os.path.join(OUT, 'fig_scoreboard_en.png')
    fig.savefig(out, dpi=350)
    plt.close(fig)
    print(' ->', out)


if __name__ == '__main__':
    S = load_pkl()
    fig_claim1(S)
    fig_claim2(S)
    fig_scoreboard(S)
    fig_mech_pr()
    fig_density()
    print('done.')
