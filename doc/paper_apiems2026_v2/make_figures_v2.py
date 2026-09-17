#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""APIEMS 2026 原稿 v2 の図 1〜5 をすべて生成する。

データソースは main_v1 の _summary_data.pkl と raw JSON、ヘルパーは analyze_v3（v1 の
make_figures_en.py と同じ）。本ディレクトリだけで v2 の図が揃うよう、図 4・図 5 も含める:

  fig_v2_concept_en.png      図1 (D, MS) 平面の模式図（H1/H2 を結果と同じ座標系で描く）
  fig_v2_front_en.png        図2 実 Pareto フロント（ta21S / ta21L, ILS-b・Mem-LS・Mem+PR）
  fig_v2_interaction_en.png  図3 演算子 none/repair/PR × 探索構造の高安定 HV
  fig_v2_mech_pr_en.png      図4 PR 経路統計（始点ごとの経路長 d0 と経路上に改善解が残る割合）
  fig_v2_scoreboard_en.png   図5 総合スコアボード（7 手法 × 8 シナリオ × 3 指標）

訪問密度差マップ（fig_density → fig_v2_density_en.png）は本文で使わない検討用で、既定の実行では生成しない。
出力先は本スクリプトと同じディレクトリの figures/。

usage: python make_figures_v2.py            # 図 1〜5
       python make_figures_v2.py --select   # 図 2 のシナリオ選定用シート（論文には載せない）
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
from matplotlib.patches import FancyArrowPatch
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import mannwhitneyu

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

FULLW = 6.93  # 本文幅 (in): A4 - 左右余白, 2 段ぶち抜き

C_ILS, C_MEM, C_MPR = 'tab:orange', 'tab:green', 'tab:brown'
BAND_COLOR = 'gold'


def load_pkl():
    with open(os.path.join(RESULTS, 'analysis', '_summary_data.pkl'), 'rb') as f:
        return pickle.load(f)


def rho_pct(prob_label):
    return int(round(A.reschedule_rate(prob_label) * 100))


# ---------- 共通: raw から trial 別の非劣解集合を作る ----------

def _baselines_of(d):
    bl = []
    if d.get('baseline') is not None:
        bl.append(list(d['baseline']))
    if d.get('baseline_rsr') is not None and list(d['baseline_rsr']) not in bl:
        bl.append(list(d['baseline_rsr']))
    return bl or None


def trial_pareto(prob, method, trial):
    """(method, trial) の 10 重み分の UEA 点を統合し、baseline を除いた非劣解 (MS, D) を返す。"""
    pts_all, bl = [], None
    for p in sorted(glob.glob(os.path.join(RESULTS, prob, 'raw', f'{method}__*__t{trial:03d}.json'))):
        d = json.load(open(p, encoding='utf-8'))
        if bl is None:
            bl = _baselines_of(d)
        pts = A.get_uea_points(d, trial)
        if len(pts):
            if bl:
                pts = A.filter_baselines(pts, bl)
            if len(pts):
                pts_all.append(np.asarray(pts, float))
    if not pts_all:
        return np.zeros((0, 2))
    P = np.unique(np.concatenate(pts_all), axis=0)
    pf = A.pareto_front(P)
    return pf[np.argsort(pf[:, 1])]  # D 昇順


def all_points_raw(prob, method):
    """v1 fig_density と同じ: 全 trial・全重みの UEA 訪問点（baseline 除去）を連結。"""
    pts_all, bl = [], None
    for p in sorted(glob.glob(os.path.join(RESULTS, prob, 'raw', f'{method}__*.json'))):
        d = json.load(open(p, encoding='utf-8'))
        if bl is None:
            bl = _baselines_of(d)
        pts = A.get_uea_points(d, 0)
        if len(pts):
            if bl:
                pts = A.filter_baselines(pts, bl)
            if len(pts):
                pts_all.append(np.asarray(pts, float))
    return np.concatenate(pts_all) if pts_all else np.zeros((0, 2))


def median_trial(S, prob, method):
    hv = np.asarray(S[prob]['union_hv_pt'][method], float)
    return int(np.argsort(hv, kind='stable')[len(hv) // 2])


# ---------- 図1: (D, MS) 平面の模式図 ----------

def _front_curve(x):
    return 0.15 + 0.80 * (1.0 - x) ** 1.8


def fig_concept():
    rng = np.random.default_rng(7)
    fig, axes = plt.subplots(1, 2, figsize=(FULLW, 2.05), sharey=True)
    band_hi = 0.30
    xs = np.linspace(0, 1, 200)

    def base(ax, title):
        ax.axvspan(0, band_hi, color=BAND_COLOR, alpha=0.18, lw=0)
        ax.text(band_hi / 2, 0.075, 'high-stability\nregion (near $S_p$)', ha='center', va='bottom',
                fontsize=6.5, color='darkgoldenrod')
        ax.plot(xs, _front_curve(xs), ls='--', color='gray', lw=0.9, zorder=1)
        # 回転テキストは rotation_mode='anchor' で基準点を文頭に固定し、破線のすぐ下に沿わせる
        ax.text(0.52, _front_curve(0.52) - 0.02, 'attainable trade-off front', fontsize=6.2,
                color='gray', rotation=-22, rotation_mode='anchor', ha='left', va='top')
        ax.plot([0], [0.95], marker='*', ms=9, color='black', zorder=6)
        ax.text(0.03, 0.975, '$S_{RSR}$: order of $S_p$ kept ($D=0$)', fontsize=6.5, va='bottom')
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(0.05, 1.08)
        ax.set_xticks([0])
        ax.set_xticklabels(['$D=0$'])
        ax.set_yticks([])
        ax.set_xlabel('Sequence deviation $D$ from $S_p$ (change of plan)')
        ax.set_title(title, loc='left')
        for s in ('top', 'right'):
            ax.spines[s].set_visible(False)

    # Memetic の解集合（両パネル共通）。図 2 の実測 ta21 と同じ形にする:
    # 小さい D には届かず（左端が ILS よりかなり手前で終わる）、届く範囲でも MS で明確に劣り、
    # 到達可能フロント（破線）に追いつくのは大きな D に限る。フロントの線は引かず点だけ置く
    mem_fx = np.array([0.27, 0.36, 0.47, 0.60, 0.75, 0.92])
    mem_fy = np.array([0.810, 0.680, 0.535, 0.380, 0.240, 0.165])

    def mem_front(x):
        return np.interp(x, mem_fx, mem_fy)

    # 非劣解に劣る個体（交叉で散った子個体）: 必ず非劣点の上側に置く
    mx = rng.uniform(0.29, 1.0, 15)
    my = mem_front(mx) + rng.uniform(0.045, 0.185, 15)

    def draw_mem(ax_, a_front, a_dom):
        ax_.scatter(mx, my, s=10, color=C_MEM, alpha=a_dom, zorder=3)
        ax_.scatter(mem_fx, mem_fy, s=12, color=C_MEM, alpha=a_front, zorder=4)

    # (a) 演算子なし: H1
    ax = axes[0]
    base(ax, '(a) Without the operator')
    # ILS の軌道: S_p から右下へ 7 点。点の間隔を広くとり、各区間に矢印を置いて進行方向を見せる。
    # フロント線上に張り付かせず、はっきりジグザグさせる。最左点は星（S_RSR）から少し離す
    # ＝この点が「PR も最終的に辿り着く水準」を表す
    ix = np.linspace(0.09, 0.47, 7)
    wob = np.array([0.030, 0.080, 0.035, 0.090, 0.040, 0.075, 0.030])
    iy = _front_curve(ix) + wob

    def arrow_chain(ax_, xs_, ys_, color, alpha=1.0):
        for i in range(len(xs_) - 1):
            ax_.annotate('', xy=(xs_[i + 1], ys_[i + 1]), xytext=(xs_[i], ys_[i]),
                         arrowprops=dict(arrowstyle='-|>', color=color, lw=0.55,
                                         mutation_scale=5, alpha=alpha, shrinkA=3.5, shrinkB=4.5),
                         zorder=5)

    def ils_chain(ax_, alpha):
        ax_.scatter(ix, iy, s=16, color=C_ILS, alpha=alpha, zorder=4)
        arrow_chain(ax_, np.r_[0.0, ix], np.r_[0.95, iy], C_ILS, alpha)  # 星（S_RSR）から出発

    draw_mem(ax, 0.95, 0.55)
    ils_chain(ax, 1.0)
    ax.text(0.03, 0.50, 'ILS (trajectory search):\nsmall moves from $S_p$\nfill the region step by step',
            fontsize=6.5, color=C_ILS, ha='left', va='top')
    ax.text(0.42, 0.93, 'Memetic (population search): offspring scattered by\ncrossover; stops short of small $D$ and is clearly worse\nin $MS$ where it does reach; low $MS$ only at large $D$',
            fontsize=6.5, color='darkgreen', ha='left', va='top')

    # (b) 集団に演算子を載せる: H2
    ax = axes[1]
    base(ax, '(b) With the operator on the population search')
    draw_mem(ax, 0.32, 0.20)
    ils_chain(ax, 0.32)
    # 引き戻し経路（ILS と逆向き）: 散った個体から S_p へ向かい、経路上の中間解が領域内フロントに落ちる。
    # 経路は ILS の最左点と同水準に到達する。PR と repair の違い（辿り切るか途中で止まるか）は §3.3 に譲り描かない
    # オフセットを単調減少にせず軽く上下させ、経路にもジグザグを持たせる（最終点で ILS の水準に到達）
    # 経路上の解は Memetic 自身のフロントより下（良い側）に来るようオフセットを決める
    pr_x = np.array([0.62, 0.50, 0.39, 0.29, 0.20, 0.115])
    pr_y = _front_curve(pr_x) + np.array([0.210, 0.115, 0.150, 0.130, 0.165, 0.032])
    ax.scatter(pr_x[1:], pr_y[1:], s=13, color=C_MPR, zorder=5)
    ax.scatter([pr_x[0]], [pr_y[0]], s=16, color=C_MEM, edgecolor='black', lw=0.5, zorder=5)
    arrow_chain(ax, pr_x, pr_y, C_MPR)
    ax.text(0.40, 0.93, 'Operator: swap the current solution back toward $S_p$,\nopposite to the ILS direction;\nsolutions on the path (brown) fill the region',
            fontsize=6.5, color=C_MPR, ha='left', va='top')
    ax.text(0.03, 0.46, 'ILS: region already filled,\nlittle left to add',
            fontsize=6.5, color=C_ILS, ha='left', va='top')

    axes[0].set_ylabel('Makespan $MS$')
    fig.tight_layout(pad=0.4, w_pad=1.0)
    out = os.path.join(OUT, 'fig_v2_concept_en.png')
    fig.savefig(out, dpi=350)
    plt.close(fig)
    print(' ->', out)


# ---------- 図2: 実 Pareto フロント（ta21 対） ----------

FRONT_PROBS = ['ta21_ta21_delay97', 'ta21_ta21_high']  # ta21 対（同一 S_p, ρ 32%→82%）
SHOW_ALL_TRIALS = False  # True にすると中央値 trial 以外の非劣解も薄い点で重ねる（本稿では非表示）
FRONT_METHODS = [('ils_baseline', 'ILS-baseline', C_ILS, 'o', 2.3, '-'),
                 ('memetic_ls', 'Memetic-LS', C_MEM, 's', 1.4, '-'),
                 ('memetic_pr', 'Memetic+PR', C_MPR, '^', 1.1, '--')]


def _draw_front_panel(ax, S, prob, panel_tag=None, band_label='wide', legend=True,
                      star_label=True):
    """1 シナリオ分の実フロントパネルを描く（fig_front と選定用シートで共用）。

    x 軸は中央値 trial のフロント（3 手法）の最大 D で切る。全 trial の薄い点はその範囲内のみ表示
    （範囲外に伸びる一部 trial の非劣解を追って軸を伸ばすと、右側が情報のない横線だけになるため）。
    """
    thr = S[prob]['thresholds']
    p50 = thr['P50']
    init_ms = float(S[prob]['init_ms'])
    xmax_med, ymin = 0.0, np.inf
    drawn = []
    for mkey, label, color, mk, lw, ls in FRONT_METHODS:
        pfs = [trial_pareto(prob, mkey, t) for t in range(10)]
        pf = pfs[median_trial(S, prob, mkey)]
        xmax_med = max(xmax_med, pf[:, 1].max())
        drawn.append((pfs, pf, label, color, mk, lw, ls))
    xlim_hi = xmax_med * 1.08
    for pfs, pf, label, color, mk, lw, ls in drawn:
        ymin = min(ymin, pf[:, 0].min())
        if SHOW_ALL_TRIALS:
            allpf = np.concatenate([p for p in pfs if len(p)])
            allpf = allpf[allpf[:, 1] <= xlim_hi]
            ymin = min(ymin, allpf[:, 0].min())
            ax.scatter(allpf[:, 1], allpf[:, 0], s=7, marker=mk, color=color, alpha=0.22,
                       lw=0, zorder=2)
        ax.step(np.r_[pf[:, 1], xlim_hi], np.r_[pf[:, 0], pf[-1, 0]], where='post',
                color=color, lw=lw, ls=ls, zorder=3)
        ax.scatter(pf[:, 1], pf[:, 0], s=16, marker=mk, color=color, edgecolor='black',
                   lw=0.4, zorder=4, label=label)
    ax.axvspan(-1e9, p50, color=BAND_COLOR, alpha=0.18, lw=0, zorder=0)
    ax.axvline(p50, color='darkgoldenrod', lw=0.7, ls=':', zorder=1)
    ax.plot([0], [init_ms], marker='*', ms=9, color='black', zorder=6)
    span = init_ms - ymin
    if star_label:
        ax.text(0.03 * xlim_hi, init_ms, '$S_{RSR}$ ($D=0$)', fontsize=6.5, va='center',
                ha='left')
    ax.set_xlim(-0.03 * xlim_hi, xlim_hi)
    ax.set_ylim(ymin - 0.06 * span, init_ms + 0.10 * span)
    if band_label == 'wide':
        ax.text(p50 / 2, ymin - 0.045 * span, 'high-stability region ($D<P_{50}$)',
                ha='center', va='bottom', fontsize=6.2, color='darkgoldenrod')
    elif band_label == 'narrow':
        ax.text(p50 * 0.2, ymin - 0.04 * span, 'high-stab. ($D<P_{50}$)',
                ha='center', va='bottom', fontsize=5.8, color='darkgoldenrod', rotation=90)
    tag = f'({panel_tag}) ' if panel_tag else ''
    ax.set_title(f'{tag}{A.problem_short_tag(prob)} ($\\rho$={rho_pct(prob)}%, '
                 f'$P_{{50}}$={p50:.0f})', loc='left')
    ax.set_xlabel('Sequence deviation $D$')
    ax.grid(alpha=0.25)
    if legend:
        ax.legend(loc='upper right', frameon=False, handletextpad=0.3)
    return p50 / xlim_hi  # 帯の幅の割合（ラベル配置の判断用）


def fig_front(S):
    fig, axes = plt.subplots(1, len(FRONT_PROBS), figsize=(FULLW, 2.1))
    for k, (ax, prob) in enumerate(zip(axes, FRONT_PROBS)):
        _draw_front_panel(ax, S, prob, panel_tag='ab'[k], band_label='wide' if k == 0 else 'narrow')
    axes[0].set_ylabel('Makespan $MS$')
    fig.tight_layout(pad=0.4, w_pad=1.0)
    out = os.path.join(OUT, 'fig_v2_front_en.png')
    fig.savefig(out, dpi=350)
    plt.close(fig)
    print(' ->', out)


def fig_front_all8(S):
    """選定用シート（論文には載せない）: 全 8 シナリオの実フロントを同じ様式で並べる。"""
    probs = A.order_prob_labels(S.keys())
    fig, axes = plt.subplots(2, 4, figsize=(FULLW * 1.6, 5.2))
    for i, (ax, prob) in enumerate(zip(axes.flat, probs)):
        frac = _draw_front_panel(ax, S, prob, band_label='none', legend=(i == 0),
                                 star_label=False)
        ax.text(0.98, 0.02, f'band = {frac*100:.0f}% of x-range', transform=ax.transAxes,
                ha='right', va='bottom', fontsize=6, color='dimgray')
        if i % 4 == 0:
            ax.set_ylabel('Makespan $MS$')
    fig.tight_layout(pad=0.5)
    out = os.path.join(OUT, 'sel_front_all8.png')
    fig.savefig(out, dpi=220)
    plt.close(fig)
    print(' ->', out)


# ---------- 検討用（本文では不使用）: 訪問密度差マップ（D 横 / MS 縦） ----------

DENSITY_PROBS = ['la36_la36_small', 'la21_la21_delay147',
                 'la36_la36_middle', 'ta21_ta21_high']


# 差分の色は他の図の手法色に合わせる（正=ILS: 橙 / 負=Memetic: 緑）
DENSITY_CMAP = LinearSegmentedColormap.from_list(
    'mem_ils', ['#1b6e2a', '#8fcf8f', '#f7f7f7', '#fdb46b', '#d95f02'])


def fig_density(S, nbins=44):
    fig, axes = plt.subplots(1, len(DENSITY_PROBS), figsize=(FULLW, 1.62))
    im = None
    for ax, prob in zip(axes, DENSITY_PROBS):
        Pi = all_points_raw(prob, 'ils_baseline')
        Pm = all_points_raw(prob, 'memetic_ls')
        if len(Pi) == 0 or len(Pm) == 0:
            ax.axis('off')
            continue
        allp = np.vstack([Pi, Pm])
        ms_e = np.linspace(allp[:, 0].min(), allp[:, 0].max(), nbins + 1)
        d_e = np.linspace(0, allp[:, 1].max(), nbins + 1)
        Hi, _, _ = np.histogram2d(Pi[:, 1], Pi[:, 0], bins=[d_e, ms_e])
        Hm, _, _ = np.histogram2d(Pm[:, 1], Pm[:, 0], bins=[d_e, ms_e])
        Hi = Hi / Hi.sum()
        Hm = Hm / Hm.sum()
        diff = Hi - Hm
        Dm = (np.sign(diff) * np.sqrt(np.abs(diff))).T  # 行=MS, 列=D
        vmax = np.nanmax(np.abs(Dm)) or 1.0
        im = ax.imshow(Dm, origin='lower', aspect='auto', cmap=DENSITY_CMAP,
                       vmin=-vmax, vmax=vmax,
                       extent=[d_e[0], d_e[-1], ms_e[0], ms_e[-1]])
        # フロントは図 2 と同じ様式（階段線＋縁取りの細い点）で、密度を隠さないよう細めに描く
        for P, color, mk in [(Pi, C_ILS, 'o'), (Pm, C_MEM, 's')]:
            pf = A.pareto_front(np.unique(P, axis=0))
            pf = pf[np.argsort(pf[:, 1])]
            ax.step(np.r_[pf[:, 1], d_e[-1]], np.r_[pf[:, 0], pf[-1, 0]], where='post',
                    color=color, lw=0.8, zorder=5)
            ax.scatter(pf[:, 1], pf[:, 0], s=7, marker=mk, color=color, edgecolor='black',
                       lw=0.3, zorder=6)
        # 高安定領域の境界（図 1・図 2 の網掛けと同じ D<P50）
        p50 = S[prob]['thresholds']['P50']
        ax.axvline(p50, color='black', lw=0.8, ls=':', zorder=6)
        ax.text(p50, ms_e[-1], ' $P_{50}$', fontsize=6, ha='left', va='top', zorder=7)
        ax.set_title(f'{A.problem_short_tag(prob)} ($\\rho$={rho_pct(prob)}%)')
        ax.set_xlabel('Sequence deviation $D$')
        ax.tick_params(labelsize=6)
        ax.yaxis.set_major_locator(plt.MaxNLocator(4))
        ax.xaxis.set_major_locator(plt.MaxNLocator(4))
    axes[0].set_ylabel('Makespan $MS$')
    fig.subplots_adjust(left=0.075, right=0.90, bottom=0.25, top=0.86, wspace=0.34)
    if im is not None:
        cax = fig.add_axes([0.915, 0.25, 0.013, 0.61])
        cb = fig.colorbar(im, cax=cax)
        cb.set_label('Density diff.\n(orange: ILS, green: Memetic)', fontsize=6)
        cb.ax.tick_params(labelsize=5.5)
    out = os.path.join(OUT, 'fig_v2_density_en.png')
    fig.savefig(out, dpi=350)
    plt.close(fig)
    print(' ->', out)


# ---------- 図3: 交互作用型プロット（none / repair / PR × ホスト） ----------

HOSTS = [('ILS', ['ils_baseline', 'ils_repair', 'ils_pr'], C_ILS, 'o'),
         ('Memetic', ['memetic_ls', 'memetic_repair', 'memetic_pr'], C_MEM, 's')]


def fig_interaction(S):
    probs = A.order_prob_labels(S.keys())
    fig, axes = plt.subplots(2, 4, figsize=(FULLW, 2.75))
    x = np.arange(3)
    for i, (ax, prob) in enumerate(zip(axes.flat, probs)):
        high = S[prob]['highstab_hv_pt']
        for host, keys, color, mk in HOSTS:
            vals = [np.asarray(high.get(k, []), float) for k in keys]
            med = np.array([np.median(v) for v in vals])
            q1 = np.array([np.percentile(v, 25) for v in vals])
            q3 = np.array([np.percentile(v, 75) for v in vals])
            ax.errorbar(x, med, yerr=[med - q1, q3 - med], marker=mk, ms=3.2, color=color,
                        lw=1.2, capsize=2, elinewidth=0.7, label=host)
            # baseline に対する両側 Mann–Whitney U。
            # trial 間の対応は乱数シード番号だけで結果は連動しないため、対応なし検定を使う
            above = host.startswith('Memetic')
            for j, k in enumerate(keys[1:], start=1):
                p = mannwhitneyu(vals[j], vals[0], alternative='two-sided').pvalue
                if p is not None and np.isfinite(p) and p < 0.05:
                    ax.annotate('*' if p >= 0.01 else ('**' if p >= 0.001 else '***'),
                                (x[j], med[j]), xytext=(0, 4 if above else -10),
                                textcoords='offset points', ha='center',
                                va='bottom' if above else 'top', fontsize=6, color=color)
        ax.set_title(f'{A.problem_short_tag(prob)} ($\\rho$={rho_pct(prob)}%)', fontsize=7.5)
        ax.set_xticks(x)
        ax.set_xticklabels(['none', 'repair', 'PR'])
        ax.set_xlim(-0.4, 2.4)
        ymax = max(float(np.max(np.asarray(high.get(k, [0.0]), float)))
                   for h in HOSTS for k in h[1])
        ax.set_ylim(0, ymax * 1.28)
        ax.tick_params(labelsize=6)
        ax.yaxis.set_major_locator(plt.MaxNLocator(4))
        ax.grid(axis='y', alpha=0.25)
        if i % 4 == 0:
            ax.set_ylabel('High-stability HV')
        if i == 0:
            ax.legend(loc='lower right', frameon=False, handletextpad=0.3)
    for ax in axes[1]:
        ax.set_xlabel('Stability-inducing operator')
    fig.tight_layout(pad=0.4, h_pad=0.8, w_pad=0.8)
    out = os.path.join(OUT, 'fig_v2_interaction_en.png')
    fig.savefig(out, dpi=350)
    plt.close(fig)
    print(' ->', out)


# ---------- 図4: PR 経路統計（構造差と非対称の機序） ----------

ILS_FIRST_KICK, ILS_LATER_KICK = 400, 10  # 表 1: 無改善 400 反復で初回、以降は 10 反復
# シナリオ名ラベルの位置 (dx, dy [pt], ha)。点が密集する d0≈7〜13 で重ならないよう個別に置く
PR_LABEL_OFFSET = {
    ('memetic_pr', 'la36S'): (0, 4, 'center'), ('memetic_pr', 'mt10'): (4, -2, 'left'),
    ('memetic_pr', 'la36M'): (-4, -2, 'right'), ('memetic_pr', 'la21'): (-4, -2, 'right'),
    ('memetic_pr', 'la40'): (4, -6, 'left'), ('memetic_pr', 'la36L'): (4, -3, 'left'),
    ('ils_pr', 'la36S'): (4, -2.5, 'left'), ('ils_pr', 'ta21S'): (-4, 1, 'right'),
}


def _ils_pr_by_start(J):
    """ILS+PR の PR 呼び出しを始点ごとにまとめ、[(d0, 始点からのいずれかの呼び出しで改善したか)] を返す。

    ILS の current は PR 発動時に必ず best と一致する（改善しなかったキックの後は best に戻り、
    insert 摂動は改善時のみ受理）。よって始点が変わるのは best 改善時だけで、停滞中は同じ始点から
    PR が繰り返し呼ばれる。呼び出し単位で数えると同じ始点の再試行が 0 として積み上がるため、
    発動タイミングを best_score の推移から再現し、始点単位に集約する（Memetic は 1 呼び出し＝1 個体）。
    再現した発動回数が保存済みの呼び出し数と一致しない run は None を返す。
    """
    ms = J.get('mech_stats') or {}
    d0, imp = ms.get('pr_d0', []), ms.get('pr_improved', [])
    bs = [h['best_score'] for h in J['history']]
    nic, kicks, start = 0, 0, []
    for i in range(len(bs) - 1):
        pr = nic >= (ILS_FIRST_KICK if kicks == 0 else ILS_LATER_KICK)
        if pr:
            nic, kicks = 0, kicks + 1
            start.append(bs[i])
        if bs[i + 1] < bs[i]:
            nic = 0
        elif not pr:
            nic += 1
    if len(start) != len(d0):
        return None
    out, s = [], 0
    while s < len(d0):
        e = s
        while e + 1 < len(d0) and start[e + 1] == start[s]:
            e += 1
        out.append((int(d0[s]), int(any(imp[s:e + 1]))))
        s = e + 1
    return out


def fig_mech_pr():
    # 単位: 始点（ILS=best が変わるまでの同じ current、Memetic=PR を掛けた個体）。
    # d0=0 は S_p そのもので経路が存在しないため除く
    agg = {}
    for prob_dir in sorted(glob.glob(os.path.join(RESULTS, '*', 'raw'))):
        prob = os.path.basename(os.path.dirname(prob_dir))
        rows = {'ils_pr': [], 'memetic_pr': []}
        for p in glob.glob(os.path.join(prob_dir, 'ils_pr__*.json')):
            g = _ils_pr_by_start(json.load(open(p, encoding='utf-8')))
            if g is None:
                raise RuntimeError(f'PR 発動の再現が保存統計と一致しない: {p}')
            rows['ils_pr'] += g
        for p in glob.glob(os.path.join(prob_dir, 'memetic_pr__*.json')):
            ms = json.load(open(p, encoding='utf-8')).get('mech_stats') or {}
            rows['memetic_pr'] += list(zip(ms.get('pr_d0', []), ms.get('pr_improved', [])))
        if rows['ils_pr'] and rows['memetic_pr']:
            agg[prob] = {m: np.asarray([r for r in v if r[0] > 0], float) for m, v in rows.items()}

    fig, ax = plt.subplots(figsize=(3.35, 2.25))
    methods = [('memetic_pr', 'Memetic+PR', C_MPR, 's'), ('ils_pr', 'ILS+PR', C_ILS, 'o')]
    zero_tags = []
    for mkey, mlabel, color, mk in methods:
        for j, prob in enumerate(A.order_prob_labels(agg.keys())):
            a = agg[prob][mkey]
            med, q1, q3 = np.percentile(a[:, 0], [50, 25, 75])
            rate = 100.0 * a[:, 1].mean()
            print(f'  {A.problem_short_tag(prob):6s} {mlabel:10s} starts={len(a):7d} '
                  f'd0 med={med:.0f} [{q1:.0f},{q3:.0f}] rate={rate:.1f}%')
            ax.plot([q1, q3], [rate, rate], color=color, lw=0.8, alpha=0.45, zorder=2)
            ax.scatter([med], [rate], s=18, marker=mk, color=color, edgecolor='black', lw=0.3,
                       zorder=3, label=mlabel if j == 0 else None)
            if mkey == 'memetic_pr' or rate >= 1.0:
                dx, dy, ha = PR_LABEL_OFFSET.get((mkey, A.problem_short_tag(prob)), (3, 2, 'left'))
                ax.annotate(A.problem_short_tag(prob), (med, rate), xytext=(dx, dy),
                            textcoords='offset points', ha=ha, va='bottom', fontsize=5.5, color=color,
                            bbox=dict(boxstyle='square,pad=0.05', fc='white', ec='none', alpha=0.85),
                            zorder=4)
            elif mkey == 'ils_pr':
                zero_tags.append((med, A.problem_short_tag(prob)))
    # 0% の ILS 点は横軸上で近接し個別ラベルが重なるため、点の左右の並び（d0 昇順）で 1 行にまとめる
    if zero_tags:
        ax.text(2.0, -8.5, ', '.join(t for _, t in sorted(zero_tags)) + ': 0%', fontsize=5.5,
                color=C_ILS, ha='left', va='bottom')
    ax.set_xscale('log')
    ax.set_xticks([2, 5, 10, 20, 50, 100])
    ax.set_xticklabels(['2', '5', '10', '20', '50', '100'])
    ax.set_xlim(1.5, 110)
    ax.set_ylim(-11, 78)
    ax.set_yticks(range(0, 80, 10))
    ax.set_xlabel('PR path length $d_0$ (median; bar = IQR)')
    ax.set_ylabel('Paths with an improving solution (%)')
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, loc='upper right', handletextpad=0.2)
    fig.tight_layout(pad=0.4)
    out = os.path.join(OUT, 'fig_v2_mech_pr_en.png')
    fig.savefig(out, dpi=350)
    plt.close(fig)
    print(' ->', out)


# ---------- 図5: 総合スコアボード（3 指標のヒートマップを縦に 3 段） ----------

SB_TITLES = {'union': '(a) Union HV (overall quality)',
             'highstab': '(b) High-stability HV (quality near $S_p$)',
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
    lbl = dict(A.METHOD_LABELS)
    prob_labels = A.order_prob_labels(S.keys())
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
        s_labels = [lbl.get(methods[j], methods[j]) for j in order]
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
    out = os.path.join(OUT, 'fig_v2_scoreboard_en.png')
    fig.savefig(out, dpi=350)
    plt.close(fig)
    print(' ->', out)


if __name__ == '__main__':
    S = load_pkl()
    if '--select' in sys.argv:
        fig_front_all8(S)
        sys.exit(0)
    fig_concept()
    fig_front(S)
    fig_interaction(S)
    fig_mech_pr()
    fig_scoreboard(S)
    print('done.')
