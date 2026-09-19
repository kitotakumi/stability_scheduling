#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""APIEMS 2026 原稿 v2 の図 1〜4 をすべて生成する。

データソースは main_v1 の _summary_data.pkl と raw JSON、ヘルパーは analyze_v3（v1 の
make_figures_en.py と同じ）。本ディレクトリだけで v2 の図が揃う:

  fig_v2_concept_en.png             図1 (D, MS) 平面の模式図（H1/H2 を結果と同じ座標系で描く）
  fig_v2_front_en.png               図2 実 Pareto フロント（ta21S / ta21L, ILS-b・Mem-LS・Mem+PR）
  fig_v2_interaction_en.png         図3 演算子 none/repair/PR × 探索構造の高安定 HV
  fig_v2_anytime_en.png             図4 アンタイム統合 HV 曲線（ta21S / la36L, 6 手法）

検討用（本文では使わない。--review で生成）:
  fig_v2_interaction_union_aoc_en.png  統合 HV・AOC の交互作用図（RPD%。図4 の旧案）
  fig_v2_scoreboard_en.png   総合スコアボード（7 手法 × 8 シナリオ × 3 指標のヒートマップ）
  fig_v2_mech_pr_en.png      PR 経路統計（始点ごとの経路長 d0 と経路上に改善解が残る割合）
  fig_v2_density_en.png      訪問密度差マップ（fig_density。--review でも生成しない）
出力先は本スクリプトと同じディレクトリの figures/。

usage: python make_figures_v2.py            # 図 1〜4
       python make_figures_v2.py --review   # 検討用（統合HV/AOC交互作用図・スコアボード・PR 経路統計）
       python make_figures_v2.py --select   # 図 2 のシナリオ選定用シート（論文には載せない）
       python make_figures_v2.py --gray     # 生成済みの図から白黒校正だけ作り直す
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

# 白黒印刷と色覚多様性に耐えるよう、意味は必ず 2 チャネル以上で表す:
#   探索構造 = 色相 ＋ グレー明度 ＋ マーカー形状（ILS: 橙/○、Memetic: 緑/□、演算子の経路: 茶/◇）
#   演算子   = 線種（なし: 実線、repair: 長破線、PR: 丸点線）
# 明度は L601=0.299R+0.587G+0.114B（PIL の convert('L') と同じ式）で、3 色の間隔を 34 以上あける。
# グレースケール版を figures/_gray/ に自動出力するので、投稿前にそちらで可読性を確認する。
C_ILS, C_MEM, C_MPR = 'tab:orange', '#1b7837', '#4a2c17'  # L601 = 152 / 85 / 51
M_ILS, M_MEM, M_MPR = 'o', 's', 'D'
# 高安定領域: 塗りは L601=231（紙白より 9.5% 暗い）。alpha ではなく実色で置き、白黒でも面として残す
BAND_FACE, BAND_EDGE = '#f2e7c8', '#8a6d1f'


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
        ax.axvspan(0, band_hi, facecolor=BAND_FACE, lw=0, zorder=0)
        ax.axvline(band_hi, color=BAND_EDGE, lw=0.7, ls=':', zorder=1)
        ax.text(band_hi / 2, 0.075, 'high-stability\nregion (near $S_p$)', ha='center', va='bottom',
                fontsize=6.5, color=BAND_EDGE)
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
        ax_.scatter(mx, my, s=9, marker=M_MEM, color=C_MEM, alpha=a_dom, zorder=3)
        ax_.scatter(mem_fx, mem_fy, s=12, marker=M_MEM, color=C_MEM, alpha=a_front, zorder=4)

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
        ax_.scatter(ix, iy, s=16, marker=M_ILS, color=C_ILS, alpha=alpha, zorder=4)
        arrow_chain(ax_, np.r_[0.0, ix], np.r_[0.95, iy], C_ILS, alpha)  # 星（S_RSR）から出発

    draw_mem(ax, 0.95, 0.55)
    ils_chain(ax, 1.0)
    ax.text(0.03, 0.50, 'ILS (trajectory search, circles):\nsmall moves from $S_p$\nfill the region step by step',
            fontsize=6.5, color=C_ILS, ha='left', va='top')
    ax.text(0.42, 0.93, 'Memetic (population search, squares): offspring scattered\nby crossover; stops short of small $D$ and is clearly worse\nin $MS$ where it does reach; low $MS$ only at large $D$',
            fontsize=6.5, color=C_MEM, ha='left', va='top')

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
    ax.scatter(pr_x[1:], pr_y[1:], s=15, marker=M_MPR, color=C_MPR, zorder=5)
    ax.scatter([pr_x[0]], [pr_y[0]], s=16, marker=M_MEM, color=C_MEM, edgecolor='black', lw=0.5,
               zorder=5)
    arrow_chain(ax, pr_x, pr_y, C_MPR)
    ax.text(0.40, 0.93, 'Operator (diamonds): swap the current solution back toward\n$S_p$, opposite to the ILS direction;\nsolutions on the path fill the region',
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

FRONT_PROBS = ['ta21_ta21_delay97', 'ta21_ta21_high']  # ta21 対（同一 S_p, 外乱の規模のみ異なる）
SHOW_ALL_TRIALS = False  # True にすると中央値 trial 以外の非劣解も薄い点で重ねる（本稿では非表示）
# 色＋マーカー形状で探索構造（橙/○: ILS、緑/□: Memetic）、線種で演算子（実線: なし、点線: PR）。
# 演算子なしは太く淡い帯、＋PR は細く濃い点線として帯の上に重ねる。主役は「＋PR がさらに $S_p$ 側へ
# 踏み込むか」なので、参照側の 2 手法にインクを使わず、重なる区間は帯の中を点線が走る形で読ませる。
# 非劣解の数は演算子なしの 2 手法でしか数えないので、マーカーもその 2 手法にだけ付ける。
# ＋PR の 2 本は高安定領域でしばしば完全に重なるので、同じ点線パターンを半周期ずらして交互に出す。
PR_DASH = (1.2, 2.4)
FRONT_METHODS = [('ils_baseline', 'ILS-baseline', C_ILS, M_ILS, 3.0, '-'),
                 ('ils_pr', 'ILS+PR', C_ILS, None, 1.7, (0.0, PR_DASH)),
                 ('memetic_ls', 'Memetic-LS', C_MEM, M_MEM, 3.0, '-'),
                 ('memetic_pr', 'Memetic+PR', C_MEM, None, 1.7, (1.8, PR_DASH))]


def _draw_front_panel(ax, S, prob, panel_tag=None, band_label='wide', legend=True,
                      star_label=True):
    """1 シナリオ分の実フロントパネルを描く（fig_front と選定用シートで共用）。

    x 軸は中央値 trial のフロント（全手法）の最大 D で切る。全 trial の薄い点はその範囲内のみ表示
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
        if SHOW_ALL_TRIALS and mk:
            allpf = np.concatenate([p for p in pfs if len(p)])
            allpf = allpf[allpf[:, 1] <= xlim_hi]
            ymin = min(ymin, allpf[:, 0].min())
            ax.scatter(allpf[:, 1], allpf[:, 0], s=7, marker=mk, color=color, alpha=0.22,
                       lw=0, zorder=2)
        ax.step(np.r_[pf[:, 1], xlim_hi], np.r_[pf[:, 0], pf[-1, 0]], where='post',
                color=color, lw=lw, ls=ls, alpha=0.40 if mk else 1.0,
                solid_capstyle='butt', dash_capstyle='round',
                zorder=3 if mk else 5, label=None if mk else label)
        if mk:
            ax.scatter(pf[:, 1], pf[:, 0], s=16, marker=mk, color=color, edgecolor='black',
                       lw=0.3, zorder=6, label=label)
    ax.axvspan(-1e9, p50, facecolor=BAND_FACE, lw=0, zorder=0)
    ax.axvline(p50, color=BAND_EDGE, lw=0.8, ls=':', zorder=1)
    ax.plot([0], [init_ms], marker='*', ms=9, color='black', zorder=6)
    span = init_ms - ymin
    if star_label:
        ax.text(0.03 * xlim_hi, init_ms, '$S_{RSR}$ ($D=0$)', fontsize=6.5, va='center',
                ha='left')
    ax.set_xlim(-0.03 * xlim_hi, xlim_hi)
    ax.set_ylim(ymin - 0.06 * span, init_ms + 0.10 * span)
    if band_label == 'wide':
        ax.text(p50 / 2, ymin - 0.045 * span, 'high-stability region ($D<P_{50}$)',
                ha='center', va='bottom', fontsize=6.2, color=BAND_EDGE)
    elif band_label == 'narrow':
        ax.text(p50 * 0.2, ymin - 0.04 * span, 'high-stab. ($D<P_{50}$)',
                ha='center', va='bottom', fontsize=5.8, color=BAND_EDGE, rotation=90)
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

HOSTS = [('ILS', ['ils_baseline', 'ils_repair', 'ils_pr'], C_ILS, M_ILS),
         ('Memetic', ['memetic_ls', 'memetic_repair', 'memetic_pr'], C_MEM, M_MEM)]


def _rpd_scale(S, prob, key):
    """そのシナリオの最良手法（全 7 手法の trial 中央値の最大）を返す。RPD%=(1-v/best)*100 の基準。

    表 2 の ARPD% と同じ基準にそろえるため、図に描かない GA も含めて最良をとる
    （実データでは GA が最良になるシナリオはないので、描画対象 6 手法での最良と一致する）。
    """
    meds = [np.median(np.asarray(v, float)) for v in S[prob][key].values() if len(v)]
    return max(meds)


def _interaction_block(axes, S, probs, key, ylabel, legend_loc='lower right',
                       norm='abs', ylim=None):
    """2×4 のパネル（シナリオ×1）に、演算子 none/repair/PR を横軸、構造を線として指標 key を描く。

    点は trial 中央値、ひげは四分位範囲。記号は演算子なしに対する両側 Mann–Whitney U
    （trial 間の対応は乱数シード番号だけで結果は連動しないため、対応なし検定）。検定は
    どちらの norm でも生の trial 値で行う（RPD 変換はシナリオ内で単調なので順位は不変）。

    norm='abs': 指標の生値。縦軸は 0 始まり（高安定 HV のように「0＝届いていない」が
        意味を持つ指標向け）。
    norm='rpd': そのシナリオの最良手法からの相対偏差 RPD%（0＝最良）。縦軸を反転して
        上を良とし、全パネルで同じ範囲を使う。統合 HV・AOC のように全手法が同程度の
        絶対値に密集する指標では、0 始まりの生値だと数 % の差が潰れるため。
    """
    x = np.arange(3)
    inv = norm == 'rpd'
    for i, (ax, prob) in enumerate(zip(axes.flat, probs)):
        per = S[prob][key]
        best = _rpd_scale(S, prob, key) if inv else None
        def t(v):  # noqa: E306  表示用の変換（RPD は単調減少なので四分位は入れ替わる）
            return (1.0 - np.asarray(v, float) / best) * 100.0 if inv else np.asarray(v, float)
        vals = {host: [np.asarray(per.get(k, []), float) for k in keys]
                for host, keys, _c, _m in HOSTS}
        meds = {host: t([np.median(v) for v in vs]) for host, vs in vals.items()}
        stars = []  # 縦軸を決めてから描く（軸の縁にかかる記号を内側へ折り返すため）
        for host, keys, color, mk in HOSTS:
            med = meds[host]
            lo = t([np.percentile(v, 75 if inv else 25) for v in vals[host]])
            hi = t([np.percentile(v, 25 if inv else 75) for v in vals[host]])
            ax.errorbar(x, med, yerr=[med - lo, hi - med], marker=mk, ms=4.2, color=color,
                        mec='white', mew=0.5, lw=1.3, capsize=2, elinewidth=0.7, label=host)
            other = next(m for h, m in meds.items() if h != host)
            for j, k in enumerate(keys[1:], start=1):
                p = mannwhitneyu(vals[host][j], vals[host][0], alternative='two-sided').pvalue
                if p is not None and np.isfinite(p) and p < 0.05:
                    # 記号は相手の線から遠い側（画面上で外側）に置き、線との重なりを避ける
                    if med[j] != other[j]:
                        above = (med[j] > other[j]) != inv
                    else:
                        above = host.startswith('Memetic')
                    stars.append((x[j], med[j], above, color,
                                  '*' if p >= 0.01 else ('**' if p >= 0.001 else '***')))
        ax.set_title(f'{A.problem_short_tag(prob)} ($\\rho$={rho_pct(prob)}%)', fontsize=7.5)
        ax.set_xticks(x)
        ax.set_xticklabels(['none', 'repair', 'PR'])
        ax.set_xlim(-0.4, 2.4)
        if inv:
            top = ylim if ylim else max(
                float(np.max(t([np.median(np.asarray(per.get(k, [0.0]), float))
                                for k in h[1]]))) for h in HOSTS)
            ax.set_ylim(top * 1.30, -top * 0.22)  # 反転（0＝最良を上に）。上下に記号の余白
        else:
            ymax = max(float(np.max(np.asarray(per.get(k, [0.0]), float)))
                       for h in HOSTS for k in h[1])
            ax.set_ylim(0, ymax * 1.28)
        lo_lim, hi_lim = ax.get_ylim()
        span = abs(hi_lim - lo_lim)  # 軸の縁に近い記号は内側へ折り返す（下側は文字高のぶん広めに）
        for xs, ys, above, color, txt in stars:
            if above and abs(ys - hi_lim) < 0.08 * span:
                above = False
            elif not above and abs(ys - lo_lim) < 0.20 * span:
                above = True
            ax.annotate(txt, (xs, ys), xytext=(0, 3 if above else -8),
                        textcoords='offset points', ha='center',
                        va='bottom' if above else 'top', fontsize=6, color=color)
        ax.tick_params(labelsize=6)
        ax.yaxis.set_major_locator(plt.MaxNLocator(4))
        ax.grid(axis='y', alpha=0.25)
        if i % 4 == 0:
            ax.set_ylabel(ylabel)
        if i == 0 and legend_loc:
            ax.legend(loc='lower left' if inv else legend_loc,
                      frameon=False, handletextpad=0.3)
    for ax in axes[1]:
        ax.set_xlabel('Stability-inducing operator')


def fig_interaction(S):
    probs = A.order_prob_labels(S.keys())
    fig, axes = plt.subplots(2, 4, figsize=(FULLW, 2.75))
    _interaction_block(axes, S, probs, 'highstab_hv_pt', 'High-stability HV')
    fig.tight_layout(pad=0.4, h_pad=0.8, w_pad=0.8)
    out = os.path.join(OUT, 'fig_v2_interaction_en.png')
    fig.savefig(out, dpi=350)
    plt.close(fig)
    print(' ->', out)


# ---------- 検討用（本文では不使用）: 統合 HV と AOC の交互作用図（図4 の旧案） ----------

def fig_interaction_union_aoc(S, norm='rpd', out=None):
    probs = A.order_prob_labels(S.keys())
    fig = plt.figure(figsize=(FULLW, 5.7))
    subs = fig.subfigures(2, 1, hspace=0.06)
    suffix = ' — RPD% from the best method (0 = best)' if norm == 'rpd' else ''
    blocks = [('(a) Union HV (overall quality)', 'union_hv_pt', 'Union HV'),
              ('(b) AOC (anytime performance)', 'aoc_pt', 'AOC')]
    for sub, (title, key, ylabel) in zip(subs, blocks):
        axes = sub.subplots(2, 4)
        ylim = None
        if norm == 'rpd':
            # 行内（同じ指標の 8 パネル）で縦軸を共有し、シナリオ間で差の大きさを比較できるようにする
            ylim = max(100.0 * (1 - np.median(np.asarray(S[p][key][k], float))
                                / _rpd_scale(S, p, key))
                       for p in probs for h in HOSTS for k in h[1])
        _interaction_block(axes, S, probs, key, ylabel + (' RPD%' if norm == 'rpd' else ''),
                           norm=norm, ylim=ylim)
        sub.suptitle(title + suffix, fontsize=8.5, y=0.995)
        sub.subplots_adjust(left=0.075, right=0.995, top=0.86, bottom=0.17,
                            hspace=0.62, wspace=0.34)
    out = out or os.path.join(OUT, 'fig_v2_interaction_union_aoc_en.png')
    fig.savefig(out, dpi=350)
    plt.close(fig)
    print(' ->', out)


# ---------- 図4: アンタイム統合 HV 曲線（統合 HV＝終点 と AOC＝立ち上がり を 1 枚で見せる） ----------

# 線種で演算子、色＋明度で探索構造を表す（図2・図3 と同じ約束）。既定の '--' / ':' は白黒だと
# 細って見分けにくいため、長破線と丸点線で signature を強くとる（パターンは線幅で自動的に拡大される）
OP_LS = {'none': '-', 'repair': (0.0, (4.2, 1.5)), 'pr': (0.0, (0.55, 1.9))}
ANYTIME_STYLE = [('ils_baseline', C_ILS, OP_LS['none']), ('ils_repair', C_ILS, OP_LS['repair']),
                 ('ils_pr', C_ILS, OP_LS['pr']),
                 ('memetic_ls', C_MEM, OP_LS['none']), ('memetic_repair', C_MEM, OP_LS['repair']),
                 ('memetic_pr', C_MEM, OP_LS['pr'])]
ANYTIME_PROBS = ('ta21_ta21_delay97', 'la36_la36_large')
RAW_PAT = __import__('re').compile(r'__(w\d+_\d+)__t(\d+)\.json$')


def _runs_by_trial(prob, method):
    """{trial: {w_label: data}} を返す。"""
    out = {}
    for p in glob.glob(os.path.join(RESULTS, prob, 'raw', f'{method}__*.json')):
        m = RAW_PAT.search(os.path.basename(p))
        if not m:
            continue
        out.setdefault(int(m.group(2)), {})[m.group(1)] = json.load(open(p, encoding='utf-8'))
    return out


def _anytime_curves(prob, methods, n_t=40, mode='union'):
    """各手法のアンタイム HV 曲線を trial ごとに作り、中央値と四分位を返す。

    mode='union': 時刻 t までに 10 重みの探索が訪問した点をすべて合併した HV(t)。
        終点が per-trial 統合 HV（表 2・本文の統合 HV）に一致する。
    mode='mean': 各重みの HV(t) を全重みで平均。AOC は各重みの HV(t) を対数時間で平均し
        全重みで平均した値なので（§3.4）、この曲線の対数時間平均が AOC にあたる。
    """
    norm = A.make_norm(np.concatenate([all_points_raw(prob, m) for m in methods]))
    runs = {m: _runs_by_trial(prob, m) for m in methods}
    info = []
    for m in methods:
        hs, ps = [], []
        for _t, by_w in sorted(runs[m].items()):
            for _wl, d in sorted(by_w.items()):
                hs.append(A.get_anytime(d))
                ps.append(A.get_uea_points_xyt(d, 0))
        info.append((m, hs, ps, 'ils', _baselines_of(next(iter(next(iter(runs[m].values())).values())))))
    t_grid = A._build_t_grid(info, n_pts=n_t)
    tl = t_grid.tolist()
    out = {}
    for m in methods:
        bl_n = A.normalize_baseline(_baselines_of(next(iter(next(iter(runs[m].values())).values()))), norm)
        per_trial = []
        for _t, by_w in sorted(runs[m].items()):
            hs = [A.get_anytime(d) for _wl, d in sorted(by_w.items())]
            ps = [A.normalize_pts(A.get_uea_points_xyt(d, 0), norm)
                  for _wl, d in sorted(by_w.items())]
            if mode == 'union':  # 10 重みの点を合併（終点＝per-trial 統合 HV）
                per_trial.append(np.asarray(
                    A._worker_union_hv_curve((hs, ps, 'ils', tl, bl_n, A.NORM_REF)), float))
            else:                # 各重みの HV(t) を全重み平均（対数時間平均＝AOC）
                per_trial.append(np.mean(np.asarray(
                    [A._worker_trial_hv_curve((h, p, 'ils', tl, bl_n, A.NORM_REF))
                     for h, p in zip(hs, ps)], float), axis=0))
        arr = np.asarray(per_trial, float)
        out[m] = (np.median(arr, axis=0), np.percentile(arr, 25, axis=0),
                  np.percentile(arr, 75, axis=0))
    return t_grid, out


def fig_anytime(S, probs=ANYTIME_PROBS, band=False, mode='union', out=None):
    # band=True で演算子なし 2 手法の四分位範囲を重ねる（検討用。本稿では中央値のみ）
    methods = [k for k, _c, _s in ANYTIME_STYLE]
    fig, axes = plt.subplots(1, 2, figsize=(FULLW, 2.45))
    for ax, prob, tag in zip(axes, probs, '(a) (b)'.split()):
        t, cur = _anytime_curves(prob, list(dict.fromkeys(methods + ['ga'])), mode=mode)
        for m, color, ls in ANYTIME_STYLE:
            med, q1, q3 = cur[m]
            ax.plot(t, med, color=color, ls=ls, lw=1.45, dash_capstyle='round',
                    label=A.METHOD_LABELS.get(m, m))
            if band and ls == '-':
                ax.fill_between(t, q1, q3, color=color, alpha=0.13, lw=0)
        ax.set_xscale('log')
        ax.set_xlim(t[0], t[-1])
        ax.set_ylim(0, None)
        ax.set_xlabel('CPU time (s, log)')
        ax.set_title(f'{tag} {A.problem_short_tag(prob)} ($\\rho$={rho_pct(prob)}%)', fontsize=8)
        ax.grid(alpha=0.25)
        ax.tick_params(labelsize=6.5)
    axes[0].set_ylabel('Union HV' if mode == 'union' else 'HV (mean over weights)')
    axes[1].legend(fontsize=5.8, frameon=False, loc='lower right', ncol=2,
                   handlelength=2.9, handletextpad=0.4, columnspacing=0.8)
    fig.tight_layout(pad=0.4, w_pad=1.0)
    out = out or os.path.join(OUT, 'fig_v2_anytime_en.png')
    fig.savefig(out, dpi=350)
    plt.close(fig)
    print(' ->', out)


# ---------- 検討用（本文では不使用）: PR 経路統計（始点ごとの経路長と改善解が残る割合） ----------

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


# ---------- 検討用（本文では不使用）: 総合スコアボード（3 指標のヒートマップを縦に 3 段） ----------

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


# ---------- 白黒校正: 生成済みの図をグレースケールに落として figures/_gray/ に置く ----------

def make_gray_proofs():
    """投稿用の図はカラー 1 種類だけ作り、白黒での可読性はこの校正で確認する。

    PIL の convert('L') は L601=0.299R+0.587G+0.114B で、モノクロ印刷のグレー変換と同じ式。
    """
    from PIL import Image
    gdir = os.path.join(OUT, '_gray')
    os.makedirs(gdir, exist_ok=True)
    for src in sorted(glob.glob(os.path.join(OUT, 'fig_v2_*.png'))):
        dst = os.path.join(gdir, os.path.basename(src))
        Image.open(src).convert('RGB').convert('L').save(dst)
        print(' ->', dst)


if __name__ == '__main__':
    if '--gray' in sys.argv:  # 生成済みの図から白黒校正だけ作り直す
        make_gray_proofs()
        sys.exit(0)
    S = load_pkl()
    if '--select' in sys.argv:
        fig_front_all8(S)
        sys.exit(0)
    if '--review' in sys.argv:
        fig_interaction_union_aoc(S)
        fig_scoreboard(S)
        fig_mech_pr()
        sys.exit(0)
    fig_concept()
    fig_front(S)
    fig_interaction(S)
    fig_anytime(S)
    make_gray_proofs()
    print('done.')
