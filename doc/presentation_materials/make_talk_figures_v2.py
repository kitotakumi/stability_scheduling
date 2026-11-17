#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""学会発表（JIMA2026 秋）v2 の結果の章の図を生成する（v1 の make_talk_figures.py はそのまま残す）。

  concept_pareto_v2.png  1 章の Pareto フロントと HV の模式図。高安定 HV の模式図と同じ非劣解と参照点までの HV に，
                         左下が良い向きの矢印を添える
  concept_hv_v2.png      1 章の高安定 HV の模式図（v1 の concept_hv から Pareto フロントの曲線とラベルを外したもの）
  concept_sp_v2.png      1 章の研究の出発点の模式図（v1 の concept_sp の★の注記を外し，★のラベルを
                         「初期スケジュール S_p」にしたもの）
  front_ta21L_v2.png     代表シナリオ ta21L の実 Pareto フロントを，(a) 演算子なし・(b) PR を組み込む の
                         2 パネルに分けた図。(b) には (a) の 2 手法を薄い帯で重ねる
  highstab_all8_v2.png   全 8 シナリオの高安定 HV（v1 の図から有意水準の * を外したもの）
  highstab_4_v2.png      演算子の効果の 4 パネル（同上）
  anytime_ta21L_v2.png   ta21L の全域 HV の推移（1 パネル）

データ・配色・集計は v1 と同じものを使う（make_talk_figures.py の load() と各定数）。
usage: python make_talk_figures_v2.py
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.legend_handler import HandlerTuple

import make_talk_figures as F
from make_talk_figures import (C_ILS, C_MEM, M_ILS, M_MEM, BAND_FACE, BAND_EDGE, INK, GUIDE, TAG, ORDER,
                               HOSTS, ALL7, despine, tight, save)

REP = 'ta21_ta21_high'   # 代表シナリオ（ta21L）．フロントと全域 HV の推移で同じシナリオを使う

# (b) に重ねる (a) の手法の帯。細い線や薄い点では下の非劣解に隠れるので，薄い帯にして縁を少しはみ出させる
GHOST_LW, GHOST_ALPHA = 6.0, 0.25
# 各パネルの手法の階段の線。手法は 2 つずつなので，帯にせず細い実線にする
FRONT_LW = 1.8


def fig_concept_pareto_v2():
    """Pareto フロントと HV のスライド用（1 章．着目点と評価指標のスライドの前）。高安定領域の帯は描かず，
    非劣解（Pareto フロント）と参照点までの HV だけを描き，両目的とも最小化なのでフロントは左下にあるほど
    良いことを矢印で示す。非劣解・参照点・塗りは次の高安定 HV の図（concept_hv）と同じにし，次の図は
    この図に高安定領域を足した形になる。
    Pareto フロントの曲線（破線）とラベルは 1 章でこの図で初めて付け，次の高安定 HV の図では描かない。"""
    fs = F.FS_PLANE
    fig, ax = plt.subplots(figsize=(6.4, 6.05))
    # 曲線とラベルは他の 1 章の模式図と同じ作り（_plane が描く）
    F._plane(ax, fs=fs, star_label=None, band=False)
    ax.xaxis.label.set_size(fs)
    ax.yaxis.label.set_size(fs)
    ref = (1.0, 1.0)
    fx = F.FX0
    fy = F._front_curve(fx)
    sx, sy = F._stair(fx, fy, ref[0])
    ax.fill_between(sx, sy, ref[1], color=F.HV_FACE, lw=0, zorder=0.5)
    ax.step(np.r_[fx, ref[0]], np.r_[fy, fy[-1]], where='post', color=INK, lw=1.3, zorder=3)
    ax.scatter(fx, fy, s=62, color=INK, zorder=6)
    ax.plot(*ref, marker='x', ms=10, mew=2.2, color='#55606B', zorder=6)
    ax.text(ref[0] - 0.02, ref[1] + 0.02, '参照点', ha='right', va='bottom', fontsize=fs - 2,
            color='#55606B')
    # 次の図の「全域 HV」と同じ位置に置く
    ax.text(0.66, 0.75, 'HV', ha='center', va='center', fontsize=fs, color=INK, fontweight='bold')
    # 両目的とも最小化：左下が良い向き
    ax.annotate('', xy=(0.10, 0.11), xytext=(0.30, 0.31),
                arrowprops=dict(arrowstyle='-|>', color='#1F4E86', lw=2.6, mutation_scale=24))
    ax.text(0.31, 0.27, '良い', ha='left', va='top', fontsize=fs, color='#1F4E86', fontweight='bold')
    fig.tight_layout(pad=0.4)
    F.mark_schematic(fig)
    save(fig, 'concept_pareto_v2.png')


def fig_concept_hv_v2():
    """高安定 HV の図（v1 の concept_hv）から，Pareto フロントの曲線とラベルを外したもの。"""
    F.fig_concept_hv(front=False, name='concept_hv_v2.png')


def fig_concept_sp_v2():
    """研究の出発点の図（v1 の concept_sp）から★の注記（探索せずに得られる）を外し，★のラベルで
    初期スケジュールであることを言う（探索せずに得られることは右の本文にある）。"""
    F.fig_concept_sp(star_label='初期スケジュール $S_p$', note=False, name='concept_sp_v2.png')


def fig_front_v2(M, S, prob=REP):
    """代表シナリオの実 Pareto フロント（各手法で全域 HV が中央値の試行）。(a) は演算子なしの ILS・Memetic，
    (b) は PR を組み込んだ両手法で，(b) には (a) の 2 手法を薄い帯で重ね，Memetic が演算子でどこから動いたかと，
    演算子なしの ILS との重なりを読めるようにする。両パネルは同じ軸で，手法の描き方も同じ。
    横軸は (b) の非劣解がすべて入る範囲で切る（(a) の Memetic の D が大きい側の点は外れ，線だけが続く）。"""
    pf = {k: M.trial_pareto(prob, k, M.median_trial(S, prob, k))
          for k in ('ils_baseline', 'memetic_ls', 'ils_pr', 'memetic_pr')}
    p50 = S[prob]['thresholds']['P50']
    init_ms = float(S[prob]['init_ms'])
    xhi = 1.08 * max(pf[k][:, 1].max() for k in ('ils_pr', 'memetic_pr'))
    ymin = min(p[:, 0].min() for p in pf.values())
    span = init_ms - ymin
    panels = [('(a) 演算子なし',
               [('ils_baseline', 'ILS', C_ILS, M_ILS), ('memetic_ls', 'Memetic', C_MEM, M_MEM)], []),
              ('(b) PR を組み込む',
               [('ils_pr', 'ILS＋PR', C_ILS, M_ILS), ('memetic_pr', 'Memetic＋PR', C_MEM, M_MEM)],
               [('ils_baseline', C_ILS), ('memetic_ls', C_MEM)])]

    def stair(ax, p, **kw):
        # 最後の点から右端の外まで水平に延ばす（軸の範囲の外の点も通る）
        ax.step(np.r_[p[:, 1], 10 * xhi], np.r_[p[:, 0], p[-1, 0]], where='post', solid_capstyle='butt', **kw)

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.5), sharex=True, sharey=True)
    for k, (ax, (title, strong, ghost)) in enumerate(zip(axes, panels)):
        for key, color in ghost:
            stair(ax, pf[key], color=color, lw=GHOST_LW, alpha=GHOST_ALPHA, zorder=2)
        for key, label, color, mk in strong:
            stair(ax, pf[key], color=color, lw=FRONT_LW, zorder=3)
            ax.scatter(pf[key][:, 1], pf[key][:, 0], s=48, marker=mk, color=color, edgecolor='black',
                       lw=0.5, zorder=6 if mk == M_MEM else 5, label=label)
        ax.axvspan(-1e9, p50, facecolor=BAND_FACE, lw=0, zorder=0)
        ax.axvline(p50, color=BAND_EDGE, lw=1.1, ls=':', zorder=1)
        ax.plot([0], [init_ms], marker='*', ms=16, color='black', zorder=7, clip_on=False)
        ax.set_xlim(-0.03 * xhi, xhi)
        ax.set_ylim(ymin - 0.10 * span, init_ms + 0.10 * span)
        ax.text(p50 / 2, ymin - 0.085 * span, '高安定領域', ha='center', va='bottom',
                fontsize=16, color=BAND_EDGE)
        ax.yaxis.set_major_locator(plt.MaxNLocator(4))
        ax.set_title(title, loc='left')
        ax.set_xlabel('順序偏差 $D$')
        ax.grid(alpha=0.25)
        despine(ax)
        if k == 0:
            ax.set_ylabel('総所要時間 $MS$')
        h, l = ax.get_legend_handles_labels()
        if ghost:
            h.append(tuple(Line2D([], [], color=c, lw=GHOST_LW, alpha=GHOST_ALPHA, solid_capstyle='butt')
                           for _, c in ghost))
            l.append('演算子なし（a）')
        ax.legend(h, l, loc='upper right', frameon=False, handlelength=2.6,
                  handler_map={tuple: HandlerTuple(ndivide=None, pad=0.3)})
    tight(fig, pad=0.4, w_pad=2.0)
    save(fig, 'front_ta21L_v2.png')


def _highstab_panel_v2(ax, per, title):
    """v1 の _highstab_panel から有意水準の * を外したもの（判定はスライドの本文で p と δ を示す）。
    HOSTS の順に，最良値で割った中央値を返す。"""
    x = np.arange(3)
    best = max(np.median(np.asarray(per[m], float)) for m in ALL7 if len(per.get(m, [])))
    meds = []
    for host, keys, color, mk, _above in HOSTS:
        vals = [np.asarray(per[m], float) / best for m in keys]
        med = np.array([np.median(v) for v in vals])
        lo = np.array([np.percentile(v, 25) for v in vals])
        hi = np.array([np.percentile(v, 75) for v in vals])
        ax.errorbar(x, med, yerr=[med - lo, hi - med], marker=mk, ms=8, color=color,
                    mec='white', mew=0.8, lw=2.4, capsize=3, elinewidth=1.0, label=host, zorder=3)
        meds.append(med)
    ax.set_title(title, fontsize=18, pad=3)
    ax.set_xticks(x)
    ax.set_xticklabels(['なし', 'PR-$k$', 'PR'], fontsize=16)
    ax.set_xlim(-0.4, 2.4)
    ax.set_ylim(0, 1.32)
    ax.set_yticks([0, 0.5, 1.0])
    ax.set_yticklabels(['0', '0.5', '1'])
    ax.grid(axis='y', alpha=0.25)
    despine(ax)
    return meds


def fig_highstab_v2(S):
    """全 8 シナリオの高安定 HV（v1 の fig_highstab から * と * の注を外したもの）。"""
    fig, axes = plt.subplots(2, 4, figsize=(12.0, 3.85), sharey=True)
    for ax, prob in zip(axes.flat, ORDER):
        _highstab_panel_v2(ax, S[prob]['highstab_hv_pt'], TAG[prob])
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc='upper center', ncol=len(l), frameon=False, fontsize=16,
               bbox_to_anchor=(0.5, 1.0), columnspacing=2.0)
    fig.text(0.005, 0.44, '高安定 HV（最良値＝1）', rotation=90, ha='left', va='center',
             fontsize=16, color=INK)
    tight(fig, pad=0.4, h_pad=0.5, w_pad=0.9, rect=(0.035, 0, 1, 0.885))
    save(fig, 'highstab_all8_v2.png')


def fig_highstab4_v2(S, tags=('la36S', 'ta21S', 'la36L', 'ta21L'), mark=('la36L', 'ta21L')):
    """演算子の効果の 4 パネル（v1 の fig_highstab4 から * と * の注を外したもの）。mark のシナリオでは，
    ILS の演算子なし → PR-k の上がりを，線の少し上に平行な青い矢印で示す。"""
    probs = {TAG[p]: p for p in S if p in TAG}
    fig, axes = plt.subplots(1, len(tags), figsize=(12.0, 2.65), sharey=True)
    for ax, tag in zip(axes, tags):
        meds = _highstab_panel_v2(ax, S[probs[tag]]['highstab_hv_pt'], tag)
        if tag in mark:
            m0, m1 = meds[0][0], meds[0][1]
            x0, x1, dy = 0.1, 0.7, 0.13
            ax.annotate('', xy=(x1, m0 + (m1 - m0) * x1 + dy), xytext=(x0, m0 + (m1 - m0) * x0 + dy),
                        arrowprops=dict(arrowstyle='-|>', color=GUIDE, lw=1.8, mutation_scale=14,
                                        shrinkA=0, shrinkB=0), zorder=6)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc='upper center', ncol=len(l), frameon=False, fontsize=16,
               bbox_to_anchor=(0.5, 1.0), columnspacing=2.0)
    fig.text(0.006, 0.45, '高安定 HV\n（最良値＝1）', rotation=90, ha='left', va='center',
             fontsize=16, color=INK, linespacing=1.2)
    tight(fig, pad=0.4, w_pad=0.9, rect=(0.065, 0, 1, 0.86))
    save(fig, 'highstab_4_v2.png')


def fig_anytime_v2(M, prob=REP):
    """代表シナリオの全域 HV の推移（v1 の fig_anytime と同じ線種で 1 パネル）。立ち上がりの速さ（AUC）と
    最終の全域 HV の両方が 1 つのシナリオで読める ta21L を使う。"""
    style_ = [('ils_baseline', C_ILS, '-'), ('ils_repair', C_ILS, (0, (5, 2))),
              ('ils_pr', C_ILS, (0, (0.6, 2.2))),
              ('memetic_ls', C_MEM, '-'), ('memetic_repair', C_MEM, (0, (5, 2))),
              ('memetic_pr', C_MEM, (0, (0.6, 2.2)))]
    methods = [s[0] for s in style_]
    fig, ax = plt.subplots(figsize=(5.9, 5.5))
    t, cur = M._anytime_curves(prob, methods + ['ga'])
    for key, color, ls in style_:
        ax.plot(t, cur[key][0], color=color, ls=ls, lw=2.8, dash_capstyle='round')
    ax.set_xscale('log')
    ax.set_xlim(t[0], M._anytime_cut(t, cur))
    ax.set_ylim(0, None)
    ax.set_title(TAG[prob], loc='left')
    ax.set_xlabel('経過時間 [s]（対数）')
    ax.set_ylabel('全域 HV')
    ax.grid(alpha=0.25)
    despine(ax)
    # 凡例は色（探索構造）→ 線種（演算子）の順に 1 列で，線のない右下（後半の低い HV）に置く
    gray = '#55606B'
    hs = [Line2D([], [], color=C_ILS, lw=2.6), Line2D([], [], color=C_MEM, lw=2.6),
          Line2D([], [], color=gray, lw=2.4), Line2D([], [], color=gray, lw=2.4, ls=(0, (5, 2))),
          Line2D([], [], color=gray, lw=2.4, ls=(0, (0.6, 2.2)), dash_capstyle='round')]
    ax.legend(hs, ['ILS', 'Memetic', '演算子なし', '＋PR-$k$', '＋PR'], loc='lower right',
              frameon=False, fontsize=16, handlelength=2.2, handletextpad=0.5)
    tight(fig, pad=0.4)
    save(fig, 'anytime_ta21L_v2.png')


if __name__ == '__main__':
    F.style()
    fig_concept_pareto_v2()   # 模式図は実データの図の文字の設定（DATA_RC）を当てる前に描く
    fig_concept_hv_v2()
    fig_concept_sp_v2()
    M, S = F.load()
    fig_front_v2(M, S)
    fig_highstab_v2(S)
    fig_highstab4_v2(S)
    fig_anytime_v2(M)
    print('done.')
