#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""学会発表（JIMA2026 秋・APIEMS2026）用の図をスライド寸法で生成する。

図はスライド上の配置寸法（inch）そのままで描く。build_pptx.py は同じ幅で貼るので、
図中の文字は指定した pt のままスライドに載る。

  concept_0.png      (D, MS) 平面：S_p・高安定領域・Pareto フロントだけ（手法なし）
  concept_hv.png     同じ平面・同じ非劣解で，全域 HV と高安定 HV の面積を塗った図
  concept_sp.png     同じ平面で ★（S_p）を主役にした特殊性の図
  concept_h1.png     同じ平面に単点探索（ILS）の軌道と多点探索（Memetic）の解集合（仮説1）
  concept_h2.png     多点探索の個体を S_p へ引き戻す安定性誘導演算子（仮説2）
  concept_op.png     演算子の経路に PR・PR-k が返す解を注記した図
  struct_ils.png     探索構造のスライドの小さな模式図：単点探索（S_p から 1 つの解を少しずつ動かす）
  struct_mem.png     同：多点探索（複数の解を保持する）
  metric_auc.png     追加の評価：AUC（anytime 性能）の説明図
  front_ta21.png     ta21S / ta21L の実 Pareto フロント（全域 HV が中央値の試行）
  highstab_all8.png  全 8 シナリオの高安定 HV（探索構造 × 演算子）
  highstab_band.png  同じ図に「最良値の 2% 以内」の帯を重ねたもの（現在は未使用）
  highstab_4.png     同じパネルを la36S・ta21S・la36L・ta21L の 4 つだけ並べた，演算子の効果の図
  anytime.png        ta21S / la36L の全域 HV の anytime 曲線

データと集計ヘルパーは APIEMS v2 の make_figures_v2.py（main_v1 の _summary_data.pkl・raw JSON）を
そのまま使う。符号化も v2 と揃える（ILS: 橙・○、Memetic: 緑・□、演算子の経路: 茶・◇）。
usage: python make_talk_figures.py [--no-data]   # --no-data は模式図だけ作る
--no-data でも highstab_all8.png と highstab_4.png は公開 CSV（per_trial_metrics.csv）から同じ統計で作る。
fig_highstab(None, load_csv(), band=True) で帯を重ねた図も同様に作れる。
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu

HERE = os.path.dirname(os.path.abspath(__file__))
V2 = os.path.normpath(os.path.join(HERE, '..', 'paper_apiems2026_v2'))
OUT = os.path.join(HERE, 'assets')
os.makedirs(OUT, exist_ok=True)

C_ILS, C_MEM, C_MPR = '#e06c00', '#0f5c28', '#2e1a0e'
M_ILS, M_MEM, M_MPR = 'o', 's', 'D'
BAND_FACE, BAND_EDGE = '#f5ebd0', '#7a5c12'
INK = '#26303A'
DPI = 250


# 実データの図はスライド上でも読めるよう，文字を模式図より大きくする（配置はほぼ等倍）
DATA_RC = {'font.size': 16, 'axes.titlesize': 18, 'axes.labelsize': 17,
           'xtick.labelsize': 16, 'ytick.labelsize': 16, 'legend.fontsize': 16,
           'figure.dpi': DPI}


def style():
    plt.rcParams.update({
        # Meiryo がない環境（Linux での確認用ビルド）では IPA ゴシックに落とす
        'font.family': ['Meiryo', 'IPAPGothic', 'IPAGothic', 'sans-serif'],
        'mathtext.fontset': 'dejavusans',
        'axes.unicode_minus': False,
        'font.size': 13,
        'axes.titlesize': 14,
        'axes.labelsize': 13,
        'xtick.labelsize': 11.5,
        'ytick.labelsize': 11.5,
        'legend.fontsize': 12,
        'text.color': INK,
        'axes.labelcolor': INK,
        'axes.edgecolor': '#55606B',
        'xtick.color': '#55606B',
        'ytick.color': '#55606B',
    })


def mark_schematic(fig):
    """実データの図と区別するため，模式図の右上に「模式図」と入れる。"""
    fig.text(0.995, 0.995, '模式図', ha='right', va='top', fontsize=12, color='#8A949E')


def tight(fig, **kw):
    """描画と tight_layout を 2 回繰り返す。軸ラベルの位置は描画時にしか更新されないため，
    1 回では軸を動かす前の位置で余白を見積もり，縦軸ラベルが図の外へはみ出す。"""
    for _ in range(2):
        fig.canvas.draw()
        fig.tight_layout(**kw)


def save(fig, name):
    path = os.path.join(OUT, name)
    fig.savefig(path, dpi=DPI, facecolor='white')
    plt.close(fig)
    print(' ->', name)


def despine(ax):
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)


# ====================================================================== 模式図

def _front_curve(x):
    return 0.15 + 0.80 * (1.0 - x) ** 1.8


BAND_HI = 0.30
XS = np.linspace(0, 1, 200)
FX0 = np.array([0.045, 0.10, 0.16, 0.23, 0.33, 0.44, 0.57, 0.72, 0.90])   # 着目点と HV の図の非劣解
HV_FACE, HS_FACE = '#E3E7EC', '#F2C58A'   # 全域 HV（淡い灰）と高安定 HV（橙）。色は主役の高安定 HV にだけ付ける
_rng = np.random.default_rng(7)
MEM_FX = np.array([0.27, 0.36, 0.47, 0.60, 0.75, 0.92])
MEM_FY = np.array([0.810, 0.680, 0.535, 0.380, 0.240, 0.165])
MX = _rng.uniform(0.29, 1.0, 15)
MY = np.interp(MX, MEM_FX, MEM_FY) + _rng.uniform(0.045, 0.185, 15)
IX = np.linspace(0.09, 0.47, 7)
IY = _front_curve(IX) + np.array([0.030, 0.080, 0.035, 0.090, 0.040, 0.075, 0.030])
# 安定性誘導演算子の経路（仮説 2 の図と演算子の図で共通）。多点探索の個体 1 つから S_p へ向かう。
# 始点は _pr_path で (PR_X[0], PR_Y[0]) に最も近い多点探索の個体に置き換える
PR_X = np.array([0.62, 0.50, 0.39, 0.29, 0.20, 0.115])
PR_Y = _front_curve(PR_X) + np.array([0.210, 0.115, 0.150, 0.130, 0.165, 0.032])


def _plane(ax, fs=12.5, star_label='初期解 $S_p$', band_label='高安定領域\n（$S_p$ の近傍）',
           front_label=True):
    """★は初期解 S_p（処理順序を保ったまま遅延を吸収した解，D = 0）。★は研究の出発点の図
    （concept_sp）で初めて描き，求められる解の図（concept_0）では描かない（star_label=None）。
    帯と Pareto フロントのラベルは最初の図で定義し，以後は帯の短い名前だけ（演算子の図では付けない）。"""
    ax.axvspan(0, BAND_HI, facecolor=BAND_FACE, lw=0, zorder=0)
    ax.axvline(BAND_HI, color=BAND_EDGE, lw=1.0, ls=':', zorder=1)
    if band_label:
        ax.text(BAND_HI / 2, 0.075, band_label, ha='center', va='bottom',
                fontsize=fs, color=BAND_EDGE, linespacing=1.15)
    ax.plot(XS, _front_curve(XS), ls='--', color='gray', lw=1.4, zorder=1)
    if front_label:   # 曲線上の点と重ならないよう，曲線から少し下げて置く
        ax.text(0.55, _front_curve(0.55) - 0.07, 'Pareto フロント', fontsize=fs - 1.5,
                color='gray', rotation=-17, rotation_mode='anchor', ha='left', va='top')
    if star_label is not None:
        ax.plot([0], [0.95], marker='*', ms=17, color='black', zorder=6, clip_on=False)
        ax.text(0.035, 0.985, star_label, fontsize=fs, va='bottom')
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(0.05, 1.10)
    # 目盛は置かず，原点の「0」は軸の左下の角に文字で添える。こうすると横軸ラベルも
    # 縦軸ラベルと同じ余白（labelpad）で軸に接する
    ax.set_xticks([])
    ax.set_yticks([])
    ax.text(-0.012, -0.015, '0', transform=ax.transAxes, ha='right', va='top',
            fontsize=fs - 1, color='#55606B')
    ax.set_xlabel('min　$S_p$ からの変更量（安定性）')
    ax.set_ylabel('min　総所要時間 $MS$（生産効率）')
    despine(ax)


def _arrow_chain(ax, xs, ys, color, alpha=1.0, lw=1.1, ms=15):
    # 矢じりは小さいとスライド上で向きが読み取れないので，線の太さに比べて大きめにする
    for i in range(len(xs) - 1):
        ax.annotate('', xy=(xs[i + 1], ys[i + 1]), xytext=(xs[i], ys[i]),
                    arrowprops=dict(arrowstyle='-|>', color=color, lw=lw, mutation_scale=ms,
                                    alpha=alpha, shrinkA=6, shrinkB=7), zorder=5)


def _ils(ax, alpha=1.0):
    ax.scatter(IX, IY, s=60, marker=M_ILS, color=C_ILS, alpha=alpha, zorder=4)
    _arrow_chain(ax, np.r_[0.0, IX], np.r_[0.95, IY], C_ILS, alpha)


def _mem(ax, alpha=0.9):
    ax.scatter(np.r_[MX, MEM_FX], np.r_[MY, MEM_FY], s=46, marker=M_MEM, color=C_MEM,
               alpha=alpha, zorder=3)


def _pr_path():
    pr_x, pr_y = PR_X.copy(), PR_Y.copy()
    i0 = int(np.argmin(np.hypot(MX - pr_x[0], MY - pr_y[0])))
    pr_x[0], pr_y[0] = MX[i0], MY[i0]
    return pr_x, pr_y


def _pr(ax):
    """演算子の経路：始点の個体（黒縁の□）と経路上の解（◇）を矢印でつなぐ。"""
    pr_x, pr_y = _pr_path()
    ax.scatter(pr_x[1:], pr_y[1:], s=52, marker=M_MPR, color=C_MPR, zorder=5)
    ax.scatter([pr_x[0]], [pr_y[0]], s=60, marker=M_MEM, color=C_MEM, edgecolor='black',
               lw=1.0, zorder=5)
    _arrow_chain(ax, pr_x, pr_y, C_MPR, lw=1.3)
    return pr_x, pr_y


# 1 章の 3 枚（concept_0・concept_hv・concept_sp）の文字の大きさ。3 枚で平面の大きさが揃うよう，
# 軸ラベルと帯の名前は同じ大きさにする（スライドには 6.1 in 幅で置くので，配置後はおよそ 0.95 倍）
FS_PLANE = 17


def fig_concept_0():
    fig, ax = plt.subplots(figsize=(6.4, 6.05))
    # 初期解 S_p（★）にはここでは触れない。★は次の研究の出発点の図で初めて出す
    _plane(ax, fs=FS_PLANE, star_label=None)
    ax.xaxis.label.set_size(FS_PLANE)
    ax.yaxis.label.set_size(FS_PLANE)
    # Pareto フロント上の非劣解を点で示す（仮説の図と同じ読み方になるように）。
    # 帯の中の解＝スケジュール変更コストが無視できない場面で求められる解を濃く，外の解を白抜きで描く
    fx = FX0
    fy = _front_curve(fx)
    inb = fx < BAND_HI
    ax.scatter(fx[~inb], fy[~inb], s=62, facecolor='white', edgecolor='#55606B', lw=1.4,
               zorder=5)
    ax.scatter(fx[inb], fy[inb], s=78, color=INK, zorder=6)
    ann = ax.annotate('スケジュール変更コストが\n無視できない場面で求められる解', xy=(fx[2] + 0.012, fy[2] + 0.012),
                      xytext=(0.36, 0.80), fontsize=FS_PLANE, color=INK, va='center', linespacing=1.3,
                      arrowprops=dict(arrowstyle='-|>', color='#55606B', lw=1.1, mutation_scale=12,
                                      shrinkA=4, shrinkB=6))
    # 注記が軸の右端を少し越えても tight_layout が軸を縮めないようにする（3 枚の平面の大きさを揃える）
    ann.set_in_layout(False)
    fig.tight_layout(pad=0.4)
    mark_schematic(fig)
    save(fig, 'concept_0.png')


def _stair(xs, ys, xmax):
    """D の昇順に並べた非劣解が支配する領域の下端（階段）。最後の段は xmax まで延ばす。"""
    sx, sy = [], []
    for k in range(len(xs)):
        sx += [xs[k], xs[k + 1] if k + 1 < len(xs) else xmax]
        sy += [ys[k], ys[k]]
    return np.array(sx), np.array(sy)


def fig_concept_hv():
    """高安定 HV のスライド用（1 章）。着目点の図と同じ平面・同じ非劣解に，参照点までの支配領域を塗る。
    全体が全域 HV，帯の中の部分（橙）が高安定 HV。P_θ や D = 0 の除外などの算出の細部は実験設定で扱うので，
    軸と帯の名前は 1 章の言葉のままにする。"""
    fig, ax = plt.subplots(figsize=(6.4, 6.05))
    # 帯の名前（S_p の近傍）と Pareto フロントのラベルは，1 章で平面を初めて見せるこの図で付ける
    _plane(ax, fs=FS_PLANE, star_label=None)
    ax.xaxis.label.set_size(FS_PLANE)
    ax.yaxis.label.set_size(FS_PLANE)
    fy = _front_curve(FX0)
    ref = (1.0, 1.0)
    sx, sy = _stair(FX0, fy, ref[0])
    ax.fill_between(sx, sy, ref[1], color=HV_FACE, lw=0, zorder=0.5)
    inb = FX0 < BAND_HI
    hx, hy = _stair(FX0[inb], fy[inb], BAND_HI)
    ax.fill_between(hx, hy, ref[1], color=HS_FACE, lw=0, zorder=0.6)
    ax.step(np.r_[FX0, ref[0]], np.r_[fy, fy[-1]], where='post', color=INK, lw=1.3, zorder=3)
    ax.scatter(FX0, fy, s=62, color=INK, zorder=6)
    ax.plot(*ref, marker='x', ms=10, mew=2.2, color='#55606B', zorder=6)
    ax.text(ref[0] - 0.02, ref[1] + 0.02, '参照点', ha='right', va='bottom', fontsize=FS_PLANE - 2,
            color='#55606B')
    # 橙の部分が全幅に広がる高さ（最初の非劣解より上）に置き，階段の線と重ねない
    ax.text(0.17, 0.935, '高安定 HV', ha='center', va='center', fontsize=FS_PLANE - 1,
            color='#8A4A14', fontweight='bold')
    ax.text(0.66, 0.75, '全域 HV', ha='center', va='center', fontsize=FS_PLANE, color=INK,
            fontweight='bold')
    ax.text(0.66, 0.665, '（高安定 HV を含む）', ha='center', va='center', fontsize=FS_PLANE - 2,
            color=INK)
    fig.tight_layout(pad=0.4)
    mark_schematic(fig)
    save(fig, 'concept_hv.png')


def fig_concept_sp():
    """研究の出発点のスライド用。同じ平面に ★（初期解 S_p）だけを描き，探索せずに得られる解であることを
    ★の注記で言う。★はこの図で初めて出す。非劣解の点・★からの矢印は描かない
    （どの探索構造が良いかはこのスライドでは言わない）。"""
    fig, ax = plt.subplots(figsize=(6.4, 6.05))
    _plane(ax, fs=FS_PLANE, star_label='初期解 $S_p$', band_label='高安定領域')
    ax.xaxis.label.set_size(FS_PLANE)
    ax.yaxis.label.set_size(FS_PLANE)
    # ★の注記は 1 行のラベルにする。長くすると tight_layout が軸を縮め，求められる解の図と平面の大きさが揃わない
    # （変更量 0・MS は最適化済み，は右の本文にある）
    ax.annotate('探索せずに得られる', xy=(0.012, 0.955),
                xytext=(0.36, 0.93), fontsize=FS_PLANE, color=INK, va='center', linespacing=1.3,
                arrowprops=dict(arrowstyle='-|>', color='#55606B', lw=1.1, mutation_scale=12,
                                shrinkA=4, shrinkB=9))
    fig.tight_layout(pad=0.4)
    mark_schematic(fig)
    save(fig, 'concept_sp.png')


# 仮説と演算子の図の文字の大きさ。仮説のスライドでは 4.7 in 幅ほどに縮めて置くので（およそ 0.8 倍），
# 平面の図より大きめにする
FS_SMALL_PLANE = 14.5


def fig_concept_h(stage):
    fig, ax = plt.subplots(figsize=(6.0, 3.95))
    _plane(ax, fs=FS_SMALL_PLANE, band_label='高安定領域', front_label=False)
    ax.xaxis.label.set_size(FS_SMALL_PLANE)
    ax.yaxis.label.set_size(FS_SMALL_PLANE)
    if stage == 1:
        _mem(ax, 0.9)
        _ils(ax, 1.0)
        ax.text(0.02, 0.43, '単点探索（ILS）', fontsize=FS_SMALL_PLANE + 1.5, color=C_ILS, ha='left',
                va='top', fontweight='bold')
        ax.text(0.5, 0.995, '多点探索（Memetic）', fontsize=FS_SMALL_PLANE + 1.5, color=C_MEM, ha='left',
                va='bottom', fontweight='bold')
        out = 'concept_h1.png'
    else:
        _mem(ax, 0.26)
        _ils(ax, 0.30)
        _pr(ax)
        ax.text(0.32, 0.995, '安定性誘導演算子：$S_p$ へ近づける', fontsize=FS_SMALL_PLANE + 0.5,
                color=C_MPR, ha='left', va='bottom', fontweight='bold')
        out = 'concept_h2.png'
    fig.tight_layout(pad=0.35)
    mark_schematic(fig)
    save(fig, out)


def fig_struct(kind):
    """探索構造のスライドで，各構造の枠の右上に添える小さな模式図。平面・帯・Pareto フロント・★は
    1 章の図と同じで，単点探索は★から 1 つの解を少しずつ動かす軌道，多点探索は解の集合だけを描く。
    スライドには 1.7 in 幅ほどで置くので，文字は軸の名前だけにする。"""
    fig, ax = plt.subplots(figsize=(2.0, 1.65))
    ax.axvspan(0, BAND_HI, facecolor=BAND_FACE, lw=0, zorder=0)
    ax.plot(XS, _front_curve(XS), ls='--', color='gray', lw=1.0, zorder=1)
    ax.plot([0], [0.95], marker='*', ms=11, color='black', zorder=6, clip_on=False)
    if kind == 'ils':
        xs, ys = np.r_[0.0, IX], np.r_[0.95, IY]
        ax.scatter(IX, IY, s=22, marker=M_ILS, color=C_ILS, zorder=4)
        for i in range(len(xs) - 1):
            ax.annotate('', xy=(xs[i + 1], ys[i + 1]), xytext=(xs[i], ys[i]),
                        arrowprops=dict(arrowstyle='-|>', color=C_ILS, lw=0.9, mutation_scale=8,
                                        shrinkA=3, shrinkB=4), zorder=5)
        out = 'struct_ils.png'
    else:
        ax.scatter(np.r_[MX, MEM_FX], np.r_[MY, MEM_FY], s=18, marker=M_MEM, color=C_MEM,
                   alpha=0.9, zorder=3)
        out = 'struct_mem.png'
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(0.05, 1.10)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel('順序偏差 $D$', fontsize=13, labelpad=2)
    ax.set_ylabel('$MS$', fontsize=13, labelpad=2)
    despine(ax)
    fig.tight_layout(pad=0.2)
    save(fig, out)


def fig_concept_op():
    """演算子のスライド用：多点探索の個体 1 つを S_p へ引き戻す経路に，PR と PR-k が返す解を
    注記する。背景の単点・多点の解集合は仮説 2 の図（concept_h2）と同じ配置を薄く残す。"""
    fig, ax = plt.subplots(figsize=(6.0, 3.95))
    _plane(ax, fs=FS_SMALL_PLANE, band_label=None, front_label=False)
    # 2 章以降の図は，問題設定で定義した順序偏差 D を横軸の名前にする
    ax.set_xlabel('min　順序偏差 $D$（安定性）')
    ax.xaxis.label.set_size(FS_SMALL_PLANE)
    ax.yaxis.label.set_size(FS_SMALL_PLANE)
    _mem(ax, 0.22)          # 単点探索の軌跡は演算子の説明には要らないので描かない
    pr_x, pr_y = _pr(ax)
    # 経路の終点は S_p（★）。最後の 1 手は薄い破線の矢印で示す
    ax.annotate('', xy=(0.0, 0.95), xytext=(pr_x[-1], pr_y[-1]),
                arrowprops=dict(arrowstyle='-|>', color=C_MPR, lw=1.1, ls='--', alpha=0.55,
                                mutation_scale=15, shrinkA=6, shrinkB=10), zorder=4)
    ax.text(pr_x[0] + 0.03, pr_y[0] + 0.05, '現在解', fontsize=FS_SMALL_PLANE, color=C_MEM,
            fontweight='bold')
    # 返す解（PR：経路上で F が最小の解，PR-k：k 回の swap で打ち切った解．例は k = 2）に輪を付け，
    # 注記は経路の下側，帯の左下の空きに置く。色は経路の茶と被らないよう本文の青にする
    BLUE = '#16385C'
    for (px, py), r in [((pr_x[-1], pr_y[-1]), 190), ((pr_x[2], pr_y[2]), 170)]:
        ax.scatter([px], [py], s=r, facecolor='none', edgecolor=BLUE, lw=1.6, zorder=6)
    # 演算子の名前はこの図が唯一の説明箇所なので，白抜きの札にして目立たせる
    from matplotlib.offsetbox import TextArea, HPacker, VPacker, AnnotationBbox

    def tagged(tag, rest, sub, xy, xytext):
        t_tag = TextArea(tag, textprops=dict(fontsize=13.5, color='white', fontweight='bold',
                                             bbox=dict(boxstyle='round,pad=0.28,rounding_size=0.7',
                                                       fc=BLUE, ec='none')))
        t_rest = TextArea(rest, textprops=dict(fontsize=13.5, color=BLUE, fontweight='bold'))
        line1 = HPacker(children=[t_tag, t_rest], align='center', pad=0, sep=7)
        t_sub = TextArea(sub, textprops=dict(fontsize=13, color=BLUE))
        box = VPacker(children=[line1, t_sub], align='left', pad=0, sep=6)
        ab = AnnotationBbox(box, xy, xybox=xytext, xycoords='data', boxcoords='data',
                            box_alignment=(0, 0.5), frameon=False, pad=0, zorder=7,
                            arrowprops=dict(arrowstyle='-|>', color='#55606B', lw=1.0,
                                            mutation_scale=10, shrinkA=3, shrinkB=9))
        ax.add_artist(ab)

    # PR-k の矢印が PR の注記の 2 行目を横切らないよう，2 つの注記の間を空ける
    tagged('PR', 'が返す解', '（経路上で $F$ 最小）', (pr_x[-1], pr_y[-1]), (0.03, 0.555))
    tagged('PR-$k$', 'が返す解', '（$k$ 回の swap で打ち切り）', (pr_x[2], pr_y[2]), (0.03, 0.27))
    ax.text(0.3, 0.995, '誘導解＝$S_p$ へ swap でたどる', fontsize=FS_SMALL_PLANE + 0.5,
            color=C_MPR, ha='left', va='bottom', fontweight='bold')
    fig.tight_layout(pad=0.35)
    mark_schematic(fig)
    save(fig, 'concept_op.png')


def fig_metric_auc():
    """追加の評価のスライド用：全域 HV の推移と AUC（anytime 性能）の模式図。AUC は全域 HV の推移の下の
    面積（対数時間）で，立ち上がりが速いほど大きい。縦軸・横軸・面積の意味はスライドの右の文で言う。
    重みごとに求めて全重みで平均するという算出の単位は，予備スライド（評価指標の詳細）で扱う。
    塗りは，全域 HV の面積の図と同じ淡い灰にする。"""
    fig, ax = plt.subplots(figsize=(6.6, 5.75))
    t = np.logspace(-1.3, 1.8, 300)
    fa = 0.78 / (1 + np.exp(-2.4 * (np.log10(t) + 0.45)))
    fb = 0.92 / (1 + np.exp(-2.6 * (np.log10(t) - 0.55)))
    face = HV_FACE
    ax.fill_between(t, 0, fa, color=face, lw=0)
    ax.plot(t, fa, color=INK, lw=2.6, label='探索A：立ち上がりが速い → AUC 大')
    ax.plot(t, fb, color='#8A949E', lw=2.6, ls='--', label='探索B：最終値は高いが遅い')
    ax.set_xscale('log')
    ax.set_ylim(0, 1.0)
    ax.set_xlabel('計算時間（対数）', fontsize=17)
    ax.set_ylabel('全域 HV', fontsize=17)
    ax.set_xticks([])
    ax.set_xticks([], minor=True)
    ax.set_yticks([])
    # 面積の名前は，凡例に塗りの見本を付けて示す（図中に文字で置くと，線の名前と読まれやすい）。
    # 見本は塗りが淡いので，縁を付けて白地の凡例でも見えるようにする
    h, l = ax.get_legend_handles_labels()
    h.append(plt.matplotlib.patches.Patch(facecolor=face, edgecolor='#A9B2BC', lw=0.8))
    l.append('探索A の AUC（曲線の下の面積）')
    ax.legend(h, l, loc='upper left', frameon=False, fontsize=16)
    despine(ax)
    fig.tight_layout(pad=0.4)
    mark_schematic(fig)
    save(fig, 'metric_auc.png')


# ====================================================================== 実データの図

TAG = {'la36_la36_small': 'la36S', 'ta21_ta21_delay97': 'ta21S', 'la40_la40_delay148': 'la40',
       'la21_la21_delay147': 'la21', 'la36_la36_middle': 'la36M', 'mt10_mt10_delay60': 'mt10',
       'la36_la36_large': 'la36L', 'ta21_ta21_high': 'ta21L'}
ORDER = list(TAG)  # 再スケジューリング率の昇順
HOSTS = [('単点探索（ILS）', ['ils_baseline', 'ils_repair', 'ils_pr'], C_ILS, M_ILS, True),
         ('多点探索（Memetic）', ['memetic_ls', 'memetic_repair', 'memetic_pr'], C_MEM, M_MEM, False)]
ALL7 = ['ils_baseline', 'ils_repair', 'ils_pr', 'ga', 'memetic_ls', 'memetic_repair', 'memetic_pr']


def load():
    sys.path.insert(0, V2)
    import make_figures_v2 as M  # noqa: E402  （import 時に rcParams を書き換えるので後で戻す）
    style()
    plt.rcParams.update(DATA_RC)
    return M, M.load_pkl()


def fig_front(M, S):
    M.PR_PERIOD_PT = 9.0  # スライド寸法では点線の周期を広げる（v2 の 6.3pt は紙面用）
    methods = [('ils_baseline', 'ILS', C_ILS, M_ILS, 4.0, '-'),
               ('ils_pr', 'ILS+PR', C_ILS, None, 3.0, M._pr_dash(3.0, 0.0)),
               ('memetic_ls', 'Memetic', C_MEM, M_MEM, 4.0, '-'),
               ('memetic_pr', 'Memetic+PR', C_MEM, None, 2.0, M._pr_dash(2.0, 0.5))]
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.2))
    for k, (ax, prob) in enumerate(zip(axes, ['ta21_ta21_delay97', 'ta21_ta21_high'])):
        p50 = S[prob]['thresholds']['P50']
        init_ms = float(S[prob]['init_ms'])
        drawn, xmax, ymin = [], 0.0, np.inf
        for key, label, color, mk, lw, ls in methods:
            pfs = [M.trial_pareto(prob, key, t) for t in range(10)]
            pf = pfs[M.median_trial(S, prob, key)]
            xmax = max(xmax, pf[:, 1].max())
            ymin = min(ymin, pf[:, 0].min())
            drawn.append((pf, label, color, mk, lw, ls))
        xhi = xmax * 1.08
        for pf, label, color, mk, lw, ls in drawn:
            ax.step(np.r_[pf[:, 1], xhi], np.r_[pf[:, 0], pf[-1, 0]], where='post', color=color,
                    lw=lw, ls=ls, alpha=0.40 if mk else 1.0, solid_capstyle='butt',
                    dash_capstyle='round', zorder=3 if mk else 5, label=None if mk else label)
            if mk:
                ax.scatter(pf[:, 1], pf[:, 0], s=48, marker=mk, color=color, edgecolor='black',
                           lw=0.5, zorder=6, label=label)
        ax.axvspan(-1e9, p50, facecolor=BAND_FACE, lw=0, zorder=0)
        ax.axvline(p50, color=BAND_EDGE, lw=1.1, ls=':', zorder=1)
        ax.plot([0], [init_ms], marker='*', ms=16, color='black', zorder=7, clip_on=False)
        span = init_ms - ymin
        ax.set_xlim(-0.03 * xhi, xhi)
        ax.set_ylim(ymin - 0.10 * span, init_ms + 0.10 * span)
        ax.text(p50 / 2, ymin - 0.085 * span, '高安定領域', ha='center', va='bottom',
                fontsize=16, color=BAND_EDGE)
        ax.yaxis.set_major_locator(plt.MaxNLocator(4))
        ax.set_title(f'({"ab"[k]}) {TAG[prob]}', loc='left')
        ax.set_xlabel('順序偏差 $D$')
        ax.grid(alpha=0.25)
        despine(ax)
        if k == 0:
            ax.set_ylabel('総所要時間 $MS$')
        if k == 1:
            ax.legend(loc='upper right', frameon=False, handlelength=2.6)
    tight(fig, pad=0.4, w_pad=2.0)
    save(fig, 'front_ta21.png')


BAND2 = '#93ABCB'   # 最良値の 2% 以内の帯（結果 2）


def _p_stars(p):
    return '*' if p >= 0.01 else ('**' if p >= 0.001 else '***')


def _highstab_panel(ax, per, title, band=False):
    """高安定 HV の 1 パネル（演算子なし・PR-k・PR × 探索構造）。全手法の中央値の最大（最良値）を 1 に
    正規化し，演算子なしとの差の有意水準を * で付ける。HOSTS の順に，最良値で割った中央値を返す。"""
    x = np.arange(3)
    best = max(np.median(np.asarray(per[m], float)) for m in ALL7 if len(per.get(m, [])))
    if band:
        ax.axhspan(0.98, 1.0, color=BAND2, lw=0, zorder=0)
    meds = []
    for host, keys, color, mk, above in HOSTS:
        vals = [np.asarray(per[m], float) / best for m in keys]
        med = np.array([np.median(v) for v in vals])
        lo = np.array([np.percentile(v, 25) for v in vals])
        hi = np.array([np.percentile(v, 75) for v in vals])
        ax.errorbar(x, med, yerr=[med - lo, hi - med], marker=mk, ms=8, color=color,
                    mec='white', mew=0.8, lw=2.4, capsize=3, elinewidth=1.0, label=host,
                    zorder=3)
        for j in (1, 2):
            p = mannwhitneyu(vals[j], vals[0], alternative='two-sided').pvalue
            if np.isfinite(p) and p < 0.05:
                ax.annotate(_p_stars(p), (x[j], med[j]), xytext=(0, 2 if above else -5),
                            textcoords='offset points', ha='center',
                            va='bottom' if above else 'top', fontsize=16, color=color)
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


def fig_highstab(M, S, band=False):
    """全 8 シナリオの高安定 HV．各シナリオで全手法の中央値の最大（最良値）を 1 に正規化する。"""
    fig, axes = plt.subplots(2, 4, figsize=(12.0, 3.85), sharey=True)
    for ax, prob in zip(axes.flat, ORDER):
        _highstab_panel(ax, S[prob]['highstab_hv_pt'], TAG[prob], band)
    h, l = axes[0, 0].get_legend_handles_labels()
    if band:
        h.append(plt.matplotlib.patches.Patch(color=BAND2))
        l.append('最良値の 2% 以内（帯）')
        fig.legend(h, l, loc='upper center', ncol=len(l), frameon=False, fontsize=16,
                   bbox_to_anchor=(0.5, 1.0), columnspacing=2.0)
    else:
        fig.legend(h, l, loc='upper left', ncol=len(l), frameon=False, fontsize=16,
                   bbox_to_anchor=(0.04, 1.0), columnspacing=2.0)
        fig.text(0.995, 0.975, '*：演算子なしとの差の有意水準（補正なし）',
                 ha='right', va='top', fontsize=15, color=INK)
    fig.text(0.005, 0.44, '高安定 HV（最良値＝1）', rotation=90, ha='left', va='center',
             fontsize=16, color=INK)
    tight(fig, pad=0.4, h_pad=0.5, w_pad=0.9, rect=(0.035, 0, 1, 0.885))
    save(fig, 'highstab_band.png' if band else 'highstab_all8.png')


CSV = os.path.join(V2, 'data', 'per_trial_metrics.csv')
CSV_METHOD = {'ILS-baseline': 'ils_baseline', 'ILS+repair': 'ils_repair', 'ILS+PR': 'ils_pr',
              'GA': 'ga', 'Memetic-LS': 'memetic_ls', 'Memetic+repair': 'memetic_repair',
              'Memetic+PR': 'memetic_pr'}


def load_csv():
    """公開データ（trial 別の 3 指標）を fig_highstab が読む形 {prob: {'highstab_hv_pt': {手法: 値}}}
    に読む。pkl がない環境（Linux での確認用ビルド）で帯の図を同じ統計で再生成するための代替。"""
    import csv
    S = {}
    with open(CSV, encoding='utf-8') as f:
        for row in csv.DictReader(f):
            S.setdefault(row['instance'], {}).setdefault('highstab_hv_pt', {}).setdefault(
                CSV_METHOD[row['method']], []).append(float(row['highstab_hv']))
    style()
    plt.rcParams.update(DATA_RC)
    return S


GUIDE = '#16385C'   # スライドの読み方のガイドの青（build_pptx.py の BLUE）


def fig_highstab4(S, tags=('la36S', 'ta21S', 'la36L', 'ta21L'), mark=('la36L', 'ta21L')):
    """演算子の効果のスライド用：全 8 シナリオの図と同じパネルを 4 つだけ 1 行に並べる。
    左 2 つは ILS が演算子なしで最良値にあるシナリオ，右 2 つは届いていないシナリオ。後者（mark）では，
    ILS の演算子なしの点を青い丸で囲む（スライドの箇条書きの「○」と対応）。パネルを全 8 シナリオの図より
    高くして，ILS の 94% → 100% の変化も読めるようにする。S は pkl でも load_csv() でもよい。"""
    probs = {TAG[p]: p for p in S if p in TAG}
    fig, axes = plt.subplots(1, len(tags), figsize=(12.0, 2.65), sharey=True)
    for ax, tag in zip(axes, tags):
        meds = _highstab_panel(ax, S[probs[tag]]['highstab_hv_pt'], tag)
        if tag in mark:
            ax.plot([0], [meds[0][0]], marker='o', ms=25, mfc='none', mec=GUIDE, mew=2.2, zorder=6)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc='upper left', ncol=len(l), frameon=False, fontsize=16,
               bbox_to_anchor=(0.04, 1.0), columnspacing=2.0)
    fig.text(0.995, 0.975, '*：演算子なしとの差の有意水準（補正なし）',
             ha='right', va='top', fontsize=15, color=INK)
    # 縦軸ラベルは全 8 シナリオの図と同じく図の左端に置く。1 行のパネルの高さには収まらないので 2 行に折る
    fig.text(0.006, 0.45, '高安定 HV\n（最良値＝1）', rotation=90, ha='left', va='center',
             fontsize=16, color=INK, linespacing=1.2)
    tight(fig, pad=0.4, w_pad=0.9, rect=(0.065, 0, 1, 0.86))
    save(fig, 'highstab_4.png')


def fig_anytime(M, S):
    style_ = [('ils_baseline', 'ILS', C_ILS, '-'), ('ils_repair', 'ILS+PR-$k$', C_ILS, (0, (5, 2))),
              ('ils_pr', 'ILS+PR', C_ILS, (0, (0.6, 2.2))),
              ('memetic_ls', 'Memetic', C_MEM, '-'),
              ('memetic_repair', 'Memetic+PR-$k$', C_MEM, (0, (5, 2))),
              ('memetic_pr', 'Memetic+PR', C_MEM, (0, (0.6, 2.2)))]
    methods = [s[0] for s in style_]
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.6))
    for k, (ax, prob) in enumerate(zip(axes, ['ta21_ta21_delay97', 'la36_la36_large'])):
        t, cur = M._anytime_curves(prob, methods + ['ga'])
        for key, label, color, ls in style_:
            ax.plot(t, cur[key][0], color=color, ls=ls, lw=2.8, dash_capstyle='round', label=label)
        ax.set_xscale('log')
        ax.set_xlim(t[0], M._anytime_cut(t, cur))
        ax.set_ylim(0, None)
        ax.set_title(f'({"ab"[k]}) {TAG[prob]}', loc='left')
        ax.set_xlabel('経過時間 [s]（対数）')
        ax.grid(alpha=0.25)
        despine(ax)
    axes[0].set_ylabel('全域 HV')
    from matplotlib.lines import Line2D
    gray = '#55606B'
    hs = [Line2D([], [], color=C_ILS, lw=2.6), Line2D([], [], color=C_MEM, lw=2.6),
          Line2D([], [], color=gray, lw=2.4),
          Line2D([], [], color=gray, lw=2.4, ls=(0, (5, 2))),
          Line2D([], [], color=gray, lw=2.4, ls=(0, (0.6, 2.2)), dash_capstyle='round')]
    fig.legend(hs, ['ILS', 'Memetic', '演算子なし', '+PR-$k$', '+PR'], loc='upper center',
               ncol=5, frameon=False, fontsize=16, handlelength=2.2, handletextpad=0.5,
               columnspacing=1.6, bbox_to_anchor=(0.5, 1.0))
    tight(fig, pad=0.4, w_pad=2.0, rect=(0, 0, 1, 0.885))
    save(fig, 'anytime.png')


if __name__ == '__main__':
    style()
    fig_concept_0()
    fig_concept_hv()
    fig_concept_sp()
    fig_concept_h(1)
    fig_concept_h(2)
    fig_concept_op()
    fig_struct('ils')
    fig_struct('mem')
    fig_metric_auc()
    if '--no-data' not in sys.argv:
        M, S = load()
        fig_front(M, S)
        fig_highstab(M, S, band=False)
        fig_highstab(M, S, band=True)
        fig_highstab4(S)
        fig_anytime(M, S)
    elif os.path.exists(CSV):
        S = load_csv()   # 高安定 HV の図は公開 CSV から同じ統計で作れる
        fig_highstab(None, S, band=False)
        fig_highstab4(S)
    print('done.')
