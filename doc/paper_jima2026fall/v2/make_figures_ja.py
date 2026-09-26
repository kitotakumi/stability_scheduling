#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""JIMA2026 秋季大会 予稿用の図を生成する（段幅 78mm 想定・日本語）。

出力: figures/fig_highstab_ja.png   高安定 HV の交互作用図（2 構造 × 演算子 3 水準 × 8 シナリオ）
      figures/_gray/                白黒校正

APIEMS v2 の図3（doc/paper_apiems2026_v2/make_figures_v2.py の fig_interaction）を
段幅 1 段に収めた版。符号化は v2 と揃える:
  探索構造 = 色相＋グレー明度＋マーカー形状（単点探索: 橙 L601=130・○／多点探索: 緑 L601=63・□）
  演算子   = 横軸の 3 水準（なし／repair／PR）
有意記号は演算子なしに対する両側 Mann-Whitney U 検定（多重比較補正なし）。置き場所は
構造で固定する（単点探索は線の上・多点探索は線の下）。どちらが上かで決めると、演算子を
載せた後の中央値がほぼ一致するシナリオで上下が入れ替わり、読み手が対応を取れなくなるため。
データ: experiments/core_comparison_v3/results/main_v1/analysis/_summary_data.pkl
usage: python make_figures_ja.py
"""
import os
import pickle

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
PKL = os.path.normpath(os.path.join(
    HERE, '..', '..', '..', 'experiments', 'core_comparison_v3', 'results',
    'main_v1', 'analysis', '_summary_data.pkl'))
FIG = os.path.join(HERE, 'figures')
os.makedirs(FIG, exist_ok=True)

# 再スケジューリング率 rho の昇順（rho は予稿本文では §4.1 のシナリオ記述にだけ出る）
ORDER = [('la36_la36_small', 'la36S'), ('ta21_ta21_delay97', 'ta21S'),
         ('la40_la40_delay148', 'la40'), ('la21_la21_delay147', 'la21'),
         ('la36_la36_middle', 'la36M'), ('mt10_mt10_delay60', 'mt10'),
         ('la36_la36_large', 'la36L'), ('ta21_ta21_high', 'ta21L')]

C_ILS, C_MEM = '#e06c00', '#0f5c28'  # L601 = 130 / 63
HOSTS = [('単点探索', ['ils_baseline', 'ils_repair', 'ils_pr'], C_ILS, 'o', True),
         ('多点探索', ['memetic_ls', 'memetic_repair', 'memetic_pr'], C_MEM, 's', False)]

plt.rcParams.update({
    'font.family': 'Meiryo',
    'axes.unicode_minus': False,
    'font.size': 6,
    'axes.titlesize': 6.4,
    'axes.labelsize': 6.5,
    'legend.fontsize': 5.8,
})


def save(fig, out):
    """保存してから上下の余白を 1pt に切り詰める（横幅は配置倍率に効くので切らない）。"""
    path = os.path.join(FIG, out)
    fig.savefig(path, dpi=400)
    plt.close(fig)
    im = Image.open(path)
    a = np.asarray(im.convert('L'))
    rows = np.where((a < 250).any(axis=1))[0]
    if len(rows):
        m = max(1, round(400 / 72.0))
        top, bot = max(0, rows[0] - m), min(a.shape[0], rows[-1] + 1 + m)
        if (top, bot) != (0, a.shape[0]):
            im.crop((0, top, im.width, bot)).save(path, dpi=(400, 400))
    gray = os.path.join(FIG, '_gray')
    os.makedirs(gray, exist_ok=True)
    Image.open(path).convert('L').save(os.path.join(gray, out))
    print(' ->', out)


def draw_interaction(out='fig_highstab_ja.png'):
    d = pickle.load(open(PKL, 'rb'))
    x = np.arange(3)
    fig, axes = plt.subplots(2, 4, figsize=(3.15, 1.66), dpi=400)
    for i, (ax, (key, tag)) in enumerate(zip(axes.flat, ORDER)):
        per = d[key]['highstab_hv_pt']
        ymax, stars = 0.0, []
        for host, methods, color, mk, above in HOSTS:
            vals = [np.asarray(per[m], float) for m in methods]
            med = np.array([np.median(v) for v in vals])
            lo = np.array([np.percentile(v, 25) for v in vals])
            hi = np.array([np.percentile(v, 75) for v in vals])
            ymax = max(ymax, float(hi.max()))
            ax.errorbar(x, med, yerr=[med - lo, hi - med], marker=mk, ms=2.6,
                        color=color, mec='white', mew=0.4, lw=1.0, capsize=1.2,
                        elinewidth=0.5, label=host)
            for j in (1, 2):
                p = mannwhitneyu(vals[j], vals[0], alternative='two-sided').pvalue
                if np.isfinite(p) and p < 0.05:
                    stars.append((x[j], med[j], above, color,
                                  '*' if p >= 0.01 else ('**' if p >= 0.001 else '***')))
        ax.set_title(tag, pad=1.5)
        ax.set_xticks(x)
        # 目盛ラベルが隣と触れるので、x 方向の余白を詰めて 3 水準の間隔を広げる
        ax.set_xticklabels(['なし', 'repair', 'PR'], fontsize=4.8)
        ax.set_xlim(-0.36, 2.36)
        ax.set_ylim(0, ymax * 1.34)
        for xs, ys, above, color, txt in stars:
            ax.annotate(txt, (xs, ys), xytext=(0, 1.5 if above else -5.5),
                        textcoords='offset points', ha='center',
                        va='bottom' if above else 'top', fontsize=5.0, color=color)
        ax.tick_params(labelsize=5.0, length=1.5, pad=1)
        ax.yaxis.set_major_locator(plt.MaxNLocator(3))
        ax.grid(axis='y', lw=0.3, color='#DDDDDD')
        ax.set_axisbelow(True)
        for s in ('top', 'right'):
            ax.spines[s].set_visible(False)
        for s in ('left', 'bottom'):
            ax.spines[s].set_linewidth(0.5)
        if i % 4 == 0:
            ax.set_ylabel('高安定HV')
        if i < 4:
            ax.tick_params(labelbottom=False)
    # 凡例はパネル内に置くとどれかの系列に必ず重なるので図の上端に出す
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=2, frameon=False,
               handlelength=1.2, handletextpad=0.3, columnspacing=1.6,
               borderpad=0.0, bbox_to_anchor=(0.5, 1.004))
    fig.tight_layout(pad=0.2, h_pad=0.45, w_pad=0.5, rect=(0, 0, 1, 0.915))
    save(fig, out)


if __name__ == '__main__':
    draw_interaction()
