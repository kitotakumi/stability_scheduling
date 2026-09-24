#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""JIMA2026 秋季大会 予稿用の図を生成する（段幅 78mm 想定・日本語）。

出力: figures/fig_highstab_ja.png   高安定 HV（2 構造 × 演算子の有無 × 8 シナリオ）
      figures/fig_union_ja.png      全域 HV（同上・本文では未使用）
      figures/_gray/                白黒校正

符号化は APIEMS v2（doc/paper_apiems2026_v2/make_figures_v2.py）と揃える:
  探索構造 = 色相＋グレー明度（単点探索: 橙 L601=130 / 多点探索: 緑 L601=63）
  演算子   = 塗りのパターン（なし: ベタ / PR: 白の斜線）
データ: experiments/core_comparison_v3/results/main_v1/analysis/_summary_data.pkl
usage: python make_figures_ja.py
"""
import os
import pickle

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
PKL = os.path.normpath(os.path.join(
    HERE, '..', '..', 'experiments', 'core_comparison_v3', 'results',
    'main_v1', 'analysis', '_summary_data.pkl'))
FIG = os.path.join(HERE, 'figures')
os.makedirs(FIG, exist_ok=True)

# 再スケジューリング率 rho の昇順
ORDER = [('la36_la36_small', 'la36S', 27), ('ta21_ta21_delay97', 'ta21S', 32),
         ('la40_la40_delay148', 'la40', 32), ('la21_la21_delay147', 'la21', 35),
         ('la36_la36_middle', 'la36M', 54), ('mt10_mt10_delay60', 'mt10', 72),
         ('la36_la36_large', 'la36L', 73), ('ta21_ta21_high', 'ta21L', 82)]

C_ILS, C_MEM = '#e06c00', '#0f5c28'  # L601 = 130 / 63
SERIES = [('ils_baseline', '単点探索', C_ILS, ''),
          ('ils_pr', '単点探索＋PR', C_ILS, '////'),
          ('memetic_ls', '多点探索', C_MEM, ''),
          ('memetic_pr', '多点探索＋PR', C_MEM, '////')]

plt.rcParams.update({
    'font.family': 'Meiryo',
    'axes.unicode_minus': False,
    'font.size': 7,
    'axes.labelsize': 7.5,
    'legend.fontsize': 6.0,
    'xtick.labelsize': 6.8,
    'ytick.labelsize': 6.8,
    'hatch.linewidth': 0.45,
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


def draw(key, ylabel, out, zero_note=False):
    d = pickle.load(open(PKL, 'rb'))
    x = np.arange(len(ORDER))
    w = 0.21
    fig, ax = plt.subplots(figsize=(3.15, 1.52), dpi=400)
    for i, (m, lab, c, hatch) in enumerate(SERIES):
        v = [float(np.median(d[k][key][m])) for k, _, _ in ORDER]
        ax.bar(x + (i - 1.5) * w, v, w, label=lab, color=c, hatch=hatch,
               edgecolor='white', linewidth=0.3)
        if zero_note:
            for xi, vi in zip(x, v):
                if vi == 0.0:
                    ax.text(xi + (i - 1.5) * w, 0.0012, '0', ha='center',
                            va='bottom', fontsize=6, color=c)
    ax.set_xticks(x)
    ax.set_xticklabels([f'{lab}\n{r}%' for _, lab, r in ORDER])
    ax.set_ylabel(ylabel)
    ax.tick_params(axis='both', length=2, pad=1.5)
    ax.grid(axis='y', lw=0.4, color='#DDDDDD')
    ax.set_axisbelow(True)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'):
        ax.spines[s].set_linewidth(0.6)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.30), ncol=2,
              frameon=False, handlelength=1.4, handleheight=0.9,
              columnspacing=1.2, borderpad=0.0, labelspacing=0.2)
    fig.tight_layout(pad=0.25)
    fig.subplots_adjust(top=0.79)
    save(fig, out)


if __name__ == '__main__':
    draw('highstab_hv_pt', '高安定HV（中央値）', 'fig_highstab_ja.png',
         zero_note=True)
    draw('union_hv_pt', '全域HV（中央値）', 'fig_union_ja.png')
