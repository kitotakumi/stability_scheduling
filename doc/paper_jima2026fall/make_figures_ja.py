#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""JIMA2026 秋季大会 予稿用の図を生成する（段幅 78mm 想定・日本語）。

出力: figures/fig_highstab_ja.png   高安定 HV（3手法 × 8シナリオ）
      figures/fig_union_ja.png      統合 HV（3手法 × 8シナリオ）
データ: experiments/core_comparison_v3/results/main_v1/analysis/_summary_data.pkl
usage: python make_figures_ja.py
"""
import os
import pickle

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

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

SERIES = [('ils_baseline', '単点探索 (ILS)', '#E07B27'),
          ('memetic_ls', '多点探索 (Memetic)', '#3E9B4F'),
          ('memetic_pr', '多点探索＋PR', '#5B5BD6')]

plt.rcParams.update({
    'font.family': 'Meiryo',
    'axes.unicode_minus': False,
    'font.size': 7,
    'axes.labelsize': 7.5,
    'legend.fontsize': 6.0,
    'xtick.labelsize': 6.8,
    'ytick.labelsize': 6.8,
})


def draw(key, ylabel, out, zero_note=False):
    d = pickle.load(open(PKL, 'rb'))
    x = np.arange(len(ORDER))
    w = 0.27
    fig, ax = plt.subplots(figsize=(3.15, 1.62), dpi=400)
    for i, (m, lab, c) in enumerate(SERIES):
        v = [float(np.median(d[k][key][m])) for k, _, _ in ORDER]
        ax.bar(x + (i - 1) * w, v, w, label=lab, color=c,
               edgecolor='white', linewidth=0.3)
        if zero_note:
            for xi, vi in zip(x, v):
                if vi == 0.0:
                    ax.text(xi + (i - 1) * w, 0.0012, '0', ha='center',
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
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.19), ncol=3,
              frameon=False, handlelength=1.1, handleheight=0.8,
              columnspacing=1.0, borderpad=0.0, labelspacing=0.25)
    fig.tight_layout(pad=0.25)
    fig.subplots_adjust(top=0.86)
    fig.savefig(os.path.join(FIG, out), dpi=400)
    plt.close(fig)
    print(' ->', out)


if __name__ == '__main__':
    draw('highstab_hv_pt', '高安定HV（中央値）', 'fig_highstab_ja.png',
         zero_note=True)
    draw('union_hv_pt', '統合HV（中央値）', 'fig_union_ja.png')
