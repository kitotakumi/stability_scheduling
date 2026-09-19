# -*- coding: utf-8 -*-
"""発表スライド用の訪問密度差図（日本語・代表3シナリオ）を生成する。

APIEMS 原稿の fig_density_en と同じ計算・同じデータソース（main_v1 の raw）だが、
ラベルを日本語にし、スライドで大きく映す前提でフォント/線を太くしてある。

出力: assets/density_ja3.png
usage: python make_density_ja.py
"""
import os
import sys
import glob
import json

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
CORE = os.path.normpath(os.path.join(HERE, '..', '..', 'experiments', 'core_comparison_v3'))
sys.path.insert(0, CORE)
import analyze_v3 as A  # noqa: E402

RESULTS = os.path.join(CORE, 'results', 'main_v1')
ASSETS = os.path.join(HERE, 'assets')
os.makedirs(ASSETS, exist_ok=True)

plt.rcParams.update({
    'font.family': 'Meiryo',
    'axes.unicode_minus': False,
    'font.size': 13,
    'axes.titlesize': 16,
    'axes.labelsize': 13.5,
})

# 低 → 中 → 高の再スケ率ラダーになる代表3シナリオ
PROBS = ['la36_la36_small', 'la36_la36_middle', 'ta21_ta21_high']


def collect(prob, method):
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


def main(nbins=44):
    fig, axes = plt.subplots(1, len(PROBS), figsize=(13.2, 4.0))
    im = None
    for ax, prob in zip(axes, PROBS):
        Pi, Pm = collect(prob, 'ils_baseline'), collect(prob, 'memetic_ls')
        allp = np.vstack([Pi, Pm])
        ms_e = np.linspace(allp[:, 0].min(), allp[:, 0].max(), nbins + 1)
        d_e = np.linspace(0, allp[:, 1].max(), nbins + 1)
        Hi, _, _ = np.histogram2d(Pi[:, 0], Pi[:, 1], bins=[ms_e, d_e])
        Hm, _, _ = np.histogram2d(Pm[:, 0], Pm[:, 1], bins=[ms_e, d_e])
        diff = Hi / Hi.sum() - Hm / Hm.sum()
        D = (np.sign(diff) * np.sqrt(np.abs(diff))).T
        vmax = np.nanmax(np.abs(D)) or 1.0
        im = ax.imshow(D, origin='lower', aspect='auto', cmap='RdBu_r',
                       vmin=-vmax, vmax=vmax,
                       extent=[ms_e[0], ms_e[-1], d_e[0], d_e[-1]])
        for P, color, lab in [(Pi, 'darkorange', '軌道(ILS) のフロント'),
                              (Pm, 'green', '集団(Memetic) のフロント')]:
            pf = A.pareto_front(np.unique(P, axis=0))
            pf = pf[np.argsort(pf[:, 0])]
            ax.step(pf[:, 0], pf[:, 1], where='post', color=color, lw=2.6,
                    zorder=5, label=lab)
        rho = int(round(A.reschedule_rate(prob) * 100))
        ax.set_title(f'{A.problem_short_tag(prob)}（再スケ率 ρ={rho}%）', pad=8)
        ax.set_xlabel('メイクスパン MS（左＝効率良）')
        ax.tick_params(labelsize=11)
        ax.xaxis.set_major_locator(plt.MaxNLocator(4))
    axes[0].set_ylabel('順位偏差 D（下＝安定）')
    axes[0].legend(loc='upper left', fontsize=11, framealpha=0.92)
    fig.subplots_adjust(left=0.062, right=0.855, bottom=0.155, top=0.90, wspace=0.22)
    cax = fig.add_axes([0.870, 0.155, 0.016, 0.745])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label('訪問密度差\n赤＝軌道(ILS)が密／青＝集団(Memetic)が密', fontsize=11.5)
    cb.ax.tick_params(labelsize=10)
    out = os.path.join(ASSETS, 'density_ja3.png')
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print('wrote', out)


if __name__ == '__main__':
    main()
