#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""統合スコアボード図: 指標ごと（統合HV/高安定HV/AOC）に
   [手法 × 問題] の「問題ごと相対偏差RPD%」ヒートマップ ＋ 右端に [Friedman順位 / ARPD%]。
   手法数・問題数はキャッシュから動的に決まる（問題追加に追従）。

analyze_v3 が summary.md 用に保存する all_summary キャッシュ（_summary_data.pkl）を読み、
スコアボードと同一の _metric_matrix / _friedman_avg_rank / _arpd_pct を再利用するので、
図の数値は summary.md / research_document の横断サマリ表と完全一致する。

セル = その問題の最良比からの相対偏差 (1 − v/best)×100%（色＝緑が最良 0%、赤が劣化大）。
ARPD列 = そのセル群の平均（同単位なので同カラースケールで着色）。順位列 = Friedman平均順位。

usage: python scoreboard_heatmap.py [results_dir]
"""
import os
import sys
import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import analyze_v3 as A

_installed = {f.name for f in font_manager.fontManager.ttflist}
for _jp in ('Yu Gothic', 'Meiryo', 'MS Gothic', 'Noto Sans CJK JP', 'IPAexGothic'):
    if _jp in _installed:
        plt.rcParams['font.family'] = _jp
        break
plt.rcParams['axes.unicode_minus'] = False

# 手法の表示順（GA → Memetic 3種 → ILS 3種）
METHOD_ORDER = ['ga', 'memetic_ls', 'memetic_repair', 'memetic_pr',
                'ils_baseline', 'ils_repair', 'ils_pr']


def _per_problem_rpd(M):
    """(N_prob, k) 中央値行列 → 各問題の最良比 RPD% (= (1 − v/best)×100, best=行内max)。
    全手法 0 の退化問題（best<=0, 例 la36S 高安定）は NaN（白）にする。
    これは analyze_v3._arpd_pct のセル単位値そのもの＝ARPD列はこの平均に一致する。"""
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


def main(results_dir):
    cache = os.path.join(results_dir, 'analysis', '_summary_data.pkl')
    with open(cache, 'rb') as f:
        all_summary = pickle.load(f)

    prob_labels = A.order_prob_labels(all_summary.keys())  # 難易度(リスケ率昇順)
    present = set()
    for pl in prob_labels:
        present.update(all_summary[pl].get('methods', []))
    methods = [m for m in METHOD_ORDER if m in present]
    labels = [A.METHOD_LABELS.get(m, m) for m in methods]
    tags = [A.problem_short_tag(pl) for pl in prob_labels]
    aoc_w = None
    for pl in prob_labels:
        aoc_w = all_summary[pl].get('aoc_weight') or aoc_w

    outdir = os.path.join(results_dir, 'analysis', 'heatmap')
    os.makedirs(outdir, exist_ok=True)

    # 指標ごとに 1 枚ずつ（その指標の Friedman 平均順位で手法を昇順ソート＝最良が上）
    for key, label, arr_key in A.SCOREBOARD_METRICS:
        M = A._metric_matrix(all_summary, prob_labels, methods, arr_key)  # (N_prob, k)
        valid = ~np.any(np.isnan(M), axis=1)
        Mv = M[valid]
        used_tags = [t for t, ok in zip(tags, valid) if ok]
        if Mv.shape[0] < 2:
            continue
        avg_rank, chi, p, W, ranks = A._friedman_avg_rank(Mv)
        arpd_mean, arpd_med = A._arpd_pct(Mv)
        RPD = _per_problem_rpd(Mv)            # (N_valid, k)

        # その指標の平均順位で手法を昇順ソート（best→worst, 上→下）
        order = list(np.argsort(avg_rank, kind='stable'))
        s_methods = [methods[j] for j in order]
        s_labels = [A.METHOD_LABELS.get(m, m) for m in s_methods]
        s_avg = [avg_rank[j] for j in order]
        s_am = [arpd_mean[j] for j in order]
        s_amed = [arpd_med[j] for j in order]
        s_RPD = RPD[:, order]                 # (N_valid, nM_sorted)
        nM, nP = len(s_methods), len(used_tags)

        # 表示行列: 問題列(RPD.T) + 順位列(NaN=色なし) + ARPD列(arpd_mean を同単位で着色)
        D = np.full((nM, nP + 2), np.nan)
        D[:, :nP] = s_RPD.T
        D[:, nP + 1] = np.array(s_am)
        vmax = np.nanmax(D) if np.any(np.isfinite(D)) else 1.0
        cmap = plt.get_cmap('RdYlGn_r').copy(); cmap.set_bad('white')

        fig, ax = plt.subplots(figsize=(1.05 * (nP + 2) + 3.0, 0.55 * nM + 1.6))
        im = ax.imshow(D, aspect='auto', cmap=cmap, vmin=0, vmax=max(vmax, 1.0))
        ax.axvline(nP - 0.5, color='black', lw=1.5)   # 問題列 と 集約列 の区切り
        ax.set_xticks(range(nP + 2))
        ax.set_xticklabels(used_tags + ['順位', 'ARPD%'], fontsize=9)
        ax.set_yticks(range(nM)); ax.set_yticklabels(s_labels, fontsize=9)
        ttl = label + (f'  (AOC weight={aoc_w})' if key == 'aoc' else '')
        ax.set_title(f'{ttl}   —   Friedman p={p:.4f}, W={W:.2f}  '
                     f'（順位ソート, 緑=その問題の最良, 数字=RPD%）', fontsize=10)

        def _txtcolor(v):
            return 'white' if (np.isfinite(v) and v > 60) else 'black'

        for i in range(nM):
            for j in range(nP):
                v = D[i, j]
                t = f'{v:.0f}' if np.isfinite(v) else '·'
                ax.text(j, i, t, ha='center', va='center', fontsize=8, color=_txtcolor(v))
            ax.text(nP, i, f'{s_avg[i]:.2f}{A._circled(rank_order_for(avg_rank, order[i]))}',
                    ha='center', va='center', fontsize=8.5, color='black')
            ax.text(nP + 1, i, f'{s_am[i]:.0f}/{s_amed[i]:.0f}',
                    ha='center', va='center', fontsize=8, color=_txtcolor(s_am[i]))

        fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02, label='RPD% (= (1−v/best)×100)')
        fig.tight_layout()
        out = os.path.join(outdir, f'scoreboard_{key}.png')
        fig.savefig(out, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f'  -> {out}')


def rank_order_for(avg_rank, i):
    """avg_rank 配列で手法 i が何位か（1=最小=最良, タイは同順）。"""
    return int(np.sum(avg_rank < avg_rank[i]) + 1)


if __name__ == '__main__':
    rd = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, 'results', 'main_v1')
    main(rd)
