#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""パッチ済み _summary_data.pkl から AOC の (i)§4.3 ILS-base vs Mem-LS 表 と
(ii)付録A.3 7手法×6問題スコアボード表 を doc 貼り付け用に出力する。"""
import os, sys, pickle
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import analyze_v3 as A

S = pickle.load(open(os.path.join('results','main_v1','analysis','_summary_data.pkl'),'rb'))
probs = sorted(S.keys())
print('aoc_weight =', S[probs[0]].get('aoc_weight'))

# (i) §4.3: ILS-baseline vs Memetic-LS per-problem
print('\n=== §4.3 AOC 表（ILS-baseline vs Memetic-LS）===')
recs = A._pairwise_per_problem(S, probs, 'aoc_pt', 'ils_baseline', 'memetic_ls')
print('| 問題 | ILS-baseline | Memetic-LS | 検定（片側 ILS>Mem） |')
print('|---|---|---|---|')
for r in recs:
    am, bm, d = r['a_med'], r['b_med'], r['d']
    if am > bm:
        v = f"$p$={r['p_ab']:.3f}, $\\delta$={d:+.2f}（ILS優）"
    else:
        v = f"Memetic 僅差・$p$={r['p_ba']:.3f}, $\\delta$={d:+.2f}"
    print(f"| {r['tag']} | {am:.3f} | {bm:.3f} | {v} |")

# (ii) 付録A.3: 7手法×6問題 + Friedman順位 + ARPD（順位ソート）
print('\n=== 付録A.3 AOC スコアボード（順位ソート）===')
order0 = [m for m in A.METHOD_ORDER]
methods = [m for m in order0 if any(m in S[p].get('aoc_pt',{}) for p in probs)]
tags = [A.problem_short_tag(p) for p in probs]
M = A._metric_matrix(S, probs, methods, 'aoc_pt')
valid = ~np.any(np.isnan(M), axis=1); Mv = M[valid]
used = [t for t,ok in zip(tags,valid) if ok]
avg_rank, chi, p, W, ranks = A._friedman_avg_rank(Mv)
am_, amed_ = A._arpd_pct(Mv)
o = list(np.argsort(avg_rank))
print(f'Friedman p={p:.4f}, W={W:.2f}')
print('| 手法 | ' + ' | '.join(used) + ' | Friedman順位 | ARPD%(平/中) |')
print('|---|' + '---|'*(len(used)+2))
for rk, j in enumerate(o, 1):
    cells = ' | '.join(f'{Mv[i,j]:.3f}' for i in range(Mv.shape[0]))
    name = A.METHOD_LABELS.get(methods[j], methods[j])
    star = '**' if rk==1 else ''
    print(f'| {star}{name}{star} | {cells} | {star}{avg_rank[j]:.2f} {A._circled(rk)}{star} | {am_[j]:.1f} / {amed_[j]:.1f} |')
