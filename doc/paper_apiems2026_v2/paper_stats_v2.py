#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""原稿 v2 の本文数値を main_v1 の raw / _summary_data.pkl から再計算する。

図の生成（make_figures_v2.py）と同じデータ・同じ前処理を使い、図に出ていない本文の数値だけを
まとめて出す。解析単位は trial（1 trial = 10 重みの掃引全体）で、代表値は trial の中央値。

  [A] §4.2 構造的原因：訪問した解（局所探索の開始点と終了点）の最小 D、
      高安定領域 0<D<P50 で訪問した異なる解の数と、訪問した異なる解全体に占める割合
  [B] §4.3 統合 HV：演算子込み Memetic と ILS-baseline の差（%）
  [C] §4.1 計算時間：1 run の壁時計時間と Memetic+PR / ILS-baseline の比
  [D] 図4 キャプション：始点単位の改善割合と、改善が出た trial 数

usage: python paper_stats_v2.py
"""
import os
import re
import sys
import glob
import json

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import make_figures_v2 as M  # noqa: E402  RESULTS・analyze_v3・PR 発動の再現を共有

A = M.A
TRIAL_PAT = re.compile(r'__t(\d+)\.json$')


def _iter_raw(prob, method):
    yield from glob.glob(os.path.join(M.RESULTS, prob, 'raw', f'{method}__*.json'))


def _by_trial_points(prob, method):
    """trial ごとに 10 重みの UEA 点を合併した (MS, D) 配列を返す（= union HV と同じ点集合）。"""
    acc = {}
    for p in _iter_raw(prob, method):
        pts = np.asarray(json.load(open(p, encoding='utf-8')).get('uea_points') or [], float)
        if len(pts):
            acc.setdefault(TRIAL_PAT.search(p).group(1), []).append(pts)
    return {t: np.unique(np.concatenate(v), axis=0) for t, v in acc.items()}


def _fmt(vals, fmt='{:.0f}'):
    return f'{fmt.format(np.median(vals))} [{fmt.format(np.min(vals))}-{fmt.format(np.max(vals))}]'


def section_a(S, probs, methods=('ils_baseline', 'memetic_ls', 'memetic_pr')):
    print('\n[A] §4.2 構造的原因（trial 単位・中央値 [最小-最大]）')
    print(f'{"scen":7s}{"P50":>5s}  {"method":13s}{"min D":>12s}{"distinct in band":>20s}{"share %":>16s}')
    span = {m: {'mind': [], 'low': [], 'share': []} for m in methods}
    for prob in probs:
        p50 = S[prob]['thresholds']['P50']
        for m in methods:
            mind, low, share = [], [], []
            for _t, u in _by_trial_points(prob, m).items():
                d = u[:, 1]
                pos = d[d > 0]                      # D=0 は自明解（HV からも除く）
                n_low = int(((d > 0) & (d < p50)).sum())
                mind.append(pos.min()); low.append(n_low)
                share.append(100 * n_low / len(pos))
            print(f'{A.problem_short_tag(prob):7s}{p50:5.0f}  {M.A.METHOD_LABELS.get(m, m):13s}'
                  f'{_fmt(mind):>12s}{_fmt(low):>20s}{_fmt(share, "{:.1f}"):>16s}')
            span[m]['mind'].append(np.median(mind)); span[m]['low'].append(np.median(low))
            span[m]['share'].append(np.median(share))
    print('  8 シナリオの trial 中央値の範囲:')
    for m in methods:
        s = span[m]
        print(f'    {M.A.METHOD_LABELS.get(m, m):13s} min D {min(s["mind"]):.0f}-{max(s["mind"]):.0f} / '
              f'distinct {min(s["low"]):.0f}-{max(s["low"]):.0f} / share {min(s["share"]):.1f}-{max(s["share"]):.1f}%')


def section_b(S, probs, base='ils_baseline', ops=('memetic_pr', 'memetic_repair')):
    print('\n[B] §4.3 統合 HV: 演算子込み Memetic の ILS-baseline に対する差（trial 中央値の比, %）')
    print(f'{"scen":7s}' + ''.join(f'{M.A.METHOD_LABELS.get(m, m):>18s}' for m in ops))
    for prob in probs:
        b = np.median(S[prob]['union_hv_pt'][base])
        cells = ''.join(f'{100 * (np.median(S[prob]["union_hv_pt"][m]) / b - 1):+17.1f}%' for m in ops)
        print(f'{A.problem_short_tag(prob):7s}{cells}')


def section_c(probs, base='ils_baseline', other='memetic_pr'):
    print('\n[C] §4.1 計算時間（1 run の壁時計、中央値）')
    print(f'{"scen":7s}{"ILS min":>10s}{"Mem+PR min":>12s}{"ratio":>8s}')
    ratios = []
    for prob in probs:
        med = {}
        for m in (base, other):
            ts = [json.load(open(p, encoding='utf-8'))['convergence']['total_cpu_time']
                  for p in _iter_raw(prob, m)]
            med[m] = float(np.median(ts))
        r = med[other] / med[base]
        ratios.append(r)
        print(f'{A.problem_short_tag(prob):7s}{med[base]/60:10.1f}{med[other]/60:12.1f}{r:8.1f}')
    print(f'  比の範囲: {min(ratios):.1f}-{max(ratios):.1f} 倍')


def section_d(probs):
    print('\n[D] 図4: 始点単位の改善割合（全 trial・全重み合算）と、改善が出た trial 数')
    print(f'{"scen":7s}{"ILS starts":>11s}{"ILS rate":>10s}{"trials>0":>10s}{"Mem starts":>12s}{"Mem rate":>10s}')
    for prob in probs:
        by_trial = {}
        for p in _iter_raw(prob, 'ils_pr'):
            g = M._ils_pr_by_start(json.load(open(p, encoding='utf-8')))
            if g is None:
                raise RuntimeError(f'PR 発動の再現が保存統計と一致しない: {p}')
            t = TRIAL_PAT.search(p).group(1)
            by_trial.setdefault(t, []).extend(r for r in g if r[0] > 0)
        n = sum(len(v) for v in by_trial.values())
        k = sum(r[1] for v in by_trial.values() for r in v)
        n_tr = sum(1 for v in by_trial.values() if any(r[1] for r in v))
        md0, mimp = [], []
        for p in _iter_raw(prob, 'memetic_pr'):
            ms = json.load(open(p, encoding='utf-8')).get('mech_stats') or {}
            d0 = np.asarray(ms.get('pr_d0', [])); imp = np.asarray(ms.get('pr_improved', []))
            sel = d0 > 0
            md0.append(int(sel.sum())); mimp.append(int(imp[sel].sum()))
        print(f'{A.problem_short_tag(prob):7s}{n:11d}{100 * k / n:9.1f}%{n_tr:9d}/10'
              f'{sum(md0):12d}{100 * sum(mimp) / sum(md0):9.1f}%')


if __name__ == '__main__':
    S = M.load_pkl()
    probs = A.order_prob_labels(S.keys())
    section_a(S, probs)
    section_b(S, probs)
    section_c(probs)
    section_d(probs)
