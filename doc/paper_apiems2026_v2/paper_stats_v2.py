#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""原稿 v2 の本文数値を main_v1 の raw / _summary_data.pkl から再計算する。

図の生成（make_figures_v2.py）と同じデータ・同じ前処理を使い、図に出ていない本文の数値だけを
まとめて出す。解析単位は trial（1 trial = 10 重みの掃引全体）で、代表値は trial の中央値。

  [B] §4.3 全域 HV：演算子込み Memetic と ILS-baseline の差（%）
  [C] §4.1 計算時間：1 run の壁時計時間と Memetic+PR / ILS-baseline の比
  [E] 表 2：Friedman 平均順位と ARPD%（7 手法 × 3 指標）を markdown で出力
  [F] §4.3 の検定数：全域 HV・AOC について、構造内（演算子 vs なし）・構造間（同じ演算子で
      ILS vs Memetic）・Memetic 内（repair vs PR）の両側 Mann–Whitney U と Cliff's δ

検討用（本文では使わない。--review で出力）:
  [A] 訪問した解（局所探索の開始点と終了点）の最小 D と、高安定領域 0<D<P50 で訪問した
      異なる解の数・割合。フロントの左端が生成解の最小 D に一致するため本文では図 2 と
      高安定 HV に吸収した
  [D] 旧図4（PR 経路統計）キャプション：始点単位の改善割合と、改善が出た trial 数
  [G] --export: trial 単位の 3 指標を data/per_trial_metrics.csv に書き出す（§4.1 の公開データ）

usage: python paper_stats_v2.py [--review] [--export]
"""
import os
import re
import sys
import glob
import json

import numpy as np
from scipy.stats import mannwhitneyu

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
            print(f'{A.problem_short_tag(prob):7s}{p50:5.0f}  {M.METHOD_LABELS.get(m, m):13s}'
                  f'{_fmt(mind):>12s}{_fmt(low):>20s}{_fmt(share, "{:.1f}"):>16s}')
            span[m]['mind'].append(np.median(mind)); span[m]['low'].append(np.median(low))
            span[m]['share'].append(np.median(share))
    print('  8 シナリオの trial 中央値の範囲:')
    for m in methods:
        s = span[m]
        print(f'    {M.METHOD_LABELS.get(m, m):13s} min D {min(s["mind"]):.0f}-{max(s["mind"]):.0f} / '
              f'distinct {min(s["low"]):.0f}-{max(s["low"]):.0f} / share {min(s["share"]):.1f}-{max(s["share"]):.1f}%')


def section_b(S, probs, base='ils_baseline', ops=('memetic_pr', 'memetic_repair')):
    print('\n[B] §4.3 全域 HV: 演算子込み Memetic の ILS-baseline に対する差（trial 中央値の比, %）')
    print(f'{"scen":7s}' + ''.join(f'{M.METHOD_LABELS.get(m, m):>18s}' for m in ops))
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


TABLE_ORDER = ['ils_baseline', 'ils_repair', 'ils_pr',
               'ga', 'memetic_ls', 'memetic_repair', 'memetic_pr']
METRIC_LABELS = {'union_hv_pt': '全域 HV', 'highstab_hv_pt': '高安定 HV', 'aoc_pt': 'AOC'}


def _cliffs_delta(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    gt = (a[:, None] > b[None, :]).sum()
    lt = (a[:, None] < b[None, :]).sum()
    return (gt - lt) / (len(a) * len(b))


def _mwu(a, b):
    """両側 Mann–Whitney U の p 値と Cliff's δ（a が b より大きいとき δ>0）。"""
    return mannwhitneyu(a, b, alternative='two-sided').pvalue, _cliffs_delta(a, b)


def _near_best_counts(S, probs, key):
    """各手法について「そのシナリオの最良手法と有意差がない」シナリオ数を返す（両側 MWU, alpha=0.05）。

    最良＝trial 中央値が最大の手法。最良手法自身も 1 と数える。多重比較の補正はかけない
    （横断の記述統計として扱う）。
    """
    cnt = {m: 0 for m in TABLE_ORDER}
    for prob in probs:
        per = S[prob][key]
        med = {m: np.median(np.asarray(per[m], float)) for m in TABLE_ORDER}
        best = max(med, key=med.get)
        for m in TABLE_ORDER:
            cnt[m] += (m == best) or (
                mannwhitneyu(np.asarray(per[m], float), np.asarray(per[best], float),
                             alternative='two-sided').pvalue >= 0.05)
    return cnt


def section_e(S, probs):
    print(chr(10) + '[E] 表 2: 平均順位／ARPD%（平均）／最良群シナリオ数。列は 全域 HV / 高安定 HV / AOC')
    cols = {}
    for key in ('union_hv_pt', 'highstab_hv_pt', 'aoc_pt'):
        Mx = A._metric_matrix(S, probs, TABLE_ORDER, key)
        avg_rank, _chi, pv, W, _ranks = A._friedman_avg_rank(Mx)
        arpd_mean, arpd_med = A._arpd_pct(Mx)
        pos = {j: int(np.sum(avg_rank < avg_rank[j] - 1e-12) + 1) for j in range(len(TABLE_ORDER))}
        cols[key] = (avg_rank, pos, arpd_mean, arpd_med, pv, W, _near_best_counts(S, probs, key))
        print(f'  {METRIC_LABELS[key]}: Friedman p={pv:.2g}, Kendall W={W:.2f}')
    print('| 手法 | ' + ' | '.join(METRIC_LABELS[k] for k in cols) + ' |')
    print('| --- |' + ' --- |' * len(cols))
    for j, m in enumerate(TABLE_ORDER):
        cells = [f'{ar[j]:.2f} ／ {am[j]:.0f} ／ {nb[m]}'
                 for _k, (ar, _pos, am, _amed, _p, _W, nb) in cols.items()]
        print(f'| {M.METHOD_LABELS.get(m, m)} | ' + ' | '.join(cells) + ' |')
    print('  平均順位の小さい順: ' + '; '.join(
        f'{METRIC_LABELS[k]} ' + ', '.join(M.METHOD_LABELS[TABLE_ORDER[j]]
                                           for j in np.argsort(cols[k][0])) for k in cols))
    print('  ARPD% 中央値: ' + '; '.join(
        f'{METRIC_LABELS[k]} ' + ', '.join(f'{M.METHOD_LABELS[m]} {cols[k][3][j]:.0f}'
                                           for j, m in enumerate(TABLE_ORDER)) for k in cols))


def _stars(p):
    return '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else 'ns'))


def section_f(S, probs):
    pairs_within = [('ILS  repair vs none', 'ils_repair', 'ils_baseline'),
                    ('ILS  PR     vs none', 'ils_pr', 'ils_baseline'),
                    ('Mem  repair vs none', 'memetic_repair', 'memetic_ls'),
                    ('Mem  PR     vs none', 'memetic_pr', 'memetic_ls'),
                    ('Mem  repair vs PR  ', 'memetic_repair', 'memetic_pr')]
    pairs_between = [('none  : ILS vs Mem', 'ils_baseline', 'memetic_ls'),
                     ('repair: ILS vs Mem', 'ils_repair', 'memetic_repair'),
                     ('PR    : ILS vs Mem', 'ils_pr', 'memetic_pr')]
    for key in ('union_hv_pt', 'aoc_pt'):
        print(f'\n[F] §4.3 {METRIC_LABELS[key]}: 中央値の相対差 %（前者/後者-1）と両側 MWU, Cliff δ（前者>後者で正）')
        for label, a, b in pairs_within + pairs_between:
            cells, n_sig_pos, n_sig_neg = [], 0, 0
            for prob in probs:
                va = np.asarray(S[prob][key][a], float)
                vb = np.asarray(S[prob][key][b], float)
                p, d = _mwu(va, vb)
                rel = 100 * (np.median(va) / np.median(vb) - 1)
                cells.append(f'{rel:+6.1f}{_stars(p):>4s}{d:+5.2f}')
                if p < 0.05:
                    n_sig_pos += d > 0
                    n_sig_neg += d < 0
            print(f'  {label} | ' + ' | '.join(cells) + f' | 有意 +{n_sig_pos}/-{n_sig_neg}')
        print('   列順: ' + ', '.join(A.problem_short_tag(p) for p in probs))


def export_csv(S, probs, path=None):
    """論文の 3 指標を trial 単位で CSV に書き出す（リポジトリ公開用, §4.1）。

    この 1 ファイルから本文・表 2 の検定と要約はすべて再計算できる。raw の run JSON は
    2 GB 超あるためリポジトリには入れない。
    """
    path = path or os.path.join(HERE, 'data', 'per_trial_metrics.csv')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    keys = [('union_hv_pt', 'union_hv'), ('highstab_hv_pt', 'highstab_hv'), ('aoc_pt', 'aoc')]
    n = 0
    with open(path, 'w', encoding='utf-8', newline='') as f:
        f.write('scenario,instance,reschedule_rate_pct,highstab_threshold_P50,method,trial,'
                + ','.join(k for _a, k in keys) + chr(10))
        for prob in probs:
            tag = A.problem_short_tag(prob)
            rho = M.rho_pct(prob)
            p50 = S[prob]['thresholds']['P50']
            for m in TABLE_ORDER:
                vals = [np.asarray(S[prob][arr_key][m], float) for arr_key, _k in keys]
                for t in range(len(vals[0])):
                    f.write(f'{tag},{prob},{rho},{p50:.0f},{M.METHOD_LABELS[m]},{t},'
                            + ','.join(f'{v[t]:.6f}' for v in vals) + chr(10))
                    n += 1
    print(f'  -> {path}  ({n} 行 = {len(probs)} シナリオ x {len(TABLE_ORDER)} 手法 x 10 trial)')


if __name__ == '__main__':
    S = M.load_pkl()
    probs = A.order_prob_labels(S.keys())
    if '--export' in sys.argv:
        print(chr(10) + '[G] trial 単位の 3 指標 CSV（リポジトリ公開用）')
        export_csv(S, probs)
        sys.exit(0)
    if '--review' in sys.argv:
        section_a(S, probs)
        section_d(probs)
        sys.exit(0)
    section_b(S, probs)
    section_c(probs)
    section_e(S, probs)
    section_f(S, probs)
