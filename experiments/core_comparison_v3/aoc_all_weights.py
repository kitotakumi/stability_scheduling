#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""AOC を「代表重み 1 点」から「全重み掃引の平均」に置き換える。

各 (問題, 手法, trial) について、全重みの per-weight AOC（log時間上のHV台形積分）を計算し、
重みをまたいで平均する＝「重み掃引全体での平均アンタイム性能」。
結果で analysis/_summary_data.pkl の aoc_pt を上書きパッチする（aoc_weight='all_mean'）。
旧 w08_02 版とのper-problem中央値・Friedman順位の比較も表示する。

usage: python aoc_all_weights.py [results_dir] [--agg mean|median] [--write]
"""
import os, sys, pickle, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import analyze_v3 as A

HERE = os.path.dirname(os.path.abspath(__file__))


def baselines_for(method_data, methods):
    bl_by = {}
    for m in methods:
        bls = []
        for wl in method_data.get(m, {}):
            for _, d in method_data[m][wl].items():
                if d.get('baseline') is not None:
                    bls.append(d['baseline'])
                b2 = d.get('baseline_rsr')
                if b2 is not None and list(b2) not in bls:
                    bls.append(list(b2))
                break
            if bls:
                break
        bl_by[m] = bls or None
    return bl_by


def norm_for(method_data, methods, bl_by):
    pts = []
    for m in methods:
        bl = bl_by.get(m)
        for wl in method_data.get(m, {}):
            for t, d in method_data[m][wl].items():
                p = A.get_uea_points(d, t)
                if bl:
                    p = A.filter_baselines(p, bl)
                if len(p):
                    pts.append(np.asarray(p, float))
    return A.make_norm(np.concatenate(pts))


def all_weight_aoc(method_data, methods, agg='mean', n_jobs=1):
    """{method: [per-trial 集計AOC]} を返す（全重みAOCを trial ごとに weights 集計）。"""
    bl_by = baselines_for(method_data, methods)
    norm = norm_for(method_data, methods, bl_by)
    w_labels = sorted({wl for m in methods for wl in method_data.get(m, {})})
    # per-weight AOC: {wl: {method: [per-trial]}}
    per_w = {}
    for wl in w_labels:
        method_info_n = []
        for m in methods:
            kind = 'ga' if m == 'ga' else 'ils'
            bl = bl_by.get(m)
            by_trial = method_data[m].get(wl, {})
            hist_list, pts_n = [], []
            for t in sorted(by_trial.keys()):
                d = by_trial[t]
                hist_list.append(A.get_anytime(d))
                pxyt = A.get_uea_points_xyt(d, t)
                pts_n.append(A.normalize_pts(pxyt, norm) if len(pxyt) else pxyt)
            method_info_n.append((m, hist_list, pts_n, kind, A.normalize_baseline(bl, norm)))
        per_w[wl] = A.compute_aoc_per_trial(method_info_n, A.NORM_REF, n_jobs=n_jobs)
    # 集計: 各 (method, trial) で全重みを agg
    aggfn = np.mean if agg == 'mean' else np.median
    out = {}
    for m in methods:
        # trial 数は最大長に合わせる
        n_tr = max((len(per_w[wl].get(m, [])) for wl in w_labels), default=0)
        vals = []
        for t in range(n_tr):
            xs = [per_w[wl][m][t] for wl in w_labels
                  if m in per_w[wl] and t < len(per_w[wl][m])]
            vals.append(float(aggfn(xs)) if xs else float('nan'))
        out[m] = vals
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('results_dir', nargs='?', default=os.path.join(HERE, 'results', 'main_v1'))
    ap.add_argument('--agg', default='mean', choices=['mean', 'median'])
    ap.add_argument('--write', action='store_true', help='_summary_data.pkl の aoc_pt を上書き')
    args = ap.parse_args()

    grouped = A.load_all_runs(args.results_dir)
    pkl = os.path.join(args.results_dir, 'analysis', '_summary_data.pkl')
    S = pickle.load(open(pkl, 'rb'))

    order = [m for m in A.METHOD_ORDER]
    new_aoc = {}
    print(f'集計法: {args.agg}\n')
    for (prob, scen), md in sorted(grouped.items()):
        pl = f'{prob}_{scen}'
        methods = [m for m in order if m in md]
        aoc_all = all_weight_aoc(md, methods, agg=args.agg)
        new_aoc[pl] = aoc_all
        # 旧(w08_02)との中央値比較
        old = S.get(pl, {}).get('aoc_pt', {})
        print(f'=== {A.problem_short_tag(pl)} （旧=w08_02 / 新=全重み{args.agg}） ===')
        for m in methods:
            om = np.median([v for v in old.get(m, []) if np.isfinite(v)]) if old.get(m) else float('nan')
            nm = np.median([v for v in aoc_all.get(m, []) if np.isfinite(v)])
            print(f'  {A.METHOD_LABELS.get(m,m):<16} 旧={om:.4f}  新={nm:.4f}')
        print()

    # 横断: 新AOCでFriedman順位とARPD
    probs = sorted(new_aoc.keys())
    methods = [m for m in order if any(m in new_aoc[p] for p in probs)]
    M = np.full((len(probs), len(methods)), np.nan)
    for i, pl in enumerate(probs):
        for j, m in enumerate(methods):
            v = [x for x in new_aoc[pl].get(m, []) if np.isfinite(x)]
            if v:
                M[i, j] = np.median(v)
    valid = ~np.any(np.isnan(M), axis=1)
    Mv = M[valid]
    avg_rank, chi, p, W, ranks = A._friedman_avg_rank(Mv)
    arpd_mean, arpd_med = A._arpd_pct(Mv)
    print(f'=== 全重み{args.agg} AOC 横断スコアボード — Friedman p={p:.4f}, W={W:.2f} ===')
    o = list(np.argsort(avg_rank))
    for rk, j in enumerate(o, 1):
        print(f'  {A.METHOD_LABELS.get(methods[j],methods[j]):<16} '
              f'順位={avg_rank[j]:.2f} ({rk})  ARPD%={arpd_mean[j]:.1f}/{arpd_med[j]:.1f}')

    if args.write:
        for pl in probs:
            if pl in S:
                S[pl]['aoc_pt'] = {m: new_aoc[pl].get(m, []) for m in new_aoc[pl]}
                S[pl]['aoc_weight'] = f'all_{args.agg}'
        pickle.dump(S, open(pkl, 'wb'))
        print(f'\n[written] {pkl} の aoc_pt を全重み{args.agg}で上書き')


if __name__ == '__main__':
    main()
