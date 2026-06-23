#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""重みごとの scalar / per-weight UEA HV / AOC を全10重みで集計して表にする。

analyze_v3 の機構（load_all_runs, get_uea_points(_xyt), normalize, hypervolume,
compute_aoc_per_trial）をそのまま再利用し、main_v1 の per-weight 出力と整合させる。

- scalar : 各 trial 最終 best_score（最適化した重み付きスカラー値）の中央値。小さいほど良い。
- HV     : per-weight UEA HV（正規化空間 [0,1]^2・参照点 NORM_REF）の中央値。大きいほど良い。
- AOC    : per-weight AOC（log時間の時間平均HV, 正規化空間）の中央値。大きいほど良い。
"""
import os
import sys
import json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import analyze_v3 as A

INPUT_DIR = os.path.join(HERE, 'results', 'main_v1')
OUT = os.path.join(INPUT_DIR, 'analysis', 'per_weight_metrics.md')

# 重み（w0=makespan, w1=stability）小さい順 w0 で並べる: w10_00(MS純) → w01_09(安定純)
W_ORDER = [f'w{int(round(w0*10)):02d}_{int(round((1-w0)*10)):02d}'
           for w0 in [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]]


def med(xs):
    xs = [x for x in xs if x is not None and not (isinstance(x, float) and np.isnan(x))]
    return float(np.median(xs)) if xs else float('nan')


def main():
    grouped = A.load_all_runs(INPUT_DIR)
    cfg = json.load(open(os.path.join(INPUT_DIR, 'config.json'), encoding='utf-8'))
    methods = cfg['methods']
    labels = {m: cfg['method_configs'][m]['label'] for m in methods}

    out_lines = ['# 重み別メトリクス（全10重み）: scalar / per-weight UEA HV / AOC',
                 '',
                 '> 生成: per_weight_table.py  データ: results/main_v1（n=10 trials の中央値）',
                 '> scalar=最終重み付きスコア(小さいほど良)  HV/AOC=正規化空間 [0,1]^2(大きいほど良)',
                 '> 重みは w0(makespan):w1(stability)。左端 w10_00=MS純最適 → 右端 w01_09=安定純最適。',
                 '']

    results = {}  # (prob,scen) -> metric -> {method: {wl: val}}

    for (prob, scen), method_data in sorted(
            grouped.items(),
            key=lambda kv: (A.reschedule_rate(f'{kv[0][0]}_{kv[0][1]}'), kv[0])):
        # baseline 収集（analyze_v3 main と同じ）。出力順は難易度(リスケ率昇順)
        baselines_by_method = {}
        for m in methods:
            bls = []
            for wl in method_data.get(m, {}):
                for t_idx, data in method_data[m][wl].items():
                    b1 = data.get('baseline')
                    b2 = data.get('baseline_rsr')
                    if b1 is not None:
                        bls.append(b1)
                    if b2 is not None and list(b2) not in bls:
                        bls.append(list(b2))
                    break
                if bls:
                    break
            baselines_by_method[m] = bls if bls else None

        # 正規化アンカー
        all_pts = []
        for m in methods:
            bl = baselines_by_method.get(m)
            for wl in method_data.get(m, {}):
                for t_idx, data in method_data[m][wl].items():
                    pts = A.get_uea_points(data, t_idx)
                    if bl:
                        pts = A.filter_baselines(pts, bl)
                    if len(pts) > 0:
                        all_pts.append(pts)
        if not all_pts:
            continue
        norm = A.make_norm(np.concatenate(all_pts))

        w_labels = [wl for wl in W_ORDER
                    if any(wl in method_data.get(m, {}) for m in methods)]

        scalar = {m: {} for m in methods}
        hv = {m: {} for m in methods}
        aoc = {m: {} for m in methods}

        for wl in w_labels:
            # --- scalar & HV ---
            for m in methods:
                bl = baselines_by_method.get(m)
                by_trial = method_data[m].get(wl, {})
                sc_vals, hv_vals = [], []
                for t_idx in sorted(by_trial.keys()):
                    data = by_trial[t_idx]
                    hist = A.get_anytime(data)
                    bs = [h.get('best_score') for h in hist
                          if h.get('best_score') is not None]
                    sc_vals.append(bs[-1] if bs else None)
                    pts = A.get_uea_points(data, t_idx)
                    if bl:
                        pts = A.filter_baselines(pts, bl)
                    pts = A.normalize_pts(pts, norm) if len(pts) else pts
                    hv_vals.append(A.hypervolume(pts, A.NORM_REF))
                scalar[m][wl] = med(sc_vals)
                hv[m][wl] = med(hv_vals)

            # --- AOC（compute_aoc_per_trial を per-weight で）---
            method_info_n = []
            for m in methods:
                kind = 'ga' if m == 'ga' else 'ils'
                bl = baselines_by_method.get(m)
                by_trial = method_data[m].get(wl, {})
                hist_list, pts_list_n = [], []
                for t_idx in sorted(by_trial.keys()):
                    data = by_trial[t_idx]
                    hist_list.append(A.get_anytime(data))
                    pts = A.get_uea_points_xyt(data, t_idx)
                    pts_list_n.append(A.normalize_pts(pts, norm) if len(pts) else pts)
                method_info_n.append((m, hist_list, pts_list_n, kind,
                                      A.normalize_baseline(bl, norm)))
            aoc_pt = A.compute_aoc_per_trial(method_info_n, A.NORM_REF, n_jobs=1)
            for m in methods:
                aoc[m][wl] = med(aoc_pt.get(m, []))

        results[(prob, scen)] = {'scalar': scalar, 'hv': hv, 'aoc': aoc,
                                 'w_labels': w_labels}

        # --- markdown 出力 ---
        out_lines.append(f'## {A.problem_short_tag(f"{prob}_{scen}")} (`{prob}_{scen}`)')
        out_lines.append('')
        metric_title = {'scalar': 'scalar 最終重み付きスコア（小さいほど良）',
                        'hv': 'per-weight UEA HV（正規化, 大きいほど良）',
                        'aoc': 'AOC（正規化, 大きいほど良）'}
        metric_fmt = {'scalar': '{:.2f}', 'hv': '{:.4f}', 'aoc': '{:.4f}'}
        for metric in ['scalar', 'hv', 'aoc']:
            d = results[(prob, scen)][metric]
            out_lines.append(f'### {metric_title[metric]}')
            out_lines.append('')
            hdr = '| 重み(MS:ST) | ' + ' | '.join(labels[m] for m in methods) + ' |'
            sep = '|' + '---|' * (len(methods) + 1)
            out_lines.append(hdr)
            out_lines.append(sep)
            for wl in w_labels:
                w0 = int(wl[1:3]); w1 = int(wl[4:6])
                cells = []
                for m in methods:
                    v = d[m].get(wl, float('nan'))
                    cells.append(metric_fmt[metric].format(v) if not np.isnan(v) else '—')
                out_lines.append(f'| {w0}:{w1} | ' + ' | '.join(cells) + ' |')
            out_lines.append('')
        out_lines.append('---')
        out_lines.append('')
        print(f'done: {prob}_{scen}')

    with open(OUT, 'w', encoding='utf-8') as f:
        f.write('\n'.join(out_lines))
    print(f'\n書き出し: {OUT}')

    # pickle も保存（後段の図/集計用）
    import pickle
    with open(os.path.join(INPUT_DIR, 'analysis', 'per_weight_metrics.pkl'), 'wb') as f:
        pickle.dump(results, f)


if __name__ == '__main__':
    main()
