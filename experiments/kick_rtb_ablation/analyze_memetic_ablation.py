#!/usr/bin/env python3
"""
memetic kick PR ablation 分析: memetic（kick なし）と memetic+PR を比較。

analyze_v2 のメトリクス (scalar / UEA HV / 領域別HV) と analyze_ablation の
ヘルパをそのまま流用する。
"""

import argparse
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, '..', 'core_comparison_v2'))

from analyze_v2 import (
    get_uea_points, region_hv, filter_baselines, compute_p33_p67,
)
from analyze_ablation import (
    load_runs, final_score, final_makespan, improved,
    per_trial_uea_hv, paired, med_iqr, fmt_stat,
)

METHOD_ORDER = ['memetic_ls', 'memetic_pr', 'memetic_pr_fi']

# (A, B): A が B より良いを対立仮説に
PAIRS = [
    ('memetic_pr', 'memetic_ls'),          # BI-PR は memetic 単体より良い？
    ('memetic_pr_fi', 'memetic_ls'),       # FI-PR は memetic 単体より良い？
    ('memetic_pr_fi', 'memetic_pr'),       # FI-PR は BI-PR と比べてどうか
]


def analyze_problem(prob, by_method, lines):
    lines.append(f'\n{"="*78}')
    lines.append(f'  問題: {prob}')
    lines.append(f'{"="*78}')

    methods = [m for m in METHOD_ORDER if m in by_method]
    baselines_by_method = {}
    all_pts = []
    for m in methods:
        bl = None
        for t in by_method[m]:
            bl = by_method[m][t].get('baseline')
            break
        baselines_by_method[m] = bl
        for t, data in by_method[m].items():
            pts = get_uea_points(data, t)
            if bl is not None:
                pts = filter_baselines(pts, bl)
            if len(pts) > 0:
                all_pts.append(pts)
    if not all_pts:
        lines.append('  有効訪問点なし。スキップ。')
        return
    concat = np.concatenate(all_pts)
    ref = (float(concat[:, 0].max()) + max(concat[:, 0].max() * 0.01, 1.0),
           float(concat[:, 1].max()) + max(concat[:, 1].max() * 0.01, 0.01))
    md = {m: {'w': {t: by_method[m][t] for t in by_method[m]}} for m in methods}
    thr = compute_p33_p67(md, baselines_by_method)
    p50 = thr['P50'] if thr else float(concat[:, 1].max()) / 2
    stab_max = thr['stab_max'] if thr else float(concat[:, 1].max())
    lines.append(f'  ref={ref[0]:.1f},{ref[1]:.2f}  P50(stab)={p50:.3f}  stab_max={stab_max:.3f}')

    score_by, ms_by, hv_by, imp_rate = {}, {}, {}, {}
    rh_high, rh_low = {}, {}
    for m in methods:
        bl = baselines_by_method[m]
        score_by[m] = {t: final_score(d) for t, d in by_method[m].items()}
        ms_by[m] = {t: final_makespan(d) for t, d in by_method[m].items()}
        hv_list = per_trial_uea_hv(by_method[m], bl, ref)
        hv_by[m] = {t: hv_list[i] for i, t in enumerate(sorted(by_method[m].keys()))}
        imp_rate[m] = np.mean([improved(d) for d in by_method[m].values()])
        union = []
        for t, data in by_method[m].items():
            pts = get_uea_points(data, t)
            if bl is not None:
                pts = filter_baselines(pts, bl)
            if len(pts) > 0:
                union.append(pts)
        if union:
            up = np.concatenate(union)
            rh_high[m] = region_hv(up, 0.0, p50, ref[0])[0]
            rh_low[m] = region_hv(up, p50, stab_max, ref[0], hi_inclusive=True)[0]
        else:
            rh_high[m] = rh_low[m] = 0.0

    lines.append('\n  [記述統計] median [Q25, Q75]   (score/MS: 小=良, HV: 大=良)')
    lines.append(f'    {"method":<18} {"score":>22} {"makespan":>20} {"UEA HV":>22} {"imp%":>6}')
    for m in methods:
        s, ms, hv = med_iqr(list(score_by[m].values())), med_iqr(list(ms_by[m].values())), med_iqr(list(hv_by[m].values()))
        lines.append(
            f'    {m:<18} {s[0]:>8.4f}[{s[1]:.4f},{s[2]:.4f}] '
            f'{ms[0]:>7.0f}[{ms[1]:.0f},{ms[2]:.0f}] '
            f'{hv[0]:>8.0f}[{hv[1]:.0f},{hv[2]:.0f}] {imp_rate[m]*100:>5.0f}')

    lines.append('\n  [領域別 HV] (union, P50 境界)')
    lines.append(f'    {"method":<18} {"高安定ゾーン[0,P50)":>22} {"低安定ゾーン[P50,max]":>24}')
    for m in methods:
        lines.append(f'    {m:<18} {rh_high[m]:>22.1f} {rh_low[m]:>24.1f}')

    lines.append('\n  [対比較 Wilcoxon + Cliff\'s delta] (左が良いと主張)')
    for ma, mb in PAIRS:
        if ma not in methods or mb not in methods:
            continue
        sa, sb = paired(score_by, ma, mb)
        ma_, mb_ = paired(ms_by, ma, mb)
        ha, hb = paired(hv_by, ma, mb)
        lines.append(f'    {ma} vs {mb} (n={len(sa)}):')
        lines.append(f'        score : {fmt_stat(sa, sb, lower_better=True)}')
        lines.append(f'        MS    : {fmt_stat(ma_, mb_, lower_better=True)}')
        lines.append(f'        UEA HV: {fmt_stat(ha, hb, lower_better=False)}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir', type=str,
                        default=os.path.join(_HERE, 'results', 'memetic_w08_02'))
    args = parser.parse_args()
    runs = load_runs(args.input_dir)
    if not runs:
        print(f'結果が見つかりません: {args.input_dir}')
        return
    lines = ['memetic kick PR ablation 分析結果',
             '  scalar=最終weighted score / 最終makespan (小=良), UEA HV (大=良)',
             '  Wilcoxon paired, * p<.05 ** p<.01 *** p<.001 ; d=Cliff\'s delta']
    for prob in sorted(runs.keys()):
        analyze_problem(prob, runs[prob], lines)
    text = '\n'.join(lines)
    print(text)
    out = os.path.join(args.input_dir, 'memetic_ablation_summary.txt')
    with open(out, 'w', encoding='utf-8') as f:
        f.write(text + '\n')
    print(f'\n→ 保存: {out}')


if __name__ == '__main__':
    main()
