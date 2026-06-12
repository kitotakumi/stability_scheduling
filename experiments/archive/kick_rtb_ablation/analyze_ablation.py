#!/usr/bin/env python3
"""
kick_rtb_ablation 分析: displacement vs return-to-best (RTB)

core_comparison_v2/analyze_v2.py のメトリクス関数をそのまま流用する:
  - scalar     : 最終 weighted score / 最終 makespan (小さいほど良い)
  - UEA HV     : 訪問点 Pareto front の hypervolume (大きいほど良い, per-trial)
  - 領域別 HV  : 高安定性 / 低安定性 ゾーンに分けた region-restricted HV (P50 境界)

displacement(現行) と RTB を対にして Wilcoxon + Cliff's delta で比較する。
"""

import argparse
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, '..', '..'))
sys.path.insert(0, os.path.join(_HERE, '..', 'core_comparison_v2'))

from analyze_v2 import (
    get_uea_points, pareto_front, hypervolume, region_hv, filter_baselines,
    compute_p33_p67, wilcoxon_paired, cliffs_delta, effect_label,
)


# 比較ペア: (A, B) は「A が B より良い」を対立仮説に置く
DISP_VS_RTB = [
    ('ils_repair_rtb', 'ils_repair_disp'),     # RTB が disp より良い？
    ('ils_pr_rtb', 'ils_pr_disp'),             # no-op PR
    ('ils_prfix_rtb', 'ils_prfix_disp'),       # 修正版 PR
]
VS_BASELINE = [
    ('ils_repair_disp', 'ils_baseline'),
    ('ils_repair_rtb', 'ils_baseline'),
    ('ils_pr_disp', 'ils_baseline'),
    ('ils_pr_rtb', 'ils_baseline'),
    ('ils_prfix_disp', 'ils_baseline'),
    ('ils_prfix_rtb', 'ils_baseline'),
]
# 修正版 PR が no-op PR より良いか（PR を機能させた効果）
FIX_VS_NOOP = [
    ('ils_prfix_disp', 'ils_pr_disp'),
    ('ils_prfix_rtb', 'ils_pr_rtb'),
]

METHOD_ORDER = ['ils_baseline', 'ils_repair_disp', 'ils_repair_rtb',
                'ils_pr_disp', 'ils_pr_rtb', 'ils_prfix_disp', 'ils_prfix_rtb']


def load_runs(input_dir):
    """{problem: {method: {trial: data}}} を返す。"""
    out = {}
    for prob_dir in sorted(os.listdir(input_dir)):
        raw_dir = os.path.join(input_dir, prob_dir, 'raw')
        if not os.path.isdir(raw_dir):
            continue
        for fn in os.listdir(raw_dir):
            if not fn.endswith('.json'):
                continue
            with open(os.path.join(raw_dir, fn), 'r', encoding='utf-8') as f:
                data = json.load(f)
            prob = f"{data['problem']}_{data['scenario']}"
            out.setdefault(prob, {}).setdefault(data['method'], {})[data['trial']] = data
    return out


def final_score(data):
    hist = data.get('history', [])
    vals = [h.get('best_score') for h in hist if h.get('best_score') is not None]
    return float(vals[-1]) if vals else float('nan')


def final_makespan(data):
    return float(data['finals'].get('makespan', np.nan))


def improved(data):
    b = data.get('baseline_score')
    if b is None:
        return False
    return final_score(data) < float(b) - 1e-6


def per_trial_uea_hv(by_trial, baseline, ref):
    """trial 順の per-trial UEA HV リスト。"""
    out = []
    for t in sorted(by_trial.keys()):
        pts = get_uea_points(by_trial[t], t)
        if baseline is not None:
            pts = filter_baselines(pts, baseline)
        if len(pts) > 0:
            out.append(float(hypervolume(pareto_front(pts), ref)))
        else:
            out.append(0.0)
    return out


def paired(metric_by_method, ma, mb):
    """共通 trial だけで (vals_a, vals_b) を返す。"""
    a, b = metric_by_method.get(ma, {}), metric_by_method.get(mb, {})
    common = sorted(set(a.keys()) & set(b.keys()))
    return [a[t] for t in common], [b[t] for t in common]


def med_iqr(vals):
    v = np.array([x for x in vals if np.isfinite(x)], dtype=float)
    if len(v) == 0:
        return float('nan'), float('nan'), float('nan')
    return float(np.median(v)), float(np.percentile(v, 25)), float(np.percentile(v, 75))


def fmt_stat(vals_a, vals_b, lower_better):
    """A vs B: lower_better なら alternative='less' (A<B=Aが良い)。
    HV のように大きいほど良いなら lower_better=False → 'greater'。"""
    alt = 'less' if lower_better else 'greater'
    stat, p = wilcoxon_paired(vals_a, vals_b, alternative=alt)
    d = cliffs_delta(vals_a, vals_b)
    sig = '***' if (np.isfinite(p) and p < 0.001) else \
          '**' if (np.isfinite(p) and p < 0.01) else \
          '*' if (np.isfinite(p) and p < 0.05) else ''
    p_str = f'{p:.3f}' if np.isfinite(p) else 'nan'
    return f'p={p_str}{sig} d={d:+.2f}({effect_label(d)})'


def analyze_problem(prob, by_method, lines):
    lines.append(f'\n{"="*78}')
    lines.append(f'  問題: {prob}')
    lines.append(f'{"="*78}')

    methods = [m for m in METHOD_ORDER if m in by_method]

    # ---- 共通参照点 & P50 閾値 ----
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

    md_for_p50 = {m: {'w': {t: by_method[m][t] for t in by_method[m]}} for m in methods}
    thr = compute_p33_p67(md_for_p50, baselines_by_method)
    p50 = thr['P50'] if thr else float(concat[:, 1].max()) / 2
    stab_max = thr['stab_max'] if thr else float(concat[:, 1].max())
    lines.append(f'  ref={ref[0]:.1f},{ref[1]:.2f}  P50(stab)={p50:.3f}  stab_max={stab_max:.3f}')

    # ---- メトリクス収集 ----
    score_by = {}    # {method: {trial: final_score}}
    ms_by = {}
    hv_by = {}
    imp_rate = {}
    region_hv_high = {}  # 高安定性ゾーン [0, P50)
    region_hv_low = {}   # 低安定性ゾーン [P50, stab_max]
    for m in methods:
        bl = baselines_by_method[m]
        score_by[m] = {t: final_score(d) for t, d in by_method[m].items()}
        ms_by[m] = {t: final_makespan(d) for t, d in by_method[m].items()}
        hv_list = per_trial_uea_hv(by_method[m], bl, ref)
        hv_by[m] = {t: hv_list[i] for i, t in enumerate(sorted(by_method[m].keys()))}
        imp_rate[m] = np.mean([improved(d) for d in by_method[m].values()])
        # 領域別 HV: 手法の全訪問点を union して region ごとに HV
        union = []
        for t, data in by_method[m].items():
            pts = get_uea_points(data, t)
            if bl is not None:
                pts = filter_baselines(pts, bl)
            if len(pts) > 0:
                union.append(pts)
        if union:
            up = np.concatenate(union)
            region_hv_high[m] = region_hv(up, 0.0, p50, ref[0])[0]
            region_hv_low[m] = region_hv(up, p50, stab_max, ref[0], hi_inclusive=True)[0]
        else:
            region_hv_high[m] = region_hv_low[m] = 0.0

    # ---- 記述統計テーブル ----
    lines.append('\n  [記述統計] median [Q25, Q75]   (score/MS: 小さいほど良い, HV: 大きいほど良い)')
    lines.append(f'    {"method":<22} {"score":>22} {"makespan":>20} {"UEA HV":>20} {"imp%":>6}')
    for m in methods:
        s = med_iqr(list(score_by[m].values()))
        ms = med_iqr(list(ms_by[m].values()))
        hv = med_iqr(list(hv_by[m].values()))
        lines.append(
            f'    {m:<22} '
            f'{s[0]:>8.4f}[{s[1]:.4f},{s[2]:.4f}] '
            f'{ms[0]:>7.0f}[{ms[1]:.0f},{ms[2]:.0f}] '
            f'{hv[0]:>8.0f}[{hv[1]:.0f},{hv[2]:.0f}] '
            f'{imp_rate[m]*100:>5.0f}')

    lines.append('\n  [領域別 HV] (union, P50 境界)')
    lines.append(f'    {"method":<22} {"高安定ゾーン[0,P50)":>22} {"低安定ゾーン[P50,max]":>24}')
    for m in methods:
        lines.append(f'    {m:<22} {region_hv_high[m]:>22.1f} {region_hv_low[m]:>24.1f}')

    # ---- 対比較 (Wilcoxon + Cliff's delta) ----
    def block(title, pairs):
        lines.append(f'\n  [{title}]')
        for ma, mb in pairs:
            if ma not in methods or mb not in methods:
                continue
            sa, sb = paired(score_by, ma, mb)
            ma_, mb_ = paired(ms_by, ma, mb)
            ha, hb = paired(hv_by, ma, mb)
            lines.append(f'    {ma} vs {mb} (n={len(sa)}):')
            lines.append(f'        score : {fmt_stat(sa, sb, lower_better=True)}')
            lines.append(f'        MS    : {fmt_stat(ma_, mb_, lower_better=True)}')
            lines.append(f'        UEA HV: {fmt_stat(ha, hb, lower_better=False)}')

    block('displacement vs RTB  (左が良いと主張)', DISP_VS_RTB)
    block('修正版PR vs no-op PR  (左が良いと主張)', FIX_VS_NOOP)
    block('各手法 vs baseline  (左が良いと主張)', VS_BASELINE)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir', type=str,
                        default=os.path.join(_HERE, 'results', 'ablation'))
    args = parser.parse_args()

    runs = load_runs(args.input_dir)
    if not runs:
        print(f'結果が見つかりません: {args.input_dir}')
        return

    lines = []
    lines.append('kick_rtb_ablation 分析結果')
    lines.append('  scalar=最終weighted score / 最終makespan (小=良) , UEA HV (大=良)')
    lines.append('  Wilcoxon paired, * p<.05 ** p<.01 *** p<.001 ; d=Cliff\'s delta')
    for prob in sorted(runs.keys()):
        analyze_problem(prob, runs[prob], lines)

    text = '\n'.join(lines)
    print(text)
    out = os.path.join(args.input_dir, 'ablation_summary.txt')
    with open(out, 'w', encoding='utf-8') as f:
        f.write(text + '\n')
    print(f'\n→ 保存: {out}')


if __name__ == '__main__':
    main()
