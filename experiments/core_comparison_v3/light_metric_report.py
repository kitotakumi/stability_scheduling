#!/usr/bin/env python3
"""
light_metric_report: ILS 3手法 (baseline / repair / PR) の指標ごとの
解の質と計算時間を軽く確認するための簡易レポート。

analyze_v3.py の検証済み関数（hypervolume / region_hv / pareto_front /
compute_union_hv_per_trial / compute_p33_p67 など）をそのまま再利用し、
指標を再実装しない。出力は図を作らずテキスト1枚に集約する。

  - HHV (統合 UEA HV, B-2a): 全重み union の per-trial Pareto front の HV
  - 領域別 HV (B-2b): P50 2分割 / P33-P67 3分割（per-trial median）
  - 計算時間: total_cpu_time（1 run 全体）と cpu_time（最良到達まで）

使い方:
  python light_metric_report.py --input-dir results/ils_metric_check
"""

import argparse
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, '..', '..'))
sys.path.insert(0, os.path.join(_HERE, '..'))

import analyze_v3 as A

try:
    from scipy.stats import wilcoxon as scipy_wilcoxon
    SCIPY_OK = True
except ImportError:
    SCIPY_OK = False


METHOD_ORDER = ['ils_baseline', 'ils_repair', 'ils_pr']


def _build_baselines(method_data, methods):
    """analyze_v3 と同一ロジックで baselines_by_method を構築。"""
    baselines_by_method = {}
    for m in methods:
        bls = []
        for wl in method_data[m]:
            for t_idx, data in method_data[m][wl].items():
                b1 = data.get('baseline')
                b2 = data.get('baseline_rsr')
                if b1 is not None:
                    bls.append(b1)
                if b2 is not None:
                    b2_entry = list(b2)
                    if b2_entry not in bls:
                        bls.append(b2_entry)
                break
            if bls:
                break
        baselines_by_method[m] = bls if bls else None
    return baselines_by_method


def _global_ref(method_data, methods, baselines_by_method):
    all_pts_flat = []
    for m in methods:
        bl = baselines_by_method.get(m)
        for wl in method_data[m]:
            for t_idx, data in method_data[m][wl].items():
                pts = A.get_uea_points(data, t_idx)
                if bl:
                    pts = A.filter_baselines(pts, bl)
                if len(pts) > 0:
                    all_pts_flat.append(pts)
    if not all_pts_flat:
        return None
    cat = np.concatenate(all_pts_flat)
    return (
        float(cat[:, 0].max()) + max(cat[:, 0].max() * 0.01, 1.0),
        float(cat[:, 1].max()) + max(cat[:, 1].max() * 0.01, 0.01),
    )


def _trial_pfs(method_data, methods, baselines_by_method):
    """method -> {trial_idx: union Pareto front (全重み)}"""
    out = {}
    for m in methods:
        bl = baselines_by_method.get(m)
        idxs = set()
        for wl in method_data[m]:
            idxs.update(method_data[m][wl].keys())
        out[m] = {}
        for t in sorted(idxs):
            up = []
            for wl in method_data[m]:
                data = method_data[m][wl].get(t)
                if data is None:
                    continue
                pts = A.get_uea_points(data, t)
                if bl:
                    pts = A.filter_baselines(pts, bl)
                if len(pts) > 0:
                    up.append(pts)
            out[m][t] = A.pareto_front(np.concatenate(up)) if up else np.zeros((0, 2))
    return out


def _region_hv_per_trial(trial_pfs, methods, regions, global_ref):
    """region -> method -> per-trial HV のリスト（median 集約は呼び出し側）。"""
    res = {m: {rn: [] for rn in regions} for m in methods}
    for m in methods:
        for t, pf in trial_pfs[m].items():
            if len(pf) == 0:
                for rn in regions:
                    res[m][rn].append(0.0)
                continue
            for rn, (lo, hi) in regions.items():
                hi_inc = (hi == global_ref[1])
                hv, _ = A.region_hv(pf, lo, hi, global_ref[0], hi_inclusive=hi_inc)
                res[m][rn].append(hv)
    return res


def _cpu_stats(method_data, methods):
    """method -> {total_cpu: [..], best_cpu: [..], best_iter: [..]}"""
    out = {}
    for m in methods:
        tot, best, biter = [], [], []
        for wl in method_data[m]:
            for t, data in method_data[m][wl].items():
                conv = data.get('convergence', {})
                if conv.get('total_cpu_time') is not None:
                    tot.append(float(conv['total_cpu_time']))
                if conv.get('cpu_time') is not None:
                    best.append(float(conv['cpu_time']))
                if conv.get('iteration') is not None:
                    biter.append(float(conv['iteration']))
        out[m] = {'total_cpu': tot, 'best_cpu': best, 'best_iter': biter}
    return out


def _wilcoxon(a, b):
    if not SCIPY_OK or len(a) != len(b) or len(a) < 2:
        return None
    diff = np.array(a) - np.array(b)
    if np.allclose(diff, 0):
        return 1.0
    try:
        _, p = scipy_wilcoxon(a, b)
        return float(p)
    except ValueError:
        return None


def report_problem(prob_key, method_data, lines):
    methods = [m for m in METHOD_ORDER if m in method_data]
    if not methods:
        return
    problem, scenario = prob_key
    prob_label = f'{problem}_{scenario}'

    baselines = _build_baselines(method_data, methods)
    global_ref = _global_ref(method_data, methods, baselines)
    if global_ref is None:
        lines.append(f'\n===== {prob_label}: 有効な点なし =====')
        return

    thr = A.compute_p33_p67(method_data, baselines)
    p33, p50, p67, stab_max = thr['P33'], thr['P50'], thr['P67'], thr['stab_max']

    union_hv = A.compute_union_hv_per_trial(method_data, baselines, global_ref, weights_subset=None)
    trial_pfs = _trial_pfs(method_data, methods, baselines)

    regions_2 = {'high_stability': (0.0, p50), 'low_stability': (p50, global_ref[1])}
    regions_3 = {'low_stab': (0.0, p33), 'mid_stab': (p33, p67), 'high_stab': (p67, global_ref[1])}
    rhv2 = _region_hv_per_trial(trial_pfs, methods, regions_2, global_ref)
    rhv3 = _region_hv_per_trial(trial_pfs, methods, regions_3, global_ref)
    cpu = _cpu_stats(method_data, methods)

    n_trials = max((len(union_hv.get(m, [])) for m in methods), default=0)

    lines.append('')
    lines.append('=' * 78)
    lines.append(f'  {prob_label}   (n_trials={n_trials}, ref={global_ref[0]:.0f}/{global_ref[1]:.2f}, '
                 f'P33={p33:.3f} P50={p50:.3f} P67={p67:.3f} stab_max={stab_max:.2f})')
    lines.append('=' * 78)

    def fmt_row(label, vals_by_m, fmt='{:>11.1f}'):
        cells = ''.join(fmt.format(vals_by_m[m]) for m in methods)
        return f'  {label:<22}{cells}'

    hdr = '  ' + ' ' * 22 + ''.join(f'{A.METHOD_LABELS.get(m, m):>11}' for m in methods)
    lines.append(hdr)
    lines.append('  ' + '-' * (22 + 11 * len(methods)))

    # ---- HHV (統合 UEA HV) ----
    lines.append('  [HHV] 統合 UEA HV (全重み union, per-trial)')
    lines.append(fmt_row('median', {m: np.median(union_hv[m]) for m in methods}))
    lines.append(fmt_row('mean', {m: np.mean(union_hv[m]) for m in methods}))
    lines.append(fmt_row('std', {m: np.std(union_hv[m]) for m in methods}))
    lines.append(fmt_row('min', {m: np.min(union_hv[m]) for m in methods}))
    lines.append(fmt_row('max', {m: np.max(union_hv[m]) for m in methods}))

    # ---- 領域別 HV (P50 2分割) ----
    lines.append('  [領域別HV] P50 2分割 (per-trial median)')
    lines.append(fmt_row('high_stab (D<P50)', {m: np.median(rhv2[m]['high_stability']) for m in methods}))
    lines.append(fmt_row('low_stab  (D>=P50)', {m: np.median(rhv2[m]['low_stability']) for m in methods}))

    # ---- 領域別 HV (P33/P67 3分割) ----
    lines.append('  [領域別HV] P33/P67 3分割 (per-trial median)')
    lines.append(fmt_row('low_stab  (D<P33)', {m: np.median(rhv3[m]['low_stab']) for m in methods}))
    lines.append(fmt_row('mid_stab', {m: np.median(rhv3[m]['mid_stab']) for m in methods}))
    lines.append(fmt_row('high_stab (D>=P67)', {m: np.median(rhv3[m]['high_stab']) for m in methods}))

    # ---- 計算時間 ----
    lines.append('  [計算時間] 秒')
    lines.append(fmt_row('total_cpu median', {m: np.median(cpu[m]['total_cpu']) for m in methods}, '{:>11.2f}'))
    lines.append(fmt_row('best_cpu median', {m: np.median(cpu[m]['best_cpu']) for m in methods}, '{:>11.2f}'))
    lines.append(fmt_row('best_iter median', {m: np.median(cpu[m]['best_iter']) for m in methods}, '{:>11.0f}'))

    # ---- 対 baseline 検定 (HHV) ----
    base = 'ils_baseline'
    if base in methods:
        lines.append(f'  [HHV Wilcoxon vs baseline] (signed-rank, n={n_trials})')
        for m in methods:
            if m == base:
                continue
            p = _wilcoxon(union_hv[m], union_hv[base])
            d_med = np.median(union_hv[m]) - np.median(union_hv[base])
            ptxt = 'n/a' if p is None else f'{p:.3f}'
            sign = '+' if d_med >= 0 else ''
            lines.append(f'    {A.METHOD_LABELS.get(m, m):<16} Δmedian={sign}{d_med:.1f}  p={ptxt}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input-dir', required=True)
    args = ap.parse_args()

    input_dir = args.input_dir
    if not os.path.isabs(input_dir):
        input_dir = os.path.join(_HERE, input_dir)

    grouped = A.load_all_runs(input_dir)
    lines = []
    lines.append('############################################################')
    lines.append('#  ILS 3手法 軽量メトリクスレポート (HHV / 領域別HV / CPU時間)')
    lines.append(f'#  input: {input_dir}')
    lines.append('############################################################')

    for prob_key in sorted(grouped.keys()):
        report_problem(prob_key, grouped[prob_key], lines)

    text = '\n'.join(lines)
    print(text)
    out_path = os.path.join(input_dir, 'light_metric_report.txt')
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(text + '\n')
    print(f'\n[saved] {out_path}')


if __name__ == '__main__':
    main()
