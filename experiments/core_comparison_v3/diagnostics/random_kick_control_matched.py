#!/usr/bin/env python3
"""内的妥当性の対照（強度マッチング修正版）: Memetic+random_matched vs repair/pr/ls。

2026-06 の random_kick_control 対照は、ランダム側の摂動強度キャップを現在解の
n_diff（=S_p からの距離）から取っていたため、S_p 方向キック（n_diff を縮める→
自己制限）とランダム方向（n_diff を縮めない→強度が増幅）の間に非対称な正の
フィードバックが生じ、計算コスト比較（7.3倍等）が歪んでいた（2026-09-20 判明）。

本スクリプトは強度キャップを main_v1 の repair 実測深さ分布に固定した
`memetic_random_matched`（run_v3.py --methods memetic_random_matched
--random-depth-pool ...）の結果を、main_v1 の ls/repair/pr とマージし、
main_v1 全 7 手法×10 重みから作った正規化アンカー・P50 閾値（論文と同一定義）
で高安定 HV・統合 HV・計算コストを比較する。

使い方:
  python diagnostics/random_kick_control_matched.py \\
      --main-v1-dir results/main_v1/mt10_mt10_delay60 \\
      --matched-dir results/random_kick_control_matched/mt10_mt10_delay60/mt10_mt10_delay60
"""
import argparse
import glob
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, '..'))
sys.path.insert(0, os.path.join(_HERE, '..', '..', '..'))

import numpy as np
from scipy.stats import mannwhitneyu

import analyze_v3 as A


def mwu_two_sided(x, y):
    """論文 §3.4 と同一プロトコルの検定: 対応なし両側 Mann-Whitney U。

    手法間は構造が異なり乱数列が直ちに分岐するため対応を用いない。n=10 対 10 の
    完全分離では両側 p = 2 / C(20,10) ~ 1.1e-5 で、対応ありの符号付順位検定
    （n=10 片側の下限 p=1/2^10~0.001）のような下限には当たらない。
    """
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    x = x[np.isfinite(x)]; y = y[np.isfinite(y)]
    if len(x) == 0 or len(y) == 0:
        return float('nan')
    return float(mannwhitneyu(x, y, alternative='two-sided').pvalue)


def load_all(prob_dir):
    """raw/<method>__<wlabel>__t<trial>.json を {method: {wlabel: {trial: data}}} に。"""
    method_data = {}
    for path in glob.glob(os.path.join(prob_dir, 'raw', '*.json')):
        fn = os.path.basename(path)[:-5]
        method, wlabel, t_tag = fn.split('__')
        trial = int(t_tag[1:])
        with open(path, encoding='utf-8') as f:
            data = json.load(f)
        method_data.setdefault(method, {}).setdefault(wlabel, {})[trial] = data
    return method_data


def baselines_for(md, methods):
    baselines_by_method = {}
    for m in methods:
        bls = []
        for wl in md.get(m, {}):
            for _, data in md[m][wl].items():
                b1, b2 = data.get('baseline'), data.get('baseline_rsr')
                if b1 is not None:
                    bls.append(b1)
                if b2 is not None and list(b2) not in bls:
                    bls.append(list(b2))
                break
            if bls:
                break
        baselines_by_method[m] = bls if bls else None
    return baselines_by_method


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--main-v1-dir', required=True,
                     help='results/main_v1/<problem_scenario> (全7手法×10重み)')
    ap.add_argument('--matched-dir', required=True,
                     help='memetic_random_matched の raw を含む出力ディレクトリ')
    args = ap.parse_args()

    md_full = load_all(args.main_v1_dir)
    full_methods = sorted(md_full.keys())
    print(f'main_v1 手法 ({len(full_methods)}): {full_methods}')

    # 正規化アンカー・P50 は main_v1 全 7 手法×10 重みから作る（論文の θ=50 と同一定義）。
    bl_full = baselines_for(md_full, full_methods)
    all_pts = []
    for m in full_methods:
        bl = bl_full.get(m)
        for wl in md_full[m]:
            for t, data in md_full[m][wl].items():
                pts = A.get_uea_points(data, t)
                if bl:
                    pts = A.filter_baselines(pts, bl)
                if len(pts) > 0:
                    all_pts.append(pts)
    norm = A.make_norm(np.concatenate(all_pts))
    thr = A.compute_p33_p67(md_full, bl_full)
    print(f'正規化アンカー(main_v1全体): MS∈[{norm[0]:.0f}, {norm[0]+norm[1]:.0f}], D∈[0, {norm[2]:.0f}]')
    print(f'P50={thr["P50"]:.2f} (高安定領域 D<P50)')
    gref = (float(np.concatenate(all_pts)[:, 0].max()) * 1.01,
            float(np.concatenate(all_pts)[:, 1].max()) * 1.01)

    # 比較対象: main_v1 の ls/repair/pr + matched-dir の random_matched。同一10重みで揃える。
    md_matched = load_all(args.matched_dir)
    md = {
        'memetic_ls': md_full['memetic_ls'],
        'memetic_repair': md_full['memetic_repair'],
        'memetic_pr': md_full['memetic_pr'],
        'memetic_random_matched': md_matched['memetic_random_matched'],
    }
    methods = list(md.keys())
    common_w = set.intersection(*[set(md[m].keys()) for m in methods])
    print(f'共通重み({len(common_w)}): {sorted(common_w)}')
    for m in methods:
        md[m] = {w: md[m][w] for w in md[m] if w in common_w}
    bl = {m: bl_full[m] for m in methods if m in bl_full}
    bl['memetic_random_matched'] = baselines_for(
        {'memetic_random_matched': md['memetic_random_matched']},
        ['memetic_random_matched'])['memetic_random_matched']

    # ===== 高安定 / 低安定 領域別 union HV =====
    rhv = A.compute_region_hv_per_trial(md, bl, gref, thr['P50'], thr['stab_max'], norm=norm)
    union_hv = A.compute_union_hv_per_trial(md, bl, norm)

    comparisons = [
        ('memetic_ls', 'memetic_random_matched', 'ls < random_matched? (多様化一般の効果)'),
        ('memetic_random_matched', 'memetic_repair', 'random_matched < repair? (S_p誘導の上乗せ)'),
        ('memetic_random_matched', 'memetic_pr',     'random_matched < pr?     (S_p誘導の上乗せ)'),
    ]

    for label, table in [('union (統合 HV, 全重み)', union_hv)]:
        print(f'\n=== {label} ===')
        print(f'{"method":<24}{"median":>10}{"mean":>10}{"n":>4}')
        for m in methods:
            v = np.array(table[m], dtype=float)
            v = v[np.isfinite(v)]
            print(f'{m:<24}{np.median(v):>10.4f}{np.mean(v):>10.4f}{len(v):>4d}')
        print('  -- 対照検定 (両側 Mann-Whitney U = 論文プロトコル, 参考で片側 Wilcoxon) --')
        for a, b, desc in comparisons:
            xa, xb = np.array(table[a]), np.array(table[b])
            p_u = mwu_two_sided(xa, xb)
            _, p_w = A.wilcoxon_paired(xa, xb, alternative='less')
            d = A.cliffs_delta(xa, xb)
            print(f'    {desc}')
            print(f'        median {a}={np.median(xa):.4f}  {b}={np.median(xb):.4f}  '
                  f'Δ={np.median(xb)-np.median(xa):+.4f}  p_U={p_u:.3e}  '
                  f'(参考 p_wilcoxon={p_w:.4f})  δ={d:+.3f} ({A.effect_label(d)})')

    for rk, rname in [('high', f'高安定領域 (D<P50={thr["P50"]:.1f})'),
                       ('low',  f'低安定領域 (D>=P50)')]:
        print(f'\n=== 領域別 union HV - {rname} ===')
        print(f'{"method":<24}{"median":>10}{"mean":>10}')
        for m in methods:
            v = np.array([x for x in rhv[m][rk] if np.isfinite(x)])
            print(f'{m:<24}{np.median(v):>10.4f}{np.mean(v):>10.4f}')
        print('  -- 対照検定 (両側 Mann-Whitney U = 論文プロトコル, 参考で片側 Wilcoxon) --')
        for a, b, desc in comparisons:
            xa = np.array(rhv[a][rk]); xb = np.array(rhv[b][rk])
            p_u = mwu_two_sided(xa, xb)
            _, p_w = A.wilcoxon_paired(xa, xb, alternative='less')
            d = A.cliffs_delta(xa, xb)
            print(f'    {desc}')
            print(f'        median {a}={np.median(xa):.4f}  {b}={np.median(xb):.4f}  '
                  f'Δ={np.median(xb)-np.median(xa):+.4f}  p_U={p_u:.3e}  '
                  f'(参考 p_wilcoxon={p_w:.4f})  δ={d:+.3f} ({A.effect_label(d)})')

    # ===== 計算コスト =====
    print('\n=== 計算コスト (total_cpu_time, 全重み s/run) ===')
    cost = {}
    for m in methods:
        cpus = []
        for wl in md[m]:
            for _, data in md[m][wl].items():
                c = data.get('convergence', {}).get('total_cpu_time')
                if c is not None:
                    cpus.append(float(c))
        cost[m] = cpus
        print(f'  {m:<24} mean={np.mean(cpus):7.1f}s  median={np.median(cpus):7.1f}s  n={len(cpus)}')
    if 'memetic_repair' in cost and 'memetic_random_matched' in cost:
        r = np.median(cost['memetic_random_matched']) / np.median(cost['memetic_repair'])
        print(f'\n  random_matched / repair (median CPU 比) = {r:.2f}x')

    # ===== 強度マッチング監査 =====
    print('\n=== 強度マッチング監査 (repair_depth 要求値 vs random_matched 実適用) ===')
    for wl in sorted(common_w):
        rdep = []
        for t, data in md['memetic_repair'][wl].items():
            rdep.extend(data['mech_stats']['repair_depth'])
        xdep, xappl = [], []
        for t, data in md['memetic_random_matched'][wl].items():
            xdep.extend(data['mech_stats']['repair_depth'])
            xappl.extend(data['mech_stats'].get('random_applied', []))
        if not rdep or not xdep:
            continue
        mismatch = (sum(1 for a2, b2 in zip(xappl, xdep) if a2 != b2) / len(xdep) * 100
                    if xappl else float('nan'))
        print(f'  {wl}: repair_depth_mean={np.mean(rdep):.2f}  '
              f'random_req_mean={np.mean(xdep):.2f}  '
              f'random_applied_mean={np.mean(xappl) if xappl else float("nan"):.2f}  '
              f'mismatch={mismatch:.1f}%')


if __name__ == '__main__':
    main()
