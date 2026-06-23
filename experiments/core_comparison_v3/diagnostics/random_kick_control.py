#!/usr/bin/env python3
"""内的妥当性の対照: Memetic+random-kick vs +repair / +PR / -LS の union HV 比較。

研究ドキュメントの「検討中」項目（同強度・ランダム方向キックの対照）への回答。
Memetic+PR/repair の利得が「S_p 方向への誘導（安定性誘導）」由来か「収束集団への一般的な
多様化」由来かを分離する。analyze_v3.py の HV 定義（正規化 [0,1]^2 空間・参照点 NORM_REF）を
そのまま再利用し、union UEA HV を per-trial で計算して Wilcoxon + Cliff's delta で比較する。

使い方:
  python diagnostics/random_kick_control.py --prob-dir results/<run>/la36_la36_large
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

import analyze_v3 as A


def load_method_data(prob_dir):
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--prob-dir', required=True, help='results/<run>/<problem_scenario>')
    args = ap.parse_args()

    md = load_method_data(args.prob_dir)
    methods = sorted(md.keys())
    print(f'問題ディレクトリ: {args.prob_dir}')
    print(f'手法: {methods}')

    # 全手法に共通する重みのみに揃える（w10_00 孤児ファイル等の混入で重み数が手法間で
    # 食い違うと union/領域 HV が不公平になるため。random は w10_00 を持たない）。
    common_w = set.intersection(*[set(md[m].keys()) for m in methods])
    for m in methods:
        md[m] = {w: md[m][w] for w in md[m] if w in common_w}
    print(f'共通重み({len(common_w)}): {sorted(common_w)}')

    # baselines_by_method（analyze_v3 と同じ: baseline + baseline_rsr）
    baselines_by_method = {}
    for m in methods:
        bls = []
        for wl in md[m]:
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

    # 正規化アンカー（全手法・全重み・全 trial の baseline 除外後訪問点）
    all_pts = []
    for m in methods:
        bl = baselines_by_method.get(m)
        for wl in md[m]:
            for t, data in md[m][wl].items():
                pts = A.get_uea_points(data, t)
                if bl:
                    pts = A.filter_baselines(pts, bl)
                if len(pts) > 0:
                    all_pts.append(pts)
    norm = A.make_norm(np.concatenate(all_pts))
    print(f'正規化アンカー: MS∈[{norm[0]:.0f}, {norm[0]+norm[1]:.0f}], D∈[0, {norm[2]:.0f}]')

    # union HV per trial
    union_hv = A.compute_union_hv_per_trial(md, baselines_by_method, norm)

    print('\n=== union UEA HV (正規化 [0,1]^2, 全重み統合) ===')
    print(f'{"method":<18}{"median":>10}{"mean":>10}{"min":>8}{"max":>8}{"n":>4}')
    for m in methods:
        v = np.array(union_hv[m], dtype=float)
        v = v[np.isfinite(v)]
        print(f'{m:<18}{np.median(v):>10.4f}{np.mean(v):>10.4f}'
              f'{np.min(v):>8.4f}{np.max(v):>8.4f}{len(v):>4d}')

    # 対照比較: random を基準に、repair/pr が random より良い（HV 大）かを検定。
    # 加えて random が memetic_ls（キックなし）より良いか（＝多様化一般の効果の有無）。
    print('\n=== Wilcoxon (paired, 片側) + Cliff\'s delta ===')
    print('  「A < B が有意」= B の HV が A より大（B が優位）。')
    rk = 'memetic_random'
    comparisons = [
        (rk, 'memetic_repair', 'random < repair?  (repair 優位なら S_p 誘導の効果)'),
        (rk, 'memetic_pr',     'random < pr?      (pr 優位なら S_p 誘導の効果)'),
        ('memetic_ls', rk,     'ls < random?      (random 優位なら多様化一般の効果)'),
    ]
    for a, b, desc in comparisons:
        if a not in union_hv or b not in union_hv:
            print(f'  [skip] {a} or {b} なし')
            continue
        xa, xb = np.array(union_hv[a]), np.array(union_hv[b])
        stat, p = A.wilcoxon_paired(xa, xb, alternative='less')  # a < b ?
        d = A.cliffs_delta(xa, xb)  # 負 = a が b より小
        med_a, med_b = np.median(xa), np.median(xb)
        print(f'  {desc}')
        print(f'      median {a}={med_a:.4f}  {b}={med_b:.4f}  '
              f'Δ(b-a)={med_b-med_a:+.4f}  p={p:.4f}  δ={d:+.3f} ({A.effect_label(d)})')

    # ===== 領域別 union HV: 高安定(D小, S_p近傍) / 低安定(D大, 効率帯) =====
    # 核心: S_p 方向キックの価値は高安定領域の充填に集中するはず。union HV は面積の大半が
    # 効率帯なので、方向効果は領域分割で初めてクリアに見える。
    thr = A.compute_p33_p67(md, baselines_by_method)
    if thr is not None:
        # global_ref（領域マスキングの raw 上端）を analyze_v3 と同様に算出
        gref = (float(np.concatenate(all_pts)[:, 0].max()) * 1.01,
                float(np.concatenate(all_pts)[:, 1].max()) * 1.01)
        rhv = A.compute_region_hv_per_trial(
            md, baselines_by_method, gref, thr['P50'], thr['stab_max'], norm=norm)
        for rk, rname in [('high', '高安定領域 (D∈[0,P50), S_p近傍)'),
                          ('low',  '低安定領域 (D∈[P50,max], 効率帯)')]:
            print(f'\n=== 領域別 union HV - {rname}  境界P50={thr["P50"]:.1f} ===')
            print(f'{"method":<18}{"median":>10}{"mean":>10}')
            for m in methods:
                v = np.array([x for x in rhv[m][rk] if np.isfinite(x)])
                print(f'{m:<18}{np.median(v):>10.4f}{np.mean(v):>10.4f}')
            print('  -- 対照検定 --')
            for a, b, desc in comparisons:
                xa = np.array(rhv[a][rk]); xb = np.array(rhv[b][rk])
                _, p = A.wilcoxon_paired(xa, xb, alternative='less')
                d = A.cliffs_delta(xa, xb)
                print(f'    {desc}  Δ(b-a)={np.median(xb)-np.median(xa):+.4f}  '
                      f'p={p:.4f}  δ={d:+.3f} ({A.effect_label(d)})')

    # ===== コスト比較: total_cpu_time（randomの方向ランダム化による再収束コスト）=====
    print('\n=== 計算コスト (total_cpu_time, 全重み平均 s/run) ===')
    for m in methods:
        cpus = []
        for wl in md[m]:
            for _, data in md[m][wl].items():
                c = data.get('convergence', {}).get('total_cpu_time')
                if c is not None:
                    cpus.append(float(c))
        if cpus:
            print(f'  {m:<18} mean={np.mean(cpus):7.1f}s  median={np.median(cpus):7.1f}s  n={len(cpus)}')


if __name__ == '__main__':
    main()
