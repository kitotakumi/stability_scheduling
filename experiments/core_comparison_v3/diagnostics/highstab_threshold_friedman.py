"""高安定 HV の閾値感度 — Friedman 平均順位（§4.4 スコアボード）版。

highstab_threshold_sensitivity.py は per-scenario の順位と pairwise 検定のみを
出力する。本スクリプトはその不足を補い、§4.4 の**横断 Friedman 平均順位**
（analyze_v3._friedman_avg_rank と同一定義: 各シナリオで手法を高安定 HV 中央値の
降順に rankdata し、全シナリオで平均）を分割点 P25/P33/P50/P67/P75 ごとに
再計算する。目的は「二分構造（上位群 vs 素の集団の最下位群）が閾値に頑健か」
「上位群内部の順位のみが動くか」を Friedman 高度で検証すること。

analyze_v3 本体は書き換えない（読み取り専用の診断）。前処理と region_hv は
highstab_threshold_sensitivity.py のヘルパを import して完全に共有する。

使い方:
  python experiments/core_comparison_v3/diagnostics/highstab_threshold_friedman.py [results_dir]
"""
import os
import sys
import numpy as np

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, '..'))   # core_comparison_v3/
sys.path.insert(0, HERE)                         # diagnostics/（sensitivity を import）
import analyze_v3 as A
import highstab_threshold_sensitivity as S

PERCENTILES = [25, 33, 50, 67, 75]
ORDER = A.METHOD_ORDER


def scenario_median_matrix(results_dir):
    """各シナリオ×手法の {P: median高安定HV} を集約。
    Returns (scen_tags, methods, med[P] = (N_scen, k_method) 行列)。"""
    grouped = A.load_all_runs(results_dir)
    # リスケ率でシナリオを整列（スコアボードと同じ並び）
    items = sorted(grouped.items(),
                   key=lambda kv: (A.reschedule_rate(f'{kv[0][0]}_{kv[0][1]}'), kv[0]))

    # 全シナリオに共通して存在する手法のみ（Friedman は完全ブロック前提）
    common = None
    per_scen = []   # (tag, method_data, methods, bl, norm, ref, stab)
    for (prob, scen), method_data in items:
        tag = f'{prob}_{scen}'
        methods = [m for m in ORDER if m in method_data] or list(method_data.keys())
        bl_by_m = S.build_baselines(method_data, methods)
        all_pts = []
        for m in methods:
            bl = bl_by_m.get(m)
            for wl in method_data[m]:
                for t, data in method_data[m][wl].items():
                    pts = A.get_uea_points(data, t)
                    if bl:
                        pts = A.filter_baselines(pts, bl)
                    if len(pts):
                        all_pts.append(pts)
        if not all_pts:
            continue
        cat = np.concatenate(all_pts)
        norm = A.make_norm(cat)
        ref = float(cat[:, 0].max()) + max(cat[:, 0].max() * 0.01, 1.0)
        stab = S.pooled_stab(method_data, methods, bl_by_m)
        if len(stab) == 0:
            continue
        pf_by = S.per_trial_union_pf(method_data, methods, bl_by_m)
        per_scen.append((tag, methods, norm, ref, stab, pf_by))
        common = set(methods) if common is None else (common & set(methods))

    methods = [m for m in ORDER if m in common]
    tags = [ps[0] for ps in per_scen]

    med = {}
    for P in PERCENTILES:
        M = np.zeros((len(per_scen), len(methods)))
        for i, (tag, ms, norm, ref, stab, pf_by) in enumerate(per_scen):
            thr = float(np.percentile(stab, P))
            hs = S.highstab_hv_per_trial(pf_by, ms, thr, ref, norm)
            for j, m in enumerate(methods):
                vals = [v for v in hs[m] if np.isfinite(v)]
                M[i, j] = np.median(vals) if vals else 0.0
        med[P] = M
    return tags, methods, med


def main(results_dir):
    tags, methods, med = scenario_median_matrix(results_dir)
    print(f'シナリオ (N={len(tags)}): ' + ', '.join(A.problem_short_tag(t) for t in tags))
    print(f'手法 (k={len(methods)}): ' + ', '.join(A.METHOD_LABELS.get(m, m) for m in methods))

    # 各パーセンタイルで Friedman 平均順位を算出
    rank_by_p, order_by_p, stats_by_p = {}, {}, {}
    for P in PERCENTILES:
        avg_rank, chi, p, W, ranks = A._friedman_avg_rank(med[P])
        rank_by_p[P] = {m: avg_rank[j] for j, m in enumerate(methods)}
        order_by_p[P] = [methods[j] for j in np.argsort(avg_rank)]
        stats_by_p[P] = (p, W)

    base_order = order_by_p[50]
    base_pos = {m: i for i, m in enumerate(base_order)}

    print('\n=== 高安定 HV の Friedman 平均順位（列=分割パーセンタイル, 小さいほど良い）===')
    hdr = f'  {"method":<16}' + ''.join(f'P{P:<9}' for P in PERCENTILES)
    print(hdr)
    for m in methods:
        cells = ''
        for P in PERCENTILES:
            r = rank_by_p[P][m]
            # P50 の順位順位置と比べて動いたら * 印
            pos = order_by_p[P].index(m)
            mark = '' if pos == base_pos[m] else '*'
            cells += f'{r:.2f}{mark:<5}'
        print(f'  {A.METHOD_LABELS.get(m,m):<16}{cells}')

    print('\n  Friedman p / Kendall W:')
    for P in PERCENTILES:
        p, W = stats_by_p[P]
        print(f'    P{P}: p={p:.4g}, W={W:.3f}')

    # ---- 二分構造の検証 ----
    print('\n=== 二分構造・順位頑健性の判定 ===')
    # 各 P で順位（昇順 = 良い順）を並べる
    for P in PERCENTILES:
        seq = ' > '.join(A.METHOD_LABELS.get(m, m) for m in order_by_p[P])
        print(f'  P{P}: {seq}')

    # 最下位2手法が全 P で {GA, Memetic-LS} か
    bottom2 = {P: set(order_by_p[P][-2:]) for P in PERCENTILES}
    all_bottom = set().union(*bottom2.values())
    bottom_stable = len({frozenset(b) for b in bottom2.values()}) == 1
    print(f'\n  最下位2手法: ' +
          '; '.join(f'P{P}={{{", ".join(A.METHOD_LABELS.get(m,m) for m in bottom2[P])}}}'
                    for P in PERCENTILES))
    print(f'  → 最下位2手法は全閾値で{"不変" if bottom_stable else "変動!"}'
          f'（{", ".join(A.METHOD_LABELS.get(m,m) for m in sorted(all_bottom))}）')

    # 上位群（top-5）のメンバーシップが全 P で不変か
    top5 = {P: set(order_by_p[P][:5]) for P in PERCENTILES}
    top_stable = len({frozenset(t) for t in top5.values()}) == 1
    print(f'  → 上位5手法のメンバーは全閾値で{"不変" if top_stable else "変動!"}')

    # 順位そのもの（完全順序）が動くか
    full_stable = len({tuple(order_by_p[P]) for P in PERCENTILES}) == 1
    print(f'  → 完全順序は{"全閾値で不変" if full_stable else "閾値で変動（上位群内で入れ替わり）"}')


if __name__ == '__main__':
    rd = sys.argv[1] if len(sys.argv) > 1 else \
        os.path.join(HERE, '..', 'results', 'main_v1')
    main(rd)
