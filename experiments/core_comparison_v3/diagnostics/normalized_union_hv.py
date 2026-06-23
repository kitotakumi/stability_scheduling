"""正規化 union HV を計算し、生 HV とランキング・検定の一致を確認する。

正規化: (problem, scenario) ごとに全手法・全重み・全trialの訪問点（baseline除外後）から
  ideal  = (MS_min, D=0)          ← 各軸の最良
  nadir  = (MS_max, D_max)        ← 各軸の最悪（観測）
を取り、各点を x'=(x-ideal)/(nadir-ideal) で [0,1]^2 に写す（MS と D を等重み）。
参照点 = (1.1, 1.1)（Ishibuchi et al. 2018 推奨の nadir 外側 10%）。

生 HV と並べて per-method 中央値 [IQR]、および主要比較の Wilcoxon p / Cliff's δ を出力。
共通アフィン正規化なので全手法が同一定数倍 → 順位・p値・δ は不変のはず（その確認が目的）。
"""
import os
import sys
import numpy as np

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, '..'))  # core_comparison_v3/ を import パスに
import analyze_v3 as A

REF = 1.1  # 正規化空間の参照点（両軸）

# 主要比較（A, B, alternative は A>B = A の HV が大きいか）
COMPARISONS = [
    ('memetic_ls', 'ils_baseline', '軌道vs集団: Memetic-LS > ILS-base'),
    ('memetic_pr', 'ils_pr',       '軌道vs集団: Memetic+PR > ILS+PR'),
    ('memetic_repair', 'ils_repair','軌道vs集団: Memetic+rep > ILS+rep'),
    ('ils_repair', 'ils_baseline', '機構(ILS): ILS+rep > ILS-base'),
    ('ils_pr', 'ils_baseline',     '機構(ILS): ILS+PR > ILS-base'),
    ('memetic_repair', 'memetic_ls','機構(Mem): Mem+rep > Mem-LS'),
    ('memetic_pr', 'memetic_ls',   '機構(Mem): Mem+PR > Mem-LS'),
]
ORDER = ['ga', 'ils_baseline', 'ils_pr', 'ils_repair',
         'memetic_ls', 'memetic_pr', 'memetic_repair']


def per_trial_union_hv(method_data, methods, bl_by_m, transform, ref):
    """各手法の per-trial union HV を返す。transform(pts)->normalized pts。"""
    n_tr = 1 + max(max(method_data[m][wl].keys())
                   for m in methods for wl in method_data[m])
    out = {}
    for m in methods:
        bl = bl_by_m[m]
        hvs = []
        for t in range(n_tr):
            up = []
            for wl in method_data[m]:
                data = method_data[m][wl].get(t)
                if data is None:
                    continue
                pts = A.get_uea_points(data, t)
                if bl:
                    pts = A.filter_baselines(pts, bl)
                if len(pts):
                    up.append(pts)
            if not up:
                continue
            pf = A.pareto_front(transform(np.concatenate(up)))
            hvs.append(A.hypervolume(pf, ref))
        out[m] = hvs
    return out


def med_iqr(v):
    v = [x for x in v if np.isfinite(x)]
    if not v:
        return float('nan'), float('nan'), float('nan')
    return np.median(v), np.percentile(v, 25), np.percentile(v, 75)


def main(results_dir, scenarios):
    grouped = A.load_all_runs(results_dir)
    for (prob, scen), method_data in sorted(grouped.items()):
        tag = f'{prob}_{scen}'
        if scenarios and not any(s in tag for s in scenarios):
            continue
        methods = [m for m in ORDER if m in method_data] or list(method_data.keys())

        bl_by_m = {}
        for m in methods:
            bls = []
            for wl in method_data[m]:
                for _t, data in method_data[m][wl].items():
                    b1 = data.get('baseline'); b2 = data.get('baseline_rsr')
                    if b1 is not None:
                        bls.append(b1)
                    if b2 is not None and list(b2) not in bls:
                        bls.append(list(b2))
                    break
                if bls:
                    break
            bl_by_m[m] = bls if bls else None

        # ideal / nadir
        allpts = []
        for m in methods:
            bl = bl_by_m[m]
            for wl in method_data[m]:
                for t, data in method_data[m][wl].items():
                    pts = A.get_uea_points(data, t)
                    if bl:
                        pts = A.filter_baselines(pts, bl)
                    if len(pts):
                        allpts.append(pts)
        cat = np.concatenate(allpts)
        ms_min, ms_max = float(cat[:, 0].min()), float(cat[:, 0].max())
        d_max = float(cat[:, 1].max())
        ms_rng = max(ms_max - ms_min, 1e-9)
        d_rng = max(d_max - 0.0, 1e-9)

        def norm(pts):
            p = np.asarray(pts, dtype=float).copy()
            p[:, 0] = (p[:, 0] - ms_min) / ms_rng
            p[:, 1] = (p[:, 1] - 0.0) / d_rng
            return p

        raw_ref = (ms_max + max(ms_max * 0.01, 1.0), d_max + max(d_max * 0.01, 0.01))
        hv_norm = per_trial_union_hv(method_data, methods, bl_by_m, norm, (REF, REF))
        hv_raw = per_trial_union_hv(method_data, methods, bl_by_m,
                                    lambda p: np.asarray(p, float), raw_ref)

        print(f'\n===== {tag} =====')
        print(f'  ideal=(MS {ms_min:.0f}, D 0)  nadir=(MS {ms_max:.0f}, D {d_max:.0f})'
              f'  正規化参照点=({REF},{REF})')
        print(f'  {"method":<16}{"生 union HV":>14}{"正規化 HV[IQR]":>26}{"順位":>6}')
        # 順位（正規化中央値で降順）
        med_n = {m: med_iqr(hv_norm.get(m, []))[0] for m in methods}
        med_r = {m: med_iqr(hv_raw.get(m, []))[0] for m in methods}
        rank_n = {m: i+1 for i, m in enumerate(
            sorted(methods, key=lambda x: -med_n[x]))}
        rank_r = {m: i+1 for i, m in enumerate(
            sorted(methods, key=lambda x: -med_r[x]))}
        for m in methods:
            mn, q1, q3 = med_iqr(hv_norm.get(m, []))
            flag = '' if rank_n[m] == rank_r[m] else f' (生={rank_r[m]})'
            print(f'  {m:<16}{med_r[m]:>14.1f}{mn:>16.4f} [{q1:.4f},{q3:.4f}]'
                  f'{rank_n[m]:>4}{flag}')

        print(f'  {"比較 (A>B)":<34}{"生: p / δ":>20}{"正規化: p / δ":>22}')
        for a, b, label in COMPARISONS:
            if a not in hv_norm or b not in hv_norm:
                continue
            xr, yr = hv_raw[a], hv_raw[b]
            xn, yn = hv_norm[a], hv_norm[b]
            # alternative='greater' → A>B: x>y。analyze の wilcoxon は 'less'=x<y。
            pr = A.wilcoxon_paired(xr, yr, alternative='greater')[1]
            pn = A.wilcoxon_paired(xn, yn, alternative='greater')[1]
            dr = A.cliffs_delta(xr, yr)
            dn = A.cliffs_delta(xn, yn)
            print(f'  {label:<34}{_fmt(pr)}/{dr:+.2f}   {_fmt(pn)}/{dn:+.2f}')


def _fmt(p):
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return ' nan '
    return f'{p:.4f}'


if __name__ == '__main__':
    rd = sys.argv[1] if len(sys.argv) > 1 else \
        'experiments/core_comparison_v3/results/main_v1'
    scen = sys.argv[2:] if len(sys.argv) > 2 else []
    main(rd, scen)
