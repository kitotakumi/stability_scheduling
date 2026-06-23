"""union HV を D（安定性偏差）バンドごとのスラブ寄与に厳密分解する診断ツール。

union HV は raw (MS, D) 空間の単一 staircase 面積。各 Pareto 点 p の寄与スラブ
  (prev_x - MS_p) * (ref_D - D_p)
を、その点の D_p が属するバンド（高安定 D<P50 / 低安定 D>=P50）に割り振る。
スラブ和は total union HV にちょうど一致する（領域別 HV と違いローカル参照点を使わない）。

目的: 「統合 HV の中で高安定領域と低安定領域が実際どれだけ面積を稼いでいるか」を、
領域別 HV（ローカル参照点・絶対値が小さく比較不能）ではなく統合 HV と同一スケールで見る。
"""
import os
import sys
import numpy as np

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, '..'))  # core_comparison_v3/ を import パスに
import analyze_v3 as A


def hv_slabs_by_band(pf, ref, p50):
    """pf を staircase 分解し、各点スラブを D バンドに割り振る。
    Returns (hv_high, hv_low, hv_total)。"""
    if len(pf) == 0:
        return 0.0, 0.0, 0.0
    pf = pf[np.argsort(pf[:, 0])]
    hv_high = hv_low = 0.0
    prev_x = ref[0]
    for p in pf[::-1]:
        if p[0] >= ref[0] or p[1] >= ref[1]:
            continue
        slab = (prev_x - p[0]) * (ref[1] - p[1])
        if p[1] < p50:
            hv_high += slab
        else:
            hv_low += slab
        prev_x = p[0]
    return hv_high, hv_low, hv_high + hv_low


def main(results_dir, scenarios):
    grouped = A.load_all_runs(results_dir)
    for (prob, scen), method_data in sorted(grouped.items()):
        tag = f'{prob}_{scen}'
        if scenarios and not any(s in tag for s in scenarios):
            continue
        methods = list(method_data.keys())

        # baselines_by_method（analyze_v3 と同一ロジック）
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

        # global_ref（全手法・全重み・全trial）
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
        ref = (float(cat[:, 0].max()) + max(cat[:, 0].max() * 0.01, 1.0),
               float(cat[:, 1].max()) + max(cat[:, 1].max() * 0.01, 0.01))
        thr = A.compute_p33_p67(method_data, bl_by_m)
        p50 = thr['P50']

        print(f'\n===== {tag} =====')
        print(f'  global_ref = (MS {ref[0]:.1f}, D {ref[1]:.2f})   P50(D)={p50:.1f}'
              f'   stab_max(D)={thr["stab_max"]:.1f}')
        print(f'  {"method":<16}{"unionHV":>10}{"高安定寄与":>12}{"低安定寄与":>12}'
              f'{"高安定%":>9}')
        # 全 trial の中央値で集計
        all_w = set()
        for m in methods:
            all_w.update(method_data[m].keys())
        n_tr = 1 + max(max(method_data[m][wl].keys())
                       for m in methods for wl in method_data[m])
        for m in methods:
            bl = bl_by_m[m]
            rows = []
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
                pf = A.pareto_front(np.concatenate(up))
                rows.append(hv_slabs_by_band(pf, ref, p50))
            if not rows:
                continue
            arr = np.array(rows)
            hi, lo, tot = np.median(arr[:, 0]), np.median(arr[:, 1]), np.median(arr[:, 2])
            print(f'  {m:<16}{tot:>10.1f}{hi:>12.1f}{lo:>12.1f}{100*hi/tot:>8.1f}%')


if __name__ == '__main__':
    rd = sys.argv[1] if len(sys.argv) > 1 else \
        'experiments/core_comparison_v3/results/main_v1'
    scen = sys.argv[2:] if len(sys.argv) > 2 else []
    main(rd, scen)
