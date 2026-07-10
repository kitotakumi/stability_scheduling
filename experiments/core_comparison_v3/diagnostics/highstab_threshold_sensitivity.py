"""高安定 HV の閾値パーセンタイル感度（査読対策: P50 の内生性ロバストネス）。

査読論点: 高安定 HV は「全手法・全trial の個別 Pareto 解プールの P50」で高安定/低安定を
分割している。分割点が手法セットに依存して動く（＝主指標がメソッド依存で定義される）ため、
分割パーセンタイルを変えても手法の順位・有意性が不変か確認する。

本スクリプトは analyze_v3 本体と同一の前処理（baseline 除外・正規化アンカー・per-trial
union PF・region_hv）を再利用し、分割点だけを P25/P33/P50/P67/P75 に振って:
  1. 各パーセンタイルでの手法順位（高安定 HV 中央値の降順）と首位手法
  2. 主要ペアの Wilcoxon p / Cliff's δ（機構・軌道vs集団・PR vs repair）
を出力する。順位・有意性が P に対して不変であれば「P50 の内生性は結論を左右しない」ことの
証拠になる。analyze_v3 は一切書き換えない（読み取り専用の診断）。

使い方:
  python experiments/core_comparison_v3/diagnostics/highstab_threshold_sensitivity.py \
      [results_dir] [scenario_substr ...]
  デフォルト results_dir = experiments/core_comparison_v3/results/main_v1
"""
import os
import sys
import numpy as np

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, '..'))  # core_comparison_v3/ を import パスに
import analyze_v3 as A

PERCENTILES = [25, 33, 50, 67, 75]
ORDER = A.METHOD_ORDER  # ['ga','ils_baseline','ils_pr','ils_repair','memetic_ls',...]

# analyze_v3 の COMPARE_FAMILIES と同一の主要ペア（A>B = A の高安定 HV が大きい）。
FAMILIES = A.COMPARE_FAMILIES


def build_baselines(method_data, methods):
    """analyze_v3 本体と同一手順で手法別 baseline リストを構築。"""
    bl_by_m = {}
    for m in methods:
        bls = []
        for wl in method_data[m]:
            for _t, data in method_data[m][wl].items():
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
        bl_by_m[m] = bls if bls else None
    return bl_by_m


def pooled_stab(method_data, methods, bl_by_m):
    """compute_p33_p67 と同一: 各 (method,w,trial) 個別 PF の stab を全プール。"""
    all_stab = []
    for m in methods:
        bl = bl_by_m.get(m)
        for wl in method_data[m]:
            for t, data in method_data[m][wl].items():
                pts = A.get_uea_points(data, t)
                if bl is not None:
                    pts = A.filter_baselines(pts, bl)
                if len(pts) == 0:
                    continue
                pf = A.pareto_front(pts)
                if len(pf) > 0:
                    all_stab.extend(pf[:, 1].tolist())
    return np.array(all_stab) if all_stab else np.zeros(0)


def per_trial_union_pf(method_data, methods, bl_by_m):
    """各手法の per-trial union Pareto front（全重み合体）を返す。"""
    all_trials = set()
    for m in methods:
        for wl in method_data[m]:
            all_trials.update(method_data[m][wl].keys())
    n_tr = (max(all_trials) + 1) if all_trials else 0
    out = {}
    for m in methods:
        bl = bl_by_m.get(m)
        pfs = []
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
            pfs.append(A.pareto_front(np.concatenate(up)) if up else np.zeros((0, 2)))
        out[m] = pfs
    return out


def highstab_hv_per_trial(pf_by_method, methods, thr, ref_ms, norm):
    """閾値 thr で高安定領域 [0,thr) の per-trial region HV を返す（analyze と同一定義）。"""
    out = {}
    for m in methods:
        vals = []
        for pf_t in pf_by_method[m]:
            if len(pf_t) == 0:
                vals.append(0.0)
                continue
            hv, _ = A.region_hv(pf_t, 0.0, thr, ref_ms, norm=norm)
            vals.append(hv)
        out[m] = vals
    return out


def _fmt_p(p):
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return ' nan '
    return f'{p:.4f}'


def main(results_dir, scenarios):
    grouped = A.load_all_runs(results_dir)
    for (prob, scen), method_data in sorted(grouped.items(),
                                            key=lambda kv: (A.reschedule_rate(f'{kv[0][0]}_{kv[0][1]}'),
                                                            kv[0])):
        tag = f'{prob}_{scen}'
        if scenarios and not any(s in tag for s in scenarios):
            continue
        methods = [m for m in ORDER if m in method_data] or list(method_data.keys())

        bl_by_m = build_baselines(method_data, methods)

        # 正規化アンカー（make_norm と同一: baseline 除外後の全訪問点）と global_ref。
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
        global_ref0 = float(cat[:, 0].max()) + max(cat[:, 0].max() * 0.01, 1.0)

        stab = pooled_stab(method_data, methods, bl_by_m)
        if len(stab) == 0:
            continue
        pf_by_method = per_trial_union_pf(method_data, methods, bl_by_m)

        rate = A.reschedule_rate(tag)
        print(f'\n========== {tag}  (リスケ率 {rate:.1%})  '
              f'{A.problem_short_tag(tag)} ==========')
        print(f'  pooled PF stab n={len(stab)}  '
              f'P25={np.percentile(stab,25):.3f} P33={np.percentile(stab,33):.3f} '
              f'P50={np.percentile(stab,50):.3f} P67={np.percentile(stab,67):.3f} '
              f'P75={np.percentile(stab,75):.3f}')

        # ---- 1. 手法順位（高安定 HV 中央値の降順）を各パーセンタイルで ----
        rank_by_p = {}
        med_by_p = {}
        for P in PERCENTILES:
            thr = float(np.percentile(stab, P))
            hs = highstab_hv_per_trial(pf_by_method, methods, thr, global_ref0, norm)
            meds = {m: (np.median([v for v in hs[m] if np.isfinite(v)])
                        if any(np.isfinite(v) for v in hs[m]) else float('nan'))
                    for m in methods}
            order = sorted(methods, key=lambda x: -(meds[x] if np.isfinite(meds[x]) else -1))
            rank_by_p[P] = {m: i + 1 for i, m in enumerate(order)}
            med_by_p[P] = meds

        base_rank = rank_by_p[50]
        print('  --- 高安定 HV 中央値による手法順位（列=分割パーセンタイル）---')
        hdr = f'    {"method":<16}' + ''.join(f'P{P:<7}' for P in PERCENTILES)
        print(hdr)
        any_rank_change = False
        for m in methods:
            cells = ''
            for P in PERCENTILES:
                r = rank_by_p[P][m]
                mark = '' if r == base_rank[m] else '*'
                if r != base_rank[m]:
                    any_rank_change = True
                cells += f'{("#"+str(r)+mark):<8}'
            print(f'    {A.METHOD_LABELS.get(m,m):<16}{cells}')
        # 首位手法の不変性
        winners = {P: min(methods, key=lambda x: rank_by_p[P][x]) for P in PERCENTILES}
        win_set = set(winners.values())
        print(f'  首位手法: ' + ', '.join(f'P{P}={A.METHOD_LABELS.get(winners[P],winners[P])}'
                                            for P in PERCENTILES))
        print(f'  → 首位{"不変" if len(win_set)==1 else "変化あり"} / '
              f'順位{"完全不変" if not any_rank_change else "一部変動(*印)"}')

        # ---- 2. 主要ペアの Wilcoxon p / Cliff's δ を各パーセンタイルで ----
        print('  --- 主要ペア Wilcoxon p (A>B) / Cliff δ（列=分割パーセンタイル）---')
        for fam_key, pairs in FAMILIES.items():
            print(f'    [{A.FAMILY_LABELS.get(fam_key, fam_key)}]')
            for (a, b) in pairs:
                if a not in pf_by_method or b not in pf_by_method:
                    continue
                cells = ''
                sig_flags = []
                for P in PERCENTILES:
                    thr = float(np.percentile(stab, P))
                    hs = highstab_hv_per_trial(pf_by_method, methods, thr, global_ref0, norm)
                    xa, xb = hs[a], hs[b]
                    p = A.wilcoxon_paired(xa, xb, alternative='greater')[1]
                    d = A.cliffs_delta(xa, xb)
                    sig_flags.append(bool(np.isfinite(p) and p < 0.05))
                    cells += f'{_fmt_p(p)}/{d:+.2f}  '
                inv = 'sig不変' if len(set(sig_flags)) == 1 else 'sig変動!'
                lab = f'{A.METHOD_LABELS.get(a,a)}>{A.METHOD_LABELS.get(b,b)}'
                print(f'      {lab:<32}{cells}[{inv}]')


if __name__ == '__main__':
    rd = sys.argv[1] if len(sys.argv) > 1 else \
        os.path.join(HERE, '..', 'results', 'main_v1')
    scen = sys.argv[2:] if len(sys.argv) > 2 else []
    main(rd, scen)
