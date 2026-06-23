# -*- coding: utf-8 -*-
"""本実験 main_v1 の予算妥当性チェック（収束 & 族間の予算感の比較）

budget_adequacy_check.md は本実験 *前* の param_sweep データで検証した記録。
本スクリプトは実際の本実験 results/main_v1 そのもの（7手法×10重み×6問題×n=10）で
  - scalar_f : 重みスカラー best_score が最後に改善した時刻 / T
  - pf_f     : per-trial Pareto front（UEA・baseline 除外後）が最後に更新した時刻 / T
  - t99_f    : 最終 HV の 99% に到達した時刻 / T（self-referenced）
  - t999_f   : 最終 HV の 99.9% に到達した時刻 / T
  - gain20   : (HV(T) - HV(0.8T)) / HV(T)。予算を +20% 増やして買える量の上界
  - T        : run の総 CPU 時間（族間の「予算感」＝壁時計の比較用）
を手法別に集計する。pf_f / t99_f が 1.0 近傍に張り付く run が多ければ予算不足。
全 run が小さい割合で頭打ちなら予算過剰。
"""
import glob
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

SRC = r"c:\Users\takum\Documents\研究\program\project01\SourceCode"
sys.path.insert(0, os.path.join(SRC, "experiments", "core_comparison_v3"))
sys.path.insert(0, os.path.join(SRC, "experiments"))
sys.path.insert(0, SRC)

from analyze_v3 import (
    _last_pf_update_time, _worker_trial_ttt, filter_baselines, pareto_front,
)

MAIN = os.path.join(SRC, "experiments", "core_comparison_v3", "results", "main_v1")

PROBLEMS = [
    "mt10_mt10_delay60",
    "la21_la21_delay147",
    "la40_la40_delay148",
    "la36_la36_large",
    "la36_la36_small",
    "ta21_ta21_delay97",
]

METHOD_ORDER = ["ga", "ils_baseline", "ils_repair", "ils_pr",
                "memetic_ls", "memetic_repair", "memetic_pr"]


def _hv_at(pts2, times, t_cut, ref):
    """t<=t_cut までに訪問した点だけで支配域 HV を計算（参照点 ref で正規化前の生 HV）。"""
    mask = times <= t_cut
    if not mask.any():
        return 0.0
    p = pts2[mask]
    pf = pareto_front(p)
    if len(pf) == 0:
        return 0.0
    # 最小化 2 目的の支配域 HV（ref が右上）。pf を ms 昇順でソートし矩形積分。
    pf = pf[np.argsort(pf[:, 0])]
    hv = 0.0
    prev_ms = ref[0]
    # ms 降順に積む（右から左）
    for ms, st in pf[::-1]:
        w = prev_ms - ms
        h = ref[1] - st
        if w > 0 and h > 0:
            hv += w * h
        prev_ms = ms
    return hv


def analyze_file(fpath):
    try:
        with open(fpath, encoding="utf-8") as f:
            d = json.load(f)
    except Exception:
        return None
    method = d["method"]
    hist = d.get("history") or []
    if not hist:
        return None
    times = [h["cpu_time"] for h in hist]
    T = times[-1]
    if not T or T <= 0:
        return None
    kind = "ga" if (method == "ga" or method.startswith("memetic")) else "ils"

    # scalar: best_score の最終改善時刻
    last_imp_t, prev = 0.0, None
    for h in hist:
        s = h.get("best_score")
        if s is None:
            continue
        if prev is None or s < prev - 1e-12:
            last_imp_t, prev = h["cpu_time"], s

    pts_raw = d.get("uea_points") or []
    ts = d.get("uea_points_t")
    pts = np.asarray(pts_raw, dtype=float)
    if len(pts) and ts is not None and len(ts) == len(pts):
        pts3 = np.hstack([pts[:, :2], np.asarray(ts, dtype=float).reshape(-1, 1)])
    else:
        pts3 = pts
    bl = d.get("baseline")

    last_pf_t = _last_pf_update_time(hist, pts3, kind, baseline=bl) or 0.0

    t99 = t999 = float("nan")
    gain20 = float("nan")
    if len(pts):
        ptsf2 = filter_baselines(pts[:, :2], bl)
        if len(ptsf2):
            ref = (float(ptsf2[:, 0].max()) * 1.01,
                   float(ptsf2[:, 1].max()) * 1.01 + 1e-9)
            r = _worker_trial_ttt((hist, pts3, kind, [0.99, 0.999], [], bl, ref, None))
            t99, t999 = r["self"]
            # gain20: 末尾 20% で稼いだ HV 割合（時刻つき点のみ）
            if pts3.shape[1] == 3:
                tt = pts3[:, 2]
                keep = np.ones(len(pts3), dtype=bool)
                bl_list = (bl if (isinstance(bl, list) and isinstance(bl[0], (list, np.ndarray)))
                           else [bl]) if bl is not None else []
                for b in bl_list:
                    keep &= ~((pts3[:, 0] >= b[0] - 1e-9) & (pts3[:, 1] >= b[1] - 1e-9))
                p2 = pts3[keep][:, :2]
                tt = tt[keep]
                if len(p2):
                    hv_full = _hv_at(p2, tt, T, ref)
                    hv_80 = _hv_at(p2, tt, 0.8 * T, ref)
                    gain20 = (hv_full - hv_80) / hv_full if hv_full > 0 else 0.0

    pf_iter = None
    if kind == "ils":
        pf_iter = int(np.searchsorted(np.asarray(times), last_pf_t, side="left"))

    return {
        "method": method,
        "T": T,
        "scalar_f": last_imp_t / T,
        "pf_f": last_pf_t / T,
        "t99_f": (t99 / T) if np.isfinite(t99) else float("nan"),
        "t999_f": (t999 / T) if np.isfinite(t999) else float("nan"),
        "gain20": gain20,
        "pf_iter": pf_iter,
        "n_units": len(hist) - 1,
    }


def agg(vals):
    a = np.asarray([v for v in vals if np.isfinite(v)], dtype=float)
    if len(a) == 0:
        return "—"
    return (f"med {np.median(a):.2f} / p90 {np.percentile(a, 90):.2f} / "
            f"max {a.max():.2f}")


def aggpct(vals):
    a = np.asarray([v for v in vals if np.isfinite(v)], dtype=float)
    if len(a) == 0:
        return "—"
    return (f"med {np.median(a)*100:.2f}% / p90 {np.percentile(a, 90)*100:.2f}% / "
            f"max {a.max()*100:.2f}%")


def main():
    only = sys.argv[1] if len(sys.argv) > 1 else None
    for prob in PROBLEMS:
        if only and only not in prob:
            continue
        raw_dir = os.path.join(MAIN, prob, "raw")
        files = sorted(glob.glob(os.path.join(raw_dir, "*.json")))
        if not files:
            continue
        print(f"\n================ {prob}  (runs: {len(files)}) ================")
        with ProcessPoolExecutor(max_workers=8) as ex:
            results = [r for r in ex.map(analyze_file, files) if r]
        by_m = {}
        for r in results:
            by_m.setdefault(r["method"], []).append(r)
        # T の中央値で族間の予算感を一覧
        print("  --- 総CPU時間 T (s) [族間の予算感] ---")
        for m in METHOD_ORDER:
            if m not in by_m:
                continue
            Ts = [r["T"] for r in by_m[m]]
            print(f"    {m:16s}: med {np.median(Ts):7.1f}  p90 {np.percentile(Ts,90):7.1f}  "
                  f"max {max(Ts):7.1f}  (n={len(Ts)})")
        for m in METHOD_ORDER:
            if m not in by_m:
                continue
            rs = by_m[m]
            print(f"\n  --- {m} (n={len(rs)}) ---")
            print(f"    scalar最終改善 t/T : {agg([r['scalar_f'] for r in rs])}")
            print(f"    PF最終更新   t/T : {agg([r['pf_f'] for r in rs])}")
            print(f"    HV99%到達    t/T : {agg([r['t99_f'] for r in rs])}")
            print(f"    HV99.9%到達  t/T : {agg([r['t999_f'] for r in rs])}")
            print(f"    残り20%で得たHV   : {aggpct([r['gain20'] for r in rs])}")
            n_tail = sum(1 for r in rs if r["pf_f"] > 0.9)
            n_tail99 = sum(1 for r in rs if r["pf_f"] > 0.99)
            print(f"    PF更新が残り10%/1%に掛かる run: {n_tail}/{len(rs)} / {n_tail99}/{len(rs)}")
            if rs[0]["pf_iter"] is not None:
                its = [r["pf_iter"] for r in rs]
                nu = rs[0]["n_units"]
                print(f"    [ILS] PF最終更新 iter: med {int(np.median(its))} / "
                      f"p90 {int(np.percentile(its, 90))} / max {max(its)} (budget {nu})")


if __name__ == "__main__":
    main()
