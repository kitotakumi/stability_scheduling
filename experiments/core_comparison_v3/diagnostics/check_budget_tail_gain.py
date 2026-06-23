# -*- coding: utf-8 -*-
"""終盤予算で買えたHV量の計測

center run ごとに HV(0.5T), HV(0.8T), HV(T) を計算し、
  gain_last50 = (HV(T) - HV(0.5T)) / HV(T)
  gain_last20 = (HV(T) - HV(0.8T)) / HV(T)
を手法別に集計する。「予算を+X%増やしたら買えるHV」は gain_last20 を上回らない
（HV(t) は凹型に飽和するため）。
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

from analyze_v3 import _worker_trial_hv_curve, filter_baselines

DIRS = {
    "la21": os.path.join(SRC, "experiments", "param_sweep_v1", "results",
                         "sweep_20260610_141059", "la21_la21_delay147", "raw"),
    "la36": os.path.join(SRC, "experiments", "param_sweep_v1", "results",
                         "sweep_la36", "la36_la36_large", "raw"),
}


def analyze_file(fpath):
    try:
        with open(fpath, encoding="utf-8") as f:
            d = json.load(f)
    except Exception:
        return None
    if d.get("tag") != "center":
        return None
    method = d["method"]
    hist = d.get("history") or []
    if not hist:
        return None
    T = hist[-1]["cpu_time"]
    if not T or T <= 0:
        return None
    kind = "ga" if (method == "ga" or method.startswith("memetic")) else "ils"

    pts = np.asarray(d.get("uea_points") or [], dtype=float)
    ts = d.get("uea_points_t")
    if len(pts) == 0:
        return None
    if ts is not None and len(ts) == len(pts):
        pts3 = np.hstack([pts[:, :2], np.asarray(ts, dtype=float).reshape(-1, 1)])
    else:
        pts3 = pts
    bl = d.get("baseline")
    ptsf = filter_baselines(pts[:, :2], bl)
    if len(ptsf) == 0:
        return None
    ref = (float(ptsf[:, 0].max()) * 1.01, float(ptsf[:, 1].max()) * 1.01 + 1e-9)

    curve = _worker_trial_hv_curve((hist, pts3, kind, [0.5 * T, 0.8 * T, T], bl, ref))
    hv50, hv80, hvT = curve
    if hvT <= 0:
        return None
    return {
        "method": method,
        "gain50": (hvT - hv50) / hvT * 100.0,
        "gain20": (hvT - hv80) / hvT * 100.0,
    }


def fmt(vals):
    a = np.asarray(vals, dtype=float)
    return f"med {np.median(a):6.3f}% / p90 {np.percentile(a, 90):6.3f}% / max {a.max():6.3f}%"


def main():
    for prob, raw_dir in DIRS.items():
        files = sorted(glob.glob(os.path.join(raw_dir, "*__center__*.json")))
        with ProcessPoolExecutor(max_workers=8) as ex:
            results = [r for r in ex.map(analyze_file, files) if r]
        by_m = {}
        for r in results:
            by_m.setdefault(r["method"], []).append(r)
        print(f"\n================ {prob} ================")
        print("（各runの最終HVに対する割合。gain_last20 ≒ 予算を2割削ったら失う量"
              "＝予算+20%で買える量の上限）")
        for m in sorted(by_m):
            rs = by_m[m]
            print(f"\n--- {m} (n={len(rs)}) ---")
            print(f"  残り50%の時間で得たHV: {fmt([r['gain50'] for r in rs])}")
            print(f"  残り20%の時間で得たHV: {fmt([r['gain20'] for r in rs])}")


if __name__ == "__main__":
    main()
