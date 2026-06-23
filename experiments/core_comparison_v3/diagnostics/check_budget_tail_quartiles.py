# -*- coding: utf-8 -*-
"""終盤予算で買えたHVの分布を四分位で出し直す（Q25/med/Q75/p90）"""
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
    hv80, hvT = _worker_trial_hv_curve((hist, pts3, kind, [0.8 * T, T], bl, ref))
    if hvT <= 0:
        return None
    # trial番号も持つ: 同一trialの全重みunion単位での影響も見る
    return {"method": method, "trial": d.get("trial"), "w": d["weights"][0],
            "gain20": (hvT - hv80) / hvT * 100.0,
            "hv80": hv80, "hvT": hvT}


def main():
    for prob, raw_dir in DIRS.items():
        files = sorted(glob.glob(os.path.join(raw_dir, "*__center__*.json")))
        with ProcessPoolExecutor(max_workers=8) as ex:
            results = [r for r in ex.map(analyze_file, files) if r]
        by_m = {}
        for r in results:
            by_m.setdefault(r["method"], []).append(r)
        print(f"\n================ {prob} ================")
        print("run単位 gain_last20 [%] と、trial単位（全重みrunの平均影響）")
        for m in sorted(by_m):
            rs = by_m[m]
            g = np.asarray([r["gain20"] for r in rs])
            q25, med, q75, p90 = (np.percentile(g, 25), np.median(g),
                                  np.percentile(g, 75), np.percentile(g, 90))
            # trial単位: そのtrialの全重みrunのgainの平均（union HVへの希釈の粗い近似）
            by_t = {}
            for r in rs:
                by_t.setdefault(r["trial"], []).append(r["gain20"])
            tg = np.asarray([np.mean(v) for v in by_t.values()])
            print(f"--- {m} (n={len(rs)} run / {len(tg)} trial) ---")
            print(f"  run単位:   Q25 {q25:6.3f} | med {med:6.3f} | Q75 {q75:6.3f} | p90 {p90:6.3f}")
            print(f"  trial単位: Q25 {np.percentile(tg,25):6.3f} | med {np.median(tg):6.3f} "
                  f"| Q75 {np.percentile(tg,75):6.3f} | max {tg.max():6.3f}")


if __name__ == "__main__":
    main()
