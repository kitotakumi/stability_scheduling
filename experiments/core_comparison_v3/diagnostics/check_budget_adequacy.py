# -*- coding: utf-8 -*-
"""ILS 3000反復 / GA・memetic 500世代の予算妥当性チェック

param_sweep_v1 の center 設定 run（現行コード・本番予算）から、各 run について
  - scalar_f : 重みスカラー best_score が最後に改善した時刻（総時間に対する割合）
  - pf_f     : per-trial Pareto front が最後に更新された時刻の割合（UEA基準＝本命）
  - t99_f    : 最終 HV の 99% に到達した時刻の割合
  - t999_f   : 最終 HV の 99.9% に到達した時刻の割合
を計算し、手法別に中央値 / P90 / 最大値を集計する。
pf_f・t99_f が 1.0 近傍に張り付く run が多ければ予算不足、
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

from analyze_v3 import _last_pf_update_time, _worker_trial_ttt, filter_baselines

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
    if len(pts):
        ptsf = filter_baselines(pts[:, :2], bl)
        if len(ptsf):
            ref = (float(ptsf[:, 0].max()) * 1.01,
                   float(ptsf[:, 1].max()) * 1.01 + 1e-9)
            r = _worker_trial_ttt((hist, pts3, kind, [0.99, 0.999], [], bl, ref, None))
            t99, t999 = r["self"]

    # ILS のみ: 最終PF更新の反復番号（history は 1 iter = 1 entry, 先頭が iter0）
    pf_iter = None
    if kind == "ils":
        pf_iter = int(np.searchsorted(np.asarray(times), last_pf_t, side="left"))

    return {
        "method": method,
        "w": f"w{int(round(d['weights'][0]*10)):02d}",
        "scalar_f": last_imp_t / T,
        "pf_f": last_pf_t / T,
        "t99_f": (t99 / T) if np.isfinite(t99) else float("nan"),
        "t999_f": (t999 / T) if np.isfinite(t999) else float("nan"),
        "pf_iter": pf_iter,
        "n_units": len(hist) - 1,
    }


def agg(vals):
    a = np.asarray([v for v in vals if np.isfinite(v)], dtype=float)
    if len(a) == 0:
        return "—"
    return (f"med {np.median(a):.2f} / p90 {np.percentile(a, 90):.2f} / "
            f"max {a.max():.2f}")


def main():
    for prob, raw_dir in DIRS.items():
        files = sorted(glob.glob(os.path.join(raw_dir, "*__center__*.json")))
        print(f"\n================ {prob}  (center runs: {len(files)}) ================")
        with ProcessPoolExecutor(max_workers=8) as ex:
            results = [r for r in ex.map(analyze_file, files) if r]
        by_m = {}
        for r in results:
            by_m.setdefault(r["method"], []).append(r)
        for m in sorted(by_m):
            rs = by_m[m]
            print(f"\n--- {m} (n={len(rs)}) ---")
            print(f"  scalar最終改善 t/T : {agg([r['scalar_f'] for r in rs])}")
            print(f"  PF最終更新   t/T : {agg([r['pf_f'] for r in rs])}")
            print(f"  HV99%到達    t/T : {agg([r['t99_f'] for r in rs])}")
            print(f"  HV99.9%到達  t/T : {agg([r['t999_f'] for r in rs])}")
            n_tail = sum(1 for r in rs if r["pf_f"] > 0.9)
            n_tail99 = sum(1 for r in rs if r["pf_f"] > 0.99)
            print(f"  PF更新が残り10%/1%に掛かる run: {n_tail}/{len(rs)} / {n_tail99}/{len(rs)}")
            if rs[0]["pf_iter"] is not None:
                its = [r["pf_iter"] for r in rs]
                nu = rs[0]["n_units"]
                print(f"  [ILS] PF最終更新 iter: med {int(np.median(its))} / "
                      f"p90 {int(np.percentile(its, 90))} / max {max(its)} (budget {nu})")
            # 重み別の PF 最終更新（どの重みが収束遅いか）
            by_w = {}
            for r in rs:
                by_w.setdefault(r["w"], []).append(r["pf_f"])
            wparts = [f"{w}:{np.median(v):.2f}" for w, v in sorted(by_w.items())]
            print(f"  重み別 pf_f 中央値: {'  '.join(wparts)}")


if __name__ == "__main__":
    main()
