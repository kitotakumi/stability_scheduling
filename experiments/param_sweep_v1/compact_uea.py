#!/usr/bin/env python3
"""既存の結果 JSON の uea_points を (ms,st) で重複除去（最早 cpu_time 保持）して書き戻す。

run_v3 の保存時 dedup を後付けで既存データに適用する一回限りのユーティリティ。
HV / HV(t) / 領域HV は重複点に不変なので分析結果は変わらず、memetic のファイルが
~99% 縮小してロード・解析が桁で速くなる。core_v3 / param_sweep どちらの結果dirにも使える。

使い方: python compact_uea.py --input-dir <results dir>  [--dry-run]
"""

import argparse
import glob
import json
import os

import numpy as np


def _dedup(pts, times):
    """(ms,st) ごとに最早 time を残す。pts:[[ms,st],...], times:[t,...]（同長）。"""
    if not pts or times is None or len(times) != len(pts):
        return pts, times, False
    arr = np.asarray(pts, dtype=float)
    tarr = np.asarray(times, dtype=float)
    if arr.ndim != 2 or arr.shape[1] < 2 or len(arr) <= 1:
        return pts, times, False
    o = np.argsort(tarr, kind='stable')
    arr, tarr = arr[o], tarr[o]
    uvals, uidx = np.unique(arr, axis=0, return_index=True)
    if len(uvals) == len(pts):
        return pts, times, False  # 重複なし
    return uvals.tolist(), tarr[uidx].tolist(), True


def main():
    ap = argparse.ArgumentParser(description='uea_points を dedup して縮約')
    ap.add_argument('--input-dir', required=True)
    ap.add_argument('--dry-run', action='store_true', help='書き換えずに削減量だけ表示')
    args = ap.parse_args()

    files = glob.glob(os.path.join(args.input_dir, '*', 'raw', '*.json'))
    n_changed = 0
    raw_before = raw_after = 0
    bytes_before = bytes_after = 0
    for f in files:
        try:
            with open(f, encoding='utf-8') as fh:
                d = json.load(fh)
        except Exception:
            continue
        pts = d.get('uea_points')
        ts = d.get('uea_points_t')
        if not pts:
            continue
        new_pts, new_ts, changed = _dedup(pts, ts)
        raw_before += len(pts)
        raw_after += len(new_pts)
        if not changed:
            continue
        n_changed += 1
        if args.dry_run:
            continue
        bytes_before += os.path.getsize(f)
        d['uea_points'] = new_pts
        d['uea_points_t'] = new_ts
        with open(f, 'w', encoding='utf-8') as fh:
            json.dump(d, fh, ensure_ascii=False)
        bytes_after += os.path.getsize(f)

    print(f'対象: {len(files)} files  縮約: {n_changed}')
    if raw_before:
        print(f'総点数: {raw_before:,} → {raw_after:,} ({raw_after/raw_before*100:.1f}%)')
    if bytes_after:
        print(f'縮約ファイルのサイズ: {bytes_before/1e6:.1f}MB → {bytes_after/1e6:.1f}MB')
    if args.dry_run:
        print('(dry-run: 書き換えていません)')


if __name__ == '__main__':
    main()
