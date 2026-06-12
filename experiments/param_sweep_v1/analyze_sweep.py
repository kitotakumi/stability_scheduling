#!/usr/bin/env python3
"""param_sweep_v1: 感度分析の集計・検定。

指標・統計検定は core_comparison_v3/analyze_v3.py を import して再利用する。
各 (問題 × 軸) について、対象手法ごとに center と各掃引値を並べ:
  - per-trial union UEA HV（主・品質）
  - 領域別 HV（高安定 D≤P50 / 低安定 D>P50）
  - TTT@95%（自己参照・per-trial union HV・速度）
  - center に対する Wilcoxon(two-sided) + Cliff's δ
を summary.md に出力する。

使い方: python analyze_sweep.py --input-dir results/main
"""

import argparse
import glob
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, '..', '..'))
sys.path.insert(0, os.path.join(_HERE, '..'))
sys.path.insert(0, os.path.join(_HERE, '..', 'core_comparison_v3'))

import analyze_v3 as A  # 指標・検定の本体


# ========== ロード ==========

def load_runs(input_dir):
    """raw/*.json をロード。
    Returns:
      problems: {prob_label: {(method,tag): {w_label: {trial: data}}}}
      cfg_meta: {(method,tag): {'axis','value','method'}}
    """
    problems = {}
    cfg_meta = {}
    for f in glob.glob(os.path.join(input_dir, '*', 'raw', '*.json')):
        try:
            with open(f, encoding='utf-8') as fh:
                d = json.load(fh)
        except Exception:
            continue
        prob = f'{d["problem"]}_{d["scenario"]}'
        key = (d['method'], d['tag'])
        wl = A._weight_label(d['weights'])
        problems.setdefault(prob, {}).setdefault(key, {}).setdefault(wl, {})[d['trial']] = d
        cfg_meta[key] = {'axis': d.get('axis'), 'value': d.get('value'), 'method': d['method']}
    return problems, cfg_meta


# ========== per-trial union 構築 ==========

def _xyt(data, trial):
    x = A.get_uea_points_xyt(data, trial)
    if len(x) == 0:
        return np.zeros((0, 3))
    if x.shape[1] < 3:  # 旧データ（時刻なし）→ 0 で埋め HV のみ使用
        x = np.hstack([x[:, :2], np.zeros((len(x), 1))])
    return x


def _dedup_xyt(xyt):
    """(ms,st) で重複除去し最早時刻のみ残す。HV/HV(t)/領域HV は重複点に不変なので
    結果は同じだが、memetic（全世代×全個体で同一(ms,st)が大量）では点数が桁で減り、
    下流の pareto_front / TTT が劇的に速くなる。"""
    if len(xyt) <= 1:
        return xyt
    o = np.argsort(xyt[:, 2], kind='stable')           # 時刻昇順
    s = xyt[o]
    uvals, uidx = np.unique(s[:, :2], axis=0, return_index=True)  # 最初=最早時刻
    return np.column_stack([uvals, s[uidx, 2]])


def union_xyt_per_trial(by_w_trial):
    """{w: {trial: data}} → {trial: union_xyt (N,3)}（全重みを trial 内で結合・重複除去）。"""
    trials = set()
    for bt in by_w_trial.values():
        trials.update(bt.keys())
    out = {}
    for t in sorted(trials):
        parts = [_xyt(bt[t], t) for bt in by_w_trial.values() if t in bt]
        parts = [p for p in parts if len(p) > 0]
        out[t] = _dedup_xyt(np.concatenate(parts)) if parts else np.zeros((0, 3))
    return out


def _baseline_of(by_w_trial):
    for bt in by_w_trial.values():
        for d in bt.values():
            return d.get('baseline')
    return None


# ========== 参照点・閾値（問題ごと） ==========

def problem_ref_p50(union_cache, baselines):
    """キャッシュ済み union(重複除去) の PF 点をプールして ref(=最大点+margin)・P50 を返す。"""
    pooled_pf = []
    for key, trial_xyt in union_cache.items():
        bl = baselines.get(key)
        for t, xyt in trial_xyt.items():
            if len(xyt) == 0:
                continue
            xy = xyt[:, :2]
            if bl:
                xy = A.filter_baselines(xy, bl)
            if len(xy) > 0:
                pooled_pf.append(A.pareto_front(xy))
    if not pooled_pf:
        return (1.0, 1.0), 0.0, 1.0
    allp = np.concatenate(pooled_pf)
    ref = (float(allp[:, 0].max()) + max(allp[:, 0].max() * 0.01, 1.0),
           float(allp[:, 1].max()) + max(allp[:, 1].max() * 0.01, 0.01))
    p50 = float(np.percentile(allp[:, 1], 50))
    return ref, p50, ref[1]


# ========== config の指標（trial 配列 + 集計） ==========

def config_trial_metrics(trial_xyt, bl, ref, p50, stab_max):
    """config の per-trial 指標配列を返す: {hv, rhv_high, rhv_low, ttt95} 各 list。
    trial_xyt は union_xyt_per_trial（重複除去済）の結果を使い回す（再構築しない）。"""
    res = {'hv': [], 'rhv_high': [], 'rhv_low': [], 'ttt95': []}
    for t in sorted(trial_xyt.keys()):
        xyt = trial_xyt[t]
        xy = xyt[:, :2]
        f = A.filter_baselines(xy, bl) if bl else xy
        pf = A.pareto_front(f) if len(f) > 0 else np.zeros((0, 2))
        res['hv'].append(float(A.hypervolume(pf, ref)) if len(pf) > 0 else 0.0)
        res['rhv_high'].append(A.region_hv(pf, 0.0, p50, ref[0], hi_inclusive=False)[0]
                               if len(pf) > 0 else 0.0)
        res['rhv_low'].append(A.region_hv(pf, p50, stab_max, ref[0], hi_inclusive=True)[0]
                              if len(pf) > 0 else 0.0)
        # TTT@95（自己参照・union HV）: _worker_trial_ttt を直接呼ぶ（Nx3 で時刻を使う）
        r = A._worker_trial_ttt(([], xyt, 'ils', [0.95], [], bl, ref, None))
        res['ttt95'].append(r['self'][0])
    return res


def _agg(vals):
    a = np.asarray(vals, dtype=float)
    fin = a[np.isfinite(a)]
    if len(fin) == 0:
        return None
    return (float(np.median(fin)), float(np.percentile(fin, 25)),
            float(np.percentile(fin, 75)), int(len(fin)))


def _fmt(agg, prec='.1f'):
    if agg is None:
        return '—'
    m, q1, q3, _ = agg
    return f'{m:{prec}} [{q1:{prec}}, {q3:{prec}}]'


# ========== 出力 ==========

def analyze(input_dir):
    problems, cfg_meta = load_runs(input_dir)
    cfg_path = os.path.join(input_dir, 'config.json')
    axes_cfg = {}
    if os.path.exists(cfg_path):
        with open(cfg_path, encoding='utf-8') as f:
            axes_cfg = json.load(f).get('axes', {})

    out_dir = os.path.join(input_dir, 'analysis_sweep')
    os.makedirs(out_dir, exist_ok=True)
    lines = ['# param_sweep_v1 感度分析サマリ', '',
             f'データ: `{input_dir}`', '',
             '速度指標 TTT@95% = 各 trial が自身の最終 union HV の 95% に到達する CPU 時間 [s]'
             '（trial 中央値[IQR]、小さいほど速い）。**最終 union HV と併読**（速いが品質の低い'
             '解への収束を区別するため）。p=Wilcoxon(two-sided, vs center), δ=Cliff (正=center超え)。', '']

    for prob in sorted(problems.keys()):
        per_config = problems[prob]
        # union(重複除去) を config×trial で一度だけ構築してキャッシュ → ref/指標/TTT で共用
        baselines = {key: _baseline_of(bw) for key, bw in per_config.items()}
        union_cache = {key: union_xyt_per_trial(bw) for key, bw in per_config.items()}
        ref, p50, stab_max = problem_ref_p50(union_cache, baselines)
        lines.append(f'## {prob}')
        lines.append(f'(ref={ref[0]:.1f},{ref[1]:.2f}  P50(D)={p50:.3f})')
        lines.append('')

        # axis -> methods（config.json があればそれ、無ければ実データから推定）
        if axes_cfg:
            axis_items = [(an, ax['methods']) for an, ax in axes_cfg.items()]
        else:
            seen = {}
            for (m, tag), meta in cfg_meta.items():
                if meta['axis'] and meta['axis'] != 'center':
                    seen.setdefault(meta['axis'], set()).add(m)
            axis_items = [(an, sorted(ms)) for an, ms in seen.items()]

        # 各 config の trial 指標を一度だけ計算してキャッシュ（union はキャッシュ流用）
        metric_cache = {key: config_trial_metrics(union_cache[key], baselines[key],
                                                   ref, p50, stab_max)
                        for key in per_config}

        for axis_name, methods in axis_items:
            lines.append(f'### 軸: `{axis_name}`')
            lines.append('')
            lines.append('| 手法 | 値 | union HV med[IQR] | rHV高安定 | rHV低安定 | TTT@95 med[IQR] | vs center |')
            lines.append('|---|---|---|---|---|---|---|')
            for m in methods:
                center_key = (m, 'center')
                center_m = metric_cache.get(center_key)
                # この軸・この手法の掃引値 config を tag 順に
                val_keys = sorted(
                    [k for k in per_config
                     if k[0] == m and cfg_meta[k]['axis'] == axis_name],
                    key=lambda k: str(cfg_meta[k]['value']))
                rows = ([(center_key, 'center')] if center_m else []) + \
                       [(k, cfg_meta[k]['value']) for k in val_keys]
                for key, val in rows:
                    mc = metric_cache.get(key)
                    if mc is None:
                        continue
                    hv = _agg(mc['hv']); rh = _agg(mc['rhv_high'])
                    rl = _agg(mc['rhv_low']); tt = _agg(mc['ttt95'])
                    # vs center
                    cell = '—'
                    if center_m is not None and key != center_key:
                        x = mc['hv']; y = center_m['hv']
                        n = min(len(x), len(y))
                        if n >= 1:
                            xx, yy = x[:n], y[:n]
                            _, p = A.wilcoxon_paired(xx, yy, alternative='two-sided')
                            # cliffs_delta(a,b)<0 は a が大きい傾向。HV は大きいほど良いので
                            # cliffs_delta(center, config) を取り「正 = config が center 超え」に揃える。
                            dlt = A.cliffs_delta(yy, xx)
                            cell = f'p={p:.3f}{A._p_star(p).strip()} δ={dlt:+.2f}({A.effect_label(dlt)})' \
                                if np.isfinite(p) else f'δ={dlt:+.2f}({A.effect_label(dlt)})'
                    lines.append(
                        f'| {A.METHOD_LABELS.get(m, m)} | {val} | {_fmt(hv, ".1f")} | '
                        f'{_fmt(rh, ".1f")} | {_fmt(rl, ".1f")} | {_fmt(tt, ".2f")} | {cell} |')
                lines.append('')

    out_md = os.path.join(out_dir, 'summary.md')
    with open(out_md, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    print(f'→ {out_md}')


def main():
    ap = argparse.ArgumentParser(description='param_sweep_v1 感度分析')
    ap.add_argument('--input-dir', required=True)
    args = ap.parse_args()
    analyze(args.input_dir)


if __name__ == '__main__':
    main()
