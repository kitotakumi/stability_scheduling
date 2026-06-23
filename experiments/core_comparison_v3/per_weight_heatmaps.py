#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""重み×手法ヒートマップ（scalar / 統合HV(per-weight) / 高安定HV / AOC）。

analyze_v3 の機構を再利用し、各 (problem, scenario) ごとに重み(行)×手法(列)の
中央値を色濃淡で示す。指標ごとに1枚（全問題を横に並べる）＋問題別も保存。

- scalar : 最終 best_score 中央値（小さいほど良 → カラーマップ反転）。
- hv     : per-weight UEA HV（正規化, 大きいほど良）。
- hsHV   : per-weight 高安定HV = D∈[0,P50) の region_hv（正規化, 大きいほど良）。
- aoc    : per-weight AOC（正規化, 大きいほど良）。
"""
import os
import sys
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import analyze_v3 as A

_installed = {f.name for f in font_manager.fontManager.ttflist}
for _jp in ('Yu Gothic', 'Meiryo', 'MS Gothic', 'Noto Sans CJK JP', 'IPAexGothic'):
    if _jp in _installed:
        plt.rcParams['font.family'] = _jp
        break
plt.rcParams['axes.unicode_minus'] = False

INPUT_DIR = os.path.join(HERE, 'results', 'main_v1')
OUTDIR = os.path.join(INPUT_DIR, 'analysis', 'heatmap')
os.makedirs(OUTDIR, exist_ok=True)

W_ORDER = [f'w{int(round(w0*10)):02d}_{int(round((1-w0)*10)):02d}'
           for w0 in [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]]

# 手法の表示順（GA → Memetic 3種 → ILS 3種）。config の順とは独立に固定する。
METHOD_ORDER = ['ga', 'memetic_ls', 'memetic_repair', 'memetic_pr',
                'ils_baseline', 'ils_repair', 'ils_pr']

METRICS = [
    ('scalar', 'scalar 最終重み付きスコア', True),   # 小さいほど良 → 反転
    ('hv',     'per-weight 統合HV',          False),
    ('hshv',   'per-weight 高安定HV (D∈[0,P50))', False),
    ('aoc',    'per-weight AOC',             False),
]


def med(xs):
    xs = [x for x in xs if x is not None and not (isinstance(x, float) and np.isnan(x))]
    return float(np.median(xs)) if xs else float('nan')


def compute():
    grouped = A.load_all_runs(INPUT_DIR)
    cfg = json.load(open(os.path.join(INPUT_DIR, 'config.json'), encoding='utf-8'))
    cfg_methods = cfg['methods']
    # 指定の表示順に並べ替え（METHOD_ORDER 優先、未収載は後ろに保険で残す）
    methods = ([m for m in METHOD_ORDER if m in cfg_methods]
               + [m for m in cfg_methods if m not in METHOD_ORDER])
    labels = [cfg['method_configs'][m]['label'] for m in methods]

    out = {}  # (prob,scen) -> {metric: 2D array [w x method]}, plus w_labels
    for (prob, scen), method_data in sorted(grouped.items()):
        baselines = {}
        for m in methods:
            bls = []
            for wl in method_data.get(m, {}):
                for _, data in method_data[m][wl].items():
                    if data.get('baseline') is not None:
                        bls.append(data['baseline'])
                    b2 = data.get('baseline_rsr')
                    if b2 is not None and list(b2) not in bls:
                        bls.append(list(b2))
                    break
                if bls:
                    break
            baselines[m] = bls if bls else None

        all_pts = []
        for m in methods:
            bl = baselines[m]
            for wl in method_data.get(m, {}):
                for t, data in method_data[m][wl].items():
                    pts = A.get_uea_points(data, t)
                    if bl:
                        pts = A.filter_baselines(pts, bl)
                    if len(pts) > 0:
                        all_pts.append(pts)
        if not all_pts:
            continue
        concat = np.concatenate(all_pts)
        norm = A.make_norm(concat)
        global_ref = (float(concat[:, 0].max()) + max(concat[:, 0].max()*0.01, 1.0),
                      float(concat[:, 1].max()) + max(concat[:, 1].max()*0.01, 0.01))
        thr = A.compute_p33_p67(method_data, baselines)
        p50 = thr['P50'] if thr else float(concat[:, 1].max())/2

        w_labels = [wl for wl in W_ORDER
                    if any(wl in method_data.get(m, {}) for m in methods)]
        nW, nM = len(w_labels), len(methods)
        scalar = np.full((nW, nM), np.nan)
        hv = np.full((nW, nM), np.nan)
        hshv = np.full((nW, nM), np.nan)
        aoc = np.full((nW, nM), np.nan)

        for wi, wl in enumerate(w_labels):
            for mi, m in enumerate(methods):
                bl = baselines[m]
                by_trial = method_data[m].get(wl, {})
                sc, hvv, hsv = [], [], []
                for t in sorted(by_trial.keys()):
                    data = by_trial[t]
                    hist = A.get_anytime(data)
                    bs = [h.get('best_score') for h in hist
                          if h.get('best_score') is not None]
                    sc.append(bs[-1] if bs else None)
                    pts = A.get_uea_points(data, t)
                    if bl:
                        pts = A.filter_baselines(pts, bl)
                    pf = A.pareto_front(pts) if len(pts) else pts
                    ptsn = A.normalize_pts(pf, norm) if len(pf) else pf
                    hvv.append(A.hypervolume(ptsn, A.NORM_REF))
                    # 高安定HV: 高安定領域 D∈[0,P50) を raw 境界でマスクし正規化空間で面積
                    h_val, _ = A.region_hv(pf, 0.0, p50, global_ref[0], norm=norm)
                    hsv.append(h_val)
                scalar[wi, mi] = med(sc)
                hv[wi, mi] = med(hvv)
                hshv[wi, mi] = med(hsv)

            # AOC（per-weight, compute_aoc_per_trial）
            method_info_n = []
            for m in methods:
                kind = 'ga' if m == 'ga' else 'ils'
                bl = baselines[m]
                by_trial = method_data[m].get(wl, {})
                hist_list, pts_n = [], []
                for t in sorted(by_trial.keys()):
                    data = by_trial[t]
                    hist_list.append(A.get_anytime(data))
                    pxyt = A.get_uea_points_xyt(data, t)
                    pts_n.append(A.normalize_pts(pxyt, norm) if len(pxyt) else pxyt)
                method_info_n.append((m, hist_list, pts_n, kind,
                                      A.normalize_baseline(bl, norm)))
            aoc_pt = A.compute_aoc_per_trial(method_info_n, A.NORM_REF, n_jobs=1)
            for mi, m in enumerate(methods):
                aoc[wi, mi] = med(aoc_pt.get(m, []))

        out[(prob, scen)] = {'scalar': scalar, 'hv': hv, 'hshv': hshv, 'aoc': aoc,
                             'w_labels': w_labels}
        print(f'computed {prob}_{scen}')
    return out, methods, labels


def wlabel_tick(wl):
    return f'{int(wl[1:3])}:{int(wl[4:6])}'


def _row_rpd(M):
    """各行（重み）を行内の最良(=最小スカラー)で割った相対偏差% (= (v-min)/min*100)。

    scalar 専用。scalar は重みごとに測る対象・達成水準が変わり絶対値を横串比較できない
    （同一重み=同一行内の比較のみ妥当）ため、行ごとに正規化する。best≈0.76-1.03 で
    0 付近にならないのでゼロ割は起きない。RPD 化で水準ドリフトが除去され、RPD 自体は
    全行・全問題で共通カラースケールに載せられる。
    """
    R = np.full_like(M, np.nan)
    for i in range(M.shape[0]):
        row = M[i, :]
        rmin = np.nanmin(row) if np.any(np.isfinite(row)) else np.nan
        if np.isfinite(rmin) and rmin > 1e-9:
            R[i, :] = (row - rmin) / rmin * 100.0
    return R


def plot_metric(out, methods, labels, key, title, invert):
    """1指標を全問題横並びで1枚に（行=重み, 列=手法）。

    scalar は「行（重み）ごとの相対偏差 RPD%」で色分けする（絶対値は横串比較不能のため）。
    hv/hsHV/AOC は正規化済みで重みをまたいで単位が共通なので、絶対値そのままで色分けする
    （絶対水準が意味を持つ。RPD 化は near-0 行の微差を過剰に増幅し誤誘導するため使わない）。
    """
    is_scalar = (key == 'scalar')
    # パネル順は難易度（リスケ率昇順, analyze_v3 と共通）
    probs = sorted(out.keys(),
                   key=lambda ps: (A.reschedule_rate(f'{ps[0]}_{ps[1]}'), ps))
    n = len(probs)
    fig, axes = plt.subplots(1, n, figsize=(3.0*n, 4.6), squeeze=False)
    axes = axes[0]

    # scalar は全問題共通の RPD カラースケール（横串比較可）。他指標は subplot ごと自動。
    gvmax = 1.0
    if is_scalar:
        for (prob, scen) in probs:
            R = _row_rpd(out[(prob, scen)][key])
            if np.any(np.isfinite(R)):
                gvmax = max(gvmax, float(np.nanmax(R)))

    for ax, (prob, scen) in zip(axes, probs):
        D = out[(prob, scen)]
        M = D[key]
        wls = D['w_labels']
        if is_scalar:
            C = _row_rpd(M)
            im = ax.imshow(C, aspect='auto', cmap='RdYlGn_r', vmin=0, vmax=gvmax)
        else:
            C = None
            im = ax.imshow(M, aspect='auto', cmap='viridis')
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=90, fontsize=7)
        ax.set_yticks(range(len(wls)))
        ax.set_yticklabels([wlabel_tick(w) for w in wls], fontsize=7)
        ax.set_title(A.problem_short_tag(f'{prob}_{scen}'), fontsize=8)
        # セル注記: scalar は RPD%（0=その重みの最良手法）、他は絶対値中央値
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                v = M[i, j]
                if np.isnan(v):
                    continue
                if is_scalar:
                    r = C[i, j]
                    txt = f'{r:.0f}' if np.isfinite(r) else ''
                    color = 'black'
                else:
                    txt = f'{v:.3f}' if key == 'hshv' else f'{v:.2f}'
                    color = 'white' if _dark(im, v) else 'black'
                ax.text(j, i, txt, ha='center', va='center', fontsize=5, color=color)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    if is_scalar:
        sub = '色=重み行ごと相対偏差RPD%（緑=その重みの最良手法, 0=最良）, セル=RPD%'
    else:
        sub = '明るい(黄)=値大=良, セル=中央値'
    fig.suptitle(f'{title}  （行=重み MS:ST  列=手法  {sub}）', fontsize=11)
    axes[0].set_ylabel('重み (MS:ST)  ↑MS純 → ↓安定純', fontsize=8)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    path = os.path.join(OUTDIR, f'heatmap_{key}_all.png')
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print(f'  -> {path}')


def _dark(im, v):
    # viridis/viridis_r: 低 norm=暗紫(白文字), 高 norm=黄(黒文字)
    return im.norm(v) < 0.5


def main():
    out, methods, labels = compute()
    for key, title, invert in METRICS:
        plot_metric(out, methods, labels, key, title, invert)
    print('\n完了:', OUTDIR)


if __name__ == '__main__':
    main()
