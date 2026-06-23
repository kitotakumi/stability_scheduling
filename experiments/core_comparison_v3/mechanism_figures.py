#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""機構レベルの証拠図（H1/H2 の構造的原因）。

(1) D値分布図 [H1]: 訪問解の順位偏差 D の分布。ILS は S_p 近傍(低D)に集中、Memetic は
    遠方(高D)に分散する——「なぜ高安定領域で ILS が勝つか」の直接証拠。
    raw の d_visit_hist（dedup前・全訪問点のD頻度）を method×problem で集計。
(2) PR経路長・改善率図 [H2]: PR 呼び出しの経路長 d0（S_p までの不一致数）と
    経路上で始点より良い解を見つけた割合。Memetic は d0 大＋改善多、ILS は d0≈0＋改善ほぼ無
    ——「なぜ機構が集団にだけ効くか」の機構的原因。raw の mech_stats を集計。

usage: python mechanism_figures.py [results_dir]
"""
import os
import sys
import glob
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

def _order_probs(candidates):
    """candidates（集合/リスト）を難易度（リスケ率昇順, analyze_v3 と共通）で整列。
    問題数が増えても自動で正しい位置に並ぶ（ハードコード順は持たない）。"""
    return A.order_prob_labels(set(candidates))


def _aggregate(results_dir):
    """{prob: {method: {'dhist': {D:count}, 'pr_d0': [...], 'pr_improved': [...],
                        'n_pr': int}}} を raw から集計。"""
    out = {}
    for prob_dir in sorted(glob.glob(os.path.join(results_dir, '*', 'raw'))):
        prob = os.path.basename(os.path.dirname(prob_dir))
        acc = {}
        for p in glob.glob(os.path.join(prob_dir, '*.json')):
            m = os.path.basename(p).split('__')[0]
            d = json.load(open(p, encoding='utf-8'))
            a = acc.setdefault(m, {'dhist': {}, 'pr_d0': [], 'pr_improved': [], 'n_pr': 0})
            for k, v in (d.get('d_visit_hist') or {}).items():
                a['dhist'][int(k)] = a['dhist'].get(int(k), 0) + int(v)
            ms = d.get('mech_stats') or {}
            a['pr_d0'] += [int(x) for x in ms.get('pr_d0', [])]
            a['pr_improved'] += [int(x) for x in ms.get('pr_improved', [])]
            a['n_pr'] += int(ms.get('n_pr_calls', 0))
        if acc:
            out[prob] = acc
    return out


def fig_d_distribution(agg, outpath):
    """D値分布: ILS-baseline vs Memetic-LS（H1 の純粋ペア）を problem 別パネルで。"""
    pair = [('ils_baseline', 'ILS-baseline', 'tab:orange'),
            ('memetic_ls', 'Memetic-LS', 'tab:green')]
    probs = _order_probs(agg.keys())
    n = len(probs)
    ncol = 3
    nrow = (n + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.4 * ncol, 3.2 * nrow), squeeze=False)
    axes = axes.ravel()
    for ax, prob in zip(axes, probs):
        for mkey, mlabel, color in pair:
            h = agg[prob].get(mkey, {}).get('dhist', {})
            if not h:
                continue
            Ds = np.array(sorted(h.keys()))
            cnt = np.array([h[d] for d in Ds], dtype=float)
            dens = cnt / cnt.sum()
            ax.plot(Ds, dens, color=color, lw=1.6, label=mlabel)
            ax.fill_between(Ds, dens, color=color, alpha=0.25)
            # 中央値D（訪問頻度で重み付け）を縦線で
            csum = np.cumsum(cnt) / cnt.sum()
            medD = Ds[np.searchsorted(csum, 0.5)]
            ax.axvline(medD, color=color, ls='--', lw=1.0, alpha=0.8)
        ax.set_title(A.problem_short_tag(prob), fontsize=10)
        ax.set_xlabel('順位偏差 D（小=S_p近傍=安定）', fontsize=8)
        ax.set_ylabel('訪問頻度（正規化）', fontsize=8)
        ax.legend(fontsize=8); ax.grid(alpha=0.3)
    for k in range(len(probs), len(axes)):
        axes[k].axis('off')
    fig.suptitle('訪問解の D 分布: ILS は S_p 近傍(低D)に集中・Memetic は遠方(高D)に分散  '
                 '(破線=訪問頻度の中央値D)  [H1 の構造的原因]', fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(outpath, dpi=140, bbox_inches='tight')
    plt.close(fig)
    print(f'  -> {outpath}')


def fig_pr_mechanism(agg, outpath):
    """PR経路長 d0 と改善発見率を Memetic+PR vs ILS+PR で problem 別に並べる（H2 の原因）。"""
    probs = _order_probs(agg.keys())
    tags = [A.problem_short_tag(p) for p in probs]
    methods = [('memetic_pr', 'Memetic+PR', 'tab:brown'),
               ('ils_pr', 'ILS+PR', 'tab:blue')]
    x = np.arange(len(probs)); w = 0.38

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 4.4))
    for i, (mkey, mlabel, color) in enumerate(methods):
        d0_mean, imp_rate = [], []
        for prob in probs:
            a = agg[prob].get(mkey, {})
            d0 = a.get('pr_d0', [])
            imp = a.get('pr_improved', [])
            d0_mean.append(float(np.mean(d0)) if d0 else 0.0)
            imp_rate.append(100.0 * np.sum(imp) / len(imp) if imp else 0.0)
        offs = (i - 0.5) * w
        b1 = axL.bar(x + offs, d0_mean, w, label=mlabel, color=color)
        b2 = axR.bar(x + offs, imp_rate, w, label=mlabel, color=color)
        for rect, v in zip(b1, d0_mean):
            axL.text(rect.get_x()+rect.get_width()/2, v, f'{v:.0f}',
                     ha='center', va='bottom', fontsize=7)
        for rect, v in zip(b2, imp_rate):
            axR.text(rect.get_x()+rect.get_width()/2, v, f'{v:.0f}',
                     ha='center', va='bottom', fontsize=7)
    axL.set_title('PR 経路長 $d_0$（S_p までの不一致数, 平均）\n大=動く余地が大', fontsize=10)
    axR.set_title('PR 改善発見率（経路上で始点より良い解を得た割合 %）', fontsize=10)
    for ax in (axL, axR):
        ax.set_xticks(x); ax.set_xticklabels(tags, fontsize=9)
        ax.legend(fontsize=9); ax.grid(axis='y', alpha=0.3)
    axL.set_ylabel('平均経路長 $d_0$'); axR.set_ylabel('改善発見率 %')
    fig.suptitle('PR の機構統計: Memetic は経路長大・改善多／ILS は経路ほぼ0・改善ほぼ無  '
                 '[H2＝機構が集団にだけ効く原因]', fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(outpath, dpi=140, bbox_inches='tight')
    plt.close(fig)
    print(f'  -> {outpath}')


def _collect_points(results_dir, prob, method):
    """(method, prob) の全 trial・全重みの訪問点を baseline 除外して縦積み (N,2)=[MS,D]。
    baseline は最初に見つかった run の baseline / baseline_rsr を使う。"""
    pts_all, bl = [], None
    for p in sorted(glob.glob(os.path.join(results_dir, prob, 'raw', f'{method}__*.json'))):
        d = json.load(open(p, encoding='utf-8'))
        if bl is None:
            bl = []
            if d.get('baseline') is not None:
                bl.append(d['baseline'])
            if d.get('baseline_rsr') is not None and list(d['baseline_rsr']) not in bl:
                bl.append(list(d['baseline_rsr']))
            bl = bl or None
        pts = A.get_uea_points(d, 0)
        if len(pts):
            if bl:
                pts = A.filter_baselines(pts, bl)
            if len(pts):
                pts_all.append(np.asarray(pts, float))
    if not pts_all:
        return np.zeros((0, 2))
    P = np.unique(np.concatenate(pts_all), axis=0)
    return P


def fig_pareto_h1(results_dir, outpath, rng_seed=0):
    """(MS, D) 空間の訪問点散布 ＋ Pareto front を ILS-baseline vs Memetic-LS で重ねる。
    高安定域（低D）で ILS が良 MS 点に届き Memetic が届かないことを直接見せる（H1）。"""
    pair = [('ils_baseline', 'ILS-baseline', 'tab:orange'),
            ('memetic_ls', 'Memetic-LS', 'tab:green')]
    probs = _order_probs(
        os.path.basename(os.path.dirname(d))
        for d in glob.glob(os.path.join(results_dir, '*', 'raw')))
    n = len(probs); ncol = 3; nrow = (n + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.0 * ncol, 3.8 * nrow), squeeze=False)
    axes = axes.ravel()
    for ax, prob in zip(axes, probs):
        pfs = {}
        for mkey, mlabel, color in pair:
            P = _collect_points(results_dir, prob, mkey)
            if len(P) == 0:
                continue
            pf = A.pareto_front(P)
            pf = pf[np.argsort(pf[:, 0])]
            pfs[mkey] = (pf, mlabel, color)
            ax.step(pf[:, 0], pf[:, 1], where='post', color=color, lw=2.2,
                    label=f'{mlabel}', marker='o', ms=4)
        # 各手法が到達した最安定点（最小D）を注記
        for mkey, (pf, mlabel, color) in pfs.items():
            jmin = int(np.argmin(pf[:, 1]))
            ax.annotate(f'D={pf[jmin,1]:.0f}', (pf[jmin, 0], pf[jmin, 1]),
                        fontsize=8, color=color, fontweight='bold',
                        xytext=(3, 3), textcoords='offset points')
        ax.set_title(A.problem_short_tag(prob), fontsize=10)
        ax.set_xlabel('メイクスパン MS（左=効率良）', fontsize=8)
        ax.set_ylabel('順位偏差 D（下=安定=$S_p$近傍）', fontsize=8)
        ax.legend(fontsize=8, loc='upper right'); ax.grid(alpha=0.3)
    for k in range(len(probs), len(axes)):
        axes[k].axis('off')
    fig.suptitle('訪問点と Pareto front (MS, D): 高安定域(低D)で ILS は良MS点に届き Memetic-LS は届かない  '
                 '[H1の直接証拠]', fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(outpath, dpi=140, bbox_inches='tight')
    plt.close(fig)
    print(f'  -> {outpath}')


def _collect_points_raw(results_dir, prob, method):
    """baseline 除外した訪問点を run をまたいで縦積み（dedupしない＝被覆濃度）。"""
    pts_all, bl = [], None
    for p in sorted(glob.glob(os.path.join(results_dir, prob, 'raw', f'{method}__*.json'))):
        d = json.load(open(p, encoding='utf-8'))
        if bl is None:
            bl = []
            if d.get('baseline') is not None:
                bl.append(d['baseline'])
            if d.get('baseline_rsr') is not None and list(d['baseline_rsr']) not in bl:
                bl.append(list(d['baseline_rsr']))
            bl = bl or None
        pts = A.get_uea_points(d, 0)
        if len(pts):
            if bl:
                pts = A.filter_baselines(pts, bl)
            if len(pts):
                pts_all.append(np.asarray(pts, float))
    return np.concatenate(pts_all) if pts_all else np.zeros((0, 2))


def fig_density_diff(results_dir, outpath, ils_m, mem_m, ils_lbl, mem_lbl, suptitle, nbins=44):
    """探索の正規化訪問密度差マップ (MS, D): 赤=ILS系が相対的に密／青=Memetic系が相対的に密 ＋ 両PF。

    各手法の訪問点を (MS,D) で2Dヒストグラム化し **手法ごと総和=1 に正規化**（探索質量の配分）。
    その差 (ILS − Memetic) を sign·√|Δ| で着色。総訪問数の違いは正規化で除去され、
    「探索を*どこに配分*したか」の相対比較になる。"""
    probs = _order_probs(
        os.path.basename(os.path.dirname(d))
        for d in glob.glob(os.path.join(results_dir, '*', 'raw')))
    n = len(probs); ncol = 3; nrow = (n + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.4 * ncol, 4.0 * nrow), squeeze=False)
    axes = axes.ravel()
    im = None
    for ax, prob in zip(axes, probs):
        Pi = _collect_points_raw(results_dir, prob, ils_m)
        Pm = _collect_points_raw(results_dir, prob, mem_m)
        if len(Pi) == 0 or len(Pm) == 0:
            ax.axis('off'); continue
        allp = np.vstack([Pi, Pm])
        ms_e = np.linspace(allp[:, 0].min(), allp[:, 0].max(), nbins + 1)
        d_e = np.linspace(0, allp[:, 1].max(), nbins + 1)
        Hi, _, _ = np.histogram2d(Pi[:, 0], Pi[:, 1], bins=[ms_e, d_e])
        Hm, _, _ = np.histogram2d(Pm[:, 0], Pm[:, 1], bins=[ms_e, d_e])
        Hi = Hi / Hi.sum(); Hm = Hm / Hm.sum()   # 手法ごと正規化（総訪問数の差を除去）
        diff = Hi - Hm
        D = (np.sign(diff) * np.sqrt(np.abs(diff))).T   # sqrt は表示用（小さい差を可視化）
        vmax = np.nanmax(np.abs(D)) or 1.0
        im = ax.imshow(D, origin='lower', aspect='auto', cmap='RdBu_r',
                       vmin=-vmax, vmax=vmax,
                       extent=[ms_e[0], ms_e[-1], d_e[0], d_e[-1]])
        for P, color in [(Pi, 'darkorange'), (Pm, 'green')]:
            pf = A.pareto_front(np.unique(P, axis=0))
            pf = pf[np.argsort(pf[:, 0])]
            ax.step(pf[:, 0], pf[:, 1], where='post', color=color, lw=2.0, zorder=5)
        ax.set_title(A.problem_short_tag(prob), fontsize=10)
        ax.set_xlabel('メイクスパン MS（左=効率良）', fontsize=8)
        ax.set_ylabel('順位偏差 D（下=安定）', fontsize=8)
    for k in range(len(probs), len(axes)):
        axes[k].axis('off')
    if im is not None:
        cb = fig.colorbar(im, ax=axes.tolist(), shrink=0.55)
        cb.set_label(f'正規化訪問密度差（赤={ils_lbl}が密／青={mem_lbl}が密, sign·√|Δ|）', fontsize=9)
    fig.suptitle(f'{suptitle}  橙線={ils_lbl} PF・緑線={mem_lbl} PF', fontsize=11)
    fig.savefig(outpath, dpi=140, bbox_inches='tight')
    plt.close(fig)
    print(f'  -> {outpath}')


def main(results_dir):
    agg = _aggregate(results_dir)
    outdir = os.path.join(results_dir, 'analysis', 'mechanism')
    os.makedirs(outdir, exist_ok=True)
    fig_d_distribution(agg, os.path.join(outdir, 'd_distribution.png'))
    fig_pr_mechanism(agg, os.path.join(outdir, 'pr_mechanism.png'))
    fig_pareto_h1(results_dir, os.path.join(outdir, 'pareto_h1.png'))
    fig_density_diff(results_dir, os.path.join(outdir, 'density_baseline.png'),
                     'ils_baseline', 'memetic_ls', 'ILS-baseline', 'Memetic-LS',
                     '探索の正規化訪問密度差 (MS, D): 純粋比較（局所探索のみ）')
    fig_density_diff(results_dir, os.path.join(outdir, 'density_pr.png'),
                     'ils_pr', 'memetic_pr', 'ILS+PR', 'Memetic+PR',
                     '探索の正規化訪問密度差 (MS, D): +PR 機構あり')


if __name__ == '__main__':
    rd = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, 'results', 'main_v1')
    main(rd)
