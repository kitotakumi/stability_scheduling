"""core_comparison_v3 の横断図・横断ランキングを生成する（旧 tools/ の4本を集約）。

旧 tools/summary_figures.py + overall_ranking.py + anytime_aoc.py + aggregate_scores.py
を1モジュールに統合。analyze_v3 の指標定義・正規化・色を共有する（import analyze_v3）。

生成物（すべて <results_dir>/analysis/ 下）:
  claim1_ils_vs_memetic.png / claim2_mechanism.png      … 主張1 / 主張2（絞った手法×6問題）
  <prob>/summary_bars.png / summary_bars_all.png         … 問題別7手法 / 横断俯瞰（付録）
  overall_ranking_union.png / overall_ranking_highstab.png … Friedman 平均順位 CD図(Demšar 2006)
  anytime_crossover.png                                  … アンタイム HV 曲線の交差
  perf_profile_AOC.png / perf_profile_統合HV.png / perf_profile_高安定HV.png … 性能プロファイル(Dolan-Moré)

数値表（スコアボード・AOC・ARPD）は analyze_v3 が summary.md に書く。本モジュールは図と
標準出力サマリのみ（summary.md と数値は一致：同じ正規化・同じ collect を使う）。

usage:
  python figures_v3.py [results_dir] [--weight w08_02] [scenario...]
  （通常は analyze_v3.main() の末尾から generate_all() が自動実行される）
"""
import os
import sys

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

import numpy as np
from scipy.stats import friedmanchisquare, rankdata
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
import analyze_v3 as A

_installed = {f.name for f in font_manager.fontManager.ttflist}
for _jp in ('Yu Gothic', 'Meiryo', 'MS Gothic', 'Noto Sans CJK JP', 'IPAexGothic'):
    if _jp in _installed:
        plt.rcParams['font.family'] = _jp
        break
plt.rcParams['axes.unicode_minus'] = False

ORDER = A.METHOD_ORDER
LABEL = {m: A.METHOD_LABELS.get(m, m) for m in ORDER}
DEFAULT_WEIGHT = 'w08_02'

# Nemenyi 用 studentized range / sqrt(2) の q_0.05（k=手法数）
Q05 = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850,
       7: 2.949, 8: 3.031, 9: 3.102, 10: 3.164}


def short_tag(prob, scen=None):
    """(prob, scen) または prob_label を略記に。analyze_v3.problem_short_tag と一致。"""
    label = prob if scen is None else f'{prob}_{scen}'
    return A.problem_short_tag(label)


def _ordered_items(grouped):
    """grouped.items() を難易度（リスケ率昇順, analyze_v3 と共通）で整列して返す。"""
    return sorted(grouped.items(),
                  key=lambda kv: (A.reschedule_rate(f'{kv[0][0]}_{kv[0][1]}'), kv[0]))


def med(v):
    v = [x for x in v if np.isfinite(x)]
    return float(np.median(v)) if v else 0.0


def star(p):
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return 'ns'
    if p < 0.001:
        return '***'
    if p < 0.01:
        return '**'
    if p < 0.05:
        return '*'
    return 'ns'


# ========== per-trial 指標の収集（summary.md / b2a と同じ正規化で再計算） ==========

def baselines_by_method(method_data, methods):
    bl_by_m = {}
    for m in methods:
        bls = []
        for wl in method_data[m]:
            for _t, data in method_data[m][wl].items():
                b1 = data.get('baseline')
                b2 = data.get('baseline_rsr')
                if b1 is not None:
                    bls.append(b1)
                if b2 is not None and list(b2) not in bls:
                    bls.append(list(b2))
                break
            if bls:
                break
        bl_by_m[m] = bls if bls else None
    return bl_by_m


def _all_visited(method_data, methods, bl_by_m):
    pts = []
    for m in methods:
        bl = bl_by_m[m]
        for wl in method_data[m]:
            for t, data in method_data[m][wl].items():
                p = A.get_uea_points(data, t)
                if bl:
                    p = A.filter_baselines(p, bl)
                if len(p):
                    pts.append(p)
    return np.concatenate(pts) if pts else np.zeros((0, 2))


def _union_hv_per_trial(method_data, methods, bl_by_m, norm):
    n_tr = 1 + max(max(method_data[m][wl].keys())
                   for m in methods for wl in method_data[m])
    out = {}
    for m in methods:
        bl = bl_by_m[m]
        hvs = []
        for t in range(n_tr):
            up = []
            for wl in method_data[m]:
                data = method_data[m][wl].get(t)
                if data is None:
                    continue
                p = A.get_uea_points(data, t)
                if bl:
                    p = A.filter_baselines(p, bl)
                if len(p):
                    up.append(p)
            if not up:
                hvs.append(0.0)
                continue
            pf = A.pareto_front(A.normalize_pts(np.concatenate(up), norm))
            hvs.append(A.hypervolume(pf, A.NORM_REF))
        out[m] = hvs
    return out


def _scalar_median(method_data, methods, weight):
    out = {}
    for m in methods:
        vals = []
        for t, data in method_data[m].get(weight, {}).items():
            hist = data.get('history', [])
            if hist:
                s = hist[-1].get('best_score')
                if s is not None and np.isfinite(float(s)):
                    vals.append(float(s))
        out[m] = float(np.median(vals)) if vals else float('nan')
    return out


def collect(method_data, weight):
    """(methods, union_pt, high_pt, scalar_med) を返す（per-trial union/高安定HV）。"""
    methods = [m for m in ORDER if m in method_data] or list(method_data.keys())
    bl_by_m = baselines_by_method(method_data, methods)
    cat = _all_visited(method_data, methods, bl_by_m)
    norm = A.make_norm(cat)
    raw_ref_ms = float(cat[:, 0].max()) + max(cat[:, 0].max() * 0.01, 1.0)
    thr = A.compute_p33_p67(method_data, bl_by_m)
    region = A.compute_region_hv_per_trial(
        method_data, bl_by_m, (raw_ref_ms, None), thr['P50'], thr['stab_max'],
        norm=norm)
    union_pt = _union_hv_per_trial(method_data, methods, bl_by_m, norm)
    high_pt = {m: list(region.get(m, {}).get('high', [])) for m in methods}
    scal = _scalar_median(method_data, methods, weight)
    return methods, union_pt, high_pt, scal


# ========== 主張バー図（claim1 / claim2 / 問題別 / 俯瞰） ==========

def _grouped_bars(ax, tags, bars_per_group, title, ylabel):
    ng = len(bars_per_group)
    x = np.arange(len(tags))
    w = 0.8 / ng
    for j, (mkey, label, data) in enumerate(bars_per_group):
        meds = [med(data[t].get(mkey, [])) for t in tags]
        ax.bar(x + (j - (ng - 1) / 2) * w, meds, w,
               label=label, color=A.get_method_color(mkey, ORDER.index(mkey)))
    ax.set_xticks(x)
    ax.set_xticklabels(tags, fontsize=9)
    ax.set_title(title, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    ax.legend(fontsize=8)


def _annotate_pair(ax, tags, data, a, b, alt='two-sided'):
    x = np.arange(len(tags))
    for i, t in enumerate(tags):
        xa, xb = data[t].get(a, []), data[t].get(b, [])
        _, p = A.wilcoxon_paired(xa, xb, alternative=alt)
        top = max(med(xa), med(xb))
        ax.annotate(star(p), (x[i], top), ha='center', va='bottom',
                    fontsize=8, color='dimgray')


def _annotate_vs_base(ax, tags, data, base, variants):
    x = np.arange(len(tags))
    w = 0.8 / 3
    for j, mk in enumerate(variants, start=1):
        for i, t in enumerate(tags):
            xa, xb = data[t].get(mk, []), data[t].get(base, [])
            _, p = A.wilcoxon_paired(xa, xb, alternative='two-sided')
            ax.annotate(star(p), (x[i] + (j - 1) * w, med(xa)),
                        ha='center', va='bottom', fontsize=7, color='dimgray')


def plot_claim1(rows, outpath):
    """rows: [(tag, union_pt, high_pt, aoc_pt), ...]。統合HV・高安定HV・AOC を ILS-b vs Mem-LS。

    aoc_pt（per-trial, all_mean 全重み平均）は _summary_data.pkl 由来。空（pkl 未生成等）の
    場合は AOC 列を落として従来の HV2枚に縮退する。"""
    tags = [r[0] for r in rows]
    union = {r[0]: r[1] for r in rows}
    high = {r[0]: r[2] for r in rows}
    aoc = {r[0]: (r[3] if len(r) > 3 else {}) for r in rows}
    has_aoc = any(aoc[t] for t in tags)
    ncol = 3 if has_aoc else 2
    fig, axes = plt.subplots(1, ncol, figsize=(6.0 * ncol, 4.6))
    _grouped_bars(axes[0], tags, [('ils_baseline', 'ILS-baseline', union),
                                  ('memetic_ls', 'Memetic-LS', union)],
                  '統合HV: ILS-baseline vs Memetic-LS（互角）', 'HV(正規化)')
    _annotate_pair(axes[0], tags, union, 'ils_baseline', 'memetic_ls')
    _grouped_bars(axes[1], tags, [('ils_baseline', 'ILS-baseline', high),
                                  ('memetic_ls', 'Memetic-LS', high)],
                  '高安定HV: ILS-baseline vs Memetic-LS（ILS優位）', 'HV(正規化)')
    _annotate_pair(axes[1], tags, high, 'ils_baseline', 'memetic_ls')
    if has_aoc:
        _grouped_bars(axes[2], tags, [('ils_baseline', 'ILS-baseline', aoc),
                                      ('memetic_ls', 'Memetic-LS', aoc)],
                      'AOC: ILS-baseline vs Memetic-LS（ILS優位・早期incumbent）',
                      'AOC(正規化, 高いほど良い)')
        _annotate_pair(axes[2], tags, aoc, 'ils_baseline', 'memetic_ls')
    fig.suptitle('主張1: 軌道(ILS) vs 集団(Memetic) ― 統合は互角・高安定とAOCはILSが優位'
                 '   (*p<.05 **p<.01 ***p<.001)', fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(outpath, dpi=140)
    plt.close(fig)


def plot_claim2(rows, outpath):
    tags = [r[0] for r in rows]
    high = {r[0]: r[2] for r in rows}
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), sharey=True)
    _grouped_bars(axes[0], tags, [('memetic_ls', 'Memetic-LS', high),
                                  ('memetic_pr', 'Memetic+PR', high),
                                  ('memetic_repair', 'Memetic+repair', high)],
                  '高安定HV Memetic: LS→+PR/+repair（機構が補う）', 'HV(正規化)')
    _annotate_vs_base(axes[0], tags, high, 'memetic_ls',
                      ['memetic_pr', 'memetic_repair'])
    _grouped_bars(axes[1], tags, [('ils_baseline', 'ILS-baseline', high),
                                  ('ils_pr', 'ILS+PR', high),
                                  ('ils_repair', 'ILS+repair', high)],
                  '高安定HV ILS: baseline→+PR/+repair（ほぼ頭打ち）', 'HV(正規化)')
    _annotate_vs_base(axes[1], tags, high, 'ils_baseline',
                      ['ils_pr', 'ils_repair'])
    fig.suptitle('主張2: PR/Repair の磨き効果は非対称 ― 集団は高安定HVを大きく改善, 軌道は頭打ち'
                 '   (vs baseline; *p<.05 **p<.01 ***p<.001)', fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(outpath, dpi=140)
    plt.close(fig)


def _bars(ax, methods, vals, title, ylabel):
    colors = [A.get_method_color(m, i) for i, m in enumerate(methods)]
    labels = [LABEL.get(m, m) for m in methods]
    x = np.arange(len(methods))
    bars = ax.bar(x, [vals[m] for m in methods], color=colors)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=7)
    ax.set_title(title, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.grid(axis='y', alpha=0.3)
    for b, m in zip(bars, methods):
        v = vals[m]
        if np.isfinite(v):
            ax.annotate(f'{v:.3f}', (b.get_x() + b.get_width() / 2, v),
                        ha='center', va='bottom', fontsize=6)


def plot_problem(tag, methods, union, high, scal, weight, outpath):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    _bars(axes[0], methods, union, '統合HV (正規化, 高いほど良い)', 'HV')
    _bars(axes[1], methods, high, '高安定HV (正規化, 高いほど良い)', 'HV')
    _bars(axes[2], methods, scal, f'scalar @{weight} (低いほど良い)', 'weighted score')
    fig.suptitle(tag, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(outpath, dpi=130)
    plt.close(fig)


# ========== 横断ランキング（Friedman + Nemenyi CD図, Demšar 2006） ==========

def _collect_medians(grouped, weight):
    probs, union_med, high_med = [], {}, {}
    for (prob, scen), md in _ordered_items(grouped):
        methods, union_pt, high_pt, _ = collect(md, weight)
        probs.append(short_tag(prob, scen))
        for m in ORDER:
            union_med.setdefault(m, []).append(med(union_pt.get(m, [])))
            high_med.setdefault(m, []).append(med(high_pt.get(m, [])))
    return probs, union_med, high_med


def _plot_cd(methods, avg_rank, cd, metric, outpath, p, W=float('nan')):
    k = len(methods)
    order = np.argsort(avg_rank)
    lo, hi = 1, k
    fig, ax = plt.subplots(figsize=(10, 0.5 * k + 1.8))
    ax.set_xlim(lo - 0.5, hi + 0.5)
    ax.set_ylim(0, k + 1)
    ax.invert_xaxis()
    ax.plot([lo, hi], [k + 0.5, k + 0.5], 'k-', lw=1)
    for x in range(lo, hi + 1):
        ax.plot([x, x], [k + 0.5, k + 0.6], 'k-', lw=1)
        ax.text(x, k + 0.75, str(x), ha='center', va='bottom', fontsize=9)
    for rank_i, j in enumerate(order):
        y = k - rank_i
        ar = avg_rank[j]
        ax.plot([ar, ar], [y, k + 0.5], 'k-', lw=0.8)
        ax.plot([ar], [y], 'o', color=A.get_method_color(methods[j],
                ORDER.index(methods[j])), ms=9)
        ax.text(ar, y, f'  {LABEL[methods[j]]} ({ar:.2f})',
                va='center', ha='left', fontsize=9)
    ax.plot([lo, lo + cd], [0.6, 0.6], 'k-', lw=2.5)
    ax.text(lo + cd / 2, 0.75, f'CD = {cd:.2f}', ha='center', fontsize=9)
    ax.set_title(f'{metric} 平均順位 (左=強い)   Friedman p={p:.4f}, '
                 f"Kendall's W={W:.2f}   CD={cd:.2f}内は有意差なし", fontsize=11)
    ax.axis('off')
    fig.tight_layout()
    fig.savefig(outpath, dpi=140)
    plt.close(fig)
    print(f"  → {outpath}")


def _rank_analysis(probs, med_by_method, metric, outpath):
    methods = [m for m in ORDER if m in med_by_method]
    k, N = len(methods), len(probs)
    mat = np.array([[med_by_method[m][i] for m in methods] for i in range(N)])
    ranks = np.array([rankdata(-row, method='average') for row in mat])
    avg_rank = ranks.mean(axis=0)
    try:
        chi, p = friedmanchisquare(*[mat[:, j] for j in range(k)])
        W = chi / (N * (k - 1))
    except Exception:
        chi, p, W = float('nan'), float('nan'), float('nan')
    cd = Q05.get(k, 2.949) * np.sqrt(k * (k + 1) / (6.0 * N))
    order = np.argsort(avg_rank)
    print(f"\n===== {metric}: 全{N}問題の平均順位 (1=最良)  "
          f"Friedman p={p:.4f}, W={W:.3f}, CD={cd:.2f} =====")
    for rnk, j in enumerate(order, 1):
        print(f"  {rnk}. {LABEL[methods[j]]:<16} 平均順位={avg_rank[j]:.2f}")
    _plot_cd(methods, avg_rank, cd, metric, outpath, p, W)


# ========== アンタイム AOC（中央値曲線）＋ 交差図 ==========

def parse_anytime(path):
    """anytime_detail_<w>.txt → {method_key: (ts[], hv_med[])}。"""
    name2key = {A.METHOD_LABELS.get(m, m): m for m in ORDER}
    out, cur = {}, None
    for raw in open(path, encoding='utf-8', errors='replace'):
        s = raw.rstrip()
        if s.startswith('## '):
            lbl = s[3:].strip()
            cur = name2key.get(lbl, lbl)
            out[cur] = ([], [])
            continue
        t = s.strip()
        if not t or t.startswith('#') or t.startswith('t(s)') or t.startswith('-'):
            continue
        f = t.split()
        if cur is None or len(f) < 6:
            continue
        try:
            ti, hv = float(f[0]), float(f[5])
        except ValueError:
            continue
        out[cur][0].append(ti)
        out[cur][1].append(hv)
    return out


def _aoc_curve(adir, weight):
    """analysis 下の各問題から AOC（中央値曲線）と最終HVを集める。"""
    probs = A.order_prob_labels(
        d for d in os.listdir(adir) if os.path.isdir(os.path.join(adir, d)))
    aoc_by = {m: {} for m in ORDER}
    curves, tags = {}, []
    for d in probs:
        path = os.path.join(adir, d, f'anytime_detail_{weight}.txt')
        if not os.path.exists(path):
            continue
        st = short_tag(d)
        tags.append(st)
        series = parse_anytime(path)
        curves[st] = series
        for m in ORDER:
            if m in series and series[m][0]:
                aoc_by[m][st] = A._aoc_from_curve(*series[m])
    return tags, aoc_by, curves


def _plot_crossover(curves, tags, weight, outpath):
    reps = [t for t in ('la21', 'la36L') if t in curves] or tags[:2]
    show = ['ils_baseline', 'memetic_ls', 'memetic_pr', 'memetic_repair']
    fig, axes = plt.subplots(1, len(reps), figsize=(6.2 * len(reps), 4.6))
    if len(reps) == 1:
        axes = [axes]
    for ax, st in zip(axes, reps):
        for m in show:
            if m in curves[st] and curves[st][m][0]:
                ts, hvs = curves[st][m]
                ax.plot(ts, hvs, marker='.', ms=4, label=LABEL[m],
                        color=A.get_method_color(m, ORDER.index(m)))
        ax.set_xscale('log')
        ax.set_xlabel('時間 [s] (log)')
        ax.set_ylabel('HV (正規化)')
        ax.set_title(f'{st}: アンタイム HV（{weight}）')
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle('アンタイム曲線の交差 ― ILSが早期リード, Memetic+機構が後で逆転', fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(outpath, dpi=140)
    plt.close(fig)
    print(f"  → {outpath}")


# ========== 性能プロファイル（Dolan-Moré 2002） ==========

def _perf_profile(tags, data, metric, outpath):
    ratios = {m: [] for m in ORDER}
    for t in tags:
        vals = {m: data[m].get(t, float('nan')) for m in ORDER}
        fin = [v for v in vals.values() if np.isfinite(v)]
        if not fin:
            continue
        best = max(fin)
        for m in ORDER:
            v = vals[m]
            if np.isfinite(v):
                ratios[m].append(best / v if v > 0 else np.inf)
    taus = np.linspace(1.0, 3.0, 120)
    fig, ax = plt.subplots(figsize=(8, 5))
    for m in ORDER:
        r = np.array(ratios[m], float)
        if len(r) == 0:
            continue
        rho = [np.mean(r <= tau) for tau in taus]
        ax.step(taus, rho, where='post',
                label=f'{LABEL[m]} (最良={np.mean(r <= 1.0 + 1e-9):.0%})',
                color=A.get_method_color(m, ORDER.index(m)), lw=1.8)
    ax.set_xlabel('τ (最良からの比率 best/値, 1=最良)')
    ax.set_ylabel('ρ(τ) = τ以内に入る問題の割合')
    ax.set_title(f'{metric} 性能プロファイル(Dolan-Moré)  τ=1切片=最良率 / 立上り=僅差さ')
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc='lower right')
    fig.tight_layout()
    fig.savefig(outpath, dpi=140)
    plt.close(fig)
    print(f"  → {outpath}")


# ========== エントリ ==========

def generate_all(results_dir, weight=DEFAULT_WEIGHT, scenarios=None):
    """summary.md と整合する全横断図を <results_dir>/analysis/ に生成する。"""
    adir = os.path.join(results_dir, 'analysis')
    os.makedirs(adir, exist_ok=True)
    grouped = A.load_all_runs(results_dir)
    if not grouped:
        print('[figures_v3] データなし。スキップ。')
        return

    # 短縮タグはこの後 claim/perf 図で辞書キーになる。衝突すると片方の問題が
    # 黙って欠落するため、ここで検出して警告する（problem_short_tag の一意性前提）。
    _tags = [short_tag(p, s) for (p, s) in grouped]
    _dups = sorted({t for t in _tags if _tags.count(t) > 1})
    if _dups:
        print(f'[figures_v3][WARN] 短縮タグが衝突: {_dups} '
              '→ analyze_v3.PROBLEM_SHORT に一意タグを追記してください（図でデータ欠落の恐れ）')

    # claim1 の AOC 列用に per-trial AOC（all_mean）を _summary_data.pkl から読む。
    # pkl は analyze_v3 が書き、aoc_all_weights.py --write で all_mean にパッチ済みの想定。
    # 無い/壊れている場合は claim1 を従来の HV2枚に縮退させる。
    aoc_by_tag = {}
    pkl_path = os.path.join(adir, '_summary_data.pkl')
    if os.path.exists(pkl_path):
        try:
            import pickle
            with open(pkl_path, 'rb') as f:
                _S = pickle.load(f)
            aoc_by_tag = {k: v.get('aoc_pt', {}) for k, v in _S.items()}
            _aw = next(iter(_S.values())).get('aoc_weight') if _S else None
            print(f"[figures_v3] claim1 AOC: _summary_data.pkl から読込 (aoc_weight={_aw})")
        except Exception as e:
            print(f'[figures_v3] AOC pkl 読込失敗（claim1 はHV2枚に縮退）: {e}')

    # --- 主張バー図 + 問題別 + 俯瞰 ---
    claim_rows, rows_full = [], []
    for (prob, scen), method_data in _ordered_items(grouped):
        tag = f'{prob}_{scen}'
        if scenarios and not any(s in tag for s in scenarios):
            continue
        methods, union_pt, high_pt, scal = collect(method_data, weight)
        st = short_tag(prob, scen)
        claim_rows.append((st, union_pt, high_pt, aoc_by_tag.get(tag, {})))
        union_med = {m: med(union_pt[m]) for m in methods}
        high_med = {m: med(high_pt[m]) for m in methods}
        out_dir = os.path.join(adir, tag)
        os.makedirs(out_dir, exist_ok=True)
        plot_problem(tag, methods, union_med, high_med, scal, weight,
                     os.path.join(out_dir, 'summary_bars.png'))
        rows_full.append((tag, methods, union_med, high_med, scal))

    if claim_rows:
        plot_claim1(claim_rows, os.path.join(adir, 'claim1_ils_vs_memetic.png'))
        plot_claim2(claim_rows, os.path.join(adir, 'claim2_mechanism.png'))
        print(f"  → {os.path.join(adir, 'claim1_ils_vs_memetic.png')}")
        print(f"  → {os.path.join(adir, 'claim2_mechanism.png')}")

    if len(rows_full) > 1:
        n = len(rows_full)
        fig, axes = plt.subplots(n, 3, figsize=(15, 3.4 * n))
        for i, (tag, methods, um, hm, sc) in enumerate(rows_full):
            _bars(axes[i, 0], methods, um, f'{tag}\n統合HV', 'HV')
            _bars(axes[i, 1], methods, hm, '高安定HV', 'HV')
            _bars(axes[i, 2], methods, sc, f'scalar @{weight}', 'score')
        fig.tight_layout()
        fig.savefig(os.path.join(adir, 'summary_bars_all.png'), dpi=120)
        plt.close(fig)
        print(f"  → {os.path.join(adir, 'summary_bars_all.png')}")

    # --- 横断ランキング CD図（統合HV / 高安定HV） ---
    probs, union_med_x, high_med_x = _collect_medians(grouped, weight)
    _rank_analysis(probs, union_med_x, '統合HV',
                   os.path.join(adir, 'overall_ranking_union.png'))
    _rank_analysis(probs, high_med_x, '高安定HV',
                   os.path.join(adir, 'overall_ranking_highstab.png'))

    # --- AOC 交差図 + 性能プロファイル(AOC/統合/高安定) ---
    tags, aoc_by, curves = _aoc_curve(adir, weight)
    if curves:
        _plot_crossover(curves, tags, weight,
                        os.path.join(adir, 'anytime_crossover.png'))
        _perf_profile(tags, aoc_by, 'AOC',
                      os.path.join(adir, 'perf_profile_AOC.png'))
    # 統合HV / 高安定HV の性能プロファイル（median 行列を tag 辞書化）
    for metric, med_x in (('統合HV', union_med_x), ('高安定HV', high_med_x)):
        data = {m: {probs[i]: med_x[m][i] for i in range(len(probs))} for m in ORDER}
        _perf_profile(probs, data, metric,
                      os.path.join(adir, f'perf_profile_{metric}.png'))
    print('[figures_v3] 完了。')


def main(results_dir, scenarios, weight):
    generate_all(results_dir, weight=weight, scenarios=scenarios or None)


if __name__ == '__main__':
    argv = list(sys.argv[1:])
    weight = DEFAULT_WEIGHT
    if '--weight' in argv:
        i = argv.index('--weight')
        weight = argv[i + 1]
        del argv[i:i + 2]
    default_rd = os.path.join(_HERE, 'results', 'main_v1')
    if argv and (os.sep in argv[0] or '/' in argv[0] or os.path.isdir(argv[0])):
        rd, scen = argv[0], argv[1:]
    else:
        rd, scen = default_rd, argv
    main(rd, scen, weight)
