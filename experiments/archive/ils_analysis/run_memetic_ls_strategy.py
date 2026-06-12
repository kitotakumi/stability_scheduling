#!/usr/bin/env python3
"""
Memetic LS strategy 比較: best-improvement (BI) vs first-improvement (FI)

指標:
  - 最終解 (makespan / stability / weighted score)  median ± std
  - CPU 時間
  - 改善成功率 (baseline 比)
  - UEA HV (探索した解集合の hypervolume)
  - C-metric (BI vs FI 相互支配率)
  - Wilcoxon signed-rank + Cliff's delta (per-weight score)
  - 等時間比較 (固定 CPU 時間でのスコア)
  - Anytime カーブ (score / MS / stability)

=== 使い方 ===
  python run_memetic_ls_strategy.py
  python run_memetic_ls_strategy.py --n-trials 5 --n-jobs 4
"""

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from experiment_utils import compute_shared_norm_params, MEMETIC_NGEN

_HERE = os.path.dirname(os.path.abspath(__file__))

N_TRIALS = 5

PROBLEM_SETS = [
    ('la21', 'la21_delay147'),
    ('la36', 'la36_delay148'),
]

WEIGHTS_LIST = [
    [0.8, 0.2],
    [0.7, 0.3],
    [0.5, 0.5],
]

METHODS = {
    'best':  {'ls_strategy': 'best',  'label': 'BI (best)',  'color': 'tab:blue',   'ls': '-'},
    'first': {'ls_strategy': 'first', 'label': 'FI (first)', 'color': 'tab:orange', 'ls': '--'},
}

SNAPSHOT_TIMES = [10.0, 30.0, 60.0, 120.0]


def _weight_label(w):
    return f"w{int(round(w[0]*10)):02d}_{int(round(w[1]*10)):02d}"


# ========== Pareto / HV / C-metric (analyze_v2 から移植) ==========

def pareto_front(points):
    if len(points) == 0:
        return np.zeros((0, 2))
    pts = np.asarray(points, dtype=float)
    idx = np.lexsort((pts[:, 1], pts[:, 0]))
    srt = pts[idx]
    if len(srt) == 1:
        return srt
    cummin_y = np.minimum.accumulate(srt[:, 1])
    on_front = np.empty(len(srt), dtype=bool)
    on_front[0] = True
    on_front[1:] = srt[1:, 1] < cummin_y[:-1]
    return srt[on_front]


def hypervolume(points, ref):
    if len(points) == 0:
        return 0.0
    pf = pareto_front(points)
    if len(pf) == 0:
        return 0.0
    pf = pf[np.argsort(pf[:, 0])]
    hv = 0.0
    prev_x = ref[0]
    for p in pf[::-1]:
        if p[0] >= ref[0] or p[1] >= ref[1]:
            continue
        hv += (prev_x - p[0]) * (ref[1] - p[1])
        prev_x = p[0]
    return hv


def c_metric(A, B):
    """C(A,B): A が弱く支配する B の点の割合。"""
    if len(B) == 0:
        return 0.0
    count = sum(1 for b in B if any(a[0] <= b[0] and a[1] <= b[1] for a in A))
    return count / len(B)


def _wilcoxon_paired(x, y):
    """paired Wilcoxon (x < y を主張). Returns (stat, p, cliffs_d)."""
    try:
        from scipy.stats import wilcoxon as scipy_wilcoxon
    except ImportError:
        return float('nan'), float('nan'), float('nan')
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    pairs = [(xi, yi) for xi, yi in zip(x, y) if np.isfinite(xi) and np.isfinite(yi)]
    if len(pairs) < 4:
        return float('nan'), float('nan'), float('nan')
    xp, yp = np.array([p[0] for p in pairs]), np.array([p[1] for p in pairs])
    diff = xp - yp
    if np.all(diff == 0):
        return float('nan'), float('nan'), 0.0
    try:
        stat, p = scipy_wilcoxon(diff, alternative='less')
    except Exception:
        stat, p = float('nan'), float('nan')
    total = len(xp) * len(yp)
    conc = sum(1 for xi in xp for yj in yp if xi < yj)
    disc = sum(1 for xi in xp for yj in yp if xi > yj)
    d = (conc - disc) / total if total > 0 else float('nan')
    return float(stat), float(p), float(d)


def _effect_label(d):
    a = abs(d)
    if np.isnan(a):
        return '?'
    if a < 0.147:
        return 'neg'
    if a < 0.330:
        return 'small'
    if a < 0.474:
        return 'med'
    return 'large'


def _p_star(p):
    if np.isnan(p):
        return '  '
    if p < 0.001: return '***'
    if p < 0.01:  return '** '
    if p < 0.05:  return '*  '
    return '   '


# ========== 並列タスク ==========

def _extract_uea_points(history):
    """pop_points から UEA 点列を抽出 (memetic は GA 形式)。"""
    pts = []
    for h in history:
        for pt in h.get('pop_points', []):
            if len(pt) >= 2:
                ms, st = float(pt[0]), float(pt[1])
                if np.isfinite(ms) and np.isfinite(st):
                    pts.append([ms, st])
    return pts


def _slim_history(history):
    out = []
    for h in history:
        t = h.get('cpu_time')
        if t is None:
            continue
        out.append({
            'cpu_time': float(t),
            'best_ms':    h.get('best_makespan'),
            'best_st':    h.get('best_stability'),
            'best_score': h.get('best_score'),
            'pop_points': h.get('pop_points', []),
        })
    return out


def _run_task(task):
    import sys as _sys, os as _os, traceback as _tb
    _sys.path.insert(0, _os.path.join(_os.path.dirname(__file__), '..'))
    _sys.path.insert(0, _os.path.join(_os.path.dirname(__file__), '..', '..'))
    from experiment_utils import run_memetic as _run_memetic

    out_path = task['out_path']
    if _os.path.exists(out_path):
        with open(out_path, 'r', encoding='utf-8') as f:
            d = json.load(f)
        return {'status': 'cached', 'key': task['key'],
                'makespan': d['finals']['makespan'],
                'stability': d['finals']['stability'],
                'cpu': d['convergence'].get('total_cpu_time', 0.0),
                'history': d['history'],
                'uea_points': d['uea_points'],
                'baseline_score': d.get('baseline_score')}

    try:
        r = _run_memetic(
            task['weights'], task['seed'], task['ngen'],
            norm_params=task['norm_params'],
            problem_name=task['problem'],
            scenario_name=task['scenario'],
            repair_prob=0.0,
            repair_strength=2,
            track_population=True,
            ls_strategy=task['ls_strategy'],
        )
        slim = _slim_history(r['history'])
        uea_pts = _extract_uea_points(r['history'])

        save = {
            'method': task['method_key'],
            'problem': task['problem'],
            'scenario': task['scenario'],
            'weights': task['weights'],
            'trial': task['trial'],
            'seed': task['seed'],
            'baseline': r.get('baseline'),
            'baseline_rsr': r.get('baseline_rsr'),
            'baseline_score': r.get('baseline_score'),
            'finals': {'makespan': r['makespan'], 'stability': r['stability']},
            'convergence': r['convergence'],
            'history': slim,
            'uea_points': uea_pts,
        }
        _os.makedirs(_os.path.dirname(out_path), exist_ok=True)
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(save, f, ensure_ascii=False)

        return {'status': 'done', 'key': task['key'],
                'makespan': r['makespan'], 'stability': r['stability'],
                'cpu': r['convergence'].get('total_cpu_time', 0.0),
                'history': slim, 'uea_points': uea_pts,
                'baseline_score': r.get('baseline_score')}
    except Exception as e:
        return {'status': 'error', 'key': task['key'],
                'error': str(e), 'traceback': _tb.format_exc()}


# ========== 分析ユーティリティ ==========

def _med(v):
    return float(np.median(v)) if v else float('nan')

def _std(v):
    return float(np.std(v)) if v else float('nan')


def _score_at_time(history, t_cut):
    """time budget t_cut での best_score を返す (補間なし、最後の有効エントリ)。"""
    val = None
    for h in history:
        if h['cpu_time'] <= t_cut:
            val = h.get('best_score')
        else:
            break
    return val


def _compute_ref_point(all_runs_by_method):
    """HV 参照点: 全手法の最大 MS と最大 Stab の 1.05 倍。"""
    ms_max, st_max = 0.0, 0.0
    for runs in all_runs_by_method.values():
        for r in runs:
            for pt in r.get('uea_points', []):
                ms_max = max(ms_max, pt[0])
                st_max = max(st_max, pt[1])
    return (ms_max * 1.05, st_max * 1.05)


# ========== テキスト出力 ==========

def print_summary(grouped, weights_list, norm_cache, out_dir):
    lines = []
    lines.append("=" * 90)
    lines.append("Memetic LS strategy 比較: BI (best-improvement) vs FI (first-improvement)")
    lines.append("repair_prob=0.0 (Memetic-LS のみ)")
    lines.append("=" * 90)

    for (problem, scenario), by_weight in sorted(grouped.items()):
        lines.append(f"\n{'─'*80}")
        lines.append(f"問題: {problem}")
        lines.append(f"{'─'*80}")

        # 各手法の全trial を収集してHV参照点を決定
        all_runs = {mk: [] for mk in METHODS}
        for wl, by_method in by_weight.items():
            for mk in METHODS:
                all_runs[mk].extend(by_method.get(mk, []))
        ref = _compute_ref_point(all_runs)
        lines.append(f"HV 参照点: MS={ref[0]:.1f}, Stab={ref[1]:.3f}")

        # ── per-weight スカラー比較 ──
        lines.append(f"\n{'重み':<10}  {'手法':<12}  {'MS':>8}  {'Stab':>8}  {'Score':>8}  {'CPU':>8}  {'改善率':>6}")
        lines.append(f"{'':─<10}  {'':─<12}  {'':─>8}  {'':─>8}  {'':─>8}  {'':─>8}  {'':─>6}")

        for w in weights_list:
            wl = _weight_label(w)
            by_method = by_weight.get(wl, {})
            for mk, cfg in METHODS.items():
                runs = by_method.get(mk, [])
                if not runs:
                    lines.append(f"{wl:<10}  {cfg['label']:<12}  (データなし)")
                    continue
                ms_v = [r['makespan'] for r in runs]
                st_v = [r['stability'] for r in runs]
                sc_v = [h['best_score'] for r in runs
                        for h in r['history'][-1:] if h.get('best_score') is not None]
                cpu_v = [r['cpu'] for r in runs]
                imp_v = [r for r in runs
                         if r.get('baseline_score') is not None
                         and r['history'] and r['history'][-1].get('best_score') is not None
                         and r['history'][-1]['best_score'] < r['baseline_score'] - 1e-6]
                imp_rate = len(imp_v) / len(runs) if runs else float('nan')
                lines.append(
                    f"{wl:<10}  {cfg['label']:<12}  "
                    f"{_med(ms_v):>8.1f}  {_med(st_v):>8.3f}  "
                    f"{_med(sc_v):>8.5f}  {_med(cpu_v):>8.1f}s  {imp_rate:>5.0%}"
                )

        # ── per-weight Wilcoxon (BI < FI ? = BI の方がスコア小さい?) ──
        lines.append(f"\n  [per-weight Wilcoxon: BI_score < FI_score?]")
        lines.append(f"  {'重み':<10}  {'n':>3}  {'BI med':>9}  {'FI med':>9}  {'p':>7}  {'sig':>4}  {'Cliff d':>8}  {'効果'}  {'方向'}")
        for w in weights_list:
            wl = _weight_label(w)
            by_method = by_weight.get(wl, {})
            bi_runs = by_method.get('best', [])
            fi_runs = by_method.get('first', [])
            if not bi_runs or not fi_runs:
                continue
            bi_sc = [r['history'][-1]['best_score'] for r in bi_runs
                     if r['history'] and r['history'][-1].get('best_score') is not None]
            fi_sc = [r['history'][-1]['best_score'] for r in fi_runs
                     if r['history'] and r['history'][-1].get('best_score') is not None]
            n = min(len(bi_sc), len(fi_sc))
            stat, p, d = _wilcoxon_paired(bi_sc[:n], fi_sc[:n])
            winner = 'BI優位' if (not np.isnan(d) and d < -0.147) else \
                     'FI優位' if (not np.isnan(d) and d > 0.147) else '差なし'
            lines.append(
                f"  {wl:<10}  {n:>3}  {_med(bi_sc):>9.5f}  {_med(fi_sc):>9.5f}  "
                f"{p:>7.4f}  {_p_star(p):>4}  {d:>8.3f}  {_effect_label(d):>5}  {winner}"
            )

        # ── 等時間比較 ──
        lines.append(f"\n  [等時間比較 (固定 CPU 時間でのスコア)]")
        for w in weights_list:
            wl = _weight_label(w)
            by_method = by_weight.get(wl, {})
            lines.append(f"  {wl}:")
            for t_cut in SNAPSHOT_TIMES:
                parts = []
                for mk, cfg in METHODS.items():
                    runs = by_method.get(mk, [])
                    sc_at_t = [s for r in runs
                               for s in [_score_at_time(r['history'], t_cut)]
                               if s is not None]
                    if sc_at_t:
                        parts.append(f"{cfg['label']}={_med(sc_at_t):.5f}")
                    else:
                        parts.append(f"{cfg['label']}=N/A")
                lines.append(f"    t={t_cut:5.0f}s: " + "  |  ".join(parts))

        # ── union HV (全重み合算) ──
        lines.append(f"\n  [union HV (全 UEA 点のハイパーボリューム, 全重み合算)]")
        lines.append(f"  {'手法':<12}  {'HV':>12}")
        for mk, cfg in METHODS.items():
            all_pts = []
            for runs in [all_runs[mk]]:
                for r in runs:
                    all_pts.extend(r.get('uea_points', []))
            hv = hypervolume(all_pts, ref) if all_pts else 0.0
            lines.append(f"  {cfg['label']:<12}  {hv:>12.4f}")

        # ── C-metric ──
        lines.append(f"\n  [C-metric (全 UEA 点)]")
        bi_pts = [pt for r in all_runs['best'] for pt in r.get('uea_points', [])]
        fi_pts = [pt for r in all_runs['first'] for pt in r.get('uea_points', [])]
        if bi_pts and fi_pts:
            bi_pf = pareto_front(bi_pts).tolist() if bi_pts else []
            fi_pf = pareto_front(fi_pts).tolist() if fi_pts else []
            c_bi_fi = c_metric(bi_pf, fi_pf)
            c_fi_bi = c_metric(fi_pf, bi_pf)
            lines.append(f"  C(BI, FI) = {c_bi_fi:.3f}  (BI Pareto が FI Pareto を支配する割合)")
            lines.append(f"  C(FI, BI) = {c_fi_bi:.3f}  (FI Pareto が BI Pareto を支配する割合)")
        else:
            lines.append("  UEA 点なし")

    text = "\n".join(lines)
    print(text)
    fpath = os.path.join(out_dir, 'summary.txt')
    with open(fpath, 'w', encoding='utf-8') as f:
        f.write(text)
    print(f"\n[保存] {fpath}")


# ========== プロット ==========

def plot_anytime(grouped, weights_list, out_dir):
    for (problem, scenario), by_weight in sorted(grouped.items()):
        # 全重み合算した anytime (score)
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        fig.suptitle(f"{problem} — Anytime (全重み平均)", fontsize=11)

        for mk, cfg in METHODS.items():
            all_histories = []
            for wl in [_weight_label(w) for w in weights_list]:
                runs = by_weight.get(wl, {}).get(mk, [])
                all_histories.extend([r['history'] for r in runs if r.get('history')])

            if not all_histories:
                continue

            t_max = min((h[-1]['cpu_time'] for h in all_histories if h), default=0)
            if t_max <= 0:
                continue
            t_grid = np.linspace(0, t_max, 300)

            for field, ax, ylabel in zip(
                    ['best_score', 'best_ms', 'best_st'],
                    axes,
                    ['Score', 'Makespan', 'Stability']):
                interp_list = []
                for hist in all_histories:
                    times = np.array([h['cpu_time'] for h in hist])
                    vals = np.array([h.get(field) if h.get(field) is not None else np.nan
                                     for h in hist], dtype=float)
                    valid = ~np.isnan(vals)
                    if valid.sum() < 2:
                        continue
                    interp_list.append(np.interp(t_grid, times[valid], vals[valid]))
                if not interp_list:
                    continue
                mean_v = np.nanmean(interp_list, axis=0)
                std_v = np.nanstd(interp_list, axis=0)
                ax.plot(t_grid, mean_v, color=cfg['color'], ls=cfg['ls'],
                        lw=2.0, label=cfg['label'])
                ax.fill_between(t_grid, mean_v - std_v, mean_v + std_v,
                                color=cfg['color'], alpha=0.15)
                ax.set_xlabel('CPU time (s)')
                ax.set_ylabel(ylabel)
                ax.set_title(ylabel)
                ax.legend(fontsize=8)
                ax.grid(alpha=0.3)

        plt.tight_layout()
        fname = os.path.join(out_dir, f'anytime_{problem}_all.png')
        plt.savefig(fname, dpi=120, bbox_inches='tight')
        plt.close()
        print(f"  [図] {fname}")

        # 重み別 anytime (score のみ)
        n_w = len(weights_list)
        fig, axes = plt.subplots(1, n_w, figsize=(4 * n_w, 4), sharey=False)
        if n_w == 1:
            axes = [axes]
        fig.suptitle(f"{problem} — per-weight Score Anytime", fontsize=11)
        for ax, w in zip(axes, weights_list):
            wl = _weight_label(w)
            ax.set_title(wl, fontsize=9)
            ax.set_xlabel('CPU time (s)')
            ax.set_ylabel('Score')
            ax.grid(alpha=0.3)
            for mk, cfg in METHODS.items():
                runs = by_weight.get(wl, {}).get(mk, [])
                histories = [r['history'] for r in runs if r.get('history')]
                if not histories:
                    continue
                t_max = min((h[-1]['cpu_time'] for h in histories if h), default=0)
                if t_max <= 0:
                    continue
                t_grid = np.linspace(0, t_max, 200)
                interp_list = []
                for hist in histories:
                    times = np.array([h['cpu_time'] for h in hist])
                    vals = np.array([h.get('best_score') if h.get('best_score') is not None
                                     else np.nan for h in hist], dtype=float)
                    valid = ~np.isnan(vals)
                    if valid.sum() < 2:
                        continue
                    interp_list.append(np.interp(t_grid, times[valid], vals[valid]))
                if not interp_list:
                    continue
                mean_v = np.nanmean(interp_list, axis=0)
                ax.plot(t_grid, mean_v, color=cfg['color'], ls=cfg['ls'],
                        lw=1.8, label=cfg['label'])
            ax.legend(fontsize=7)
        plt.tight_layout()
        fname = os.path.join(out_dir, f'anytime_{problem}_per_weight.png')
        plt.savefig(fname, dpi=120, bbox_inches='tight')
        plt.close()
        print(f"  [図] {fname}")


def plot_hv_boxplot(grouped, weights_list, out_dir):
    """UEA HV の boxplot (per problem, per weight)。"""
    for (problem, scenario), by_weight in sorted(grouped.items()):
        # 全手法の全 UEA 点から HV 参照点を決定
        all_runs = {mk: [] for mk in METHODS}
        for wl in [_weight_label(w) for w in weights_list]:
            for mk in METHODS:
                all_runs[mk].extend(by_weight.get(wl, {}).get(mk, []))
        ref = _compute_ref_point(all_runs)

        fig, axes = plt.subplots(1, len(weights_list), figsize=(3 * len(weights_list), 4))
        if len(weights_list) == 1:
            axes = [axes]
        fig.suptitle(f"{problem} — UEA HV per trial", fontsize=11)

        for ax, w in zip(axes, weights_list):
            wl = _weight_label(w)
            data = []
            labels = []
            for mk, cfg in METHODS.items():
                runs = by_weight.get(wl, {}).get(mk, [])
                hv_list = [hypervolume(r.get('uea_points', []), ref) for r in runs]
                data.append(hv_list)
                labels.append(cfg['label'])
            bp = ax.boxplot(data, labels=labels, patch_artist=True)
            for patch, mk in zip(bp['boxes'], METHODS.keys()):
                patch.set_facecolor(METHODS[mk]['color'])
                patch.set_alpha(0.6)
            ax.set_title(wl, fontsize=9)
            ax.set_ylabel('UEA HV')
            ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        fname = os.path.join(out_dir, f'hv_boxplot_{problem}.png')
        plt.savefig(fname, dpi=120, bbox_inches='tight')
        plt.close()
        print(f"  [図] {fname}")


# ========== メイン ==========

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-trials', type=int, default=N_TRIALS)
    parser.add_argument('--n-jobs', type=int, default=4)
    parser.add_argument('--ngen', type=int, default=MEMETIC_NGEN)
    args = parser.parse_args()

    out_dir = os.path.join(_HERE, 'results', 'memetic_ls_strategy')
    os.makedirs(out_dir, exist_ok=True)

    # norm_params
    norm_cache = {}
    for problem, scenario in PROBLEM_SETS:
        print(f'[norm_params] {problem}/{scenario}')
        norm_cache[(problem, scenario)] = compute_shared_norm_params(problem, scenario)

    # タスク展開
    tasks = []
    for problem, scenario in PROBLEM_SETS:
        np_params = norm_cache[(problem, scenario)]
        for w in WEIGHTS_LIST:
            wl = _weight_label(w)
            for mk, cfg in METHODS.items():
                for trial in range(args.n_trials):
                    raw_dir = os.path.join(out_dir, f'{problem}_{scenario}', 'raw')
                    out_path = os.path.join(raw_dir, f'{mk}__{wl}__t{trial:03d}.json')
                    tasks.append({
                        'key': f"{problem}/{wl}/{mk}",
                        'method_key': mk,
                        'problem': problem,
                        'scenario': scenario,
                        'weights': w,
                        'ls_strategy': cfg['ls_strategy'],
                        'seed': trial * 100 + 7,
                        'ngen': args.ngen,
                        'norm_params': np_params,
                        'trial': trial,
                        'out_path': out_path,
                    })

    skip = sum(1 for t in tasks if os.path.exists(t['out_path']))
    print(f"\n総タスク: {len(tasks)}  スキップ(済): {skip}  実行: {len(tasks)-skip}")

    # 並列実行
    # grouped[(problem, scenario)][w_label][method_key] = list of run dicts
    grouped = {}
    done = 0

    with ProcessPoolExecutor(max_workers=args.n_jobs) as ex:
        futures = {ex.submit(_run_task, t): t for t in tasks}
        for fut in as_completed(futures):
            r = fut.result()
            done += 1
            key = r['key']
            if r['status'] in ('done', 'cached'):
                t = futures[fut]
                prob_key = (t['problem'], t['scenario'])
                wl = _weight_label(t['weights'])
                mk = t['method_key']
                grouped.setdefault(prob_key, {}).setdefault(wl, {}).setdefault(mk, []).append(r)
                status = 'cached' if r['status'] == 'cached' else 'done'
                print(f"  [{done}/{len(tasks)}] {key}  MS={r['makespan']}  "
                      f"Stab={r['stability']:.3f}  CPU={r['cpu']:.1f}s  [{status}]")
            else:
                print(f"  [ERROR] {key}: {r.get('error')}")
                for line in r.get('traceback', '').splitlines()[-5:]:
                    print(f"    {line}")

    print("\n分析中...")
    print_summary(grouped, WEIGHTS_LIST, norm_cache, out_dir)

    print("\nグラフ生成中...")
    plot_anytime(grouped, WEIGHTS_LIST, out_dir)
    plot_hv_boxplot(grouped, WEIGHTS_LIST, out_dir)

    print(f"\n完了: {out_dir}")


if __name__ == '__main__':
    main()
