"""
GA/ILS共通の評価関数モジュール

安定性関数・メイクスパン計算・正規化・重み付き目的関数を提供する。
GA・ILSの両方がこのモジュールを使用することで、公平な比較を保証する。
"""

import copy
import random

import gantt_chart_operation


# ========== 安定性関数 ==========

# 安定性 = 機械ごとのジョブ処理順序（順列）の偏差 D = Σ ω·|init_pos − cur_pos|。
# 元スケジュール S_p から処理順がどれだけ動いたかを測る（小さいほど安定）。
#
# 重み指数 β（ω = 1/(cur_pos+1)^β）で 2 つの指標を切り替える:
#   β = 0.0（既定）: 重みなしの順列偏差 D = Σ|init_pos − cur_pos|（修論で採用）。
#   β = 1.25       : 早期投入ジョブの順序変動を重くみる重み付き偏差（卒論の提案指標）。
STABILITY_RANK_WEIGHT_BETA = 0.0


def compute_stability_from_orders(init_orders, current_orders):
    """machine_ordersから安定性を計算（ILS用）

    Parameters:
        init_orders: 初期解の {machine_id: [(job_id, op_idx), ...]}
        current_orders: 現在解の同形式

    Returns:
        float: 順列偏差 Σ|init_pos − cur_pos|
    """
    total = 0.0
    for m_idx in init_orders:
        if m_idx not in current_orders:
            continue
        init_jobs = [op[0] for op in init_orders[m_idx]]
        current_jobs = [op[0] for op in current_orders[m_idx]]
        total += _rank_deviation(init_jobs, current_jobs)
    return total


def compute_stability_from_gantt(init_gantt, current_gantt, fixed_gantt):
    """ガントチャートから安定性を計算（GA用）

    Parameters:
        init_gantt: 初期ガントチャート（delayed_gantt）
        current_gantt: リスケ後ガントチャート
        fixed_gantt: 確定済みガントチャート

    Returns:
        float: 順列偏差 Σ|init_pos − cur_pos|
    """
    init_changed = _extract_changed_gantt(init_gantt, fixed_gantt)
    current_changed = _extract_changed_gantt(current_gantt, fixed_gantt)
    total = 0.0
    for machine in range(len(current_changed)):
        if not current_changed[machine] or not init_changed[machine]:
            continue
        init_jobs = [task[2] for task in init_changed[machine]]
        current_jobs = [task[2] for task in current_changed[machine]]
        total += _rank_deviation(init_jobs, current_jobs)
    return total


def compute_stability_stat(init_gantt, current_gantt, fixed_gantt):
    """安定性の詳細統計を返す（分析用）

    Returns:
        tuple: (rank_deviation, changed_count, distance_mean, distance_sum,
                per_position_distances[10])
    """
    init_changed = _extract_changed_gantt(init_gantt, fixed_gantt)
    current_changed = _extract_changed_gantt(current_gantt, fixed_gantt)
    rank_diff_sum = 0.0
    changed_count = 0
    distance_sum = 0
    per_position = [0] * 10

    for machine in range(len(current_changed)):
        if not current_changed[machine] or not init_changed[machine]:
            continue
        init_jobs = [task[2] for task in init_changed[machine]]
        current_jobs = [task[2] for task in current_changed[machine]]
        for init_pos, job_id in enumerate(init_jobs):
            current_pos = current_jobs.index(job_id)
            diff = abs(init_pos - current_pos)
            rank_diff_sum += diff / (current_pos + 1) ** STABILITY_RANK_WEIGHT_BETA
            if diff != 0:
                changed_count += 1
                distance_sum += diff
                if current_pos < 10:
                    per_position[current_pos] += diff

    distance_mean = distance_sum / changed_count if changed_count > 0 else 0
    return (rank_diff_sum, changed_count, distance_mean, distance_sum, *per_position)


def _rank_deviation(init_jobs, current_jobs):
    """順位（処理順）偏差のコア計算: Σ |init_pos − cur_pos| / (cur_pos+1)^β

    β = STABILITY_RANK_WEIGHT_BETA。既定 β=0 では重みが消え、単純な順列偏差になる。
    """
    total = 0.0
    for init_pos, job_id in enumerate(init_jobs):
        current_pos = current_jobs.index(job_id)
        total += abs(init_pos - current_pos) / (current_pos + 1) ** STABILITY_RANK_WEIGHT_BETA
    return total


def _extract_changed_gantt(gantt, fixed_gantt):
    """ガントチャートから確定済み部分を除外する。

    標準 JSSP では (machine, job) が一意なのでジョブ番号で照合する。
    タプル全体の時刻一致比較は、check_disturbance が right-shift 後の時刻で
    fixed_gantt を返すようになったため一致しないケースがあり使えない。
    """
    changed = []
    for gantt_machine, fixed_machine in zip(gantt, fixed_gantt):
        fixed_jobs = {op[2] for op in fixed_machine}
        changed_ops = [op for op in gantt_machine if op[2] not in fixed_jobs]
        changed.append(changed_ops)
    return changed


# ========== メイクスパン ==========

def compute_makespan_from_gantt(gantt):
    """ガントチャートからメイクスパンを計算"""
    makespan = 0
    for machine_tasks in gantt:
        if machine_tasks:
            makespan = max(makespan, machine_tasks[-1][1])
    return makespan


# ========== 正規化 ==========

def normalize_value(value, min_val, max_val):
    """min-max正規化 → [1, 2] の範囲

    min_val → 1.0, max_val → 2.0
    value < min_val の場合は < 1.0 になる（探索中に発見した良い解）
    """
    if max_val == min_val:
        return 1.0
    return 1 + (value - min_val) / (max_val - min_val)


def weighted_objective(makespan, stability, weights, norm_params):
    """重み付き目的関数

    Parameters:
        makespan: メイクスパン値
        stability: 安定性値
        weights: [効率性の重み, 安定性の重み]
        norm_params: dict with keys 'min_eff', 'max_eff', 'max_stab'

    Returns:
        float: 重み付き正規化スコア（小さいほど良い）
    """
    norm_eff = normalize_value(makespan, norm_params['min_eff'], norm_params['max_eff'])
    if norm_params['max_stab'] > 0:
        norm_stab = 1 + stability / norm_params['max_stab']
    else:
        norm_stab = 1.0
    return weights[0] * norm_eff + weights[1] * norm_stab


# ========== 共通正規化パラメータ推定 ==========

def estimate_normalization_params(jm_table, fixed_gantt, reschedule_time,
                                  delayed_gantt, base_gene, n_samples=200):
    """GT法ランダムサンプリングによる正規化パラメータ推定（GA/ILS共通）

    ランダムな遺伝子（ジョブ順列）を生成し、GT法（アクティブスケジュール）で
    デコードしてメイクスパン・安定性のサンプルを収集する。

    Parameters:
        jm_table: JobMachineTableBase インスタンス
        fixed_gantt: 確定済みガントチャート
        reschedule_time: リスケジューリング開始時刻
        delayed_gantt: 遅延ガントチャート（安定性計算の基準）
        base_gene: シャッフル元のジョブ順列（リスケ対象操作の遺伝子表現）
        n_samples: ランダムサンプル数

    Returns:
        dict: {'min_eff', 'max_eff', 'max_stab'}
    """
    eff_samples = []
    stab_samples = []

    # 初期解（シャッフルなし）を含める
    gantt = gantt_chart_operation.get_gantt_reactive(
        jm_table, base_gene, fixed_gantt, reschedule_time)
    eff_samples.append(compute_makespan_from_gantt(gantt))
    stab_samples.append(0.0)

    # Phase 1: GT法ランダムサンプリング
    for _ in range(n_samples):
        shuffled = list(base_gene)
        random.shuffle(shuffled)
        gantt = gantt_chart_operation.get_gantt_reactive(
            jm_table, shuffled, fixed_gantt, reschedule_time)
        ms = compute_makespan_from_gantt(gantt)
        st = compute_stability_from_gantt(delayed_gantt, gantt, fixed_gantt)
        eff_samples.append(ms)
        stab_samples.append(st)

    # Phase 2: 最良サンプル周辺で追加サンプリング（局所探索的な精緻化）
    # 上位10%のサンプルを取得し、それらを軽くシャッフルして再サンプリング
    eff_with_genes = []
    gene_copy = list(base_gene)
    for _ in range(n_samples):
        random.shuffle(gene_copy)
        gantt = gantt_chart_operation.get_gantt_reactive(
            jm_table, gene_copy, fixed_gantt, reschedule_time)
        ms = compute_makespan_from_gantt(gantt)
        eff_with_genes.append((ms, list(gene_copy)))

    eff_with_genes.sort(key=lambda x: x[0])
    top_k = max(1, n_samples // 10)
    for ms_val, gene in eff_with_genes[:top_k]:
        # 上位個体を軽い摂動で5回サンプリング
        for _ in range(5):
            g = list(gene)
            # 2-3箇所をスワップ
            for _ in range(random.randint(2, 3)):
                i, j = random.sample(range(len(g)), 2)
                g[i], g[j] = g[j], g[i]
            gantt = gantt_chart_operation.get_gantt_reactive(
                jm_table, g, fixed_gantt, reschedule_time)
            ms = compute_makespan_from_gantt(gantt)
            eff_samples.append(ms)

    # パラメータ算出
    eff_samples.sort()
    min_eff = eff_samples[0]
    p90_idx = int(len(eff_samples) * 0.9)
    max_eff = eff_samples[p90_idx] if p90_idx < len(eff_samples) else eff_samples[-1]
    max_eff = max(max_eff, min_eff + 1)  # ゼロ除算防止
    max_stab = max(stab_samples) if stab_samples else 1.0
    max_stab = max(max_stab, 1e-6)

    return {'min_eff': min_eff, 'max_eff': max_eff, 'max_stab': max_stab}
