"""
ILS (Iterated Local Search) による再スケジューリング

解表現: 機械ごとの作業順序 (Machine Order)
局所探索: N5近傍 (クリティカルブロック境界の交換, 閉路チェック不要)
評価: メイクスパン (効率性) + 順位偏差 (安定性) の重み付き和
"""

import copy
import random
import sys
import time
from collections import deque

import job_shop_scheduling
import gantt_chart_operation
import analysis


class ILSSolver:
    """反復局所探索法によるジョブショップ再スケジューリングソルバー"""

    def __init__(self, jm_table, fixed_gantt, reschedule_gantt, reschedule_time, weights):
        """
        Parameters:
            jm_table: JobMachineTableBase インスタンス
            fixed_gantt: 確定済みガントチャート (変更不可)
            reschedule_gantt: リスケ対象操作のガントチャート
            reschedule_time: リスケジューリング開始時刻
            weights: [効率性の重み, 安定性の重み]
        """
        self.jm_table = jm_table
        self.fixed_gantt = fixed_gantt
        self.reschedule_time = reschedule_time
        self.weights = weights

        num_jobs = jm_table.get_job_count()
        num_machines = jm_table.get_machine_count()

        # 各ジョブの確定済み工程数
        self.fixed_op_count = [0] * num_jobs
        for tasks in fixed_gantt:
            for task in tasks:
                self.fixed_op_count[task[2]] += 1

        # 各ジョブの最早開始時刻 (reschedule_time or fixed_ganttの最終終了時刻)
        self.job_earliest = [reschedule_time] * num_jobs
        for tasks in fixed_gantt:
            for task in tasks:
                job_id = task[2]
                self.job_earliest[job_id] = max(self.job_earliest[job_id], task[1])

        # 各機械の最早開始時刻 (fixed_ganttの最終終了時刻)
        self.machine_earliest = [0] * num_machines
        for m_idx, tasks in enumerate(fixed_gantt):
            for task in tasks:
                self.machine_earliest[m_idx] = max(self.machine_earliest[m_idx], task[1])

        # fixed_ganttのメイクスパン
        self.fixed_makespan = 0
        for tasks in fixed_gantt:
            for task in tasks:
                self.fixed_makespan = max(self.fixed_makespan, task[1])

        # 初期解の machine_orders を抽出
        self.initial_machine_orders = self._gantt_to_machine_orders(reschedule_gantt)

        # 正規化パラメータ (estimate_normalization_params で設定)
        self.max_eff = 1
        self.min_eff = 0
        self.max_stab = 1

    # ========== 表現変換 ==========

    def _gantt_to_machine_orders(self, reschedule_gantt):
        """ガントチャートからリスケ対象の machine_orders を抽出
        ※ 各ジョブが各機械を1回だけ訪問する標準JSSPを前提
        """
        machine_orders = {}
        for m_idx, tasks in enumerate(reschedule_gantt):
            if not tasks:
                continue
            sorted_tasks = sorted(tasks, key=lambda t: t[0])
            ops = []
            for task in sorted_tasks:
                job_id = task[2]
                # このジョブのこの機械での工程番号を特定
                op_idx = self.jm_table.m_table[job_id].index(m_idx)
                ops.append((job_id, op_idx))
            machine_orders[m_idx] = ops
        return machine_orders

    def _copy_orders(self, machine_orders):
        return {m: list(ops) for m, ops in machine_orders.items()}

    # ========== ガント構築 (前方パス, semi-active schedule) ==========

    def build_gantt(self, machine_orders):
        """machine_ordersからsemi-active scheduleを構築 (Kahn's algorithm)
        Returns: op_times {(job_id, op_idx): (start, end, machine)} or None (閉路)
        """
        job_end = list(self.job_earliest)
        machine_end = list(self.machine_earliest)

        # 全操作の依存関係を構築
        all_ops = []
        in_degree = {}
        machine_succ = {}  # op -> 同一機械の次操作
        job_succ = {}      # op -> 同一ジョブの次工程

        for m_idx, ops in machine_orders.items():
            for pos, op in enumerate(ops):
                all_ops.append(op)
                deg = 0
                # 機械内先行制約
                if pos > 0:
                    deg += 1
                    machine_succ[ops[pos - 1]] = op
                # ジョブ内先行制約 (前工程がリスケ対象の場合のみ)
                job_id, op_idx = op
                if op_idx > 0 and op_idx - 1 >= self.fixed_op_count[job_id]:
                    prev_op = (job_id, op_idx - 1)
                    deg += 1
                    job_succ[prev_op] = op
                in_degree[op] = deg

        total_ops = len(all_ops)

        # トポロジカルソート
        queue = deque(op for op in all_ops if in_degree[op] == 0)
        op_times = {}
        scheduled = 0

        while queue:
            op = queue.popleft()
            job_id, op_idx = op
            m_idx = self.jm_table.m_table[job_id][op_idx]
            pt = self.jm_table.pt_table[job_id][op_idx]

            start = max(job_end[job_id], machine_end[m_idx])
            end = start + pt

            op_times[op] = (start, end, m_idx)
            job_end[job_id] = end
            machine_end[m_idx] = end
            scheduled += 1

            for succ_dict in (machine_succ, job_succ):
                if op in succ_dict:
                    succ = succ_dict[op]
                    in_degree[succ] -= 1
                    if in_degree[succ] == 0:
                        queue.append(succ)

        if scheduled < total_ops:
            return None  # 閉路検出

        return op_times

    def get_makespan(self, op_times):
        if not op_times:
            return self.fixed_makespan
        rescheduled_ms = max(end for _, end, _ in op_times.values())
        return max(self.fixed_makespan, rescheduled_ms)

    def to_gantt_chart(self, op_times):
        """op_times + fixed_gantt → 2Dガントチャート (可視化用)"""
        num_machines = self.jm_table.get_machine_count()
        gantt = [list(tasks) for tasks in self.fixed_gantt]
        while len(gantt) < num_machines:
            gantt.append([])
        for (job_id, op_idx), (start, end, m_idx) in op_times.items():
            gantt[m_idx].append([start, end, job_id])
        for m in gantt:
            m.sort(key=lambda x: x[0])
        return gantt

    # ========== 評価関数 ==========

    def compute_stability(self, machine_orders):
        """安定性: 初期解との順位偏差 (ガント構築不要)"""
        total = 0.0
        for m_idx in self.initial_machine_orders:
            if m_idx not in machine_orders:
                continue
            init_jobs = [op[0] for op in self.initial_machine_orders[m_idx]]
            current_jobs = [op[0] for op in machine_orders[m_idx]]
            for init_pos, job_id in enumerate(init_jobs):
                current_pos = current_jobs.index(job_id)
                rank_diff = init_pos - current_pos
                total += abs(rank_diff) / (current_pos + 1) ** 1.25
        return total

    def evaluate(self, machine_orders, op_times=None):
        """正規化した重み付き評価値"""
        if op_times is None:
            op_times = self.build_gantt(machine_orders)
        if op_times is None:
            return float('inf')

        makespan = self.get_makespan(op_times)
        stability = self.compute_stability(machine_orders)

        if self.max_eff == self.min_eff:
            norm_eff = 1.0
        else:
            norm_eff = 1 + (makespan - self.min_eff) / (self.max_eff - self.min_eff)

        norm_stab = 1 + stability / self.max_stab if self.max_stab > 0 else 1.0

        return self.weights[0] * norm_eff + self.weights[1] * norm_stab

    def evaluate_pareto(self, machine_orders):
        """(メイクスパン, 安定性) を返す"""
        op_times = self.build_gantt(machine_orders)
        if op_times is None:
            return float('inf'), float('inf')
        return self.get_makespan(op_times), self.compute_stability(machine_orders)

    # ========== クリティカルパス ==========

    def find_critical_path(self, op_times, machine_orders):
        """クリティカルパス上のリスケ対象操作を特定 (終端から逆引き)"""
        makespan = 0
        last_op = None
        for op, (start, end, m) in op_times.items():
            if end > makespan:
                makespan = end
                last_op = op

        if last_op is None:
            return set()

        # 機械内先行操作のルックアップ
        machine_pred = {}
        for m_idx, ops in machine_orders.items():
            for pos in range(1, len(ops)):
                machine_pred[ops[pos]] = ops[pos - 1]

        critical_path = set()
        stack = [last_op]
        visited = {last_op}

        while stack:
            op = stack.pop()
            critical_path.add(op)
            job_id, op_idx = op
            start = op_times[op][0]

            # ジョブ内先行 (リスケ対象のもののみ)
            if op_idx > 0:
                prev = (job_id, op_idx - 1)
                if prev in op_times and prev not in visited:
                    if op_times[prev][1] == start:
                        visited.add(prev)
                        stack.append(prev)

            # 機械内先行 (リスケ対象のもののみ)
            if op in machine_pred:
                prev = machine_pred[op]
                if prev not in visited and prev in op_times:
                    if op_times[prev][1] == start:
                        visited.add(prev)
                        stack.append(prev)

        return critical_path

    def find_critical_blocks(self, critical_path, machine_orders):
        """クリティカルブロック: 同一機械上で連続するクリティカル操作の列"""
        blocks = []
        for m_idx, ops in machine_orders.items():
            current_block = []
            for op in ops:
                if op in critical_path:
                    current_block.append(op)
                else:
                    if len(current_block) >= 2:
                        blocks.append((m_idx, current_block))
                    current_block = []
            if len(current_block) >= 2:
                blocks.append((m_idx, current_block))
        return blocks

    # ========== N5近傍 ==========

    def generate_n5_neighbors(self, machine_orders, op_times=None):
        """N5近傍を生成
        ブロック境界の交換は閉路を作らない (Nowicki & Smutnicki, 1996)
        """
        if op_times is None:
            op_times = self.build_gantt(machine_orders)
            if op_times is None:
                return []

        critical_path = self.find_critical_path(op_times, machine_orders)
        blocks = self.find_critical_blocks(critical_path, machine_orders)

        neighbors = []
        for m_idx, block in blocks:
            # 先頭2つの交換
            new_orders = self._copy_orders(machine_orders)
            ops = new_orders[m_idx]
            idx_a = ops.index(block[0])
            idx_b = ops.index(block[1])
            ops[idx_a], ops[idx_b] = ops[idx_b], ops[idx_a]
            neighbors.append(new_orders)

            # 末尾2つの交換 (ブロックサイズ > 2 の場合のみ)
            if len(block) > 2:
                new_orders = self._copy_orders(machine_orders)
                ops = new_orders[m_idx]
                idx_a = ops.index(block[-2])
                idx_b = ops.index(block[-1])
                ops[idx_a], ops[idx_b] = ops[idx_b], ops[idx_a]
                neighbors.append(new_orders)

        return neighbors

    # ========== 局所探索 ==========

    def local_search(self, machine_orders, strategy='best'):
        """N5近傍による山登り法
        strategy: 'best' = 最良改善, 'first' = 最初改善
        """
        current = self._copy_orders(machine_orders)
        current_score = self.evaluate(current)
        steps = 0

        while True:
            op_times = self.build_gantt(current)
            if op_times is None:
                break

            neighbors = self.generate_n5_neighbors(current, op_times)
            if not neighbors:
                break

            if strategy == 'best':
                best_neighbor = None
                best_score = current_score
                for neighbor in neighbors:
                    score = self.evaluate(neighbor)
                    if score < best_score:
                        best_score = score
                        best_neighbor = neighbor
                if best_neighbor is None:
                    break
                current = best_neighbor
                current_score = best_score

            elif strategy == 'first':
                random.shuffle(neighbors)
                improved = False
                for neighbor in neighbors:
                    score = self.evaluate(neighbor)
                    if score < current_score:
                        current = neighbor
                        current_score = score
                        improved = True
                        break
                if not improved:
                    break

            steps += 1

        return current, current_score, steps

    # ========== 摂動 ==========

    def perturb(self, machine_orders, method='swap', strength=1):
        """摂動: 局所最適解から脱出するためのキック操作
        method: 'swap' = N5近傍のスワップをstrength回連続適用
                'insert' = 操作を抜き取り別の位置に挿入
                'path_relink' = 初期順序への部分的復元
        strength: 操作の回数 (大きいほど強い摂動)
        """
        for _ in range(20):  # 実行可能解が見つかるまでリトライ
            new_orders = self._copy_orders(machine_orders)

            if method == 'swap':
                # N5近傍のスワップをstrength回連続適用
                # 各ステップでクリティカルパスを再計算し、N5近傍からランダムに1つ選ぶ
                for _ in range(strength):
                    op_times = self.build_gantt(new_orders)
                    if op_times is None:
                        break
                    neighbors = self.generate_n5_neighbors(new_orders, op_times)
                    if not neighbors:
                        break
                    new_orders = random.choice(neighbors)

            elif method == 'insert':
                # 操作を抜き取り別の位置に挿入
                for _ in range(strength):
                    machines = [m for m in new_orders if len(new_orders[m]) >= 2]
                    if not machines:
                        break
                    m = random.choice(machines)
                    ops = new_orders[m]
                    i = random.randrange(len(ops))
                    op = ops.pop(i)
                    j = random.randrange(len(ops) + 1)
                    ops.insert(j, op)

            elif method == 'path_relink':
                # 初期順序への部分的復元
                for _ in range(strength):
                    machines = list(new_orders.keys())
                    m = random.choice(machines)
                    current_ops = new_orders[m]
                    initial_ops = self.initial_machine_orders[m]
                    diffs = [i for i in range(len(current_ops))
                             if current_ops[i] != initial_ops[i]]
                    if not diffs:
                        continue
                    pos = random.choice(diffs)
                    target_op = initial_ops[pos]
                    current_pos = current_ops.index(target_op)
                    current_ops.pop(current_pos)
                    current_ops.insert(pos, target_op)

            if self.build_gantt(new_orders) is not None:
                return new_orders

        return machine_orders  # フォールバック

    # ========== 正規化パラメータ推定 ==========

    def estimate_normalization_params(self, n_samples=100):
        """正規化用パラメータを推定

        Phase 1: ランダム摂動 + 局所探索でサンプルを収集
        Phase 2: 効率性は局所探索後の解の分布から推定（外れ値を除外）
                 安定性は全サンプルの最大値を使用
        """
        eff_samples = []
        stab_samples = []

        # 初期解を含める
        op_times = self.build_gantt(self.initial_machine_orders)
        if op_times is not None:
            eff_samples.append(self.get_makespan(op_times))
            stab_samples.append(0.0)

        # Phase 1: ランダム摂動でサンプル生成
        saved_weights = list(self.weights)
        count = 0
        for _ in range(n_samples * 10):
            if count >= n_samples:
                break
            sample = self._copy_orders(self.initial_machine_orders)
            for _ in range(random.randint(1, 5)):
                machines = [m for m in sample if len(sample[m]) >= 2]
                if not machines:
                    break
                m = random.choice(machines)
                ops = sample[m]
                i = random.randrange(len(ops))
                op = ops.pop(i)
                j = random.randrange(len(ops) + 1)
                ops.insert(j, op)

            op_times = self.build_gantt(sample)
            if op_times is None:
                continue

            ms = self.get_makespan(op_times)
            st = self.compute_stability(sample)
            eff_samples.append(ms)
            stab_samples.append(st)
            count += 1

        # Phase 2: 摂動+局所探索でmin_effをより正確に推定
        self.weights = [1.0, 0.0]
        ls_makespans = []
        current_best = self._copy_orders(self.initial_machine_orders)
        for i in range(30):
            # swap摂動（N5連続適用）で多様な出発点を生成
            strength = 2 + (i // 5)
            sample = self.perturb(current_best, 'swap', strength=min(strength, 8))
            ls_result, _, _ = self.local_search(sample, 'best')
            op_times = self.build_gantt(ls_result)
            if op_times is not None:
                ms = self.get_makespan(op_times)
                ls_makespans.append(ms)
                if ms < (min(ls_makespans) if len(ls_makespans) > 1 else float('inf')):
                    current_best = self._copy_orders(ls_result)
        self.weights = saved_weights

        # 効率性の範囲: min=局所探索の最良値, max=ランダムサンプルの90パーセンタイル
        eff_samples.sort()
        min_eff = min(ls_makespans) if ls_makespans else eff_samples[0]
        p90_idx = int(len(eff_samples) * 0.9)
        max_eff = eff_samples[p90_idx] if p90_idx < len(eff_samples) else eff_samples[-1]
        max_eff = max(max_eff, min_eff + 1)  # ゼロ除算防止

        # 安定性の範囲
        max_stab = max(stab_samples) if stab_samples else 1.0

        self.max_eff = max_eff
        self.min_eff = min_eff
        self.max_stab = max(max_stab, 1e-6)

        print(f"正規化パラメータ推定完了 (サンプル数: {count}, LS探索: {len(ls_makespans)})")
        print(f"  効率性 (Makespan): [{min_eff}, {max_eff}] (LS min={min(ls_makespans) if ls_makespans else 'N/A'})")
        print(f"  安定性 (Max): {max_stab:.2f}")

    # ========== ILSメインループ ==========

    def run(self, max_iterations=50, strategy='best', perturb_method='swap',
            initial_strength=2, max_strength=5):
        """
        ILSメインループ
        1. 摂動 (bestから) → 2. 局所探索 → 3. 最良解更新 → 1に戻る
        常にbest解から摂動を行う標準的なILS受理判定
        """
        start_time = time.time()
        current = self._copy_orders(self.initial_machine_orders)

        # 初期局所探索
        best, best_score, ls_steps = self.local_search(current, strategy)

        ms, st = self.evaluate_pareto(best)
        print(f"\n初期局所探索完了 ({ls_steps}ステップ): "
              f"Makespan={ms}, Stability={st:.2f}, Score={best_score:.4f}")

        strength = initial_strength
        no_improve_count = 0
        total_ls_steps = ls_steps

        for i in range(max_iterations):
            # 常にbest解から摂動
            perturbed = self.perturb(best, perturb_method, strength)

            # 局所探索
            ls_result, ls_score, ls_steps = self.local_search(perturbed, strategy)
            total_ls_steps += ls_steps

            # 最良解の更新
            if ls_score < best_score:
                best = self._copy_orders(ls_result)
                best_score = ls_score
                ms, st = self.evaluate_pareto(best)
                print(f"  Iter {i+1}: 改善! Makespan={ms}, Stability={st:.2f}, "
                      f"Score={best_score:.4f} (strength={strength})")
                strength = initial_strength
                no_improve_count = 0
            else:
                no_improve_count += 1
                # 改善が続かなければ摂動を強化 (3回で増加)
                if no_improve_count % 3 == 0:
                    strength = min(strength + 1, max_strength)

        elapsed = time.time() - start_time
        print(f"\nILS完了: {max_iterations}反復, {total_ls_steps}局所探索ステップ, "
              f"{elapsed:.2f}秒")

        return best, best_score


# ==========================================
# メインブロック
# ==========================================

if __name__ == "__main__":
    jsp_name = "MT10_10"
    weights = [0.5, 0.5]

    jm_table = job_shop_scheduling.get_jm_table(jsp_name)
    init_gantt = jm_table.initial_gantt()
    delayed_gantt = jm_table.delayed_gantt()

    fixed_gantt, reschedule_gantt, reschedule_time, msg = \
        gantt_chart_operation.check_disturbance(init_gantt, delayed_gantt)

    if reschedule_time == 0:
        print("リスケジューリング不要です")
        sys.exit()

    print(f"問題: {jsp_name}")
    print(msg)
    print(f"リスケジュール時刻: {reschedule_time}")
    print(f"重みベクトル: 効率性={weights[0]}, 安定性={weights[1]}")

    # ソルバー作成
    solver = ILSSolver(jm_table, fixed_gantt, reschedule_gantt, reschedule_time, weights)

    # 初期解の評価
    init_ms, init_st = solver.evaluate_pareto(solver.initial_machine_orders)
    print(f"\n初期解: Makespan={init_ms}, Stability={init_st:.2f}")

    # 正規化パラメータ推定
    solver.estimate_normalization_params(n_samples=100)

    # ILS実行
    best_orders, best_score = solver.run(
        max_iterations=50,
        strategy='best',
        perturb_method='swap',
    )

    # 結果表示
    makespan, stability = solver.evaluate_pareto(best_orders)
    print(f"\n===== 最終結果 =====")
    print(f"Makespan:  {init_ms} → {makespan} (改善: {init_ms - makespan})")
    print(f"Stability: {stability:.4f}")
    print(f"Weighted Score: {best_score:.4f}")

    # ガントチャート表示
    op_times = solver.build_gantt(best_orders)
    gantt = solver.to_gantt_chart(op_times)
    analysis.plot_gantt(
        gantt, jm_table.get_job_count(), jm_table.get_machine_count(),
        "ILS Result", reschedule_time
    )
