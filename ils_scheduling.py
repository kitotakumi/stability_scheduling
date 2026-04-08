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
import evaluation
import analysis


class ILSSolver:
    """反復局所探索法によるジョブショップ再スケジューリングソルバー"""

    def __init__(self, jm_table, fixed_gantt, reschedule_gantt, reschedule_time, weights,
                 active_schedule=False, taillard_acceleration=True):
        """
        Parameters:
            jm_table: JobMachineTableBase インスタンス
            fixed_gantt: 確定済みガントチャート (変更不可)
            reschedule_gantt: リスケ対象操作のガントチャート
            reschedule_time: リスケジューリング開始時刻
            weights: [効率性の重み, 安定性の重み]
            active_schedule: Trueの場合、GT法（左詰め挿入）でアクティブスケジュールを構築。
                             実験的機能。N5近傍（semi-active前提）との理論的整合性がないため、
                             基本は False (semi-active) を使用すること。
            taillard_acceleration: Trueの場合、N5近傍評価にTaillardの高速化を使用（推奨）。
                                   active_schedule=True のときは自動的に無効化される。
        """
        self.jm_table = jm_table
        self.fixed_gantt = fixed_gantt
        self.reschedule_time = reschedule_time
        self.weights = weights
        self.active_schedule = active_schedule
        self.taillard_acceleration = taillard_acceleration

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

        # Active方式用: 初期解をGT法デコードした実際の機械順序を保存
        # （安定性計算の基準として使用）
        if self.active_schedule:
            init_op_times = self._build_gantt_active(self.initial_machine_orders)
            if init_op_times is not None:
                self.initial_actual_orders = self._op_times_to_machine_orders(init_op_times)
            else:
                self.initial_actual_orders = self.initial_machine_orders
        else:
            self.initial_actual_orders = self.initial_machine_orders

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

    # ========== ガント構築 ==========

    def build_gantt(self, machine_orders):
        """machine_ordersからスケジュールを構築
        active_schedule=True の場合はGT法（左詰め挿入）でアクティブスケジュールを生成。
        Returns: op_times {(job_id, op_idx): (start, end, machine)} or None (閉路)
        """
        if self.active_schedule:
            return self._build_gantt_active(machine_orders)
        return self._build_gantt_semi_active(machine_orders)

    def _build_gantt_semi_active(self, machine_orders):
        """semi-active scheduleを構築 (Kahn's algorithm)"""
        job_end = list(self.job_earliest)
        machine_end = list(self.machine_earliest)

        all_ops = []
        in_degree = {}
        machine_succ = {}
        job_succ = {}

        for m_idx, ops in machine_orders.items():
            for pos, op in enumerate(ops):
                all_ops.append(op)
                deg = 0
                if pos > 0:
                    deg += 1
                    machine_succ[ops[pos - 1]] = op
                job_id, op_idx = op
                if op_idx > 0 and op_idx - 1 >= self.fixed_op_count[job_id]:
                    prev_op = (job_id, op_idx - 1)
                    deg += 1
                    job_succ[prev_op] = op
                in_degree[op] = deg

        total_ops = len(all_ops)

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
            return None

        return op_times

    def _build_gantt_active(self, machine_orders):
        """machine_ordersからactive scheduleを構築（GT法デコード）

        手順:
        1. machine_ordersをトポロジカルソートしてジョブ番号列を生成
        2. get_gantt_reactive（左詰め挿入）でアクティブスケジュールを構築
        3. 結果をop_times形式に変換
        """
        # --- Step 1: トポロジカルソートでジョブ列を生成 ---
        all_ops = []
        in_degree = {}
        machine_succ = {}
        job_succ = {}

        for m_idx, ops in machine_orders.items():
            for pos, op in enumerate(ops):
                all_ops.append(op)
                deg = 0
                if pos > 0:
                    deg += 1
                    machine_succ[ops[pos - 1]] = op
                job_id, op_idx = op
                if op_idx > 0 and op_idx - 1 >= self.fixed_op_count[job_id]:
                    prev_op = (job_id, op_idx - 1)
                    deg += 1
                    job_succ[prev_op] = op
                in_degree[op] = deg

        total_ops = len(all_ops)
        queue = deque(op for op in all_ops if in_degree[op] == 0)
        job_sequence = []
        scheduled = 0

        while queue:
            op = queue.popleft()
            job_id, op_idx = op
            job_sequence.append(job_id)
            scheduled += 1

            for succ_dict in (machine_succ, job_succ):
                if op in succ_dict:
                    succ = succ_dict[op]
                    in_degree[succ] -= 1
                    if in_degree[succ] == 0:
                        queue.append(succ)

        if scheduled < total_ops:
            return None  # 閉路検出

        # --- Step 2: GT法でアクティブスケジュール構築 ---
        gantt = gantt_chart_operation.get_gantt_reactive(
            self.jm_table, job_sequence, self.fixed_gantt, self.reschedule_time)

        # --- Step 3: ガントからop_times形式に変換 ---
        fixed_tasks = set()
        for m_idx, tasks in enumerate(self.fixed_gantt):
            for task in tasks:
                fixed_tasks.add((task[0], task[1], task[2], m_idx))

        op_times = {}
        for m_idx, tasks in enumerate(gantt):
            for task in tasks:
                if (task[0], task[1], task[2], m_idx) in fixed_tasks:
                    continue
                job_id = task[2]
                op_idx = self.jm_table.m_table[job_id].index(m_idx)
                op_times[(job_id, op_idx)] = (task[0], task[1], m_idx)

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

    def _op_times_to_machine_orders(self, op_times):
        """op_timesから実際の機械上の処理順序を復元する

        Active方式ではGT法の左詰め挿入でmachine_ordersと実際の順序が
        異なるため、安定性を正しく計算するにはこの変換が必要。
        """
        machine_ops = {}
        for (job_id, op_idx), (start, end, m_idx) in op_times.items():
            if m_idx not in machine_ops:
                machine_ops[m_idx] = []
            machine_ops[m_idx].append((start, (job_id, op_idx)))
        # 開始時刻順にソート
        actual_orders = {}
        for m_idx, ops in machine_ops.items():
            ops.sort(key=lambda x: x[0])
            actual_orders[m_idx] = [op for _, op in ops]
        return actual_orders

    def compute_stability(self, machine_orders, op_times=None):
        """安定性: 初期解との順位偏差

        active_schedule=True の場合はop_timesから実際の機械順序を復元して比較。
        （machine_ordersはGT法デコード前の順序であり、実際のスケジュールと異なるため）
        """
        if self.active_schedule and op_times is not None:
            actual_orders = self._op_times_to_machine_orders(op_times)
            return evaluation.compute_stability_from_orders(
                self.initial_actual_orders, actual_orders)
        return evaluation.compute_stability_from_orders(
            self.initial_machine_orders, machine_orders)

    def evaluate(self, machine_orders, op_times=None):
        """正規化した重み付き評価値（evaluation.pyの共通関数を使用）"""
        if op_times is None:
            op_times = self.build_gantt(machine_orders)
        if op_times is None:
            return float('inf')

        makespan = self.get_makespan(op_times)
        stability = self.compute_stability(machine_orders, op_times)
        norm_params = {
            'min_eff': self.min_eff,
            'max_eff': self.max_eff,
            'max_stab': self.max_stab,
        }
        return evaluation.weighted_objective(makespan, stability, self.weights, norm_params)

    def evaluate_pareto(self, machine_orders):
        """(メイクスパン, 安定性) を返す"""
        op_times = self.build_gantt(machine_orders)
        if op_times is None:
            return float('inf'), float('inf')
        return self.get_makespan(op_times), self.compute_stability(machine_orders, op_times)

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

    # ========== Taillard高速化 ==========

    def _compute_heads_and_tails(self, op_times, machine_orders):
        """各オペレーションのhead, tail_job, tail_machine を計算 (Taillard 1994)

        head[op]: 最早開始時刻 (= op_times[op].start)
        tail_job[op]: ジョブ後続のみを通る最長パス長
        tail_machine[op]: 機械後続のみを通る最長パス長

        スワップ時は後続の接続先が変わるため、ジョブ成分と機械成分を分離して保持する。
        """
        # head は op_times の start そのもの
        head = {}
        for op, (start, end, m) in op_times.items():
            head[op] = start

        # 先行・後続関係の構築
        machine_succ_map = {}
        for m_idx, ops in machine_orders.items():
            for pos in range(len(ops) - 1):
                machine_succ_map[ops[pos]] = ops[pos + 1]

        job_succ_map = {}
        for op in op_times:
            job_id, op_idx = op
            succ = (job_id, op_idx + 1)
            if succ in op_times:
                job_succ_map[op] = succ

        # tail の計算（逆トポロジカル順 = 終了時刻降順）
        tail_job = {}      # ジョブ後続のみ経由の最長パス
        tail_machine = {}  # 機械後続のみ経由の最長パス
        tail = {}          # 合計 (= max(tail_job, tail_machine))

        sorted_ops = sorted(op_times.keys(), key=lambda op: -op_times[op][1])

        for op in sorted_ops:
            tj = 0
            if op in job_succ_map:
                succ = job_succ_map[op]
                s_job, s_op = succ
                tj = tail[succ] + self.jm_table.pt_table[s_job][s_op]

            tm = 0
            if op in machine_succ_map:
                succ = machine_succ_map[op]
                if succ in op_times:
                    s_job, s_op = succ
                    tm = tail[succ] + self.jm_table.pt_table[s_job][s_op]

            tail_job[op] = tj
            tail_machine[op] = tm
            tail[op] = max(tj, tm)

        return head, tail, tail_job, tail_machine

    def _taillard_estimate_swap(self, head, tail_job, tail_machine,
                                 machine_orders, u, v, m_idx):
        """Taillardの高速化: 隣接するu,v (uが先) のスワップ後のメイクスパンを計算

        スワップ後の順序: ...pred, v, u, succ...
        接続変化:
          - v の機械後続: u (元は u→v だったのが v→u に)
          - u の機械後続: succ (元の v の機械後続)

        f(v,u) = max(
            r'(v) + p_v + q_J(v),      # v → ジョブ後続
            r'(u) + p_u + q_J(u),      # u → ジョブ後続
            r'(u) + p_u + q_M(v)       # u → 元のvの機械後続
        )
        """
        u_job, u_op = u
        v_job, v_op = v
        p_u = self.jm_table.pt_table[u_job][u_op]
        p_v = self.jm_table.pt_table[v_job][v_op]

        # uの機械上の先行者の完了時刻（= スワップ後のvの機械先行者）
        ops = machine_orders[m_idx]
        u_pos = ops.index(u)
        if u_pos > 0:
            pred = ops[u_pos - 1]
            pred_job, pred_op = pred
            r_machine_pred_u = head[pred] + self.jm_table.pt_table[pred_job][pred_op]
        else:
            r_machine_pred_u = self.machine_earliest[m_idx]

        # vのジョブ内先行者の完了時刻
        r_job_v = self.job_earliest[v_job]
        if v_op > 0:
            prev_job_op = (v_job, v_op - 1)
            if prev_job_op in head:
                r_job_v = head[prev_job_op] + self.jm_table.pt_table[v_job][v_op - 1]

        # uのジョブ内先行者の完了時刻
        r_job_u = self.job_earliest[u_job]
        if u_op > 0:
            prev_job_op = (u_job, u_op - 1)
            if prev_job_op in head:
                r_job_u = head[prev_job_op] + self.jm_table.pt_table[u_job][u_op - 1]

        # スワップ後のhead
        r_v_new = max(r_job_v, r_machine_pred_u)
        r_u_new = max(r_job_u, r_v_new + p_v)

        # Taillard公式: 3つのパスの最大値
        est = max(
            r_v_new + p_v + tail_job[v],      # v → ジョブ後続
            r_u_new + p_u + tail_job[u],       # u → ジョブ後続
            r_u_new + p_u + tail_machine[v],   # u → 元のvの機械後続
        )

        return max(est, self.fixed_makespan)

    def _generate_n5_with_taillard(self, machine_orders, op_times):
        """N5近傍を生成し、Taillard高速化でメイクスパンを推定して返す

        Returns: list of (neighbor_orders, estimated_makespan)
        """
        critical_path = self.find_critical_path(op_times, machine_orders)
        blocks = self.find_critical_blocks(critical_path, machine_orders)

        if not blocks:
            return []

        head, tail, tail_job, tail_machine = self._compute_heads_and_tails(
            op_times, machine_orders)

        results = []
        for m_idx, block in blocks:
            # 先頭2つの交換
            u, v = block[0], block[1]
            est_ms = self._taillard_estimate_swap(
                head, tail_job, tail_machine, machine_orders, u, v, m_idx)
            new_orders = self._copy_orders(machine_orders)
            ops = new_orders[m_idx]
            idx_a = ops.index(u)
            idx_b = ops.index(v)
            ops[idx_a], ops[idx_b] = ops[idx_b], ops[idx_a]
            results.append((new_orders, est_ms))

            # 末尾2つの交換
            if len(block) > 2:
                u, v = block[-2], block[-1]
                est_ms = self._taillard_estimate_swap(
                    head, tail_job, tail_machine, machine_orders, u, v, m_idx)
                new_orders = self._copy_orders(machine_orders)
                ops = new_orders[m_idx]
                idx_a = ops.index(u)
                idx_b = ops.index(v)
                ops[idx_a], ops[idx_b] = ops[idx_b], ops[idx_a]
                results.append((new_orders, est_ms))

        return results

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

        taillard_acceleration有効時:
          メイクスパンのみの評価(安定性の重みが0)の場合、Taillardの高速化で
          ガント再構築なしにメイクスパンを推定。安定性も使う場合は、Taillardで
          メイクスパンを推定しつつ安定性は直接計算（ガント不要）して評価。
        """
        current = self._copy_orders(machine_orders)
        current_score = self.evaluate(current)
        steps = 0
        eval_count = 1  # 初期解の評価

        use_taillard = self.taillard_acceleration and not self.active_schedule

        while True:
            op_times = self.build_gantt(current)
            if op_times is None:
                break

            if use_taillard:
                # Taillardスクリーニング: 推定値で有望な近傍を絞り込み、
                # 有望なもののみフル評価（build_gantt）で正確に評価する
                neighbors_with_ms = self._generate_n5_with_taillard(current, op_times)
                if not neighbors_with_ms:
                    break

                # 推定メイクスパンの下界でスクリーニング
                current_ms = self.get_makespan(op_times)
                candidates = [(n, est) for n, est in neighbors_with_ms
                              if est <= current_ms]

                if not candidates:
                    # 下界でも改善見込みなし → 全近傍を通常評価にフォールバック
                    neighbors = [n for n, _ in neighbors_with_ms]
                else:
                    neighbors = [n for n, _ in candidates]

                if strategy == 'best':
                    best_neighbor = None
                    best_score = current_score
                    for neighbor in neighbors:
                        score = self.evaluate(neighbor)
                        eval_count += 1
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
                        eval_count += 1
                        if score < current_score:
                            current = neighbor
                            current_score = score
                            improved = True
                            break
                    if not improved:
                        break
            else:
                neighbors = self.generate_n5_neighbors(current, op_times)
                if not neighbors:
                    break

                if strategy == 'best':
                    best_neighbor = None
                    best_score = current_score
                    for neighbor in neighbors:
                        score = self.evaluate(neighbor)
                        eval_count += 1
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
                        eval_count += 1
                        if score < current_score:
                            current = neighbor
                            current_score = score
                            improved = True
                            break
                    if not improved:
                        break

            steps += 1

        self._last_eval_count = eval_count
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
                # 初期順序への部分的復元（ランダム選択）
                # NOTE: greedy版（全候補を評価してスコア最良を選択）を試したが、
                # 摂動の多様性が失われ局所最適から脱出できなくなり効果がゼロになった。
                # 摂動はキック操作なのでランダム性が重要。(2026-04-04)
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

    def set_normalization_params(self, norm_params):
        """事前計算済みの正規化パラメータを設定"""
        self.min_eff = norm_params['min_eff']
        self.max_eff = norm_params['max_eff']
        self.max_stab = norm_params['max_stab']

    def estimate_normalization_params(self, n_samples=100, norm_params=None):
        """正規化用パラメータを推定

        norm_params が指定されればそれを使用（共通パラメータ）。
        指定されなければ従来方式で自前推定。
        """
        if norm_params is not None:
            self.set_normalization_params(norm_params)
            return

        # 従来方式: ランダム摂動でサンプル生成
        eff_samples = []
        stab_samples = []

        op_times = self.build_gantt(self.initial_machine_orders)
        if op_times is not None:
            eff_samples.append(self.get_makespan(op_times))
            stab_samples.append(0.0)

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

        self.weights = [1.0, 0.0]
        ls_makespans = []
        current_best = self._copy_orders(self.initial_machine_orders)
        for i in range(30):
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

        eff_samples.sort()
        min_eff = min(ls_makespans) if ls_makespans else eff_samples[0]
        p90_idx = int(len(eff_samples) * 0.9)
        max_eff = eff_samples[p90_idx] if p90_idx < len(eff_samples) else eff_samples[-1]
        max_eff = max(max_eff, min_eff + 1)

        max_stab = max(stab_samples) if stab_samples else 1.0

        self.max_eff = max_eff
        self.min_eff = min_eff
        self.max_stab = max(max_stab, 1e-6)

    # ========== ILSメインループ ==========

    def run(self, max_iterations=800, strategy='best', perturb_method='swap',
            initial_strength=2, max_strength=5, verbose=True,
            path_relink_mode=False, relink_trigger=50):
        """
        ILSメインループ

        Args:
            path_relink_mode: Trueの場合、主摂動で停滞したらpath_relinkに切り替える
            relink_trigger: path_relinkに切り替えるまでの無改善反復数

        Returns:
            best, best_score, convergence_info, history
            history: 反復ごとの記録リスト [{cpu_time, evaluations, iteration,
                     best_makespan, best_stability, best_score,
                     ls_makespan, ls_stability, ls_score, accepted, perturb_used}]
        """
        start_time = time.time()
        current = self._copy_orders(self.initial_machine_orders)
        eval_count = 0
        history = []

        # 初期局所探索
        best, best_score, ls_steps = self.local_search(current, strategy)
        eval_count = self._last_eval_count if hasattr(self, '_last_eval_count') else ls_steps

        ms, st = self.evaluate_pareto(best)
        best_ms, best_st = ms, st
        if verbose:
            print(f"\n初期局所探索完了 ({ls_steps}ステップ): "
                  f"Makespan={ms}, Stability={st:.2f}, Score={best_score:.4f}")

        best_iteration = 0
        best_cpu_time = time.time() - start_time
        best_eval_count = eval_count

        history.append({
            'cpu_time': time.time() - start_time,
            'evaluations': eval_count,
            'iteration': 0,
            'best_makespan': best_ms,
            'best_stability': best_st,
            'best_score': best_score,
            'ls_makespan': ms,
            'ls_stability': st,
            'ls_score': best_score,
            'accepted': True,
            'perturb_used': 'init',
        })

        strength = initial_strength
        no_improve_count = 0
        no_improve_since_switch = 0  # relink切替後の無改善カウント
        total_ls_steps = ls_steps
        using_relink = False  # path_relinkモード中かどうか
        relink_burst = 30  # relinkを試す反復数（これを超えたら主摂動に戻る）

        for i in range(max_iterations):
            # path_relink_mode: 停滞時にpath_relinkへ切り替え
            if path_relink_mode and not using_relink and no_improve_count >= relink_trigger:
                using_relink = True
                no_improve_since_switch = 0
                strength = initial_strength
                if verbose:
                    print(f"  Iter {i+1}: path_relinkモードに切替 "
                          f"({no_improve_count}反復停滞)")

            # relinkバーストが終了 → 主摂動に戻す
            if using_relink and no_improve_since_switch >= relink_burst:
                using_relink = False
                no_improve_count = 0  # リセットして主摂動を再試行
                strength = initial_strength
                if verbose:
                    print(f"  Iter {i+1}: path_relinkバースト終了、主摂動に復帰")

            current_method = 'path_relink' if using_relink else perturb_method
            perturbed = self.perturb(best, current_method, strength)

            ls_result, ls_score, ls_steps = self.local_search(perturbed, strategy)
            total_ls_steps += ls_steps
            eval_count += self._last_eval_count if hasattr(self, '_last_eval_count') else ls_steps

            ls_ms, ls_st = self.evaluate_pareto(ls_result)
            accepted = False

            if ls_score < best_score:
                best = self._copy_orders(ls_result)
                best_score = ls_score
                best_ms, best_st = ls_ms, ls_st
                best_iteration = i + 1
                best_cpu_time = time.time() - start_time
                best_eval_count = eval_count
                accepted = True
                if verbose:
                    print(f"  Iter {i+1}: 改善! Makespan={ls_ms}, Stability={ls_st:.2f}, "
                          f"Score={best_score:.4f} (method={current_method}, strength={strength})")
                strength = initial_strength
                no_improve_count = 0
                no_improve_since_switch = 0
                # path_relink中に改善 → 主摂動に戻る
                if using_relink:
                    using_relink = False
                    if verbose:
                        print(f"  Iter {i+1}: path_relink改善成功、主摂動に復帰")
            else:
                no_improve_count += 1
                if using_relink:
                    no_improve_since_switch += 1
                if no_improve_count % 3 == 0:
                    strength = min(strength + 1, max_strength)

            history.append({
                'cpu_time': time.time() - start_time,
                'evaluations': eval_count,
                'iteration': i + 1,
                'best_makespan': best_ms,
                'best_stability': best_st,
                'best_score': best_score,
                'ls_makespan': ls_ms,
                'ls_stability': ls_st,
                'ls_score': ls_score,
                'accepted': accepted,
                'perturb_used': current_method,
            })

        elapsed = time.time() - start_time
        if verbose:
            print(f"\nILS完了: {max_iterations}反復, {total_ls_steps}局所探索ステップ, "
                  f"{elapsed:.2f}秒")

        convergence_info = {
            'cpu_time': best_cpu_time,
            'evaluations': best_eval_count,
            'iteration': best_iteration,
            'total_cpu_time': elapsed,
            'total_evaluations': eval_count,
            'total_iterations': max_iterations,
        }
        if verbose:
            print(f"最良解到達: 反復={best_iteration}, 評価回数={best_eval_count}, "
                  f"CPU時間={best_cpu_time:.2f}秒")

        return best, best_score, convergence_info, history


# ==========================================
# メインブロック
# ==========================================

if __name__ == "__main__":
    problem_name = "mt10"
    scenario_name = "mt10_delay60"
    weights = [0.5, 0.5]

    jm_table = job_shop_scheduling.get_jm_table(problem_name, scenario_name)
    init_gantt = jm_table.initial_gantt()
    delayed_gantt = jm_table.delayed_gantt()

    fixed_gantt, reschedule_gantt, reschedule_time, msg = \
        gantt_chart_operation.check_disturbance(init_gantt, delayed_gantt)

    if reschedule_time == 0:
        print("リスケジューリング不要です")
        sys.exit()

    print(f"問題: {problem_name} / シナリオ: {scenario_name}")
    print(msg)
    print(f"リスケジュール時刻: {reschedule_time}")
    print(f"重みベクトル: 効率性={weights[0]}, 安定性={weights[1]}")

    # ソルバー作成 (semi-active + Taillard高速化)
    solver = ILSSolver(jm_table, fixed_gantt, reschedule_gantt, reschedule_time, weights,
                       active_schedule=False, taillard_acceleration=True)

    # 初期解の評価
    init_ms, init_st = solver.evaluate_pareto(solver.initial_machine_orders)
    print(f"\n初期解: Makespan={init_ms}, Stability={init_st:.2f}")

    # 正規化パラメータ推定
    solver.estimate_normalization_params(n_samples=100)

    # ILS実行
    best_orders, best_score, conv_info, history = solver.run(
        max_iterations=800,
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
