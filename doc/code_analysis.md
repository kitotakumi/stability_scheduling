# ILS スケジューリング コード解析・検証・改善提案書

> **このドキュメントの役割**: `ils_scheduling.py`のコードを処理順に詳細解説し、研究論文との整合性を検証し、コードレビュー結果と改善提案をまとめたもの。実装の正しさの根拠資料として使用する。

**対象ファイル**: `ils_scheduling.py`
**作成日**: 2026-03-25
**最終更新**: 2026-03-31

---

## 目次
1. [処理フロー全体像](#1-処理フロー全体像)
2. [コードレベルの詳細解説](#2-コードレベルの詳細解説)
3. [研究論文との整合性検証](#3-研究論文との整合性検証)
4. [コードレビュー（バグ・改善点）](#4-コードレビューバグ改善点)
5. [実験結果の分析と改善提案](#5-実験結果の分析と改善提案)

---

## 1. 処理フロー全体像

```
メインブロック実行フロー:

1. 問題データ読み込み (job_shop_scheduling.get_jm_table)
2. 初期ガント・遅延ガント取得
3. 外乱検知 (check_disturbance) → fixed_gantt / reschedule_gantt / reschedule_time
4. ILSSolver 初期化
   4.1 fixed_op_count 計算（各ジョブの確定済み工程数）
   4.2 job_earliest 計算（各ジョブの最早開始時刻）
   4.3 machine_earliest 計算（各機械の最早開始時刻）
   4.4 reschedule_gantt → initial_machine_orders 変換
5. 正規化パラメータ推定 (estimate_normalization_params)
6. ILSメインループ (run)
   6.1 初期局所探索
   6.2 反復: 摂動 → 局所探索 → 最良解更新
7. 結果出力・可視化
```

---

## 2. コードレベルの詳細解説

### 2.1 `__init__` (L23-72): コンストラクタ

#### 入力パラメータ
- `jm_table`: `JobMachineTableBase` インスタンス。`m_table[job][op] = machine_id`、`pt_table[job][op] = processing_time`
- `fixed_gantt`: 確定済みガントチャート。`fixed_gantt[machine] = [[start, end, job_id], ...]`
- `reschedule_gantt`: リスケ対象ガントチャート（同形式）
- `reschedule_time`: リスケ開始時刻（int）
- `weights`: `[効率性の重み, 安定性の重み]`

#### 処理詳細

**L41-44: `fixed_op_count` の計算**
```python
self.fixed_op_count = [0] * num_jobs
for tasks in fixed_gantt:
    for task in tasks:
        self.fixed_op_count[task[2]] += 1
```
- 各ジョブについて確定済みの工程数をカウント
- `fixed_gantt`は機械ごとにタスクを持つため、全機械を走査して各ジョブのタスク数を合計
- **用途**: `build_gantt`でジョブ内先行制約を構築する際、前工程が確定済み（リスケ対象外）かどうかの判定に使用
- **前提**: 確定済み工程はジョブの先頭から連続（操作0, 1, ..., k-1）。スケジュールの時間的性質から保証される

**L47-51: `job_earliest` の計算**
```python
self.job_earliest = [reschedule_time] * num_jobs
for tasks in fixed_gantt:
    for task in tasks:
        job_id = task[2]
        self.job_earliest[job_id] = max(self.job_earliest[job_id], task[1])
```
- 初期値は `reschedule_time`（リスケ開始時刻以前には開始不可）
- 確定済みタスクの終了時刻 `task[1]` と比較し、最大値を採用
- **意味**: 各ジョブの最初のリスケ対象工程は、`max(reschedule_time, 最後の確定済み工程の終了時刻)` 以降にしか開始できない

**L54-57: `machine_earliest` の計算**
```python
self.machine_earliest = [0] * num_machines
for m_idx, tasks in enumerate(fixed_gantt):
    for task in tasks:
        self.machine_earliest[m_idx] = max(self.machine_earliest[m_idx], task[1])
```
- 各機械の最早利用可能時刻 = その機械上の最後の確定済みタスクの終了時刻
- **注意**: `reschedule_time`とは独立。機械によってはリスケ開始時刻より前に空いている場合もある（job_earliestが制約となる）

**L60-63: `fixed_makespan` の計算**
```python
self.fixed_makespan = 0
for tasks in fixed_gantt:
    for task in tasks:
        self.fixed_makespan = max(self.fixed_makespan, task[1])
```
- 確定済みガントの最大終了時刻。リスケ後の全体メイクスパンはこれ以上になる

**L66: 初期解のmachine_orders抽出**
```python
self.initial_machine_orders = self._gantt_to_machine_orders(reschedule_gantt)
```

---

### 2.2 `_gantt_to_machine_orders` (L75-91): ガント → 機械順序変換

```python
def _gantt_to_machine_orders(self, reschedule_gantt):
    machine_orders = {}
    for m_idx, tasks in enumerate(reschedule_gantt):
        if not tasks:
            continue
        sorted_tasks = sorted(tasks, key=lambda t: t[0])  # 開始時刻でソート
        ops = []
        for task in sorted_tasks:
            job_id = task[2]
            op_idx = self.jm_table.m_table[job_id].index(m_idx)  # ★
            ops.append((job_id, op_idx))
        machine_orders[m_idx] = ops
    return machine_orders
```

#### 処理の流れ
1. 各機械について、リスケ対象タスクを開始時刻順にソート
2. 各タスクから `(job_id, op_idx)` タプルを生成
3. `op_idx` は `m_table[job_id].index(m_idx)` で特定（このジョブがこの機械を使う工程番号）
4. **前提**: 各ジョブは各機械を1回だけ訪問する標準JSSP

#### 出力形式
```python
{
    0: [(6, 5), (5, 3), (2, 7)],   # Machine 0: Job6のOp5, Job5のOp3, Job2のOp7の順
    1: [(6, 0), (2, 0), (7, 4)],   # Machine 1: ...
    ...
}
```

---

### 2.3 `build_gantt` (L98-158): ガント構築（トポロジカルソート）

**目的**: `machine_orders` から semi-active schedule を構築する

#### アルゴリズム（Kahn's algorithm によるトポロジカルソート）

**ステップ1: 依存関係グラフの構築 (L106-125)**
```python
for m_idx, ops in machine_orders.items():
    for pos, op in enumerate(ops):
        all_ops.append(op)
        deg = 0
        # 機械内先行制約: 同じ機械の前の操作が完了しないと開始不可
        if pos > 0:
            deg += 1
            machine_succ[ops[pos - 1]] = op
        # ジョブ内先行制約: 同じジョブの前工程が完了しないと開始不可
        # ただし前工程がfixed（確定済み）なら制約不要（job_earliestで吸収）
        job_id, op_idx = op
        if op_idx > 0 and op_idx - 1 >= self.fixed_op_count[job_id]:
            prev_op = (job_id, op_idx - 1)
            deg += 1
            job_succ[prev_op] = op
        in_degree[op] = deg
```

各操作の入次数（依存する先行操作の数）を計算:
- **機械内先行**: 同じ機械の直前の操作 → +1
- **ジョブ内先行**: 同じジョブの直前の工程（リスケ対象の場合のみ）→ +1
- 直前工程がfixed（`op_idx - 1 < self.fixed_op_count[job_id]`）の場合は辺を作らない（`job_earliest`で暗黙的に制約）

**ステップ2: トポロジカルソートによるスケジューリング (L130-153)**
```python
queue = deque(op for op in all_ops if in_degree[op] == 0)
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
    # 後続操作の入次数を減らし、0になったらキューに追加
    for succ_dict in (machine_succ, job_succ):
        if op in succ_dict:
            succ = succ_dict[op]
            in_degree[succ] -= 1
            if in_degree[succ] == 0:
                queue.append(succ)
```

- 入次数0の操作（全先行操作が完了済み）をキューから取り出し
- `start = max(ジョブの最後の終了時刻, 機械の最後の終了時刻)` でsemi-active
- 後続操作の入次数を更新、0になればキューに追加
- **閉路検出**: `scheduled < total_ops` なら閉路あり（実行不可能解）

**Semi-active schedule の意味**: 各操作を可能な限り早く開始するが、隙間への挿入（左詰め）は行わない。GAの`get_gantt_reactive`（active schedule）との違い。

---

### 2.4 `compute_stability` (L180-192): 安定性評価

```python
def compute_stability(self, machine_orders):
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
```

#### 計算式
$$\text{Stability} = \sum_{m} \sum_{j \in m} \frac{|r_{\text{init}}(j,m) - r_{\text{current}}(j,m)|}{(r_{\text{current}}(j,m) + 1)^{1.25}}$$

- $r_{\text{init}}(j,m)$: 初期解での機械$m$上のジョブ$j$の投入順位（0-indexed）
- $r_{\text{current}}(j,m)$: 現在解での同順位
- **分母の意味**: 投入順序の早いジョブ（小さい`current_pos`）の順序変更を重くペナルティ
- **ガント構築不要**: machine_ordersから直接計算可能で高速

---

### 2.5 `evaluate` (L194-211): 重み付き評価関数

```python
def evaluate(self, machine_orders, op_times=None):
    if op_times is None:
        op_times = self.build_gantt(machine_orders)
    if op_times is None:
        return float('inf')

    makespan = self.get_makespan(op_times)
    stability = self.compute_stability(machine_orders)

    # min-max正規化で [1, 2] の範囲にマッピング
    if self.max_eff == self.min_eff:
        norm_eff = 1.0
    else:
        norm_eff = 1 + (makespan - self.min_eff) / (self.max_eff - self.min_eff)
    norm_stab = 1 + stability / self.max_stab if self.max_stab > 0 else 1.0

    return self.weights[0] * norm_eff + self.weights[1] * norm_stab
```

#### 正規化の仕組み
- 効率性: `norm_eff ∈ [1, 2]`。min_effのとき1、max_effのとき2
- 安定性: `norm_stab ∈ [1, 1 + stability/max_stab]`。0のとき1
- 両指標を同じスケール [1, 2] に揃えてから重み付き和を取る
- GAの`weight_function`（`pareto_operation.py`）と**同一の正規化方式**

---

### 2.6 `find_critical_path` (L222-266): クリティカルパス探索

```python
def find_critical_path(self, op_times, machine_orders):
    # 1. メイクスパンを定義する最終操作を特定
    makespan = 0
    last_op = None
    for op, (start, end, m) in op_times.items():
        if end > makespan:
            makespan = end
            last_op = op

    # 2. 逆方向探索: 先行操作の終了時刻 == 当該操作の開始時刻なら「タイト」
    machine_pred = {}  # 機械内先行操作のルックアップテーブル
    for m_idx, ops in machine_orders.items():
        for pos in range(1, len(ops)):
            machine_pred[ops[pos]] = ops[pos - 1]

    critical_path = set()
    stack = [last_op]
    while stack:
        op = stack.pop()
        critical_path.add(op)
        job_id, op_idx = op
        start = op_times[op][0]
        # ジョブ内先行（終了時刻==開始時刻ならクリティカル）
        if op_idx > 0:
            prev = (job_id, op_idx - 1)
            if prev in op_times and op_times[prev][1] == start:
                stack.append(prev)
        # 機械内先行（同条件）
        if op in machine_pred:
            prev = machine_pred[op]
            if prev in op_times and op_times[prev][1] == start:
                stack.append(prev)
```

#### アルゴリズム
- メイクスパンを決定する最終操作から逆方向にたどる
- 「先行操作の終了時刻 == 当該操作の開始時刻」（= 余裕時間0、タイトな辺）を再帰的にたどる
- 結果はクリティカルパス上の操作の集合

---

### 2.7 `find_critical_blocks` (L268-282): クリティカルブロック抽出

```python
def find_critical_blocks(self, critical_path, machine_orders):
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
```

- 同一機械上でクリティカルパスに**連続して**含まれる操作の列をブロック化
- 1操作のみのブロックは除外（スワップ対象がないため）
- **N5近傍の定義域**: 各ブロックの先頭2つ・末尾2つのスワップ

---

### 2.8 `generate_n5_neighbors` (L286-317): N5近傍生成

```python
def generate_n5_neighbors(self, machine_orders, op_times=None):
    # クリティカルパス → クリティカルブロック → 近傍生成
    critical_path = self.find_critical_path(op_times, machine_orders)
    blocks = self.find_critical_blocks(critical_path, machine_orders)

    neighbors = []
    for m_idx, block in blocks:
        # 先頭2つの交換（必ず生成）
        new_orders = self._copy_orders(machine_orders)
        ops = new_orders[m_idx]
        idx_a = ops.index(block[0])
        idx_b = ops.index(block[1])
        ops[idx_a], ops[idx_b] = ops[idx_b], ops[idx_a]
        neighbors.append(new_orders)

        # 末尾2つの交換（ブロックサイズ > 2 の場合のみ）
        if len(block) > 2:
            new_orders = self._copy_orders(machine_orders)
            ops = new_orders[m_idx]
            idx_a = ops.index(block[-2])
            idx_b = ops.index(block[-1])
            ops[idx_a], ops[idx_b] = ops[idx_b], ops[idx_a]
            neighbors.append(new_orders)
    return neighbors
```

#### Nowicki & Smutnicki (1996) のN5近傍
- **先頭スワップ**: ブロックの最初の操作と2番目の操作を交換
- **末尾スワップ**: ブロックの最後から2番目と最後の操作を交換
- **閉路チェック不要**: ブロック境界のスワップはディスジャンクティブグラフに閉路を作らないことが数学的に証明されている
- **近傍サイズ**: 最大 $2m$ 個（$m$ = 機械数。各機械に1ブロックとして）

---

### 2.9 `local_search` (L321-366): 局所探索（山登り法）

```python
def local_search(self, machine_orders, strategy='best'):
    current = self._copy_orders(machine_orders)
    current_score = self.evaluate(current)
    while True:
        op_times = self.build_gantt(current)
        neighbors = self.generate_n5_neighbors(current, op_times)
        if not neighbors:
            break
        if strategy == 'best':
            # 全近傍を評価し、最良のものに移動
            best_neighbor = None
            best_score = current_score
            for neighbor in neighbors:
                score = self.evaluate(neighbor)
                if score < best_score:
                    best_score = score
                    best_neighbor = neighbor
            if best_neighbor is None:
                break  # 改善なし → 局所最適
            current = best_neighbor
            current_score = best_score
        elif strategy == 'first':
            # 近傍をランダム順に評価し、最初に改善したものに移動
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
    return current, current_score, steps
```

- **最良改善 (best)**: 全N5近傍を評価し、最も改善幅の大きいものに移動
- **最初改善 (first)**: ランダム順に近傍を評価し、最初に改善したものに移動
- 改善がなくなるまで反復 → 局所最適解を返す

---

### 2.10 `perturb` (L370-422): 摂動

3つの摂動手法を実装。いずれも実行可能解になるまで最大20回リトライ。

#### swap (L378-387)
```python
# strength回、ランダムな機械のランダムな隣接操作を交換
m = random.choice(machines)
i = random.randrange(len(new_orders[m]) - 1)
new_orders[m][i], new_orders[m][i + 1] = new_orders[m][i + 1], new_orders[m][i]
```

#### insert (L389-400)
```python
# strength回、ランダムな機械のランダムな操作を抜き取り、ランダムな位置に挿入
m = random.choice(machines)
ops = new_orders[m]
i = random.randrange(len(ops))
op = ops.pop(i)
j = random.randrange(len(ops) + 1)
ops.insert(j, op)
```

#### path_relink (L402-417)
```python
# strength回、初期解の順序と異なる位置を1つ選び、初期解と同じ位置に強制的に戻す
pos = random.choice(diffs)  # 差異がある位置
target_op = initial_ops[pos]
current_pos = current_ops.index(target_op)
current_ops.pop(current_pos)
current_ops.insert(pos, target_op)
```
- 初期スケジュールの構造を部分的に復元する「Path Relinking」
- 安定性の改善に特化した摂動

---

### 2.11 `estimate_normalization_params` (L426-471): 正規化パラメータ推定

```python
def estimate_normalization_params(self, n_samples=100):
    # 初期解を含める
    op_times = self.build_gantt(self.initial_machine_orders)
    ms = self.get_makespan(op_times)
    max_eff, min_eff = ms, ms

    # n_samples個のランダム摂動解でサンプリング
    for _ in range(n_samples * 10):
        if count >= n_samples:
            break
        sample = self._copy_orders(self.initial_machine_orders)
        # 1〜5回のランダムinsert摂動
        for _ in range(random.randint(1, 5)):
            # ... ランダムinsert ...
        op_times = self.build_gantt(sample)
        if op_times is None:
            continue  # 実行不可能解はスキップ
        ms = self.get_makespan(op_times)
        st = self.compute_stability(sample)
        max_eff = max(max_eff, ms)
        min_eff = min(min_eff, ms)
        max_stab = max(max_stab, st)
    self.max_eff = max_eff
    self.min_eff = min_eff
    self.max_stab = max(max_stab, 1e-6)
```

- ランダムなinsert摂動でサンプルを生成し、メイクスパンと安定性の範囲を推定
- GAの`get_max_min`と同様の考え方（ランダム集団でサンプリング）
- **問題点**: 後述（セクション4.2）

---

### 2.12 `run` (L475-529): ILSメインループ

```python
def run(self, max_iterations=50, strategy='best', perturb_method='insert',
        initial_strength=2, max_strength=5):
    current = self._copy_orders(self.initial_machine_orders)

    # ステップ1: 初期局所探索
    best, best_score, ls_steps = self.local_search(current, strategy)
    current = self._copy_orders(best)

    strength = initial_strength
    no_improve_count = 0

    for i in range(max_iterations):
        # ステップ2: 摂動（改善停滞時に手法をローテーション）
        method = perturb_method if no_improve_count < 3 else \
                 perturb_methods[no_improve_count % len(perturb_methods)]
        perturbed = self.perturb(current, method, strength)

        # ステップ3: 局所探索
        ls_result, ls_score, ls_steps = self.local_search(perturbed, strategy)

        # ステップ4: 最良解更新 + 受理判定
        if ls_score < best_score:
            best = self._copy_orders(ls_result)
            best_score = ls_score
            current = self._copy_orders(ls_result)  # ★改善時: bestを起点に
            strength = initial_strength
            no_improve_count = 0
        else:
            current = self._copy_orders(ls_result)  # ★非改善時: 新解を起点に
            no_improve_count += 1
            if no_improve_count % 3 == 0:
                strength = min(strength + 1, max_strength)

    return best, best_score
```

#### 受理判定（Acceptance Criterion）
- **改善時**: best更新、currentをbest解に設定、強度リセット
- **非改善時**: currentを新しい局所最適に設定（常に受理）
- **摂動強度制御**: 3回連続非改善で強度+1、摂動手法もローテーション

---

## 3. 研究論文との整合性検証

### 3.1 解表現 ✅ 正しい
- **論文**: 機械ごとの作業順序（Machine Order）
- **実装**: `machine_orders = {m_idx: [(job_id, op_idx), ...], ...}`
- GT法の遺伝子を使わない理由（設計文書記載）も妥当

### 3.2 局所探索（N5近傍）✅ 正しい
- **論文**: クリティカルブロックの先頭2つ/末尾2つを交換。閉路チェック不要（Nowicki & Smutnicki, 1996）
- **実装**: `generate_n5_neighbors`が正確に実装。ブロックサイズ>2のときのみ末尾スワップ生成
- **移動規則**: 最良改善/最初改善の選択可能 → 論文の「比較実験した上で決定」に対応

### 3.3 デコード方式 ✅ 正しい（ただし注意点あり）
- **論文**: 「Taillardのアルゴリズム（or ガントチャート再生成）」
- **実装**: 前方パス（semi-active schedule）でのガント構築（Phase 1）
- **設計文書**: Phase 2（Taillardの高速化）は「性能ボトルネック確認後に導入」とあり、未実装は妥当
- **GAとの差異**: GAは`get_gantt_reactive`（active schedule、左詰め挿入あり）を使用。ILSはsemi-active schedule。N5近傍の理論的保証はsemi-active前提なので**理論的に正しい選択**

### 3.4 安定性関数 ✅ 正しい
- **論文**: 「ジョブの投入順序の変更量」「投入順序の早いジョブの順序変更量の抑制」
- **実装**: $\sum |r_{\text{init}} - r_{\text{current}}| / (r_{\text{current}} + 1)^{1.25}$
- **GA版との比較**: `pareto_operation.py`の`stability_function_v3`と同一の計算式。ただしGA版はガントチャート経由で比較、ILS版はmachine_ordersから直接計算（高速版）

### 3.5 目的関数・正規化 ✅ 正しい
- **論文**: 重みパラメータ法、min-max正規化
- **実装**: `evaluate`関数がGA版の`weight_function`と同一の `[1, 2]` 範囲正規化を採用

### 3.6 摂動 ⚠️ 一部不一致
- **論文/設計文書**: 「クリティカルパス上のk個のジョブを交換」（critical_swap）
- **実装**: `perturb`の`swap`メソッドは**ランダムな**隣接操作の交換であり、クリティカルパス上に限定していない
- これは設計変更と思われるが、クリティカルパス上に限定した摂動も有効な選択肢

---

## 4. コードレビュー（バグ・改善点）

### 4.1 ✅ 修正済み — 受理判定（Acceptance Criterion）の問題

**修正内容**: 常にbest解から摂動する標準的なILS受理判定に変更（2026-03-31）

```python
# 常にbest解から摂動
perturbed = self.perturb(best, perturb_method, strength)
```

### 4.2 🔴 重要度: 高 — 正規化パラメータ推定の偏り

**箇所**: `estimate_normalization_params` メソッド

**問題**:
1. ランダムinsertのみでサンプリング → 到達可能な解空間の一部しかカバーしない
2. 摂動回数が1〜5と小さいため、min_effが過大評価（真の最適makespanより大きい値になる）
3. 結果として効率性指標の正規化が不適切になり、改善余地が過小評価される

**修正案**:
```python
def estimate_normalization_params(self, n_samples=100):
    max_eff, min_eff = float('-inf'), float('inf')
    max_stab = 0.0

    # 初期解を含める
    op_times = self.build_gantt(self.initial_machine_orders)
    if op_times is not None:
        ms = self.get_makespan(op_times)
        max_eff, min_eff = ms, ms

    count = 0
    for _ in range(n_samples * 10):
        if count >= n_samples:
            break
        sample = self._copy_orders(self.initial_machine_orders)
        # 異なる摂動手法を混合使用
        method = random.choice(['insert', 'swap'])
        n_perturbations = random.randint(1, 8)  # より広い範囲
        for _ in range(n_perturbations):
            machines = [m for m in sample if len(sample[m]) >= 2]
            if not machines:
                break
            m = random.choice(machines)
            ops = sample[m]
            if method == 'insert':
                i = random.randrange(len(ops))
                op = ops.pop(i)
                j = random.randrange(len(ops) + 1)
                ops.insert(j, op)
            else:
                i = random.randrange(len(ops) - 1)
                ops[i], ops[i + 1] = ops[i + 1], ops[i]

        op_times = self.build_gantt(sample)
        if op_times is None:
            continue
        ms = self.get_makespan(op_times)
        st = self.compute_stability(sample)
        max_eff = max(max_eff, ms)
        min_eff = min(min_eff, ms)
        max_stab = max(max_stab, st)
        count += 1

    # 局所探索でmin_effをより正確に推定
    for _ in range(5):
        sample = self._copy_orders(self.initial_machine_orders)
        for _ in range(random.randint(3, 8)):
            machines = [m for m in sample if len(sample[m]) >= 2]
            if not machines:
                break
            m = random.choice(machines)
            ops = sample[m]
            i = random.randrange(len(ops))
            op = ops.pop(i)
            j = random.randrange(len(ops) + 1)
            ops.insert(j, op)
        # 効率性のみで局所探索
        old_weights = self.weights
        self.weights = [1.0, 0.0]
        ls_result, _, _ = self.local_search(sample)
        self.weights = old_weights
        op_times = self.build_gantt(ls_result)
        if op_times is not None:
            ms = self.get_makespan(op_times)
            min_eff = min(min_eff, ms)

    self.max_eff = max_eff
    self.min_eff = min_eff
    self.max_stab = max(max_stab, 1e-6)
```

### 4.3 🟡 重要度: 中 — `evaluate`でのbuild_gantt二重呼び出し

**箇所**: `local_search` → `evaluate` → `build_gantt`

**問題**: `local_search`では`build_gantt`を明示的に呼び出してop_timesを取得し（L330）、N5近傍生成に使用。しかし各近傍の`evaluate`呼び出し時に再度`build_gantt`が呼ばれる（L196-197）。

```python
# local_search内
op_times = self.build_gantt(current)          # 1回目
neighbors = self.generate_n5_neighbors(current, op_times)
for neighbor in neighbors:
    score = self.evaluate(neighbor)            # evaluate内でbuild_ganttが再度呼ばれる
```

**修正案**: `evaluate`にop_timesを渡す仕組みは既にあるので、局所探索内で活用:
```python
# 近傍の場合はbuild_ganttは避けられないが、currentの評価は使い回せる
# （現状のコードでcurrentの再評価は不要なのでこれは近傍の話）
```
実はこの問題は近傍に対してのものなので、各近傍ごとにbuild_ganttが必要。ただし**Taillardのアルゴリズム（Phase 2）の導入でO(n)に削減可能**。

### 4.4 🟡 重要度: 中 — `find_critical_path`のvisitedチェック漏れ

**箇所**: L254, L263

**問題**: 現在のコードは`visited`チェックを正しく行っているが、ジョブ先行と機械先行の両方から同じ操作に到達した場合にstackに2回追加される可能性がある。

```python
# ジョブ内先行
if prev in op_times and prev not in visited:
    if op_times[prev][1] == start:
        visited.add(prev)
        stack.append(prev)
# 機械内先行
if op in machine_pred:
    prev = machine_pred[op]
    if prev not in visited and prev in op_times:
        if op_times[prev][1] == start:
            visited.add(prev)
            stack.append(prev)
```

実際には`visited`チェックがあるので二重追加はない。しかし、ジョブ先行のprevと機械先行のprevが**異なる変数スコープ**に書かれているため、ジョブ先行で`prev`を設定した後、機械先行の条件で同じ`prev`変数が使われる可能性がある。

→ 実際にはL258で`prev`を再代入しているので安全だが、変数名を変えた方が明確:

```python
# ジョブ内先行
if op_idx > 0:
    job_prev = (job_id, op_idx - 1)
    if job_prev in op_times and job_prev not in visited:
        if op_times[job_prev][1] == start:
            visited.add(job_prev)
            stack.append(job_prev)
# 機械内先行
if op in machine_pred:
    mach_prev = machine_pred[op]
    if mach_prev not in visited and mach_prev in op_times:
        if op_times[mach_prev][1] == start:
            visited.add(mach_prev)
            stack.append(mach_prev)
```

### 4.5 ✅ 修正済み — `perturb`のswapがクリティカルパスを無視

**修正内容**: swap摂動をN5近傍のstrength回連続適用に変更（2026-03-31）。各ステップでクリティカルパスを再計算し、N5近傍からランダムに1つ選んで適用する。

```python
if method == 'swap':
    for _ in range(strength):
        op_times = self.build_gantt(new_orders)
        if op_times is None:
            break
        neighbors = self.generate_n5_neighbors(new_orders, op_times)
        if not neighbors:
            break
        new_orders = random.choice(neighbors)
```

### 4.6 🟢 重要度: 低 — `_copy_orders`の不要なコピー

**箇所**: `run`メソッド内の複数箇所

```python
best, best_score, ls_steps = self.local_search(current, strategy)
current = self._copy_orders(best)  # local_searchが新しいオブジェクトを返すので不要
```

`local_search`は内部で`self._copy_orders(machine_orders)`を呼んで新しいオブジェクトを作成しているため、返り値に対する追加コピーは冗長。ただし安全性のためあえて残す選択もあり。

---

## 5. 実験結果の分析と改善提案

### 5.1 なぜ初期解を改善できないのか

実験ログからの事実:
1. 初期解: Makespan=1080, Stability=0.00
2. **全N5近傍でメイクスパンが悪化**（+30〜+103）
3. 摂動→LSで1080に戻るか、より悪い局所最適（1110, 1219）に到達

#### 根本原因の分析

**原因1: 外乱が小規模すぎる**
- 遅延量+60（全体のメイクスパン1080の約5.6%）
- 元のスケジュール構造がほぼ最適なまま維持されている
- これは研究仮説1「小規模外乱では元のスケジュール構造が優秀」の確認であり、**ILSの正当性を証明する結果**ではある

**原因2: N5近傍の限界**
- N5は「クリティカルブロック境界のスワップ」に限定
- 初期解が既にN5局所最適 → 局所探索が1ステップも進まない
- 摂動で解を壊しても、LSが元に戻すか、より悪い局所最適に収束

**原因3: 安定性ペナルティの影響**
- 初期解のStability=0（完璧）→ **どんな変更もペナルティ増加**
- weights=[0.5, 0.5]の場合、Makespanを改善しても安定性ペナルティで相殺される

**原因4: 正規化の不十分さ**
- min_effが過大評価（真の最適より大きい値）→ 効率性改善の評価値が圧縮される

### 5.2 改善提案

#### 提案1: より大規模な外乱での検証（最優先）

```python
# delayed_gantt_v2 等、異なる外乱シナリオで検証
jsp_name = "MT10_10"
jm_table = job_shop_scheduling.get_jm_table(jsp_name)
init_gantt = jm_table.initial_gantt()

# 複数の外乱シナリオ
for scenario in ['delayed_gantt', 'delayed_gantt_v1', 'delayed_gantt_v2']:
    delayed_gantt = getattr(jm_table, scenario)()
    fixed_gantt, reschedule_gantt, reschedule_time, msg = \
        gantt_chart_operation.check_disturbance(init_gantt, delayed_gantt)
    if reschedule_time == 0:
        continue
    solver = ILSSolver(jm_table, fixed_gantt, reschedule_gantt, reschedule_time, weights)
    solver.estimate_normalization_params()
    best_orders, best_score = solver.run(max_iterations=100)
    # ... 結果比較
```

#### 提案2: 拡張近傍の導入

N5近傍が狭すぎて初期解から脱出できない場合、より広い近傍を追加:

```python
def generate_extended_neighbors(self, machine_orders, op_times=None):
    """N5 + 挿入近傍（クリティカルパス上の操作を別位置に挿入）"""
    # まずN5近傍
    neighbors = self.generate_n5_neighbors(machine_orders, op_times)

    if op_times is None:
        op_times = self.build_gantt(machine_orders)
        if op_times is None:
            return neighbors

    critical_path = self.find_critical_path(op_times, machine_orders)

    # クリティカルパス上の操作に対する挿入近傍
    for op in critical_path:
        job_id, op_idx = op
        m_idx = self.jm_table.m_table[job_id][op_idx]
        if m_idx not in machine_orders:
            continue
        ops = machine_orders[m_idx]
        current_pos = ops.index(op)

        # 1つ前/1つ後ろへの挿入（N5とは異なる動き）
        for new_pos in [current_pos - 2, current_pos + 2]:
            if 0 <= new_pos <= len(ops) - 1 and new_pos != current_pos:
                new_orders = self._copy_orders(machine_orders)
                new_ops = new_orders[m_idx]
                moved = new_ops.pop(current_pos)
                new_ops.insert(new_pos, moved)
                if self.build_gantt(new_orders) is not None:
                    neighbors.append(new_orders)

    return neighbors
```

#### 提案3: 受理判定の改善（実装コスト低、効果大）

```python
def run(self, max_iterations=50, strategy='best', perturb_method='insert',
        initial_strength=2, max_strength=5, restart_threshold=5):
    # ... 前略 ...

    for i in range(max_iterations):
        method = perturb_method if no_improve_count < 3 else \
                 perturb_methods[no_improve_count % len(perturb_methods)]
        perturbed = self.perturb(current, method, strength)
        ls_result, ls_score, ls_steps = self.local_search(perturbed, strategy)
        total_ls_steps += ls_steps

        if ls_score < best_score:
            best = self._copy_orders(ls_result)
            best_score = ls_score
            current = self._copy_orders(ls_result)
            strength = initial_strength
            no_improve_count = 0
        else:
            no_improve_count += 1
            if no_improve_count % 3 == 0:
                strength = min(strength + 1, max_strength)

            # ★改善: 一定回数改善なしならbestから再スタート
            if no_improve_count >= restart_threshold:
                current = self._copy_orders(best)
                strength = min(initial_strength + 1, max_strength)
                no_improve_count = 0
            else:
                current = self._copy_orders(ls_result)

    return best, best_score
```

#### 提案4: 多重重みベクトルによるパレート近似

GA版のように複数重みで探索し、パレートフロントを近似:

```python
def run_pareto(self, weight_vectors, max_iterations_per_weight=20, **kwargs):
    """複数重みベクトルでILSを実行し、パレートフロントを近似"""
    pareto_front = []  # [(makespan, stability, machine_orders), ...]

    for w in weight_vectors:
        self.weights = w
        best_orders, best_score = self.run(
            max_iterations=max_iterations_per_weight, **kwargs
        )
        ms, st = self.evaluate_pareto(best_orders)
        pareto_front.append((ms, st, self._copy_orders(best_orders)))

    # 非支配ソートでパレートフロントを抽出
    non_dominated = []
    for i, (ms_i, st_i, _) in enumerate(pareto_front):
        dominated = False
        for j, (ms_j, st_j, _) in enumerate(pareto_front):
            if i != j and ms_j <= ms_i and st_j <= st_i and (ms_j < ms_i or st_j < st_i):
                dominated = True
                break
        if not dominated:
            non_dominated.append(pareto_front[i])

    return non_dominated
```

#### 提案5: 初期解の多様化（マルチスタート）

初期解が既にN5最適の場合、異なる初期解から出発:

```python
def run_multistart(self, n_starts=5, max_iterations_per_start=20, **kwargs):
    """複数の初期解からILSを実行し、最良を返す"""
    overall_best = None
    overall_best_score = float('inf')

    for s in range(n_starts):
        if s == 0:
            # 最初は元の初期解から
            start = self._copy_orders(self.initial_machine_orders)
        else:
            # 強めの摂動で異なる初期解を生成
            start = self.perturb(
                self.initial_machine_orders, 'insert',
                strength=random.randint(3, 6)
            )

        # 一時的にinitial_machine_ordersは変えない（安定性基準は同じ）
        solver_copy = self._copy_orders(start)
        best, best_score = self.run(
            max_iterations=max_iterations_per_start, **kwargs
        )

        if best_score < overall_best_score:
            overall_best = self._copy_orders(best)
            overall_best_score = best_score

    return overall_best, overall_best_score
```

#### 提案6: 実験設計の改善

現在の実験では「ILSが改善を見つけられない」というネガティブな結果しか得られていない。以下の追加実験を推奨:

| 実験 | 目的 | 期待される結果 |
|------|------|--------------|
| 大規模外乱 (+90, +120) | ILSの改善能力の検証 | N5最適でない初期解からの改善 |
| GAとの同一計算時間比較 | 仮説2の検証 | ILSの方が同じ時間でより良い解 |
| 複数問題インスタンス (MT6_6) | 汎化性の検証 | 小規模問題での安定した改善 |
| 効率性のみ (w=[1,0]) | N5近傍の効果検証 | メイクスパン改善の上界把握 |
| 反復回数増加 (500, 1000) | 収束性の検証 | 何反復で確実に収束するか |
| `first` vs `best` 戦略比較 | 移動規則の選択 | 多様性vs貪欲のトレードオフ |

### 5.3 実装優先度まとめ

| 優先度 | 項目 | 理由 |
|--------|------|------|
| ★★★ | 大規模外乱での検証 | 現在の問題設定では改善余地がほぼない。まず問題を変える |
| ★★★ | 受理判定の改善（提案3） | 実装コスト最小で効果大。bestへの回帰で探索効率向上 |
| ★★☆ | クリティカルパス指向の摂動（4.5） | 摂動の質を上げることで有効な近傍への到達確率向上 |
| ★★☆ | 正規化パラメータ推定の改善（4.2） | 目的関数のバランスが取れていないと最適解を見逃す |
| ★☆☆ | 拡張近傍（提案2） | N5が狭すぎる場合の保険。実装コスト中 |
| ★☆☆ | パレート近似（提案4） | GA版との公平な比較のために必要 |
