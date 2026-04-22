# ILS スケジューリング 実装説明書

> **このドキュメントの役割**: `ils_scheduling.py` を中心としたコードが何をしているかを説明する。アルゴリズムの根拠・実装上の判断理由もあわせて記録する。

---

## 1. 研究概要

ジョブショップスケジューリング問題（JSSP）の再スケジューリングを対象に、**効率性（メイクスパン）と安定性（ジョブ投入順序の変更量）の2目的**を重み付き合成スコアで最適化する反復局所探索法（ILS）を実装する。

**核心的な仮説**: GAの交叉は良質な初期スケジュールの構造を破壊する。ILSは「局所探索（深掘り）」と「摂動（脱出）」が分離されており、安定性を制御しつつ効率的に探索できる。

---

## 2. ファイル構成

| ファイル | 役割 |
|---|---|
| `ils_scheduling.py` | ILSソルバー本体 |
| `evaluation.py` | GA/ILS共通: 安定性関数・正規化・重み付き目的関数 |
| `ga_scheduling.py` | GAソルバー（比較手法） |
| `job_shop_scheduling.py` | 問題データ（m_table, pt_table, gantt） |
| `gantt_chart_operation.py` | 外乱検知・デコード |

---

## 3. 解表現: Machine Orders

ILSは **機械ごとの作業順序（machine_orders）** を解表現として使う。GAの遺伝子列（GT法）を使わない理由は、N5近傍が機械ごとの順序を直接操作するため machine_orders 表現が自然で、遺伝子との相互変換はコストが高くバグの温床になるため。

```python
machine_orders = {
    0: [(job_id, op_idx), (job_id, op_idx), ...],  # Machine 0 の処理順序
    1: [(job_id, op_idx), ...],                      # Machine 1 の処理順序
    ...
}
```

各エントリは `(ジョブID, 工程番号)` のタプル。リスケジュール対象の操作のみ含む（確定済み操作は `fixed_gantt` で別管理）。

---

## 4. ILSアルゴリズム全体フロー

```
1. 初期解の生成
   - delayed_gantt → check_disturbance → fixed_gantt, reschedule_gantt
   - reschedule_gantt → machine_orders に変換 (_gantt_to_machine_orders)

2. 正規化パラメータの推定 (estimate_normalization_params)
   - ランダムなmachine_ordersをサンプリングして max_eff, min_eff, max_stab を推定

3. ILSメインループ (max_iterations回)
   3.1 摂動 (Perturbation)
       - current から出発して摂動を加え新たな出発点を生成
       - 摂動の強さは段階的に制御（non_improve_count % 3 == 0 で +1）

   3.2 局所探索 (Local Search)
       - N5近傍を列挙 → best/first improvement で移動
       - 改善がなくなるまで繰り返す → 局所最適解

   3.3 受理判定 (Acceptance)
       - ls_score < best_score なら best と current を両方更新
       - stagnation_threshold 指定時: 停滞が続くと current のみ悪化方向に移動可能

4. 結果出力
   - best_machine_orders → ガント構築 → メイクスパン・安定性を返す
```

---

## 5. ガント構築: `build_gantt`

**目的**: machine_orders → semi-active schedule を構築し、各操作の `(start, end, machine)` を返す。

**アルゴリズム**: Kahnのアルゴリズム（トポロジカルソート）

```
各操作の入次数（依存する先行操作の数）を計算:
  - 機械内先行: 同じ機械の直前の操作 → +1
  - ジョブ内先行: 同じジョブの直前の工程（リスケ対象のみ） → +1
    ※ 直前工程が fixed なら辺を作らず job_earliest で暗黙的に制約

入次数0の操作をキューに入れ、順に処理:
  start = max(job_end[job_id], machine_end[m_idx])
  end   = start + processing_time

後続操作の入次数を減らし、0になったらキューに追加。
スケジュール済み数 < 総操作数 なら閉路（実行不可能）→ None を返す
```

**Semi-active の意味**: 各操作を可能な限り早く開始するが、隙間への挿入（左詰め）は行わない。N5近傍との組み合わせが理論的に整合する（Nowicki & Smutnicki 1996 はこの一致を前提とする）。

### 初期値の設定（`__init__`）

| 変数 | 意味 | 計算方法 |
|---|---|---|
| `job_earliest[j]` | ジョブ j の最早開始時刻 | `max(reschedule_time, 確定済み最終タスクの終了時刻)` |
| `machine_earliest[m]` | 機械 m の最早利用可能時刻 | 確定済みタスクの最後の終了時刻 |
| `fixed_makespan` | 確定済みガントのメイクスパン | 全確定タスクの最大終了時刻 |
| `fixed_op_count[j]` | ジョブ j の確定済み工程数 | fixed_gantt を走査してカウント |

---

## 6. 評価関数

### メイクスパン（効率性）

```python
makespan = max(end for _, end, _ in op_times.values())
makespan = max(makespan, self.fixed_makespan)
```

### 安定性

```python
# evaluation.compute_stability_from_orders を呼ぶ
Stability = Σ_m Σ_j |rank_init(j,m) - rank_current(j,m)| / (rank_current(j,m) + 1)^1.25
```

- `rank_init`: 初期解での機械 m 上のジョブ j の投入順位（0-indexed）
- `rank_current`: 現在解での同順位
- 分母: 先頭に近いジョブの順序変更を重くペナルティ（β=1.25）
- **ガント構築不要**: machine_orders から直接計算できるため高速

### 重み付き目的関数（`evaluation.weighted_objective`）

```python
norm_eff  = 1 + (makespan  - min_eff)  / (max_eff  - min_eff)  # [1, 2] にマッピング
norm_stab = 1 + stability / max_stab                             # [1, 1+α] にマッピング
score = weights[0] * norm_eff + weights[1] * norm_stab           # 小さいほど良い
```

正規化パラメータ（min_eff, max_eff, max_stab）はランダムサンプリングで推定する（`estimate_normalization_params`）。GA・ILSで共通の推定値を使うことで公平な比較を保証する。

---

## 7. クリティカルパスと N5 近傍

### クリティカルパスの探索（`find_critical_path`）

メイクスパンを決定する最終操作から逆方向にたどる。  
「先行操作の終了時刻 == 当該操作の開始時刻」（余裕時間ゼロ）を満たす辺のみを追跡。  
fixed_gantt との境界: `prev not in op_times` なら追わない（変更不可なので正しい動作）。

### クリティカルブロックの抽出（`find_critical_blocks`）

同一機械上でクリティカルパスに連続して含まれる操作の列をブロックとして抽出。  
ブロックサイズ ≥ 2 のものだけ近傍生成の対象にする。

### N5 近傍（`generate_n5_neighbors`）

各クリティカルブロックの **先頭2つの操作を交換** と **末尾2つの操作を交換**（ブロックサイズ > 2 の場合）を生成する。

**閉路チェック不要の根拠**: Nowicki & Smutnicki (1996) により、ブロック境界の操作の交換はディスジャンクティブグラフに閉路を作らないことが証明されている。

**計算量**: クリティカルブロック数は高々 m 個（機械数）、各ブロックから高々2近傍 → 1ステップあたり最大 2m 個。

---

## 8. Taillardスクリーニング

### 目的

N5近傍の全候補に対して `build_gantt`（重い）を呼ぶ前に、有望でない候補を高速に除外する。

### head と tail の事前計算（`_compute_heads_and_tails`）

| 値 | 意味 |
|---|---|
| `head[op]` | その操作の最早開始時刻 |
| `tail_job[op]` | ジョブ後続パスの最長残り時間 |
| `tail_machine[op]` | 機械後続パスの最長残り時間 |
| `tail[op]` | `max(tail_job, tail_machine)` |

### スワップ後のMS推定（`_taillard_estimate_swap`）

隣接する u→v を v→u に交換したとき、ガントを再構築せずに MS の **下界** を推定する。

```
r_v_new = max(r_job_v, r_machine_pred_u)        # スワップ後の v の最早開始
r_u_new = max(r_job_u, r_v_new + p_v)           # スワップ後の u の最早開始

est = max(
    r_v_new + p_v + tail_job[v],      # v → ジョブ後続
    r_u_new + p_u + tail_job[u],      # u → ジョブ後続
    r_u_new + p_u + tail_machine[v],  # u → 元のvの機械後続
)
```

u, v 以外への波及効果を無視するため `est ≤ 実際のMS` が保証される（下界）。

### 合成スコア下界フィルタ（`_score_lower_bound`）

```python
def _score_lower_bound(self, est_ms, machine_orders):
    stability = self.compute_stability(machine_orders)  # 正確な値（ガント不要）
    return weighted_objective(est_ms, stability, weights, norm_params)
```

- `est_ms ≤ actual_ms` かつ stability は正確な値
- → `score_lb ≤ actual_score` が保証される

**フィルタ基準**: `score_lb <= current_score` の近傍のみフル評価する。  
`weights[1] == 0` のとき `est_ms <= current_ms`（旧動作）と等価。

**変更理由**: 旧実装では `est_ms <= current_ms`（MSのみ）でフィルタしていたため、「MSは若干悪化するが安定性改善で総合スコアが改善する近傍」が除外されていた。安定性の重み > 0 の場合に実質無効化されていた問題を修正（2026-04-14）。

---

## 9. 局所探索（`local_search`）

N5近傍による山登り法。改善がなくなるまでループ。

```
while True:
    1. build_gantt で op_times を構築
    2. Taillard スクリーニングで candidates を絞り込み
    3. candidates が空なら全近傍にフォールバック
    4. best/first improvement で最良近傍を選択
    5. 改善があれば current を更新、なければ break
```

**strategy='best'（デフォルト）**: 全近傍を評価して最良を選ぶ。  
**strategy='first'**: 改善する近傍を最初に見つけた時点で移動（未実験、大規模問題で有利な可能性）。

---

## 10. 摂動（`perturb`）

局所最適から脱出するためのキック操作。

| 手法 | 動作 |
|---|---|
| `swap`（デフォルト） | N5スワップを strength 回連続適用。各ステップでクリティカルパスを再計算してN5からランダムに1つ選ぶ |
| `insert` | 操作を抜き取り、同一機械の別位置に挿入。リトライガードあり |

**strength の制御**: `no_improve_count % 3 == 0` で +1、上限 `max_strength=5`（デフォルト）。

---

## 11. 受理判定（best/current 分離）

`best`（全体最良解）と `current`（次の摂動の出発点）を分離する。

| 状態 | best | current |
|---|---|---|
| 改善時（`ls_score < best_score`） | 更新 | 更新 |
| 非改善・通常時 | 更新しない | 更新しない |
| 停滞時（`stagnation_threshold` 指定） | 更新しない | 悪化方向に移動可能 |

**`stagnation_threshold` のデフォルトは `None`（悪化受理なし）**。実験（2026-04-14）で「悪化受理はMS改善とSt悪化のトレードオフを生み、重み付きスコアでは大半の問題でデフォルトが有利」と確認。
