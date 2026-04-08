# ILS スケジューリング手法の解説

## 1. スケジュールの分類

ジョブショップスケジューリングにおける実行可能スケジュールの包含関係：

```
Non-delay ⊂ Active ⊂ Semi-active ⊂ Feasible
```

### 1.1 Semi-active schedule（セミアクティブスケジュール）

**定義**: 処理順序を変更せずにはどのオペレーションも左（より早い開始時刻）に移動できないスケジュール。

機械上の処理順序を固定した状態で、各操作を可能な限り早く配置したもの。本プロジェクトでは `_build_gantt_semi_active()` (Kahnのアルゴリズムによるトポロジカルソート) で構築。

**構築アルゴリズム**:
1. `machine_orders` から各オペレーションの入次数（先行制約数）を計算
2. 入次数0のオペレーションをキューに追加
3. キューから取り出し、`start = max(ジョブ前工程終了, 機械前タスク終了)` で配置
4. 後続の入次数を減らし、0になったらキューに追加

**特徴**:
- 機械上の順序を厳密に保持
- N5近傍との組み合わせが理論的に整合（Nowicki & Smutnicki 1996 の証明はこの一致を前提とする）

### 1.2 Active schedule（アクティブスケジュール）

**定義**: 他のオペレーションの開始時刻を遅延させずにはどのオペレーションも左に移動できないスケジュール。

機械の空き時間（アイドルタイム）にフィットするオペレーションがあれば積極的に挿入する。**最適解は必ずアクティブスケジュールの集合に含まれる** (Giffler & Thompson, 1960)。

**本プロジェクトでの実装 (`_build_gantt_active`)**:

`machine_orders` → トポロジカルソート → ジョブ列 → `get_gantt_reactive`（左詰め挿入） → `op_times` の3段階で構築。

```
machine_orders
  → トポロジカルソート（ジョブ先行 + 機械先行を両方制約）→ job_sequence
  → get_gantt_reactive（左詰め挿入）→ gantt
  → op_times
```

**重要な特性**:
- トポロジカルソートが machine_orders の機械順序を反映したジョブ列を生成するため、純粋なGT法（ジョブ先行制約のみ）に比べて左詰めで順序が変わる余地は少ない
- 左詰め挿入により実際の機械順序が machine_orders と異なる場合があり、安定性は op_times から逆算した実際の順序で計算する
- N5近傍が machine_orders を変更すると、生成されるジョブ列（＝GT法への入力）が変わり、別のアクティブスケジュールが得られる

**semi-activeとの比較**:

| | semi-active | active (direct GT) |
|---|---|---|
| 機械順序 | machine_orders を厳守 | 変わりうる（左詰め許容） |
| ready判定 | ジョブ先行 + 機械先行の両方 | ジョブ先行のみ |
| 構築アルゴリズム | トポロジカルソート O(n) | GTループ O(n²) |
| N5との理論的整合 | 整合（順序が一致） | 不整合（探索空間の意味が変わる）|

---

## 2. 評価手法の比較

### 2.1 通常評価（ガントチャート再構築）

各近傍解の評価時にガントチャートを完全に再構築する方式。

```
evaluate(neighbor):
  1. build_gantt(neighbor)          → O(n*m)
  2. get_makespan(op_times)         → O(n*m)
  3. compute_stability(neighbor)    → O(n*m)
  4. weighted_objective(...)        → O(1)
```

**計算量**: O(n*m) per neighbor

### 2.2 Taillardの高速化（Taillard, 1994）

N5近傍のスワップ評価を、ガント再構築なしに行う手法。**semi-active専用**（active modeでは使用不可）。

**原理**: 各オペレーションについて事前計算：
- **head (r)**: 最早開始時刻（前方からの最長パス）
- **tail_job (q_J)**: ジョブ後続のみを通る最長パス長（後方）
- **tail_machine (q_M)**: 機械後続のみを通る最長パス長（後方）

隣接オペレーション u（先）, v（後）のスワップ後メイクスパン推定：

```
r'(v) = max(r_job(v), r_machine_pred(u))
r'(u) = max(r_job(u), r'(v) + p_v)

f(v,u) = max(
    r'(v) + p_v + q_J(v),
    r'(u) + p_u + q_J(u),
    r'(u) + p_u + q_M(v)
)
```

この公式は**下界**を与える（下流の変化を追わないため実際値はこれ以上になりうる）。

### 2.3 スクリーニング方式（本実装）

Taillardの推定値を直接使うと下界の不正確さで誤選択が生じる。そこで**スクリーニング方式**を採用：

1. Taillardの推定値で全近傍をスクリーニング（推定MS ≤ 現在のMSの近傍のみ残す）
2. 通過した近傍のみ `build_gantt` でフル評価
3. 有望な近傍がない場合は全近傍をフル評価にフォールバック

これにより：
- **解の品質**: 通常評価と完全に同一
- **計算コスト**: 評価回数が約40%削減（14,176 → 8,657）、CPU時間が約24%削減

---

## 3. 実験結果

### 3.1 実験設定

- 問題: MT10_10（10ジョブ × 10機械）
- ILS: N5近傍, swap摂動, 800反復, best改善戦略
- 10試行（シード: 7, 107, 207, ..., 907）
- 比較手法:
  - **Active (direct GT)**: `active_schedule=True`（GT直接駆動, 新実装）
  - **Semi-Active**: `active_schedule=False, taillard_acceleration=False`
  - **Semi-Active + Taillard**: `active_schedule=False, taillard_acceleration=True`

### 3.2 結果サマリー

#### 重み [1.0, 0.0]（効率性のみ）

| 手法 | MS平均 | MS最良 | Stab平均 | BestCPU平均 | TotalCPU平均 | 評価回数平均 |
|------|--------|--------|----------|------------|-------------|------------|
| Active (direct GT)     | 1051.0 | 1051 | 5.99 |  8.53s | 29.52s |  7,517 |
| Semi-Active            | 1051.0 | 1051 | 6.29 |  2.57s | 10.09s | 14,176 |
| Semi-Active + Taillard | 1051.0 | 1051 | 6.29 |  1.94s |  7.64s |  8,657 |

#### 重み [0.9, 0.1]（安定性を少し考慮）

| 手法 | MS平均 | MS最良 | Stab平均 | BestCPU平均 | TotalCPU平均 | 評価回数平均 |
|------|--------|--------|----------|------------|-------------|------------|
| Active (direct GT)     | 1051.0 | 1051 | 5.99 |  7.52s | 29.19s |  7,609 |
| Semi-Active            | 1053.9 | 1051 | 5.39 |  3.87s | 10.47s | 14,697 |
| Semi-Active + Taillard | 1053.9 | 1051 | 5.39 |  3.01s |  7.88s |  9,154 |

### 3.3 考察

#### メイクスパン品質
- w=[1.0, 0.0]: 全3手法で全試行1051達成（同等）
- w=[0.9, 0.1]: Active (direct GT) は全試行1051達成。Semi-Active系はTrial 6で1080（安定性との重み付きスコアで初期解が局所最適になるため）
- **Active (direct GT) の方が安定性考慮時のメイクスパン品質が高い**

#### 安定性
- Active (direct GT): 全試行で5.99（ばらつきなし）。GT法が一定のパターンで再配置するため
- Semi-Active系: 5.99〜6.74（Trial 6で0.00、1080から動けないケース）
- Active (direct GT) の安定性は op_times から逆算した実際の機械順序で計算（machine_ordersとの乖離を正しく処理）

#### 計算効率
- **Semi-Active + Taillard が最速**（TotalCPU: 7.64s）
- Semi-Active通常版（10.09s）より約24%高速化
- Active (direct GT) は最も遅い（29.52s）。GT法のO(n²)ループが各近傍評価で動くため

Active (direct GT) の計算コスト増加の内訳：
- Semi-Active: トポロジカルソート O(n) × 評価回数
- Active (direct GT): GTループ O(n²) × 評価回数（評価回数は少ない7,517回だが1回のコストが高い）

#### まとめ

| 観点 | Active (direct GT) | Semi-Active + Taillard |
|---|---|---|
| メイクスパン品質（安定性考慮時） | ◎ 全試行1051 | △ 1試行で1080 |
| 安定性 | △ 5.99（固定的） | ◎ 5.99〜6.74（探索的） |
| 計算速度 | × 約29s | ◎ 約7.6s |
| 理論的整合性 | △ N5とactive化の乖離あり | ◎ 完全整合 |

**推奨**: 計算速度・理論整合性・安定性のすべてで優れる **Semi-Active + Taillard**。
ただし安定性考慮時のメイクスパン安定度が重要な場合は **Active (direct GT)** も選択肢となる。

---

## 4. 使い方

```python
# Semi-Active + Taillard（推奨）
solver = ILSSolver(jm_table, fixed_gantt, reschedule_gantt, reschedule_time, weights,
                   active_schedule=False, taillard_acceleration=True)

# Active (direct GT)
solver = ILSSolver(jm_table, fixed_gantt, reschedule_gantt, reschedule_time, weights,
                   active_schedule=True)

# Semi-Active 通常
solver = ILSSolver(jm_table, fixed_gantt, reschedule_gantt, reschedule_time, weights,
                   active_schedule=False, taillard_acceleration=False)
```

---

## 5. 参考文献

- Giffler, B. & Thompson, G. L. (1960). Algorithms for solving production-scheduling problems. *Operations Research*, 8(4), 487-503.
- Taillard, É. (1994). Parallel taboo search techniques for the job shop scheduling problem. *ORSA Journal on Computing*, 6(2), 108-117.
- Nowicki, E. & Smutnicki, C. (1996). A fast taboo search algorithm for the job shop problem. *Management Science*, 42(6), 797-813.
