# ILS (反復局所探索法) スケジューリング 設計文書

> **このドキュメントの役割**: ILSソルバーの設計方針・アーキテクチャ・アルゴリズム仕様をまとめた設計書。実装の「なぜこうなっているか」の根拠資料として使用する。コード変更時はこのドキュメントも更新すること。

## 研究概要

ジョブショップスケジューリング問題(JSSP)における、ジョブ投入順序の変更量を考慮した反復局所探索法(ILS)による再スケジューリング手法の提案。

### 核心的な仮説
- GAベースの手法は交叉によって「良質な初期スケジュール」の構造を破壊する
- ILSなら「局所探索(深掘り)」と「摂動(脱出)」が分離されており、安定性を制御しつつ効率的に探索できる

### 目的関数
- 重みパラメータ法による2目的最適化
  - 効率性: メイクスパン (総所要時間)
  - 安定性: 順位偏差関数 (ジョブ投入順序の変更量)
- min-max正規化で重みスケールを合わせる

---

## 解表現

### GA (既存手法) との違い

| | GA (既存: main_pareto_scheduling.py) | ILS (提案手法: ils_scheduling.py) |
|---|---|---|
| 解表現 | GT法の遺伝子 (ジョブ番号混合列) | **機械ごとの作業順序 (Machine Order)** |
| 探索操作 | 交叉・突然変異 | N5近傍 (機械内順序の直接スワップ) |
| デコード | `get_gantt_reactive` (左詰め挿入) | **前方パス (semi-active schedule)** |
| 解の表現例 | `[3, 0, 5, 1, 3, ...]` | `{M0: [(j8,op5), (j0,op3), ...], M1: [...]}` |

### Machine Order 表現
```python
# 各機械について、リスケ対象操作の処理順序を保持
machine_orders = {
    0: [(job_id, op_index), (job_id, op_index), ...],  # Machine 0
    1: [(job_id, op_index), (job_id, op_index), ...],  # Machine 1
    ...
}
```

### GT法の遺伝子をILSで使わない理由
1. ILSではGAの交叉を使わないため、1本のジョブ番号列にエンコードするメリットがない
2. N5近傍は「機械ごとの作業順序」を直接操作するため、machine_order表現が自然
3. 遺伝子⇔machine_orderの相互変換（トポロジカルソート等）は計算コストが高くバグの温床

---

## 評価関数の実装方針

### メイクスパン (効率性)

**Phase 1: 前方パスによるガント構築**
- machine_ordersから各操作の開始・終了時刻を計算
- `start = max(同一機械の前操作の終了時刻, 同一ジョブの前工程の終了時刻)`
- メイクスパン = 全操作の最大終了時刻
- MT10_10 (100操作) では十分高速

**Phase 2: Taillardのアルゴリズム (高速化が必要な場合)**
- 各操作のhead (最早開始時刻) / tail (残り最長パス) を事前計算
- N5スワップ後、影響範囲のみ再計算 → O(n) での近傍評価が可能
- 全近傍のガント再構築を回避できる
- 実装が複雑なため、Phase 1で性能ボトルネックを確認してから導入

### 安定性

**machine_ordersから直接計算 (ガント構築不要)**
- 初期スケジュールと候補解のmachine_orders上のジョブ順序を直接比較
- 既存の stability_function_v3 と同じ順位偏差計算:
  `Σ |rank_diff| / (position + 1)^1.25`
- ガントチャートの再構築が不要なため高速

---

## アーキテクチャ

### 作成/書き換えファイル
- **`ils_scheduling.py`**: 全面書き換え (唯一の変更ファイル)

### ファイル構成 (2026-04-01 リファクタリング後)
| ファイル | 役割 |
|---|---|
| `evaluation.py` | **GA/ILS共通**: 安定性関数・正規化・重み付き目的関数 |
| `ils_scheduling.py` | ILSソルバー（evaluation.pyを使用）|
| `ga_scheduling.py` | GAソルバー（evaluation.pyを使用）|
| `genetic_operation.py` | GA遺伝子操作（交叉・突然変異・選択）|
| `job_shop_scheduling.py` | 問題データ（m_table, pt_table, gantt）|
| `gantt_chart_operation.py` | ガントチャート操作（外乱検知・デコード）|
| `analysis.py` | 可視化 |
| `pareto_scheduling/` | パレート最適化関連（アーカイブ）|
| `achirve/` | 旧コード（バックアップ）|

### ILS内部で自前実装するもの
1. ガントチャート → machine_orders 変換 (初期解生成時のみ)
2. machine_orders → ガントチャート構築 (前方パス、メイクスパン評価用)
3. 安定性評価 (machine_ordersから直接計算)
4. クリティカルパス探索
5. クリティカルブロック抽出
6. N5近傍生成 (閉路チェック不要: Nowicki & Smutnicki証明)
7. 局所探索 (最良改善 / 最初改善を選択可能)
8. 摂動 (critical_swap / insert / path_relink)
9. ILSメインループ

---

## クラス設計

```
class ILSSolver:
    # --- 初期化 ---
    __init__(jm_table, fixed_gantt, reschedule_time, weights,
             max_eff, min_eff, max_stab)

    # --- 表現変換 ---
    gantt_to_machine_orders(gantt) → machine_orders
        # ガントチャートからリスケ対象のmachine_ordersを抽出

    # --- ガント構築 (前方パス) ---
    build_gantt(machine_orders) → gantt, op_times
        # machine_orders + fixed_gantt + jm_table から
        # semi-active scheduleを構築
        # op_times: {(job, op): (start, end, machine)}

    # --- 評価 ---
    compute_makespan(machine_orders) → int
        # build_ganttを呼んでmax終了時刻を返す

    compute_stability(machine_orders) → float
        # 初期解との順位偏差を直接計算 (ガント不要)

    evaluate(machine_orders) → float
        # 正規化した重み付き評価値

    # --- クリティカルパス ---
    find_critical_path(op_times) → set of (job, op)
        # 終端からバックトラックでクリティカルパスを特定

    find_critical_blocks(critical_path, machine_orders) → [Block, ...]
        # 同一機械上の連続するクリティカル操作をブロック化

    # --- N5近傍 ---
    generate_n5_neighbors(machine_orders) → [(machine_orders, swap_info), ...]
        # 各ブロックの先頭2つ / 末尾2つを交換
        # 閉路チェック不要 (Nowicki & Smutnicki)

    # --- 局所探索 ---
    local_search(machine_orders, strategy='best') → (machine_orders, score)
        # strategy: 'best' = 最良改善, 'first' = 最初改善

    # --- 摂動 ---
    perturb(machine_orders, method, strength) → machine_orders
        # method: 'critical_swap' / 'insert' / 'path_relink'
        # strength: 摂動の強さ (段階的に増加)

    # --- ILSメインループ ---
    run(initial_machine_orders, max_iterations) → best_machine_orders
```

---

## ILSアルゴリズムのフロー

```
1. 初期解の生成
   - delayed_gantt → check_disturbance → fixed_gantt, reschedule_gantt
   - reschedule_gantt → machine_orders に変換 (初期解)

2. 正規化パラメータの推定
   - ランダムなmachine_ordersをサンプリングして max_eff, min_eff, max_stab を推定

3. ILSメインループ (max_iterations回)
   3.1 摂動 (Perturbation / Kick)
       - **常にbest解**に摂動を加えて新たな出発点を生成
       - 摂動の強さは段階的に制御（非改善3回で+1）

   3.2 局所探索 (Local Search)
       - N5近傍を列挙
       - 最良改善 (or 最初改善) で移動
       - 改善がなくなるまで繰り返し → 局所最適解

   3.3 最良解の更新 (Update Best)
       - 局所最適解が全体最良を更新したら記録

4. 結果出力
   - best_machine_orders → ガント構築 → 可視化
```

---

## 摂動の設計

### 摂動手法
1. **Swap (デフォルト)**: N5近傍のスワップをstrength回連続適用。各ステップでクリティカルパスを再計算し、N5近傍からランダムに1つ選んで適用する。N5近傍の理論的保証（閉路が発生しない）を活かした摂動であり、クリティカルパス上の操作を直接操作するため効果的。
2. **Insert**: 操作を抜き取り、同一機械の別の位置に挿入。実行不可能解が発生しうるが、同一機械内の順序入れ替えのみなのでほぼ発生しない。リトライガードあり。
3. **Path Relinking**: 初期スケジュールの順序に一部を強制的に戻す。安定性改善方向への摂動。

### 摂動手法の選択方針
- まずSwapのみで検証し、効果を確認してからInsertを追加検討する
- Path Relinkingは安定性重視の重みで有効になりうるが、優先度は低い

### 受理判定
- **常にbest解から摂動する**（標準的なILS受理判定）
- 非改善が続いたら摂動強度を段階的に増加（3回で+1）

### 摂動強度の制御
- 安定性の破壊を最小限に抑えるため、摂動の規模を小→大と段階的に増加
- 局所最適が改善されなくなったら摂動を強める

---

## N5近傍の詳細

### 定義
クリティカルブロック (同一機械上でクリティカルパスに連続して含まれる操作の列) の:
- **先頭2つの操作を交換**
- **末尾2つの操作を交換**

### 閉路チェック不要の根拠
Nowicki & Smutnicki (1996) により、ブロック境界にある操作の交換は
ディスジャンクティブグラフに閉路を作らないことが数学的に証明されている。

### 計算量
- クリティカルブロック数は高々 m 個 (機械数)
- 各ブロックから高々2つの近傍 → 1ステップあたり最大 2m 個の近傍
