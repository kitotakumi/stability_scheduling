# コードレビュー（ils/memetic/evaluation 系）2026-06-23

> **範囲**: `ils_scheduling.py` / `memetic_scheduling.py` / `evaluation.py` / `genetic_operation.py` / `ga_scheduling.py` / `experiments/experiment_utils.py`。観点=理論的正しさ・最適化余地・無駄。3エージェント並列レビュー＋主要所見を本人が再検証。**コードは未変更（レビューのみ）**。
>
> **総括**: **致命的な実行中バグは検出されず**。コア数値（β=0 順列偏差の定義、`_stab_swap_delta` の差分一致、infeasible(閉路)判定、N5近傍、active/semi-active の delta ガード）は健全。主な論点は (1) 手法横断の公平性（GA と ILS/memetic の stability 基準・norm_params 経路）、(2) 速度最適化の余地（PR/decode 重複）。検証状態を各所見に明記する。

凡例: 検証= ✅本人確認済 / 🟡構造的に確からしいが数値未確認 / ⬜エージェント所見（未再検証）。

---

## A. 正しさ・研究妥当性

### A1. 🟡 [minor〜要確認] GA(B) と ILS/memetic(A) の stability が同一スケールか
- 実装: ILS/memetic は `evaluation.compute_stability_from_orders`（A, init=`reschedule_gantt`由来の `initial_machine_orders`）。GA は `compute_stability_from_gantt`（B, init=`delayed_gantt`から `_extract_changed_gantt` で確定除外）。
- **両者とも「可リスケ工程のみ」に対し同じ `_rank_deviation`（β=0: Σ|init_pos−cur_pos|）を適用**するので、可リスケ工程集合と初期順序が一致するなら数値も一致する（reschedule_gantt ＝ delayed−fixed は同じはず）。構造的には一致見込み。
- **未確認**: 同一解を A と B に与えて数値一致するかの突合せはしていない。
- **影響範囲**: 中心主張 **ILS vs Memetic は両方 A なので不影響**。影響は GA（ベースライン）の横断 stability スケールのみ。
- **推奨**: 同一シナリオ・同一順序で A と B が一致することの**回帰テストを1本**追加（mt10）。一致すれば本件クローズ。

### A2. ✅ [latent footgun] norm_params 未指定時に memetic/GA が非決定の正規化推定にフォールバック
- `run_memetic`/`run_ga` は `norm_params=None` だと `genetic_operation.estimate_normalization_params`（グローバル乱数依存・シード非固定・max_stab 推定が evaluation 版と別アルゴリズム）に落ちる。
- ✅ **core_v3 本番は run_v3 が常に seeded な `compute_shared_norm_params` を渡すため到達しない**（[[norm_params_unseeded]] の前提は守られている）。
- **推奨**: `run_*` で「本番は norm_params 必須」を assert 化し、ad-hoc 実行での取り違えを防止（latent な再現性事故の予防）。

### A3. ✅ [minor/保守性] `evaluation.compute_stability_stat` が β を直書きで再実装（`_rank_deviation` 非共有）
- L91 が `diff/(current_pos+1)**β` を独自実装。✅ β=0 なので**現状の数値は本番 stability と完全一致＝実害なし**。将来 β を変えると stat だけ同期漏れする二重定義リスク。stat は分析(mech_stats)用。
- **推奨**: L88-96 を `_rank_deviation` 呼び出しへ一本化（エージェントは critical 判定だが、β=0 現状は実害ゼロ＝minor に訂正）。

### A4. ✅ [問題なし・確認] `_stab_swap_delta` は β=0 で `compute_stability` と厳密一致
- 2エージェントが独立に検算。swap で動く2ジョブの項のみ差し引き、他ジョブ項は位置不変で相殺＝厳密。LS/PR の高速差分評価の土台は健全。
- **メモのみ**: docstring に「β 任意で厳密（他ジョブ項相殺）」と明記し、β を戻す将来に備えると安全。

### A5. ⬜ [minor] makespan と stability の正規化が非対称
- makespan は p90 レンジ正規化（`normalize_value`）、stability は `1 + stab/max_stab`（min=0 固定・max=最悪）。同じ重みでも実効スケールが非対称。設計意図の可能性が高いが、重み掃引の解釈に効くので **doc に仕様明記**を推奨。

### A6. ⬜ [minor] memetic エリート保存 `offspring[0]` 無条件上書き（memetic_scheduling.py:399付近）
- `offspring[0] = selBest(offspring+[best_prev],1)[0]`。best は保護されるので**悪化解は残らない**が、評価済み個体を捨てる/重複個体混入で多様性が僅かに低下。`selWorst` を置換対象にするのが安全。正しさ影響は軽微。

---

## B. 最適化の余地（速度のみ・解は不変）

> いずれも**出力（解・HV）は不変、時間ベース指標(AOC等)は変わりうる**。[[pr_decode_cost_and_aoc_consistency]] の整合性方針に従い、採用は現バッチ完走後＋diff0検証後。

### B1. ⬜ [major] path_relinking / `_escape_infeasible` の候補ごと full copy（O(diffs²)）
- 既知（[[pr_decode_cost_and_aoc_consistency]]）。`step_strategy='best'` 掃引時に特に顕著。repair の in-place 方式へ統一可能。

### B2. ⬜ [major] memetic `_refine_individual` の build_gantt 重複デコード
- 1個体で gene→orders（GT法 deepcopy 込み）＋ kick評価＋gene書き戻し＋最終評価 で 3〜5 回デコード。さらに `skip_first_ls` かつ kick 不発の個体も先に decode してから捨てる（kick確率抽選を decode 前に引けば回避可）。`kick_point`/`prels_point` は `track_population` 時のみに限定可。

### B3. ⬜ [major] GA `_evaluate_individual` が makespan と stability で 2 回フルデコード
- `compute_makespan` と `compute_stability` が各々 `get_gantt_reactive` を呼ぶ。GA 500世代×pop50 で支配的。1回デコードを ms/st 両方に流す共通関数で半減可能（GA ランタイムのみ・本番ボトルネックは memetic_pr）。

### B4. ⬜ [major/要ベンチ] Taillard スクリーニング通過後にメイクスパンで全候補を再デコード
- `_score_lower_bound`(est_ms下界・decode無)で安価に絞るが、通過候補は `evaluate` で `build_gantt` を回す。best 戦略で est_ms 昇順＋確定 best_score 未満を est で枝刈りすれば partial decode 回避の余地（要実測）。

### B5. ⬜ [minor] local_search で毎反復 `compute_stability(current)` をフル再計算
- 近傍は差分評価しているが、ループ頭の基準 cur_stab を毎回フル算出。採用 swap の `n_stab` を次反復へ引き継げば削減可。

### B6. ⬜ [minor] N5/Taillard 推定で `ops.index()` 線形探索を反復
- ブロック生成時に既知の位置を引き回せば `.index()` 除去可。

---

## C. 無駄・デッドコード（軽微）

- ✅ `ils_scheduling.py:9` `import copy` 未使用（`_copy_orders` は手書き dict-comp）。
- ⬜ `ils_scheduling.py:325` `_score_lower_bound` の `stability is None` 分岐は実呼び出し（必ず注入）から到達不能。
- ⬜ `_compute_heads_and_tails` の戻り値 `tail` は外部未使用（内部漸化には必要）。

---

## D. 推奨アクションの優先順位

| 優先 | 項目 | 種別 | いつ |
|---|---|---|---|
| 1 | A1: GA(B) vs ILS/memetic(A) の stability 一致を回帰テストで確認 | 妥当性 | 次の空き時間（結果を信じる前に1回）|
| 2 | A2: 本番 norm_params 必須の assert 化 | 妥当性(予防) | いつでも（低リスク）|
| 3 | A3: compute_stability_stat を _rank_deviation に一本化 | 保守性 | いつでも |
| 4 | B1-B4: PR/decode 重複の最適化 | 速度 | **現バッチ完走後**＋diff0検証＋AOC方針決定後 |
| 5 | C: デッドコード除去 | 整理 | いつでも |

**今は触らない**（実験稼働中＝再現性保護）。速度系(B)は [[pr_decode_cost_and_aoc_consistency]] の通り AOC 整合性の判断とセットで。
