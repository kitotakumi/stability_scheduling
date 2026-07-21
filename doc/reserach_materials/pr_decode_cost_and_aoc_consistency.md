# 計算時間削減（PR/decode）と AOC 整合性

> **目的**: 計算時間（特に ta21_high の memetic_pr ~45分/run）の削減余地と、最適化が AOC 等の時間系指標に与える影響を整理する。コードレビュー（2026-06-23, 3エージェント＋本人検証）の計算時間系所見を集約。
>
> **結論（2026-06-23）**: **B1（PR候補lazy化）は実測 ~10% で見送り。計算時間最適化は一旦クローズ**（現状維持＋脚注）。正しさレビューは**バグなし・コア健全**（§5）。
>
> 関連: [[memetic_pr_time_investigation]] / [[pr_decode_cost_and_aoc_closed]]。検証済パッチ=`doc/patches/B1_path_relinking_lazy_candidates.patch`。

---

## 1. PR decode が重い原因（O(diffs²)）

Path Relinking（[`ils_scheduling.py:909`](../ils_scheduling.py#L909)）は現在解 S_cur から初期解 S_ref へ、食い違い位置を direct swap で1つずつ直してたどる。経路長＝差分数 `diffs`（≈リスケ数。ta21_high は 82%×400op ≈ 数百）。

| 層 | 内容 | コスト | 状態 |
|---|---|---|---|
| ① 経路を歩く（最低1 decode/手） | O(n·diffs) | 不可避（PRの本質）|
| ② best 選択（毎手 全候補をデコード） | ×diffs = O(n·diffs²) | 既定 'random' で回避済 |
| ③ 候補を毎手フルコピー（random でも全件 copy 後に1件採用） | ×diffs = O(n·diffs²) | **B1の対象（→§2で見送り）** |
| ④ top-k LS×3（best-improvement, O(n²)） | 定数3倍 | 設計上の選択。**実は主コスト（§2）** |

repair（`_perturb_repair`, [`ils_scheduling.py:762`](../ils_scheduling.py#L762)）は in-place swap＋infeasibleなら戻す方式で③の無駄が無い（PRが真似れば③は消える）。ta21_high で repair も遅い(30分)のは④系＝広大な自由化領域への LS が重い本質コスト。

## 2. B1 実測結果（見送り確定）

worktree 隔離で B1（候補の lazy 生成＝③除去）を実装し memetic_pr で OLD/NEW 比較:

- **等価性 ✅ 完全一致**: mt10・la36_large とも finals・uea_points・history（cpu_time除く）が bit 一致。乱数列を保つ等価リファクタ（変わるのは時間系のみ）。
- **速度短縮（cpu, 2本平均）**: mt10(低diffs) **3%** / la36_large(73%) **10%** / ta21_high(82%) 外挿 ~15-25%。
- **解釈**: ③のコピーは memetic_pr の主コストではない。実コストの大半は **④ top-k LS×3 と①の経路 decode**。
- **判断**: ~10%のために PR系2手法を 10-17h 再走（しかも変わるのは AOC のみ・HV不変）は非効率 → **B1見送り**。検証済パッチは保存（将来 ta21_high の速度が律速になったら適用可）。AOC を本気で縮めるなら効くのは ④ だが、これは解が変わる別物。

## 3. 他の計算時間削減候補と手法スコープ

| | 内容 | 効く手法 |
|---|---|---|
| B1 | PR候補のフルコピー O(diffs²) | ils_pr, memetic_pr |
| B2 | memetic `_refine_individual` の build_gantt 重複（1個体3〜5回 decode） | memetic_ls/repair/pr |
| B3 | GA `_evaluate_individual` が makespan/stability で2回デコード | ga |
| B4 | LS の Taillard スクリーニング通過後に全候補を再デコード | ga 以外の6手法（LS経由）|

**B4 が ga 以外の全手法を巻き込む**ため、B1〜B4 を全部やると **7手法すべて**の時間が変わる＝事実上フル再走。

## 4. 再走コストと AOC 整合性【最重要】

最適化は**解は不変・到達時間が縮む**。指標が2種に分かれる:

- **不変（解ベース）**: 統合HV / 高安定HV / scalar / C-metric / カバー率 / 改善率 / 各 p値・順位。
- **変わる（時間ベース）**: **AOC** / anytime HV(t) / TTT / cpu_time（速くなる＝AOC良化）。

→ 「コードだけ直して既存データ温存」が成立するのは**解ベース指標のみ**。AOC を最適化後コードで揃えるなら**PRを含む実験の回し直しが必要**（旧遅コードと混在は不公平）。再走 wall 見積（現コード速度＝上限・8並列）:

| シナリオ | 再走対象 | wall |
|---|---|---|
| B1のみ | ils_pr+memetic_pr ×8問題 | ~17.5h（B1適用で実質短縮） |
| 全B | 7手法フル | ~36h |

**方針（確定）**: 現 main_v1 は全手法・全問題が同一コード＝AOC は内部的に公平な apples-to-apples。**現状維持＋脚注**（「PR実装は O(diffs²) のコピー余地あり・high-reschedule で高速化可能（future work）」）で十分。

## 5. コードレビュー 正しさ（要約：問題なし）

3エージェント＋本人検証で **致命的な実行中バグは検出なし**。コア数値は健全:
- β=0 順列偏差の定義（`_rank_deviation` を全実装が共有）・`_stab_swap_delta` の差分一致・infeasible(閉路)判定・N5近傍・active/semi-active の delta ガード ＝いずれも確認済。
- 唯一の妥当性確認事項: **GA(B) と ILS/memetic(A) の stability スケール一致**。両者とも「可リスケ工程に β=0順列偏差」で構造的に一致見込みだが数値突合せ未実施 → 気になれば mt10 で回帰テスト1本（中心主張 ILS vs Memetic は両方Aで不影響）。
- 衛生（任意・解にもAOCにも無影響）: 本番 norm_params 必須の assert 化 / `compute_stability_stat` の β 直書きを `_rank_deviation` に一本化 / dead `import copy` 除去。
