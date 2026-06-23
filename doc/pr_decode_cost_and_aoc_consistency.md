# Path Relinking の計算コストと AOC 整合性メモ

> **目的**: ta21_high（20×20×82%）で Memetic+PR / ILS+PR が極端に重くなる原因（PR decode の計算量）を整理し、最適化の余地・回避可否、および「最適化すると**時間ベースの指標(AOC等)が変わる**」という実験結果整合性の論点を記録する。
>
> **作成**: 2026-06-23。実装は [`ils_scheduling.py`](../ils_scheduling.py) `path_relinking` / `_perturb_repair`、関連は [`doc/theory/path_relinking.md`](theory/path_relinking.md)・[[memetic_pr_time_investigation]]。

---

## 1. 背景: ta21_high で PR が爆発する

主実験 main_v1 に追加した `ta21_high`（リスケ率82%）の実測 CPU/run:

| 手法 | la36_small(27%) | la36_middle(54%) | ta21_high(82%) |
|---|---|---|---|
| memetic_pr | 約4.5分 | 約9.5分 | **約45分** |
| memetic_repair | 約2.5分 | 約5.8分 | 約30分 |
| ils 系 / ga | 0.4〜1.3分 | 1〜1.4分 | 2.5〜3.5分 |

memetic_pr が突出。原因は PR の経路長＝差分数 `diffs`（≈自由化工程数≈リスケ数）に対する計算量。ta21_high は 82%×400op ≈ diffs が数百になる。

---

## 2. PR decode の計算量モデル

Path Relinking は現在解 S_cur から初期解 S_ref へ、食い違っている位置を direct swap で1つずつ直してたどり、経路上の最良中間解を返す（[`ils_scheduling.py:909` `path_relinking`](../ils_scheduling.py#L909)）。

各ステップで「まだ食い違っている全位置」について候補スワップを生成（[`ils_scheduling.py:1005`付近](../ils_scheduling.py#L1005)）。ステップ数 ≈ diffs、各ステップの候補 ≈ 残り diffs。

### 3層に分解

| 層 | 内容 | コスト | 回避可否 |
|---|---|---|---|
| ① 経路を歩く | diffs 手・各手 最低1デコード | O(n·diffs) | **不可避**（PR の本質。n=工程数）|
| ② best 選択 | `step_strategy='best'` は毎手 全候補を**評価(デコード)** | ×diffs = O(n·diffs²) | **既定 'random' で回避済**（1デコード/手, TS-PR流, Peng et al. 2015）。pilot で best と HV 有意差なし(p=0.72)・3〜13倍高速を確認 |
| ③ 候補コピー構築 | 'random' でも候補生成ループが**残り全位置ぶんのフルコピー(`_copy_orders`)を毎手作ってから**1個を選ぶ | ×diffs = O(n·diffs²) | **回避可能・未対応**（現状の残る無駄）|
| ④ top-k LS | `pr_ls_top_k=3` で経路上位3中間解に best-improvement LS | 定数3倍（diffs² ではない）| 設計上の選択 |

**結論: PR は②(デコード)は最適化済みだが、③(コピー構築)に O(diffs²) が残っている。** ta21_high のような極端な高 diffs でだけ③が支配的になりうる（diffs² なので）。1回のコピーは安い（μ秒級）が、ta21_high では diffs~300 → 1 PR呼び出しで ~9万コピー、PR は1runで数百回発火するため累積する。

> 注: ③が memetic_pr の**第1の時間要因かは未 profiling**。top-k LS×3（O(n²)級×3）も重い。確定には profiling が必要。

---

## 3. repair は無関係（既に効率パターン）

repair（`_perturb_repair`, [`ils_scheduling.py:762`](../ils_scheduling.py#L762)）は**コピーを作らない**:

- 不一致を**タプル列挙**（コピーなし）→ シャッフル → **1個ずつ in-place スワップ → デコード → infeasible なら戻す**（[`ils_scheduling.py:790-799`](../ils_scheduling.py#L790)）。

つまり repair は③の無駄を持たない（PR が真似すべき効率パターンを既に採用）。ta21_high で repair も遅い（30分）のは**キック後の広大な自由化領域に対する LS** が重い別要因で、本質的コスト。

→ **③のコピー最適化は PR(memetic_pr / ils_pr)にのみ効き、repair には無関係。** ils_pr も `pr_step_strategy='random'`（[`ils_scheduling.py:1279`](../ils_scheduling.py#L1279)）で同じ path_relinking を通るため、memetic_pr と同じ最適化が同じ等価性で効く。

---

## 4. 最適化案（③の除去）

PR の候補生成を repair 方式に書き換える:

- 現状: 食い違い全位置ぶんの**フルコピー候補**を作ってから選択。
- 改善: スワップ `(m, i, q)` の**タプルだけ列挙** → シャッフル → 実際に評価する1個だけ in-place スワップ→デコード→ feasible なら採用、infeasible なら戻して次へ。
- 効果: コピー構築が O(diffs²) → **O(diffs)**。'best' は全評価が必須なので O(diffs²) のまま（既定は 'random'）。

### 等価性（最重要）

乱数を消費するのは `random.shuffle(scan_order)` のみ。同じシャッフル順で「最初の feasible」を採るロジックを保てば**乱数列・経路・出力は完全同一**（コピーを遅延構築するだけ）。

→ **採用前に小問題（mt10・数試行）で旧コードと diff 0 を検証する。** diff 0 が取れれば下記5の整合性問題のうち「品質指標」は不変が保証される。

---

## 5. AOC 整合性の論点【重要】

③を最適化すると PR が速くなる。**解(見つかるスケジュール)は同一だが、到達時間が縮む**ため、指標が2種に分かれる:

| 指標 | 種類 | 最適化で | 理由 |
|---|---|---|---|
| 統合HV / 高安定HV / scalar / C-metric / カバー率 / 改善率 / 各 p値・順位 | 解ベース | **不変** | 最終点集合・パレートフロントが同一 |
| **AOC** | 時間ベース | **変わる(良化)** | HV対 log時間 の時間平均。PR が速い＝同じHVに早く到達＝AOC上昇 |
| anytime HV(t) 曲線・交差図 | 時間ベース | 変わる | snapshot の cpu_time が縮む |
| TTT / QRTD | 時間ベース | 変わる | 到達時刻が縮む |
| convergence.total_cpu_time | 時間ベース | 変わる | run 自体が高速化 |

### 帰結

- **「コードだけ直して既存データ温存」が成立するのは HV系など解ベース指標に限る。**
- **AOC を最適化後コードで揃えたいなら、PR を含む実験の回し直しが必要**（旧コード=遅い計時 と 新コード=速い計時 の混在は手法間で不公平）。
- 現 main_v1 は**全手法・全問題が同一(旧)コード**で走っており、AOC は内部的に公平な apples-to-apples。「この実装での各手法の anytime 性能」として論文掲載は妥当。

### 方針の選択肢

1. **現状維持（最適化を今回の論文に入れない）** ＝ 無難。AOC は実装込みの実測として正直に出し、脚注で「PR 実装は O(diffs²) のコピー余地があり高速化可能（future work）」と明記。
2. **最適化して AOC も正しく測る** ＝ PR系を回し直す（ta21_high は ~20h/再走）。③のコピーは PR の**本質でなく実装の無駄**（repair はやっていない）ため、「アルゴリズムの公平比較」を厳密に求めるなら最適化後の計時が正しい、という論拠は立つ。

判断軸: **AOC が主張の中心か**。claim1 の3枚目として補助的に出す程度なら選択肢1（現状維持＋脚注）が妥当。AOC で PR の時間優位/劣位を強く主張するなら選択肢2。

---

## 5.5. B1 実測結果（2026-06-23・見送り確定）

worktree 隔離で B1（候補の lazy 生成）を実装し memetic_pr で OLD/NEW を実測:

- **等価性: ✅ 完全一致**。mt10・la36_large とも finals・uea_points・history（cpu_time除く）が bit 一致。乱数列を保つ等価リファクタであることを実証（変わるのは時間系のみ＝設計どおり）。
- **速度短縮（cpu, 競合下・各1本）**: mt10(低diffs) ~3% / **la36_large(73%) ~12%**。ta21_high(82%) 外挿でも ~15〜25% 程度。
- **解釈**: O(diffs²) コピーは memetic_pr の主コストではなかった。実コストの大半は **top-k LS×3（best-improvement, O(n²)）と経路 decode**。コピー削減はそこに効かない。

**判断: B1 は見送り**。~12%のために PR系2手法を 10〜17h 再走（しかも変わるのは AOC のみ・HVは不変）は非効率。現状結果＋脚注（「PR実装はO(diffs²)のコピー余地ありfuture work」）で確定。検証済みパッチは `doc/patches/B1_path_relinking_lazy_candidates.patch` に保存（将来 ta21_high の速度が律速になったら適用可・等価性実証済み）。AOC を本気で縮めるなら効くのは B1 でなく top-k LS（ただし解が変わる別物）。

## 6. アクション（保留中）

- [ ] 現バッチ完走を待つ（走行中コードは触らない＝再現性保護）。
- [ ] ③の in-place 化を実装 → mt10 等で **diff 0 検証**。
- [ ] AOC の扱いを§5で決定（現状維持＋脚注 / 回し直し）。
- [ ] profiling で③ vs top-k LS のどちらが memetic_pr の主因か確認（最適化の費用対効果見積もり）。
