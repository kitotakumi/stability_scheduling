# 評価・実験設計

> **このドキュメントの役割**: 再スケジューリング比較研究の評価方法論と実験設計を定義する。「何をどう測るか」「どの実験をどの優先度で走らせるか」の判断根拠を残す。多目的最適化・評価指標の理論的背景は [theory/multiobjective_optimization.md](theory/multiobjective_optimization.md) を参照。

---

## 1. 研究の主張

3 軸が独立に成立するロバスト構造。1 軸崩れても論文は成立する。

| 軸 | 主張 | 主指標 | 1 軸崩れた場合 |
|---|---|---|---|
| **(A) 速度** | ILS は GA より構造的に高速（同 CPU 時間で優れた解に到達） | per-weight anytime scalar / UEA HV curve | B-1/B-2 が残れば成立 |
| **(B-1) per-weight 質** | weight 別に ILS は GA より高品質。GA は高 stab 重みで degenerate | per-weight scalar 値 + 改善成功率 | B-2 や A が残る |
| **(B-2) 統合 Pareto 質** | weighted sum sweep の UEA 解集合の Pareto 覆域で ILS が上（2 側面） | 下記 B-2a / B-2b 参照 | B-1 や A が残る |

**B-2 の 2 側面（どちらも weighted sum sweep の union UEA 解集合に対して分析）**:

| サブ主張 | 主張内容 | 主指標 |
|---|---|---|
| **(B-2a) 総合的な Pareto 質** | ILS は全領域にわたって Pareto 覆域が広い | per-trial union UEA HV + C-metric |
| **(B-2b) 高安定性領域での質** | ILS は高安定性の解を GA より多く・良質に発見できる | 領域別 HV + 差分 EAF + 条件付き MS Wilcoxon |

**副次観察（独立主張ではなく補強）**:
- repair 摂動は baseline ILS の Pareto 安定性側を系統的に拡張する（B-1/B-2 内で確認）
- MS と安定性は構造的トレードオフ関係にある（PR 実験で確立）

**名称の注意**: アルゴリズムは「weighted sum scalarization sweep」。「MOEA/D」は neighborhood + 解共有を持つ別手法なので使わない（→ [multiobjective_optimization.md](theory/multiobjective_optimization.md)）。

---

## 2. 独立変数（実験で振るもの）

| レベル | 変数 | 値域 | 振る目的 |
|---|---|---|---|
| **アルゴリズム** | 手法 | ILS（±repair, ±PR）, GA, (メメティック) | 手法間比較 |
| 問題 | JSSP instance | mt10, la21, la36, la40 | サイズ・種類への一般性 |
| 外乱 | 遅延シナリオ | `_delay{60, 100, 147, 148, ...}` | 外乱規模への頑健性 |
| 目的関数 | weights = [eff, stab] | `[1.0,0], [0.9,0.1], ..., [0.1,0.9], [0.0,1.0]`（0.1 刻み 11 点） | 重み依存性・Pareto 集約 |
| 算法パラメータ | `repair_trigger`, `repair_strength` | 掃引で決定 | repair 効果最大化 |
| 予算 | 反復数 / 世代数 | ILS `max_iterations`, GA `ngen`（自然収束まで余裕ある値） | 速度は anytime curve で事後抽出 |

### 手法の説明

| 手法 | 説明 | 位置付け |
|---|---|---|
| ILS-baseline | 重み付き ILS（insert 摂動） | ベースライン |
| ILS+repair | baseline + 停滞時に repair キック（P-1） | 提案手法（安定性拡張） |
| ILS+PR | baseline + Path Relinking（elite archive との経路探索） | 提案手法（探索多様化） |
| GA | 重み付き GA（既存実装） | 主比較対象 |
| メメティック（任意） | GA + LS（必要に応じて） | 査読対応用、優先度低 |

### 固定する変数（実験済み・根拠確定）

| 変数 | 固定値 | 根拠 |
|---|---|---|
| `strategy` (LS 戦略) | `'best'` | FI vs BI で有意差なし（experiment_plan 2026-04-15） |
| `active_schedule` | `False` | N5 の理論保証が semi-active 前提 |
| `stagnation_threshold` | `None` | 悪化受理は重み付きスコアで不利（2026-04-14） |
| `taillard_acceleration` | `True` | 合成スコア下界フィルタで正しく動作（2026-04-14 修正後） |
| ILS `max_iterations` | `800`（自然収束まで余裕） | 収束後の最終品質を測りたいので CPU 打ち切りはしない |
| GA `ngen`, `pop_size` | `500`, `50`（自然収束まで余裕） | 同上 |
| 正規化方式 | min-max（問題・外乱ごと共通パラメータ） | `experiment_utils.compute_shared_norm_params` |

---

## 3. 従属変数（測定するもの）

### 3.1 速度（主張 A）

| 指標 | プロット形式 | 見る対象 |
|---|---|---|
| CPU 時間 × per-weight scalar 値 | anytime curve（代表 weights） | deployment 視点の速度 |
| CPU 時間 × per-weight UEA HV | anytime curve（代表 weights） | 特定 weight 方向の探索覆域の速度 |

スナップショット T = 5, 10, 20, 40 秒で Wilcoxon signed-rank（paired per-trial 設計）。

### 3.2 per-weight 品質（主張 B-1）

| 指標 | 計算 | 役割 |
|---|---|---|
| **per-weight scalar 値** | w₁·MS + w₂·Stab（最終解） | 基本性能 |
| **per-weight UEA HV** | 同一 weight 内で ILS vs GA の HV 比較 | 探索覆域（方向偏りが同一なので比較可能） |
| **改善成功率** | 初期解から動けた trial 数 / 総 trial 数 | degeneracy 検出（Fisher's exact） |
| ΔMS 中央値、Stab 中央値 | 動いた trial の改善量 | (ii)MS 改善+Stab 犠牲 vs (iii)MS 改善+Stab 維持 を区別 |

scalar 値だけでは GA の degenerate 状態（init 張り付き）と改善状態を区別できないため、改善成功率との併用が必要。

### 3.3 統合 Pareto 品質（主張 B-2）

**B-2a: 総合的な Pareto 質（全領域）**

| 指標 | 計算 | 役割 |
|---|---|---|
| **per-trial union UEA HV** | trial 内で N weights の UEA を統合 → non-dominated → HV | **主筋**。純粋なアルゴリズム比較 |
| **C-metric** | C(ILS, GA) と C(GA, ILS) | dominance 直接測定（参照点不要） |

**B-2b: 高安定性領域での質**

| 指標 | 計算 | 役割 |
|---|---|---|
| **カバー率** | stab ≥ 閾値 の解を 1 個以上持つ trial の割合 | 高 stab 解に到達できるかの基本確認（Step 1） |
| **領域別 HV** | stab 軸を P33/P67 で 3 分割 → 各領域内の HV | 安定性帯ごとの覆域定量化（スカラー指標） |
| **条件付き MS Wilcoxon** | stab ≥ P67 の解の中で最小 MS → trial 間 Wilcoxon | 「高安定性を確保しつつどこまで MS を下げられるか」の統計検定 |
| **差分 EAF** | per-trial union UEA の α-attainment surface 差分 | 高 stab 領域での確率的優位の視覚証拠 |

### 3.4 補強（オプション）

| 指標 | 目的 |
|---|---|
| TCH sweep での上記指標 | scalarization 方式に依存しない優位性の検証 |
| N=3 vs N=6 の union UEA HV 比較 | weight 数への結論の頑健性（lucky punch 対策） |

---

## 4. 評価方法論

### 4.1 予算設定: 自然収束まで走らせ、anytime で切る

本研究の最重要主張は**速度**だが、実行そのものを CPU 時間で打ち切る設計は取らない。代わりに:

- **実行時は反復数/世代数で自然収束まで走らせる**（ILS `max_iterations=1500`、GA `ngen=500`）
- 履歴（iteration × cpu_time）を保存しておき、**速度比較は anytime curve で任意時刻 T を切り出して行う**
- **品質比較は収束後（最終反復/世代）の値で行う**

| アプローチ | 採否 | 備考 |
|---|---|---|
| 同反復数/世代数で打ち切り比較 | ❌ | iter/gen が等価でない |
| 同 CPU 時間で打ち切り | ❌ | 収束後の最終品質が失われる |
| **自然収束まで走らせ、anytime で CPU 時間断面を切る** | ✅ **本研究の基本** | 速度も最終品質も同一データから取れる |

### 4.2 評価フレームワーク: UEA scenario

Pareto front の構築に **Unbounded External Archive (UEA) scenario** を採用。探索過程の全非劣解を使う（final population scenario の運依存性を回避）。

- **ILS**: LS 訪問点の全評価解
- **GA**: 全世代の全 population

両手法で同一シナリオを採用することが「同一条件比較」の前提。理論的根拠 → [multiobjective_optimization.md §3](theory/multiobjective_optimization.md)

### 4.3 Anytime curve と snapshot

```
履歴（LS 訪問点の時系列）
  ├─ Anytime scalar curve（代表 weights 別）    ← deployment 視点
  ├─ Anytime per-weight UEA HV curve            ← 特定 weight 方向の覆域時系列
  ├─ T 秒 snapshot 表                           ← 要約数値
  └─ T 秒での Pareto front plot                 ← 視覚的比較
```

**weights の扱い**:

| 曲線 | weights の扱い | 曲線の本数/手法 |
|---|---|---|
| Anytime scalar | 代表 weights 各々で計算（平均しない） | **代表 weights 数** |
| Anytime per-weight UEA HV | 同一 weight 内で ILS vs GA | 代表 weights 数 |

### 4.4 per-weight UEA HV の位置付け

N=1 UEA scenario で得た HV は「特定 weight 方向での探索が副次的に発見した非劣解集合の覆域」を測る。

| 使い方 | 妥当性 |
|---|---|
| 同一 weight 内で ILS vs GA の HV 比較 | **妥当**（方向偏りが打ち消し合う） |
| 同一 weight で anytime curve | **妥当** |
| 異なる weight の HV を直接比較 | **不当**（方向偏りが異なる） |
| 「Pareto 全体性能」として主張 | **不当**（1 方向の覆域では front 全体を測れない → union UEA で） |

### 4.5 per-trial union UEA HV（主筋）

各 trial で N weights の UEA を統合 → non-dominated 抽出 → HV。trial 間で中央値 + IQR + Wilcoxon。

**Lucky punch 対策**: union 集約は分散の大きい手法が N 増加で有利になる。対策:
- **N sensitivity check**: N=6（代表 6 weights）と N=11（全 weights）で結論が変わらないことを確認
- **median と IQR を報告**（mean のみ不可）
- **trial 数 30 を目標**（10 でパイロット → 30 に拡大）

### 4.6 Region-restricted HV と高安定性領域分析

#### 領域境界の定義

**全手法の個別 Pareto フロント解の stab 値の P33/P67 で 3 分割**する。

具体的な手順:
1. 各手法・各 trial ごとに、その trial の訪問点から**trial 個別の Pareto フロント**を構築する
2. 全手法・全 trial の個別 Pareto フロント解を **cross-method dominance フィルタなし**で一括プールした集合 S を作る
3. S の stab 値の P33・P67 パーセンタイルを境界とする（**1 実験につき 1 回だけ計算し固定**）

**trial をまたいで固定する理由**: 閾値が trial ごとに変動すると、paired Wilcoxon 比較で「trial i の ILS」と「trial i の GA」を異なる物差しで測ることになり、比較が成立しない。全 trial を一括プールして 1 つの閾値を決め、全 trial・全手法に同一の境界を適用する。

| 領域名 | 範囲 | 意味 |
|---|---|---|
| `low_stab` | stab ∈ [0, P33] | 解集合の下位 1/3（安定性低め） |
| `mid_stab` | stab ∈ (P33, P67] | 解集合の中位 1/3 |
| `high_stab` | stab ∈ (P67, stab_max] | 解集合の上位 1/3（安定性高め） |

**union PF を使わない理由**: cross-method dominance で除外された解（ある手法では Pareto 最適だが別手法の解に支配される解）も stab 範囲の代表点として含めるため。等幅分割（max/3）より各領域に解が均等に入るため HV = 0 を回避できる。

境界値 P33/P67/stab_max の実値は論文に併記する。

#### 領域別 HV の計算

各手法の**全訪問点**を stab でフィルタし、領域内で Pareto 再構築してから HV を計算。参照点は `(init_ms, P_upper + margin)`。

#### 高安定性領域分析の 2 段階設計（B-2b）

```
Step 1: カバー率の比較
  stab ≥ P67 の解を 1 個以上持つ trial の割合
  → 手法間の到達可能性の比較（Fisher's exact test または単純割合比較）

Step 2: 条件付き MS 比較（Wilcoxon）
  Step 1 で解が存在する trial に限定して
  「stab ≥ P67 を満たす解の中の最小 MS」を各 trial から 1 値取得
  → 手法間で paired Wilcoxon signed-rank + Cliff's delta
```

**閾値の感度分析**: P67 固定だけでなく P50・P75 でも同じ分析を行い、「どの閾値でも同じ結論が出る」ことを示す（p-hacking 対策）。

### 4.7 集約レンジ: 全 weights vs fair range

GA は高 stab 重み（目安: stab ≥ 0.3〜0.5）で degenerate するため、集約レンジの選び方で HV の意味が変わる。両ケースを報告:

| ケース | 集約対象 weights | 用途 |
|---|---|---|
| **案 A: 全 weights** | `[1.0,0]〜[0.0,1.0]` 11 点 | メイン。degeneracy 含めた特性差を正直に反映 |
| **案 B: fair range** | 両手法とも動く範囲（例: `[1.0,0]〜[0.7,0.3]` 前後） | 補足。degeneracy 抜きでも ILS が勝つ証拠 |

「ILS は degeneracy の恩恵で勝ってるだけでは？」反論を事前に封じる 2 段構え。

### 4.8 統計検定の設計

| 場面 | 検定 | 備考 |
|---|---|---|
| per-weight ILS vs GA | Wilcoxon signed-rank（paired per-trial） | Cliff's delta で効果量 |
| per-trial union UEA HV | Wilcoxon signed-rank（paired） | median + IQR 報告 |
| 改善成功率（binary） | Fisher's exact test | weight × 手法のヒートマップ |
| 高 stab 領域カバー率 | Fisher's exact test または割合比較 | Step 1: 到達可能性の確認 |
| 条件付き MS 比較（stab ≥ P67） | Wilcoxon signed-rank（paired） | Step 2: Cliff's delta で効果量。P50/P75 でも実施して感度確認 |
| 複数 weight 同時比較 | Holm 補正 | ファミリー = 同一主張に対する検定群 |

---

## 5. GA の degeneracy 問題の扱い

### 現象

GA は stab 重みが高くなる（目安: stab ≥ 0.3〜0.5）と、初期解から動けなくなる degeneracy を示す。

原因:
- 初期解は Stab = 0（定義上の参照点）
- 任意の変化は Stab を悪化させる
- 高 stab 重みでは集団内の全個体が「初期解が最良」に収束
- GA の crossover/mutation は disruptive で脱出能力が低い

### 研究的な扱い

**隠さず、独立した findings として主張化する**。weights 空間を 3 レジームに分けて報告:

| レジーム | GA | ILS | 主張 |
|---|---|---|---|
| MS 偏重 (stab ≤ 0.1) | 探索する | 探索する | 速度で比較 clean |
| 中程度 (0.1 < stab ≤ 0.3) | 弱探索 | 探索する | ILS の構造的優位 |
| stab 偏重 (stab ≥ 0.5) | 固着 | 問題依存で動く | GA の構造的限界 |

### Pareto 比較への影響

固着した重みでは GA の union Pareto は初期解 1 点しか出ないため、ILS の union Pareto の方が広くなる。これは**算法の構造的差異の公平な反映**であり、GA への有利な補正は不要（案 B の fair range 比較は補足として提示）。

---

## 6. 実験一覧

### 実験 1: コア比較（主張 A/B-1/B-2 の同時検証）

**目的**: 速度・品質・頑健性の包括的比較。本研究の中核。

| 因子 | 水準数 | 値 |
|---|---|---|
| 問題 | 4 | mt10, la21, la36, la40 |
| weights | 11 | `[1.0,0], [0.9,0.1], [0.8,0.2], ..., [0.1,0.9], [0.0,1.0]`（0.1 刻み） |
| 手法 | 4 | ILS-baseline, ILS+repair, ILS+PR, GA |
| trial | 30 | seed 固定で再現性確保（lucky punch 対策。まず 10 でパイロット） |
| 予算 | 自然収束 | ILS 800 iter / GA 500 gen。anytime curve で CPU 時間断面を事後抽出 |

**規模**: 4 × 11 × 4 × 30 = **5280 run**（パイロットは 4 × 11 × 4 × 10 = 1760 run）

**出力**:

| カテゴリ | 指標 | 主張 |
|---|---|---|
| 速度 | per-weight anytime scalar curve（代表 3 weights） | (A) |
| 速度 | per-weight anytime UEA HV curve（代表 3 weights） | (A) |
| 速度 | T=5/10/20/40s snapshot + Wilcoxon（paired） + Cliff's delta | (A) |
| B-1 | per-weight scalar 値（Wilcoxon + Cliff's delta, Holm 補正） | (B-1) |
| B-1 | per-weight UEA HV（同一 weight 内で手法間比較） | (B-1) |
| B-1 | 改善成功率ヒートマップ（weight × 手法） + Fisher's exact | (B-1 degeneracy) |
| B-2a | per-trial union UEA HV（median/IQR + Wilcoxon paired） | (B-2a) **主筋** |
| B-2a | C-metric（C(ILS,GA) と C(GA,ILS)） | (B-2a) |
| B-2b | カバー率（stab ≥ P67 の解を持つ trial 割合） | (B-2b) Step 1 |
| B-2b | 領域別 HV（low/mid/high stab, P33/P67 境界） | (B-2b) |
| B-2b | 条件付き MS Wilcoxon（stab ≥ P67, 感度: P50/P75） | (B-2b) Step 2 |
| 視覚化 | 差分 EAF（高 stab 領域を重点的に確認） | (B-2b) 視覚証拠 |
| 感度 | N=3 vs N=6 の union UEA HV 比較 | lucky punch 対策 |

**補強実験（時間余裕次第）**:
- TCH sweep（2 問題 × 3 weights × 5 trial）→ scalarization 方式に依存しない優位性
- WS 凸限界の empirical 確認（1 問題で NSGA-II または TCH sweep の front 形状確認）

### 実験 2: repair パラメータ掃引（前段）

**目的**: 実験 1 で使う `repair_trigger`、`repair_strength` の決定。

| 因子 | 水準数 | 値 |
|---|---|---|
| 問題 | 2 | la36, la40（repair が効く問題） |
| weights | 1 | `[0.8, 0.2]` |
| trigger × strength | 12 | trigger ∈ {10, 30, 50, 100}, strength ∈ {1, 2, 3} |
| trial | 10 | |

**規模**: 2 × 1 × 12 × 10 = **240 run**

**出力**: HV ヒートマップ（trigger × strength）、最適パラメータ。

**実行順**: **実験 1 の前に必ず走らせる**。

### 実験 3: 外乱スケール感度（頑健性検証）

**目的**: 主張が遅延量に依存しないことの確認。

| 因子 | 水準数 | 値 |
|---|---|---|
| 問題 | 2 | la36, la40 |
| 外乱 | 3 | 各問題で遅延を小・中・大（例: delay=100, 150, 200） |
| weights | 2 | `[0.9, 0.1]`、`[0.8, 0.2]` |
| 手法 | 4 | ILS-baseline, ILS+repair, ILS+PR, GA |
| trial | 5 | |

**規模**: 2 × 3 × 2 × 3 × 5 = **180 run**

**優先度**: 中（実験 1/2 完了後）。

### 実験 4: メメティック比較（任意）

**目的**: GA + LS の組合せとの比較。「LS の力だけなのでは？」という反論への対応。

**設計**: 既存 GA 実装に LS を組み込んだメメティック版 vs ILS 系。実験 1 のサブセット問題で実施。

**優先度**: 最低（査読コメント次第）。初期投稿時は省略、必要なら revision で追加。

### 実験 5: NSGA-II 比較（任意）

**目的**: 最新多目的手法との Pareto 覆域比較。

**優先度**: 最低。現状の主張（重み付き ILS vs 重み付き GA）は自己完結しており必須ではない。

---

## 7. 実行順序

```
Step 1: 実験 2（パラメータ掃引）
        └→ repair_trigger, repair_strength 確定

Step 2: 実験 1 パイロット（1 問題 × 5〜10 trial）
        └→ データ分布・実行時間の確認、UEA/anytime 実装の動作確認

Step 3: 実験 1 本格実行（確定パラメータで 30 trial）
        └→ 主張 A/B-1/B-2 の主結果を取得

Step 4: 実験 1 の分析
        - anytime scalar / UEA HV curve
        - per-weight scalar 値 + UEA HV（B-1）
        - 改善成功率ヒートマップ（degeneracy）
        - per-trial union UEA HV + Regional HV + C-metric（B-2）
        - 差分 EAF（視覚化）
        - N sensitivity check
        └→ 主要図表の確定

Step 5: (任意) 実験 3（外乱感度）
Step 6: (任意) 実験 4（メメティック）or 実験 5（NSGA-II）

Step 7: 論文執筆
```

---

## 8. 未決事項

- [ ] **HV 参照点の統一方法**: 複数 weights 集約 Pareto での参照点をどう定義するか（現状 per-(問題, weight) で nadir + マージン）
- [ ] **Trial 数**: 10 でパイロット → 30 に拡大の判断基準（パイロット結果で分散を確認）
- [ ] **GA 側の履歴保存**: 現状 GA は最終解のみ。anytime curve に必要なので拡張が要る（実験 1 の前提条件）
- [ ] **TCH 補強実験の要否**: WS 凸限界の defense として有効。半日〜数日の実装コスト
- [ ] **外乱シナリオの体系化**: 実験 3 では 1 問題 × 3 外乱量。遅延量の決め方（固定 / ジョブ依存 / 相対比率）
- [ ] **実験 1 で ILS-swap 系も入れるか**: insert 主軸で swap は 1-2 weights のみ補助、が現状の方針

---

## 9. 変更履歴

| 日付 | 変更 |
|---|---|
| 2026-04-18 | 議論を反映して全面改訂。主張構造 (A)〜(D)、4 実験構成、anytime + Pareto 併用の評価方法論を確立 |
| 2026-04-19 | 指標分類を refine。速度 / 品質（総合探索力・最終選択値・探索構造・頑健性）の 4 軸に再編 |
| 2026-04-21 | 予算設計を変更: CPU 時間打ち切りを廃止し、自然収束まで走らせて anytime curve で事後抽出する方針に |
| 2026-04-21 | §4.5 Region-restricted HV を具体化: stab 軸 quartile 3 分割、全手法共通 union Pareto から境界を機械的に決定 |
| 2026-05-19 | v3 方法論文書を統合。主張を (A)(B)(C)(D) 4 軸 → **(A)(B-1)(B-2) 3 軸**に再編。UEA scenario 明示、per-weight UEA HV の位置付け整理、lucky punch 対策（trial 30、N sensitivity check）、統計検定設計（paired Wilcoxon + Cliff's delta + Holm）を追加。理論的背景を theory/multiobjective_optimization.md に分離。weights を 0.1 刻み 11 点に変更、手法に ILS+PR を追加 |
| 2026-05-21 | B-2 を B-2a（総合 Pareto 質）と B-2b（高安定性領域での質）に分割。B-2b の指標として領域別 HV・カバー率・条件付き MS Wilcoxon・差分 EAF を追加。§4.6 の領域境界定義を等幅分割から **各手法個別 PF 解集合の P33/P67 分位点**に変更。2 段階分析設計（カバー率→条件付き MS 比較）と感度分析（P50/P67/P75）を明記 |
