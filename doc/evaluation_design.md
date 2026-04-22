# 評価・実験設計

> **このドキュメントの役割**: 再スケジューリング比較研究の評価方法論と実験設計を定義する。「何をどう測るか」「どの実験をどの優先度で走らせるか」の判断根拠を残す。

---

## 1. 研究の主張

本研究は以下を主張する。

| ID | 主張 | 位置付け |
|---|---|---|
| **(A)** | ILS は GA より構造的に高速（同 CPU 時間で優れた解に到達） | **最重要** |
| (B) | ILS（+repair） の Pareto 覆域は GA を凌駕する（CPU 時間予算に依らず、探索力として上回る） | 品質裏付け |
| (C) | repair 摂動は baseline ILS の Pareto を安定性側へ系統的に拡張する | 機構貢献 |
| (D) | ILS は重み設定に対して頑健、GA は高 stab 重みで degenerate | 新規な発見 |

**副次的な観察**:
- MS と安定性は構造的トレードオフ関係にある（PR 実験で確立、repair 実験でも再確認）
- ILS の単一探索軌跡内では「MS 維持で安定性だけ改善」する解は見つからない（post-hoc 修復困難）。ただし GA のような集団探索で得られる可能性は未検証。

---

## 2. 独立変数（実験で振るもの）

| レベル | 変数 | 値域 | 振る目的 |
|---|---|---|---|
| **アルゴリズム** | 手法 | ILS（±repair）, GA, (メメティック) | 手法間比較 |
| 問題 | JSSP instance | mt10, la21, la36, la40 | サイズ・種類への一般性 |
| 外乱 | 遅延シナリオ | `_delay{60, 100, 147, 148, ...}` | 外乱規模への頑健性 |
| 目的関数 | weights = [eff, stab] | `[1.0,0], [0.9,0.1], [0.8,0.2], [0.7,0.3], [0.5,0.5], [0.3,0.7]` | 重み依存性・Pareto 集約 |
| 算法パラメータ | `repair_trigger`, `repair_strength` | 掃引で決定 | repair 効果最大化 |
| 予算 | 反復数 / 世代数 | ILS `max_iterations`, GA `ngen`（いずれも自然収束まで余裕ある値） | 収束後の最終品質を比較。速度は anytime curve で任意時刻を切り出す |

### 手法の説明

| 手法 | 説明 | 位置付け |
|---|---|---|
| ILS-baseline | 重み付き ILS（swap or insert 摂動） | ベースライン |
| ILS+repair | baseline + 停滞時に repair キック（P-1） | 提案手法 |
| GA | 重み付き GA（既存実装） | 主比較対象 |
| メメティック（任意） | GA + LS（必要に応じて） | 査読対応用、優先度低 |

### 固定する変数（実験済み・根拠確定）

| 変数 | 固定値 | 根拠 |
|---|---|---|
| `strategy` (LS戦略) | `'best'` | FI vs BI で有意差なし（experiment_plan 2026-04-15） |
| `active_schedule` | `False` | N5 の理論保証が semi-active 前提 |
| `stagnation_threshold` | `None` | 悪化受理は重み付きスコアで不利（2026-04-14） |
| `taillard_acceleration` | `True` | 合成スコア下界フィルタで正しく動作（2026-04-14 修正後） |
| ILS `max_iterations` | `800`（自然収束まで余裕ある値） | 収束後の最終品質を測りたいので CPU 打ち切りはしない |
| GA `ngen`, `pop_size` | `500`, `50`（自然収束まで余裕ある値） | 同上 |
| 正規化方式 | min-max（問題・外乱ごと共通パラメータ） | `experiment_utils.compute_shared_norm_params` |

---

## 3. 従属変数（測定するもの）

### 3.1 速度

| 指標 | プロット形式 | 見る対象 |
|---|---|---|
| 実 CPU 時間 × HV（full） | anytime curve | 総合品質到達の速度 |
| 実 CPU 時間 × HV（conditional: 低 stab 領域） | anytime curve | 安定性方向探索の速度 |
| 実 CPU 時間 × scalar 値 | anytime curve（代表 weights） | deployment 視点の速度 |

**評価の仕方**: 同一 CPU 時間での断面（T = 5, 10, 20, 40 秒等）で各指標を比較。

### 3.2 品質

#### 総合探索力
- **HV（per-trial 平均, union）**: Pareto 覆域のスカラー集約
- **C-metric（per-trial, union）**: dominance 関係の直接測定（参照点不要）

#### 最終選択値（= 実用性）
- **scalar 値**（代表 weights）: 重み付きスコア
- **メイクスパン（MS）**: 効率性
- **安定性評価値（Stab）**: 初期解との順位偏差

#### 探索構造
- **Region-restricted HV**（領域別 HV）: stab 軸 quartile で 3 分割した各領域（低/中/高 stab）内の HV。詳細は §4.5
- **attainment surface**（MS 予算別 stab 到達深度）: 50% 等の quantile で描く曲線。"MS = X での stab 到達深度" が見える
- **差分 EAF**（領域別優劣の視覚化）: 領域ごとに A/B のどちらが有利か色分け。定量化は Region-restricted HV で代替可能（数値としては等価）

#### 頑健性
- **改善成功率**: trial 中で「init 解から動けなかった（= 最終 MS = init_MS）」比率。GA の degeneracy を定量化

---

## 4. 評価方法論

### 4.1 予算設定: 反復数/世代数で自然収束、速度は anytime で切る

本研究の最重要主張は**速度**だが、実行そのものを CPU 時間で打ち切る設計は取らない。代わりに:

- **実行時は反復数/世代数で自然収束まで走らせる**（ILS `max_iterations=800`, GA `ngen=500`）
- 履歴（iteration × cpu_time）を保存しておき、**速度比較は anytime curve で任意時刻 T を切り出して行う**
- **品質比較は収束後（最終反復/世代）の値で行う**

この設計の利点:
- 同一実行から「速度」と「最終品質」の両方が得られる（再実行不要）
- 各手法が自分のペースで収束するので、中途半端な CPU 時間打ち切りでランダムに順位が入れ替わる現象を避けられる
- Snapshot（T=5/10/20/40 秒）はすべて anytime 履歴から事後抽出可能

| アプローチ | 採否 | 備考 |
|---|---|---|
| 同反復数/世代数で打ち切り比較 | ❌ | iter/gen が等価でない |
| 同 CPU 時間で打ち切り | ❌ | 収束後の最終品質が失われる |
| **自然収束まで走らせ、anytime で CPU 時間断面を切る** | ✅ **本研究の基本** | 速度も最終品質も同一データから取れる |

### 4.2 Anytime curve と snapshot

履歴保存機構を活用し、同一データから以下を生成:

```
履歴（LS 訪問点の時系列）
  ├─ Anytime HV curve (full)          ← 総合品質の時系列
  ├─ Anytime HV curve (conditional)   ← 領域別品質の時系列
  ├─ Anytime scalar curve             ← deployment 視点
  ├─ T 秒 snapshot 表                 ← 要約数値
  └─ T 秒での Pareto front plot       ← 視覚的比較
```

**Snapshot 時刻**: `T ∈ {5, 10, 20, 40 秒}` 等を複数点取り、収束の時系列を示す。

**weights の扱い**: 指標ごとに異なる。

| 曲線 | weights の扱い | 曲線の本数/手法 |
|---|---|---|
| Anytime HV (full) | 全 weights × 全 trial を集約 → union Pareto → HV | **1 本** |
| Anytime HV (conditional) | 同上（領域限定で計算） | 1 本 |
| Anytime scalar | 代表 weights 各々で計算、平均しない | **代表 weights 数** |

HV は多目的集約指標なので全 weights のデータを混ぜて 1 本の曲線にする。scalar は weights 依存なので代表 weights（例: `[0.9, 0.1]`, `[0.8, 0.2]`）各々で独立プロット。

### 4.3 品質評価: 多 weights 掃引 + Pareto 集約

**アルゴリズムは重み付き scalar のまま**（ILS も GA も）、**評価段階で多 weights の結果を集約して Pareto front を作る**。

```
1 weights × trial × 10 の最終解を比較（単純）
  ↓
weights × 5〜7 × trial × 10 の全解を履歴から抽出
  ↓ per-trial Pareto 抽出
  ↓ 手法ごとに union Pareto 作成
HV, EAF, C-metric で比較
```

これにより:
- アルゴリズムは無改造（PILS 化等不要）
- weights 依存性を封じた品質評価
- GA も重み付きのまま評価 =**「重み付き vs 重み付き」の土俵を維持**

#### 集約レンジ: 2 ケース併記

GA は高 stab 重み（目安 stab ≥ 0.3）で degenerate するため、**集約レンジの選び方で HV の意味が変わる**。両方を報告する:

| ケース | 集約対象 weights | 用途 |
|---|---|---|
| **案 A: 全 weights 集約** | `[1.0,0], [0.9,0.1], [0.8,0.2], [0.7,0.3], [0.5,0.5], [0.3,0.7]` 等 6 点 | メイン比較。algo の特性差（degeneracy 含む）を正直に反映 |
| **案 B: fair range 集約** | 両手法とも動く範囲、例: `[1.0,0], [0.9,0.1], [0.8,0.2], [0.7,0.3]` の 4 点 | 補足比較。degeneracy 抜きでも ILS が勝つことの証拠 |

**論文構成上の位置付け**:
- メイン表: 案 A の union HV → 「ILS の方が Pareto が広い（degeneracy の帰結も含めて）」
- 補足表: 案 B の union HV → 「degeneracy 効果を差し引いても ILS が勝つ」
- 「ILS は degeneracy の恩恵で勝ってるだけでは？」反論を事前に封じる 2 段構え

### 4.4 HV と EAF の使い分け

数学的に `∫∫_R α(p) dp = E[HV_R(trial)]`（領域 R 内で同値）。

| 道具 | 用途 |
|---|---|
| **HV** | スカラー集約（テーブル、曲線の Y 軸） |
| **EAF（差分プロット）** | 視覚的説明（「どこで勝ってるか」） |
| 個別 EAF (0-1) | 差分 EAF の 0-0 と 1-1 を区別する補助 |

**定量化は HV 系で統一**、**視覚化で差分 EAF を併用**、が実用的。

### 4.5 領域別の分析（安定性方向優位の主張）: Region-restricted HV

主張 (B) を支える領域別指標。一般名は **Region-restricted HV**（領域限定 HV）。

| 指標 | 計算 | 用途 |
|---|---|---|
| **Region-restricted HV**（低/中/高 stab 領域） | stab 軸 quartile で 3 分割した各領域内の HV | **定量**（主力） |
| **Attainment surface 差分（MS 軸）** | `Δ_50(ms) = A_50(ms) - B_50(ms)` の曲線 + 符号別面積 | 補助の定量 + 可視化 |
| **差分 EAF プロット** | 領域ごとの A/B 優劣を色分けしたヒートマップ | **可視化のみ** |

**差分 EAF の定量化は Region-restricted HV と等価なので別途行わない**（数学的には `∫∫_R (α_A − α_B) dp = E[HV_R(A)] − E[HV_R(B)]`）。差分 EAF は視覚的説明に特化させる。

#### Region-restricted HV の具体計算

**領域分割（stab 軸 quartile, 全手法共通）**:

1. **全手法の全 trial 訪問点を結合**し、non-dominated set（= cross-method union Pareto）を抽出
2. union Pareto の **stab 軸で Q1 (25%)、Q3 (75%)** を計算
3. 3 領域に分割:

   | 領域名 | 範囲 | 意味 |
   |---|---|---|
   | `low_stab` | `stab ∈ [0, Q1]` | 安定性重視（MS 維持しつつ順序変更小） |
   | `mid_stab` | `stab ∈ (Q1, Q3]` | 中間（トレードオフの中核） |
   | `high_stab` | `stab ∈ (Q3, stab_max]` | 効率性重視（順序大幅変更で MS 改善） |

**各手法の各領域での HV 計算**:

1. その手法の union Pareto から、領域 R 内の点だけをフィルタ
2. 参照点を `(init_ms, R_upper)` に固定（R_upper は領域の stab 上限）
3. 通常の 2D HV 公式を適用

**設計上の利点**:
- quartile 境界が全手法共通データから機械的に決まる → 恣意性最小化
- 参照点も問題ごとに自動（`init_ms` は JSSP 再スケジューリングでの自然な上限）
- 「低 stab 領域での ILS の優位」という主張 (B) が定量値で直接出る
- 複数 weights の場合も「全 weights × 全手法の union Pareto」で quartile を取る（weights も横断）

**境界値の報告**: 論文・レポートには `Q1, Q3, stab_max` の実値も併記する（問題ごとに異なるため）。

### 4.6 EAF の実装上の注意

- per-trial Pareto front を抽出してから集約（最終解 1 点ベースは偏る）
- グリッドは `init_ms` まで伸ばす（未描画白と同値白を区別）
- Stab 軸は 0 を必ず含める（初期解参照点を可視化）
- 軸範囲は Pareto 点 + `init_ms` に引き締める（exploration 外れ値で拡大しない）

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

多 weights 掃引で GA の Pareto を作ると、固着した重みでは初期解 1 点しか出ないため、結果として ILS の union Pareto の方が広くなる。これは**算法の構造的差異の公平な反映**であり、GA に有利な補正は不要。

---

## 6. 実験一覧

### 実験 1: コア比較（主張 A/B/C/D の同時検証）

**目的**: 速度・品質・頑健性の包括的比較。本研究の中核。

| 因子 | 水準数 | 値 |
|---|---|---|
| 問題 | 4 | mt10, la21, la36, la40 |
| weights | 5〜7 | `[1.0,0], [0.9,0.1], [0.8,0.2], [0.7,0.3], [0.5,0.5], [0.3,0.7]` |
| 手法 | 3〜5 | ILS-insert, ILS-insert+repair, GA, (+ ILS-swap 系) |
| trial | 10 | seed 固定で再現性確保 |
| 予算 | 自然収束 | ILS 800 iter / GA 500 gen（余裕ある値で自然収束）。anytime curve で CPU 時間断面を事後抽出 |

**規模**: 4 × 6 × 4 × 10 ≈ **1000 run**

**出力**:
- 速さ: anytime HV curve（full / conditional）, anytime scalar curve
- 品質: union Pareto per method, HV per trial 分布, C-metric 表
- 探索構造: 差分 EAF, attainment surface 曲線, Region-restricted HV（stab quartile 3 分割）
- 頑健性: 重み × 手法の「init 解固着率」ヒートマップ
- Snapshot: T = 5, 10, 20, 40 秒での各指標

### 実験 2: repair パラメータ掃引（前段）

**目的**: 実験 1 で使う `repair_trigger`, `repair_strength` の決定。

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
| weights | 2 | `[0.9, 0.1]`, `[0.8, 0.2]` |
| 手法 | 3 | ILS-insert, ILS-insert+repair, GA |
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

Step 2: 実験 1（コア比較）を確定パラメータで実行
        └→ 主張 A/B/C/D の主結果を取得

Step 3: 実験 1 の分析
        - anytime HV curve（full / conditional）
        - anytime scalar curve
        - Snapshot 表・Pareto plot
        - 差分 EAF, attainment surface, Region-restricted HV
        - 改善成功率ヒートマップ（degeneracy）
        - C-metric 表
        └→ 主要図表の確定

Step 4: (任意) 実験 3（外乱感度）
Step 5: (任意) 実験 4（メメティック）or 実験 5（NSGA-II）

Step 6: 論文執筆
```

---

## 8. 主要アウトプット（想定図表）

| ID | 種類 | 内容 | 主張への貢献 |
|---|---|---|---|
| Fig 1 | anytime curve | CPU 時間 vs HV（full）(手法別) | (A) 速度 + (D) 品質 |
| Fig 2 | anytime curve | CPU 時間 vs HV（conditional: 低 stab 領域） | (A) + (B) |
| Fig 3 | anytime curve | CPU 時間 vs scalar score（代表 weights） | (A) deployment 視点 |
| Fig 4 | Pareto snapshot | T=10s, 40s での各手法の union Pareto | (D) 品質 |
| Fig 5 | 差分 EAF | +repair vs base（安定性側領域の視覚化） | (B) |
| Fig 6 | attainment surface | 50% AS の MS 別 stab 到達深度（手法別）| (B) 方向別比較 |
| Fig 7 | 重み × 手法ヒートマップ | 改善成功率（degeneracy） | (C) |
| Fig 8 | 外乱感度（任意） | 外乱規模 vs 各手法の性能 | 頑健性 |
| Table 1 | Snapshot 数値 | T=5/10/20/40 秒での HV, scalar, Stab | (A)(D) 総括 |
| Table 2 | Union HV 一覧 | 問題 × 手法の union HV（full / conditional） | (D) 品質 |
| Table 3 | C-metric 表 | 手法間 dominance（T_final 時点） | (B)(D) |

---

## 9. 実装状況

| 要素 | 実装状況 | 備考 |
|---|---|---|
| 履歴保存（LS 訪問点） | ✅ 完了 | `run_repair_perturb_experiment.py` が `(ls_ms, ls_st, accepted)` を JSON 保存 |
| per-trial Pareto 抽出 | ✅ 完了 | `analyze_eaf.py` の `pareto_front_2d` |
| 差分 EAF + 個別 EAF | ✅ 完了 | `analyze_eaf.py` |
| HV（per-trial, union） | ✅ 完了 | `analyze_eaf.py` |
| C-metric | ✅ 完了 | `analyze_eaf.py` |
| Anytime scalar curve | ⚠️ 部分 | `fi_vs_bi` にあり、コア実験用に再利用 |
| Anytime HV curve（full） | ❌ 未 | 実験 1 分析で新規実装 |
| Anytime HV curve（conditional） | ❌ 未 | 領域指定 + 時系列 |
| Snapshot Pareto plot | ❌ 未 | 任意時刻で per-trial Pareto 抽出 |
| Attainment surface 曲線 | ❌ 未 | 50% AS の 1D プロット |
| Attainment surface 差分 | ❌ 未 | 2 手法 AS の差分曲線 + 積分 |
| Region-restricted HV（領域別） | ❌ 未 | stab 軸 quartile 3 分割、参照点 `(init_ms, R_upper)` |
| 改善成功率ヒートマップ | ❌ 未 | degeneracy 可視化 |
| 多 weights 掃引 + Pareto 集約 | ❌ 未 | 実験 1 用分析スクリプト新規 |
| GA の anytime 対応 | ❌ 未 | GA 側も履歴保存機構の拡張が必要 |
| メメティック実装 | ❌ 未 | 任意、優先度低 |

---

## 10. 未決事項

- [ ] **HV 参照点の統一方法**: 現状 per-(問題, weights) で nadir + マージン。複数 weights 集約 Pareto での参照点をどう定義するか。
- [ ] **実験 1 で ILS-swap 系も入れるか**: insert 主軸で、swap は 1-2 weights のみ補助、が現状の方針。
- [ ] **外乱シナリオの体系化**: 実験 3 では 1 問題 × 3 外乱量。遅延量の決め方（固定 / ジョブ依存 / 相対比率）。
- [ ] **メメティック実装の要否**: 査読コメント次第で後付け。
- [ ] **GA 側の履歴保存**: 現状 GA は最終解のみ。anytime curve に必要なので対応が要る。

---

## 11. 変更履歴

| 日付 | 変更 |
|---|---|
| 2026-04-18 | 議論を反映して全面改訂。主張構造 (A)〜(D)、4 実験構成、anytime + Pareto 併用の評価方法論を確立 |
| 2026-04-19 | 指標分類を refine。速度 / 品質（総合探索力・最終選択値・探索構造・頑健性）の 4 軸に再編。メメティックを手法候補に追加（優先度低）。領域別分析（conditional HV, attainment surface 差分）を明文化 |
| 2026-04-19 | 指標の重複整理: 差分 EAF は可視化のみと明記（conditional HV と数値的に等価のため）。§4.2 に weights の扱い表追加（HV は全集約 / scalar は代表 weights 別）。§4.3 に品質評価の集約レンジ 2 ケース（案 A: 全 weights / 案 B: fair range）を明記 |
| 2026-04-21 | 予算設計を変更: CPU 時間打ち切りを廃止し、自然収束まで走らせて anytime curve で CPU 時間断面を事後抽出する方針に。§2/§4.1/§6 を更新、§10 の T_max 未決項目を削除 |
| 2026-04-21 | §4.5 領域別分析を具体化: 名称を「Region-restricted HV」に統一、stab 軸 quartile 3 分割（低/中/高 stab）、全手法共通 union Pareto から境界を機械的に決定、参照点 `(init_ms, R_upper)` を明文化。§3/§6 の用語も統一 |
