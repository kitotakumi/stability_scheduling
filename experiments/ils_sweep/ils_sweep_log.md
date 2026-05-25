# ILS パラメータ掃引 実験ログ（Stage 1〜2-A）

> 設計詳細は [doc/ils_parameter_sweep.md](../../doc/ils_parameter_sweep.md)。
> このログは **実行結果の確定値と知見** を記録する。raw データは `results/<stage>_<timestamp>/`。

---

## TL;DR — 確定設定

ILS 本体（Stage 1）と repair 拡張（Stage 2-A）の合計 4 variants を、実験 1（コア比較）の比較対象として確定する：

| variant | perturb | strategy | initial_strength | strength_delta | repair_mode | repair_trigger | repair_strength | 主用途 |
|---|---|---|---|---|---|---|---|---|
| **ILS-swap** | swap | best | 2 | 3 | False | — | — | 大問題（la36 系）|
| **ILS-insert** | insert | best | 2 | 3 | False | — | — | 小・中問題（mt10, la21）|
| **ILS-swap-repair** | swap | best | 2 | 3 | True | 50 | 2 | la36 系で微改善（限定的）|
| **ILS-insert-repair** | insert | best | 2 | 3 | True | 50 | 1 | 提案手法のコア（small で激変、la36 で low_stab 拡張）|

共通設定:
- `max_iterations = 1500`（Stage 1 実測の p_max × 1.5 マージン）
- `active_schedule = False`（N5 理論保証）
- `stagnation_threshold = None`（δ 受理は不利）
- `path_relink_mode = False`（PR は本研究では不採用）

対象問題（パラメータ掃引用）:
- `mt10_delay60`、`la21_delay147`、`la36_delay148`、`la36_multi3_x15`
- **`la21` は飽和傾向**（OFAT で全 config 同結果）→ 補助的位置
- **`la40` 系は除外**（saturation 問題、§5.2 of design doc）

---

## 1. 実行履歴

| 日付 | Stage | 結果ディレクトリ | 規模 | 備考 |
|---|---|---|---|---|
| 2026-04-23 | 1-A 旧 | `stage1a_20260423_192045` | 4 問題 × 8 (perturb×init_strength) × 10 trial = 320 run | 旧設計、後に `perturb × strategy` に変更 |
| 2026-04-23 | 1-B 旧 | `stage1b_20260423_192801` | 4 問題 × 5 × 5 = 100 run | strategy + max_strength の OFAT |
| 2026-04-27 | 1-A | `stage1a_20260427_101456` | 4 問題 × 4 × 10 = 160 run | **新設計** (perturb × strategy 2D) |
| 2026-04-27 | 1-B | `stage1b_20260427_101947` | 4 問題 × 6 × 5 = 120 run | initial_strength + strength_delta OFAT |
| 2026-04-27 | 2-A | `stage2a_20260427_155921` | 3 問題 × 34 × 5 = 510 run | repair (trigger×strength) × variants |

（2026-04-23 の旧 1-A/1-B は探索段階。2026-04-27 のものが確定実験）

---

## 2. Stage 1-A 結果（perturb × strategy）

### 2.1 HV per-trial median ランキング

問題ごとに最良 perturb が異なる：

| 順位 | mt10 | la21 | la36-single | la36-multi3 |
|---|---|---|---|---|
| 1 | **insert_first** 298 | **insert_best** 1109 | **swap_best** 597 | **swap_best** 597 |
| 2 | insert_best 47 | insert_first 1095 | swap_first 590 | swap_first 590 |
| 3 | swap_best 0 ❌ | swap_first 535 | insert_best 561 | insert_first 566 |
| 4 | swap_first 0 ❌ | swap_best 534 | insert_first 559 | insert_best 561 |

❌ 全 trial で init から動けず

### 2.2 主要発見

#### (a) **問題サイズ依存の perturb 最適性**

- 小・中問題 (mt10 10×10, la21 15×10): **insert** が支配的（swap は近傍が枯渇して動けないことすらある）
- 大問題 (la36 15×15): **swap** が +5〜7% で勝つ
- → **「両 variant をメインとして残す」設計が正解**

#### (b) **strategy (FI/BI) の差は perturb 依存だが小さい**

| 問題 | swap: best vs first | insert: best vs first |
|---|---|---|
| mt10 | 0 vs 0（両方詰まり）| 47 vs 298 |
| la21 | 534 vs 535 | 1109 vs 1095（best 1.2% 勝ち）|
| la36-single | 597 vs 590（best 1.3% 勝ち）| 561 vs 559（ほぼ同じ）|
| la36-multi3 | 597 vs 590 | 561 vs 566 |

**la36 では BI/FI の差は ≤1.3%**（実質誤差範囲）。mt10 の outlier (insert_first=298) を除けば **BI 統一で問題なし**。
→ 簡潔さと再現性のため `strategy='best'` で両 variant を統一。

#### (c) **Region-restricted HV — perturb の質的差**

| 領域 | swap の挙動 | insert の挙動 |
|---|---|---|
| low_stab | 21.3 で安定 | 13.4 程度（劣る）|
| mid_stab | 122-127 で安定 | 122 |
| high_stab | **0**（到達できない） | わずかに到達 |

→ swap は low/mid_stab で精緻化、insert は stab 軸方向に若干広く探索する **構造的差** が見える。これは Stage 2-A の repair 効果の解釈にも繋がる。

---

## 3. Stage 1-B 結果（initial_strength + strength_delta OFAT）

base = `(insert, best)` 固定で各軸を振る。

### 3.1 Score ランキング

| 問題 | base | init=1 | init=3 | init=4 | δ=1 | δ=6 |
|---|---|---|---|---|---|---|
| mt10 | 0.984 | 0.968 | **0.939** | 0.953 | 0.968 | **0.937** |
| la21 | 1.030 | 1.030 | 1.030 | 1.030 | 1.030 | 1.030 |
| la36-single | 0.965 | 0.976 | **0.962** | 0.964 | **1.007** | 0.966 |
| la36-multi3 | 0.968 | 0.978 | **0.965** | 0.969 | **1.008** | 0.968 |

太字＝base より良い、下線＝base より劇的に悪い。

### 3.2 主要発見

#### (a) **`initial_strength` の感度は低い**（差 ≤2%）

- 1〜4 で大きな差なし、`initial_strength=2` の慣例値で OK
- la36 で init=3 が僅差で勝つが、効果は小さい（< 1%）

#### (b) **`strength_delta=1` は致命的**（適応強度は必須）

la36 で base 比 +5%、init より悪化することも。**δ=1 = 実質固定強度**で、N5 近傍が枯渇したときに脱出できない。

#### (c) **`strength_delta=6` ≒ `delta=3`**（小問題でやや有利）

- la36 系: δ=3 と δ=6 はほぼ同等
- mt10: δ=6 が明確に勝ち（0.937 vs 0.984）

→ **δ=3 を default、適応の上限を欲しければ δ=6 でも可**。

### 3.3 max_iter 安全値

`convergence_safety_cross.txt` から：

```
problem                 p50    p95    p99  p_max  max_iter
mt10                    111    888    965    966     1000
la21                    553    919    962    971     1000
la36 single             824    987    995    996     1000
la36 multi3             761    967    990    995     1000

全問題横断 max(p95) = 987
推奨 max_iter = 1500 (p_max × 1.5 マージン)
```

→ **`ILS_MAX_ITER = 1500` に設定済み**（[experiment_utils.py](../experiment_utils.py)）。

---

## 4. Stage 2-A 結果（repair grid × ILS variant）

base = ILS-swap (swap+best) と ILS-insert (insert+best) で repair_trigger × repair_strength を grid 振り。

### 4.1 HV ランキング（la36_delay148）

| 順位 | config | HV_med | baseline 比 |
|---|---|---|---|
| 1 | **swap_t50_s2** | 527.0 | swap_baseline (524.5) +0.5% |
| 2 | swap_t100_s2 | 526.7 | +0.4% |
| 3 | swap_t50_s3 | 525.7 | +0.2% |
| ... | | | |
| - | swap_baseline | 524.5 | (基準) |
| - | insert_baseline | 506.6 | (基準) |
| 13 | insert_t30_s1 | 507.5 | insert_baseline +0.2% |
| 32 | swap_t10_s2 | **222.9** | -57%（致命的）|
| 34 | insert_t10_s3 | **38.5** | -92%（壊滅）|

### 4.2 HV ランキング（mt10_delay60）

| 順位 | config | HV_med | baseline 比 |
|---|---|---|---|
| 1 | **insert_t50_s1** | 389.8 | insert_baseline (41.8) **+832%** |
| 2 | insert_t50_s3 | 389.7 | +832% |
| 3 | insert_t100_s3 | 389.5 | +831% |
| - | insert_baseline | 41.8 | (基準) |
| - | swap 全 cell | **0** | swap は変わらず動けない |

### 4.3 Region-restricted HV: low_stab の改善

[evaluation_design.md](../../doc/evaluation_design.md) 主張 (C) 検証の核心：

| la36 / config | low_stab | mid_stab | high_stab |
|---|---|---|---|
| swap_baseline | 21.3 | 122.0 | 0 |
| **insert_baseline** | **13.4** | 121.9 | 0 |
| **insert_t50_s2** | **21.3** ★ | 111.3 | 0 |
| insert_t30_s4 | 18.3 | 92.8 | 0 |

★ **insert+repair は low_stab を 13.4 → 21.3 (+60%) に拡張、swap baseline と同水準まで到達**。
mid_stab で若干劣るので net HV はほぼ変わらないが、**Pareto front の low_stab 端を確実に押し下げる効果が観測**。

mt10 の high_stab：

| config | high_stab |
|---|---|
| insert_baseline | 3.4 |
| insert_t50_s1 | **17.9** (+426%) |

### 4.4 主要発見

#### (a) **repair 効果は variant で両極化**

| variant + 問題 | 効果 |
|---|---|
| **insert + mt10** | HV 9 倍、構造的脱出 |
| **insert + la36** | net HV 同等、ただし low_stab で +60%（Pareto 拡張）|
| **swap + la36** | HV +0.5%（ほぼ無効）|
| **swap + mt10** | 元から動けず、repair も無効 |

→ **repair は insert 用の機構**と理解するのが妥当。swap+repair は実装上残せるが論文では補助。

#### (b) **`trigger=10` は危険**

両 variant・両問題で trigger=10 のセル群が大幅に HV を悪化（最悪 -92%）。
**理由**: 10 反復毎に発動 → 探索を破壊。
**教訓**: trigger ≥ 30、推奨 50。

#### (c) **swap 自体は repair なしで十分**

swap baseline (524.5) に対し、swap+repair で +0.5% しか改善しない。
swap の N5 近傍探索が既に la36 で十分機能しているため、**追加の stagnation escape は不要**。

#### (d) **insert+repair は提案手法のコアとして妥当**

- 小問題で局所最適脱出を実現（HV 9 倍）
- la36 で low_stab Pareto 拡張（+60%）
- 主張 (C) を直接的に支持

---

## 5. 副次観察

### la40 系の除外確定

- la40 は ILS が単一最適解に強収束する saturation 問題
- 単一・多遅延いずれも 5 trial で同一 (MS, St) に到達
- 詳細は [doc/ils_parameter_sweep.md §5.2](../../doc/ils_parameter_sweep.md)

### la21 の飽和

- Stage 1-B で全 OFAT cell が同じ Score 1.030
- Stage 2-A は実行対象から除外（mt10 + la36 系 3 問題のみ）

### Region-restricted HV の per-trial 化

- 旧実装は trial 全体を union → 単一の Pareto front から HV 計算
- 新実装は per-trial median + IQR
- 効果: 「insert_first だけが high_stab=188」のような外れ値依存の見せかけが消え、**統計的に堅実な比較**になった

---

## 6. 出力ファイル一覧（Stage 1-A の例）

```
results/<stage>_<timestamp>/
├── config.json                          # 掃引設定
├── cross_summary.txt                    # 全問題の数値サマリ
├── convergence_safety_cross.txt         # 全問題横断の max_iter 妥当性
└── <problem>_<scenario>/
    ├── results.json                     # 全 config × 全 trial の履歴
    ├── summary.txt
    └── analysis/
        ├── summary_table.txt            # config × HV/Score per-trial 集約
        ├── hv_heatmap.png               # Stage 1-A 専用
        ├── tornado.png                  # Stage 1-B 専用
        ├── repair_heatmap_swap.png      # Stage 2-A 専用
        ├── repair_heatmap_insert.png    # Stage 2-A 専用
        ├── repair_lift.png              # Stage 2-A 専用
        ├── region_restricted_hv.png     # 全 stage
        ├── region_restricted_hv.txt
        ├── acceptance_breakdown.png
        ├── strength_trace.png
        ├── last_improve_iter_cdf.png
        ├── convergence_safety.txt
        ├── anytime_best_score.png
        ├── anytime_best_ms.png
        ├── anytime_best_stab.png
        ├── anytime_hv.png
        └── pareto_overlay.png
```

---

## 7. 実行コマンド（再現用）

```bash
# Stage 1-A: perturb × strategy
python experiments/ils_sweep/run_ils_sweep.py --stage 1a --analyze

# Stage 1-B: initial_strength + strength_delta OFAT (default base = insert+best)
python experiments/ils_sweep/run_ils_sweep.py --stage 1b --analyze
# 別の base で再走したい場合（例: swap+first）
python experiments/ils_sweep/run_ils_sweep.py --stage 1b --base swap,first --analyze

# Stage 2-A: repair grid (両 variant)
python experiments/ils_sweep/run_ils_sweep.py --stage 2a --analyze
# variant 限定
python experiments/ils_sweep/run_ils_sweep.py --stage 2a --variant insert --analyze
```

---

## 8. 次フェーズへの input

実験 1（コア比較、[evaluation_design.md §6](../../doc/evaluation_design.md)）に渡す ILS 設定：

```python
ILS_VARIANTS = {
    'ils_swap':         dict(perturb='swap',   strategy='best',
                             initial_strength=2, max_strength=5,
                             repair_mode=False),
    'ils_insert':       dict(perturb='insert', strategy='best',
                             initial_strength=2, max_strength=5,
                             repair_mode=False),
    'ils_swap_repair':  dict(perturb='swap',   strategy='best',
                             initial_strength=2, max_strength=5,
                             repair_mode=True, repair_trigger=50, repair_strength=2),
    'ils_insert_repair': dict(perturb='insert', strategy='best',
                              initial_strength=2, max_strength=5,
                              repair_mode=True, repair_trigger=50, repair_strength=1),
}
```

実験 1 では上記 4 variants + GA を比較し、主張 (A)〜(D) を検証する。

---

## 9. 変更履歴

| 日付 | 変更 |
|---|---|
| 2026-04-27 | Stage 1-A〜2-A の確定結果をまとめ、初版作成 |
