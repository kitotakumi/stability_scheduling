# ILS パラメータ掃引 設計ドキュメント

> **このドキュメントの役割**: 提案手法 ILS のパラメータ掃引（感度分析）の設計を定義する。「何を振り、何を見て、どう判断するか」を明文化し、[evaluation_design.md](evaluation_design.md) の実験 1（コア比較）に入力するパラメータを確定するための枠組みとする。

---

## 1. 方針

本研究の提案手法は **ILS 本体** であり、`repair` / `path_relink` は本体の上に載せる拡張機構である。したがって掃引は **2 段階** で進める：

| Stage | 対象 | 位置づけ |
|---|---|---|
| **Stage 1** | **ILS 本体のパラメータ** | 提案手法のベースラインを確定する。主軸 |
| **Stage 2** | **拡張機構（repair, path relinking）のパラメータ** | Stage 1 で確定した ILS 本体の上で、拡張機構の最適設定を決める。ネクストステップ |

Stage 1 を先に完結させてから Stage 2 に進む。Stage 2 では Stage 1 の確定設定を固定して拡張機構だけを振る。

### 1.1 分析基盤

掃引の評価には既存の 2 系統の分析基盤を併用する：

- **[experiments/core_comparison/](../experiments/core_comparison/)**: anytime HV curve, Region-restricted HV, C-metric, 差分 EAF, 改善成功率ヒートマップ
- **[experiments/ils_analysis/](../experiments/ils_analysis/)**: 反復トレース（[plot_iteration_trace](../experiments/experiment_utils.py#L136)）, 探索軌跡（[plot_trajectory](../experiments/experiment_utils.py#L162)）, 受理/棄却・摂動種別の記録

解の品質（HV / scalar）と探索の挙動（受理内訳 / 軌跡）を両面から見て判断する。

---

## 2. Stage 1: ILS 本体のパラメータ

ILS の振る舞いを決めるコアパラメータ群。これを確定しないと拡張機構の効果も公平に測れない。

### 2.1 掃引対象

| パラメータ | 候補値 | 定性的な意味 |
|---|---|---|
| `perturb_method` | `swap` / `insert` | 摂動の近傍構造。insert の方が disruptive |
| `strategy` | `best` / `first` | LS の近傍選択規則。FI は確率的で軌跡多様性が増える |
| `initial_strength` | {1, 2, 3, 4} | 初期キック強度。小さいほど集中、大きいほど広域 |
| `strength_delta` | {1, 3, 6} | 適応幅 = `max_strength − initial_strength`。停滞時に何段階まで強度を上げられるか |

**`max_strength` は掃引パラメータではなく `max_strength = initial_strength + strength_delta` として導出する**。こうする理由は次の通り：

- `max_strength` を絶対値で固定してしまうと、`initial_strength=4` のセルは適応幅 `+1`（ほぼ固定強度）、`initial_strength=1` のセルは適応幅 `+4` となり、initial_strength の効果と適応幅の効果が交絡する
- 適応幅 `δ` を明示的に揃える・振るほうが、Stage 1-A と Stage 1-B の役割分担が明瞭になる

### 2.1.1 `max_iterations` の扱い（掃引対象外）

`max_iterations` はパラメータとして振らない。予算は [evaluation_design.md §4.1](evaluation_design.md) の方針通り「自然収束まで余裕ある値」を固定し、速度比較は anytime curve で事後抽出する。

ただし **「余裕ある値」がどこなのかは把握しておく必要がある**。analyze_ils_sweep.py の `convergence_safety_cross.txt` で全問題横断の last_improvement_iter 分布（p50, p95, p99, max）を出力し、判断する。

**現状の知見（2026-04-27 時点、max_iter=1000 で 4 問題 × Stage 1 全体）**：

| 問題 | p50 | p95 | p99 | p_max | max_iter |
|---|---|---|---|---|---|
| la21 | 553 | 919 | 962 | 971 | 1000 |
| la36 (single) | 824 | 987 | 995 | 996 | 1000 |
| la36 (multi3) | 761 | 967 | 990 | 995 | 1000 |
| mt10 | 111 | 888 | 965 | 966 | 1000 |

**1000 でも p99 が 995 など max_iter ギリ**（特に la36）→ **`max_iter = 1500`** に引き上げ（p_max × 1.5 マージン）。anytime curve の尾も切れず収束を観察可能。Stage 1 / Stage 2 / 実験 1 のデフォルトに採用。

### 2.2 掃引戦略

#### Stage 1-A: 主因子の 2D グリッド（本命）

ILS の質的挙動を決定する支配的 2 因子（perturb_method × strategy）を優先的に：

| x 軸 | y 軸 | 目的 |
|---|---|---|
| `perturb_method` (2) | `strategy` (2) | 近傍構造と LS 選択規則の組合せ評価（FI/BI の効果は perturb 依存しうる） |

- 2 × 2 = 4 セル
- `initial_strength` は **2 固定**（Stage 1-B で振る）
- `strength_delta` は **3 固定**（Stage 1-B で振る）→ `max_strength = 5`
- 問題: 4 問題（[ils_sweep](../experiments/ils_sweep/run_ils_sweep.py) の `DEFAULT_PROBLEM_SETS`：mt10, la21, la36-single, la36-multi3）
- 重み: [0.8, 0.2] 固定
- trial: 10
- **規模: 4 × 4 × 10 = 160 run**
- 出力: HV ヒートマップ + Region-restricted HV + 受理内訳 + anytime curves

#### Stage 1-B: OFAT 感度（補助）

Stage 1-A で確定した perturb / strategy をベースとし、定量パラメータ（initial_strength, strength_delta）を 1 軸ずつ振る：

| 因子 | 水準 | 期待される差 |
|---|---|---|
| `initial_strength` | {1, 2, 3, 4} | 初期キック強度の感度 |
| `strength_delta` | {1, 3, 6} | 適応幅。+1=ほぼ固定、+3=中庸（base）、+6=広い |

- 問題: 4 問題
- 重み: [0.8, 0.2]
- trial: 5
- **規模: 4 × (1 base + 3 strength + 2 delta) = 4 × 6 × 5 = 120 run**
- base の (perturb, strategy) は CLI `--base` で上書き可能（例: `--base swap,first`）

→ Stage 1 合計で **約 280 run**。

### 2.3 Stage 1 の成果物

- ILS 本体の確定パラメータ（Stage 2 および実験 1 の入力）
- Tornado plot（OFAT での各因子の影響幅）
- 代表パラメータの受理内訳・軌跡プロット

### 2.4 Stage 1 確定結果（2026-04-27）

#### Stage 1-A の結論

| 因子 | 結論 | 確信度 |
|---|---|---|
| `perturb_method` | **問題サイズ依存**: 大問題 (la36) は `swap`、小・中問題 (mt10, la21) は `insert` | ★★★ |
| `strategy` | **`best` (BI) で統一**。FI vs BI の HV 差は ≤1.3% で実質誤差範囲（mt10 / insert のみ FI 有意だが小問題で実験 1 のメインではない）| ★★ |

**実験 1 (コア比較) の ILS variants**:
- `ILS-swap`: perturb=swap, strategy=best, init=2, δ=3 — 大問題向け
- `ILS-insert`: perturb=insert, strategy=best, init=2, δ=3 — 小・中問題向け

#### Stage 1-B の結論

| 因子 | 確定値 | 確信度 |
|---|---|---|
| `initial_strength` | **2** | ★ | 1〜4 で大差なし、慣例値 |
| `strength_delta` | **3** | ★★★ | δ=1 は致命的（init より悪化）、δ=6 は同等または僅差で勝ち。互換性のため 3 を default |
| `max_iter` | **1500** | ★★ | p_max × 1.5 マージン |

#### 副次的な観察（論文記述用）

- **insert は high_stab 領域への独占的アクセス**を示す（Region-restricted HV で la36/mt10/la21 ともに insert (特に first) のみが high_stab に到達）。これは perturb の質的差異の構造的根拠。
- **swap は low/mid_stab 領域で精緻化、insert は stab 多様性で勝負**という棲み分け。
- **la21 は飽和**（OFAT 全 config で同じ結果）。Stage 2 / 実験 1 では補助的位置に。

---

## 3. Stage 2: 拡張機構のパラメータ

Stage 1 完結後、Stage 1 確定の **2 ILS variants** (`ILS-swap` = swap+best, `ILS-insert` = insert+best) を固定して repair の効果を確認する。

PR (Path Relinking) は repair と機構が重複する（どちらも停滞時のキック）ので、本研究では **repair 一本に絞る方針**。Stage 2-B (PR 掃引) は実施しない（必要なら repair 結果次第で後付け）。

### 3.1 Stage 2-A: Repair パラメータグリッド

#### 掃引対象

| 因子 | 水準 | 数 |
|---|---|---|
| `repair_trigger` | {10, 30, 50, 100} | 4 |
| `repair_strength` | {1, 2, 3, 4} | 4 |
| ILS variant | {ILS-swap, ILS-insert} | 2 |
| 問題 | mt10, la36-single, la36-multi3 | 3 |
| trial | 5 | 5 |

各 variant に **baseline cell**（`repair_mode=False`）を含めて、現状の Stage 1 確定 ILS との直接比較を可能にする。

#### 規模

- (4×4 grid + 1 baseline) × 2 variants × 3 problems × 5 trial = **510 run**
- 想定時間: ~60 分（4 並列、~30s/run）

#### 対象問題（la21, la40 除外）

- **la21**: Stage 1 で完全飽和（全 OFAT 同結果）→ repair でも変化なしと予想
- **la40**: saturation 問題で除外確定（§5.2）

#### 評価指標（per-trial 統一）

| 指標 | repair の主張への対応 |
|---|---|
| **Union HV** (per-trial median) | repair が baseline の HV を改善するか |
| **Region-restricted HV - low_stab** ★ | repair の主目的（安定性側拡張）の効果 |
| Region-restricted HV - mid/high_stab | 他領域への悪影響なし確認 |
| Score (per-trial median + IQR) | 重み付き実用性能 |
| `repair_heatmap_<variant>.png` | trigger × strength の HV / Score ヒートマップ（baseline 越え cell に `*`）|
| `repair_lift.png` | baseline → 最良 grid cell の Region-restricted HV 改善量（low/mid/high）|
| キック発動率・貢献率 | 機構の実効性 |

★ **low_stab 領域での HV 改善が repair の存在意義の主たる証拠**。

#### 実装

- [run_ils_sweep.py](../experiments/ils_sweep/run_ils_sweep.py) `--stage 2a`
- variant 絞り込み: `--variant swap` / `--variant insert` / `--variant both` (default)
- 問題 default: `STAGE_2A_DEFAULT_PROBLEM_SETS` (la21 除外版)

### 3.2 Stage 2-B: PR パラメータ（保留）

メイン提案 = repair なので、PR は実施しない。**もし repair が期待した効果を示さなかった場合**にのみ、PR を代替候補として検討。

### 3.3 Stage 2 の成果物

- ILS variant ごとの確定 repair パラメータ（実験 1 の入力）
- repair 効果の検証: HV / Region-restricted HV での baseline vs grid 比較
- repair メカニズム挙動: キック発動率・貢献率

---

## 4. 判断基準（指標の 3 層化）

単一指標ではなく「集約指標・挙動指標・軌跡指標」の 3 層で判断する。

### 4.1 集約指標（パラメータ間比較の主力）

[analyze_ils_sweep.py](../experiments/ils_sweep/analyze_ils_sweep.py) で出力済みの指標：

| 指標 | 出力ファイル | 用途 |
|---|---|---|
| **Union HV (per-trial median + std)** | `summary_table.txt`, `hv_heatmap.png` | 総合 Pareto 覆域 |
| **Region-restricted HV (per-trial median + IQR)**（stab 3 領域）| `region_restricted_hv.png` / `.txt` | 領域ごとの強み（low/mid/high stab）|
| **Scalar score**（[0.8, 0.2]） | `summary_table.txt` | 実用性能 |
| **Anytime HV / Score / MS / Stab**（反復軸）| `anytime_*.png` | iter ごとの median + IQR 推移 |
| **改善成功率** | （`summary_table` の MS 統計から確認） | init 固着の有無 |
| **収束安全値** | `convergence_safety.txt`, `convergence_safety_cross.txt` | last_improvement_iter の p50/p95/p99/max → max_iter 妥当性 |
| **Stage 2-A repair heatmap** | `repair_heatmap_<variant>.png` | trigger × strength の HV/Score、baseline 越え cell に `*` |
| **Stage 2-A repair lift** | `repair_lift.png` | baseline → 最良 grid cell の Region-restricted HV 改善量 |

**HV はすべて per-trial 計算で統一**（trial ごとに Pareto 抽出 → HV 計算 → trial 間で集約）。trial 間 union を取らないことで lucky 試行に膨らまない、より統計的に堅実な値になる。

**使い分け**: Stage 1-A は HV ヒートマップ + Region-restricted HV が主、OFAT は scalar + HV を併記。Stage 2-A は repair_heatmap + repair_lift がメイン図表。

### 4.2 挙動指標（なぜ差が出るかの説明）

`history` に既に保存されている情報を集計する。集計スクリプトを新規作成する。

| 指標 | 計算 | 用途 |
|---|---|---|
| **受理内訳ヒストグラム** | `accepted=True` のうち `perturb_used` 別の回数 | どの摂動が実際に貢献したか |
| **棄却率** | rejected / total iterations | 摂動強度の妥当性 |
| **キック発動回数** | `perturb_used ∈ {repair, path_relink}` の回数 | 機構の稼働状況（Stage 2 用） |
| **キック貢献率** | キック発動時の best 改善回数 / キック発動回数 | 機構の実効性（Stage 2 用） |
| **強度軌跡** | `strength` の時系列（max_strength 張り付き率） | 適応ルールの妥当性 |
| **Best 改善 iteration 分布** | `best_score` が更新された iteration の箱ひげ | 収束プロファイル |

→ 「差が HV に出た」→「挙動指標で原因を特定」の 2 段で説明できるようにする。

### 4.3 軌跡指標（視覚的説明）

[experiments/ils_analysis](../experiments/ils_analysis/) の既存プロットを流用。

| プロット | 用途 |
|---|---|
| **Iteration trace**（best_ms, best_st × iteration） | 収束プロファイルの視覚比較 |
| **Trajectory plot**（受理/棄却点を MS-stab 平面に散布） | 探索の "歩き方" の比較 |
| **Pareto overlay**（代表パラメータ複数を 1 枚） | 最終的な到達範囲 |

軌跡プロットは **代表パラメータ 3〜4 点を 1 枚に重ねる** と差がわかりやすい。

---

## 5. 基準点（確定）

Stage 1 結果を反映した、Stage 2 / 実験 1 で使う ILS 設定：

| 項目 | ILS-swap | ILS-insert |
|---|---|---|
| `problem` | la36 系（大問題） | mt10, la21, la36 系（小〜大） |
| `weights` | [0.8, 0.2] 等 | 同左 |
| `perturb_method` | `swap` | `insert` |
| `strategy` | `best` | `best` |
| `initial_strength` | 2 | 2 |
| `strength_delta` | 3（max_strength=5） | 3 |
| `max_iterations` | **1500** | **1500** |
| `repair_mode` | False（Stage 2 で確定） | False |
| `path_relink_mode` | False（Stage 2 で確定） | False |

`active_schedule` は理論保証（N5 が semi-active 前提）のため常に `False` で固定。

### 5.1 掃引対象外パラメータのメモ

実装には存在するが、本掃引では振らないパラメータ：

| パラメータ | 扱い | 理由 |
|---|---|---|
| `stagnation_threshold` | `None` で無効化 | **canonical な ILS の受理規則ではない**（Lourenço et al. 2003 の Better / RW / LSMC のいずれにも該当しない δ ベースの独自規則）。もともと PR 機構のサポート用途で実装されたもので、重み付きスコアでは不利であることが別実験で確認済み。機構としてコードには残すが、本研究の評価では常に無効 |
| `accept_delta` | `stagnation_threshold=None` なら参照されない | 上と同じ理由 |
| `active_schedule` | `False` 固定 | N5 近傍の理論保証が semi-active 前提 |

### 5.2 対象問題から除外: la40 系（saturation 問題）

**la40 + 既存初期スケジュール（`la40_delay148.json` 由来）の組み合わせは、本研究のパラメータ掃引から除外する。**

| シナリオ | distinct (MS, St) / 5 trial | 状態 |
|---|---|---|
| la40_delay148 | 1/5 | saturated |
| la40_multi3_x15 | 1/5 | saturated |
| la40_multi5_x15 | 1/5 | saturated |

#### 観察された事実

- ILS は改善を見つける（例: la40_multi3_x15 で 1887 → 1815, 改善 72）
- だが **5 trial 全てが完全に同じ最終 (MS, St) に収束**
- 単一遅延（`la40_delay148`）でも複数遅延（`la40_multi3_x15`, `la40_multi5_x15`）でも同症状
- la36 系は 4/5 distinct で識別力あり

#### 原因（推定）

la40 + この初期スケジュールの post-reschedule 問題に **構造的な単一最適解**が存在する。N5 近傍探索が seed に依らず同じ盆地に落ちる。複数遅延化で reschedule_time を早めて search space を広げても解消せず、シナリオ設計レベルでは対処不能と判断。

#### 対応

- **Stage 1 / Stage 2 の掃引対象から la40 を外す**（[ils_sweep/run_ils_sweep.py](../experiments/ils_sweep/run_ils_sweep.py) の `DEFAULT_PROBLEM_SETS` を la36_delay148 + la36_multi3_x15 に更新済み）
- 実験 1（コア比較、[evaluation_design.md](evaluation_design.md)）でのみ「saturation 状態の例」として残す可能性あり（手法非依存に同じ点に収束する状況の確認）

---

## 6. 出力フォーマット

各 Stage の出力は `experiments/ils_sweep/results/<stage>_<timestamp>/` に保存する。

```
results/<stage>_<timestamp>/
├── config.json                      # 掃引設定（Stage, 因子, 候補値, 基準点）
├── <problem>/
│   ├── results_<config_id>.json     # 全 trial の履歴・最終値
│   └── summary_<config_id>.txt
├── aggregate.json                   # 全設定の集約指標
└── analysis/
    ├── heatmap_<factor_x>_<factor_y>.png      # Stage 1-A, 2-A
    ├── tornado_<baseline>.png                  # Stage 1-B, 2-B OFAT
    ├── acceptance_breakdown_<config>.png       # 挙動指標
    ├── iteration_trace_<config>.png            # 軌跡指標
    ├── trajectory_<config>.png
    └── pareto_overlay_<stage>.png
```

### 論文図表への対応

| 図表 ID（想定） | 種類 | 対応 Stage |
|---|---|---|
| Table S1 | Stage 1-A HV ヒートマップ数値 | Stage 1-A |
| Fig S1 | Tornado plot（Stage 1-B OFAT） | Stage 1-B |
| Table S2 | Stage 2-A repair ヒートマップ数値 | Stage 2-A |
| Fig S2 | 受理内訳比較（代表パラメータ） | 各 Stage |
| Fig S3 | Trajectory overlay（代表パラメータ） | Stage 1-B / 2 |

---

## 7. 実装 TODO

- [ ] **`experiments/ils_sweep/`** 新規ディレクトリ
  - [ ] `run_ils_sweep.py`: Stage 1-A / 1-B / 2-A / 2-B を flag で切り替え実行
  - [ ] `analyze_sweep.py`: 全 trial の history JSON から集約指標・挙動指標を集計
- [ ] **可視化関数**
  - [ ] Tornado plot（OFAT での各因子の影響幅）
  - [ ] Acceptance breakdown stacked bar
  - [ ] HV ヒートマップ（2D グリッド用）

---

## 8. 実行順序

```
Step 1: Stage 1-A（ILS 本体の主因子 2D グリッド）
        └→ perturb_method, initial_strength 確定
        └→ 以降の OFAT 用基準点確立

Step 2: Stage 1-B（ILS 本体の OFAT）
        └→ strategy, strength_delta 確定
        └→ ILS 本体の最終パラメータ確定

Step 3: Stage 2-A（repair パラメータグリッド）
        └→ repair_trigger, repair_strength 確定

Step 4: Stage 2-B（PR パラメータ）
        └→ relink_trigger, pr_ls_strategy 確定

Step 5: evaluation_design 実験 1 を確定パラメータで実行
```

Stage 1 を完結させてから Stage 2 に入る。Stage 1 の結論が出ないうちに拡張機構の掃引に進むと、ILS 本体の影響と拡張機構の影響が切り分けられなくなるため。

---

## 9. 意思決定ルール

各 Stage での「どの値を採用するか」の判断は以下の優先順位：

1. **集約指標の中央値が最良**（HV、scalar、conditional HV）
2. **集約指標の分散が小さい**（trial 間のばらつきが少ない = 頑健）
3. **挙動指標が説明可能**（受理内訳・キック貢献率が定性的に妥当）
4. **軌跡指標で視覚的に納得できる**（極端な固着・振動がない）

**1 と 2 で齟齬がある場合はロバストなセルを優先する**。HV 中央値が 1% 下がっても IQR が半減するならそちらを選ぶ。

---

## 10. 変更履歴

| 日付 | 変更 |
|---|---|
| 2026-04-23 | 初版。ILS 本体の掃引（Stage 1）と拡張機構（repair/PR）の掃引（Stage 2）の 2 段階構成。判断基準を 3 層化（集約・挙動・軌跡） |
| 2026-04-23 | `max_strength` 絶対値掃引を `strength_delta`（適応幅 = max − initial）掃引に変更。Stage 1-A では δ=3 固定で initial の効果だけを分離、Stage 1-B で δ ∈ {1, 3, 6} を掃引する設計へ整理 |
| 2026-04-23 | `check_disturbance` を複数遅延対応に修正（最遅遅延の right-shift 後 end + 1 を reschedule_time とする統一ルール）。多遅延シナリオ（la36/la40 の multi3_x15, multi5_x15）を追加。la40 は ILS が単一最適解に収束する saturation 問題が判明したため掃引対象から除外（§5.2 追記、ils_sweep の `DEFAULT_PROBLEM_SETS` 更新） |
| 2026-04-27 | Stage 1 の構成を再設計: Stage 1-A を **perturb × strategy** の 4 セルに簡略化（initial_strength と strength_delta は固定）、Stage 1-B を initial_strength + strength_delta の OFAT に変更。`max_iter` を 800 → 1000 に引き上げ（4 問題実測で 800 では一部 trial が打ち切り直前まで改善継続、p95×1.2 で算出）。analyze に Region-restricted HV と convergence_safety レポートを追加 |
| 2026-04-27 | Stage 1 確定: ILS variants を `ILS-swap` (swap+best) と `ILS-insert` (insert+best) の 2 種に整理（FI vs BI は HV 差 ≤1.3% で BI に統一）。`max_iter` を 1000 → **1500** に引き上げ（max_iter=1000 でも la36 で p99=995 と限界寸前、p_max×1.5 マージン）。§2.4 に Stage 1 結果サマリ、§5 を確定基準点表に更新 |
| 2026-04-27 | Stage 2-A 設計確定 (510 run): `repair_trigger × repair_strength × variant` グリッド + baseline。問題は mt10 + la36 系 3 つ（la21 除外）、両 ILS variant で repair 効果を比較。Stage 2-B (PR) は repair に集中するため不実施。Region-restricted HV を **per-trial median + IQR** に修正（union から変更、lucky 試行による inflation 排除）。analyze に `plot_repair_heatmap_stage2a` と `plot_repair_lift_stage2a` を実装 |
