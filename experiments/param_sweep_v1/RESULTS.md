# param_sweep_v1 実験結果記録

OAT感度分析（la21_delay147 / la36_large、各 重み6点 × n=10）の結果と、それを踏まえた**確定デフォルト値**の記録。指標は per-trial union UEA HV（中央値）、検定は Wilcoxon(two-sided, vs center)＋Cliff's δ。

---

## 1. 確定デフォルト値（2026-06-12 確定）

**当初の center 設定から変更するのは memetic+PR の `pr_ls_top_k` (1→3) のみ。** 他の全軸は center 値を維持する（感度分析で有意改善が無い／ベースライン限定／コスト過大のいずれか）。

| パラメータ | 対象手法 | 確定値 | 当初center | 判断根拠 |
|---|---|---|---|---|
| **pr_ls_top_k** | **memetic+PR** | **3** ← 変更 | 1 | la21有意+(p=0.002 δ=+1.00)・la36も中央値+(ns)・コストはkに線形でO(d²)非増幅（§4） |
| pr_ls_top_k | ILS+PR | 1 | 1 | la21 ns（ILSでは効かない） |
| pr_step_strategy | 全PR系 | random | random | best: la21は品質+(δ=+0.97)だがla36で**18倍コスト**＝実用不可（§5） |
| mutpb | GA | 0.1 | 0.1 | regime B（文献標準値固定）。GAは0.2志向だがmemeticは頑健＝ベースライン限定効果 |
| cxpb | GA | 0.85 | 0.85 | regime B / 両方向 ns |
| pop_size | GA・memetic | 50 | 50 | regime B。memetic-LSは30/80とも ns（頑健） |
| kick_prob | memetic | 0.3 | 0.3 | 0.1〜0.7 全て ns または負方向 |
| relink_trigger / repair_trigger | ILS | 10 | 10 | 5/20/40 全て ns |
| kick_trigger_first | ILS | 400 | 400 | 100/200/600 全て ns |
| repair_strength | memetic・ILS | 0 | 0 | 2/4/8 は負方向（悪化） |
| max_strength | ILS-baseline | 5 | 5 | 3/8 とも ns |
| perturb | ILS-baseline | insert | insert | swap は壊滅（p=0.002 δ=-1.00） |
| initial_strength | ILS | 2 | 2 | 掃引対象外 |

> **下流への含意**: memetic+PR を k=3 に変えるため、論文の主比較（core_comparison_v3）の memetic+PR は **k=3 で再取得**が必要。感度分析自体は center=1 のまま「k=1→3 の効果」として提示してよいが、報告するヘッドライン構成は k=3 に揃える。memory の旧決定「既定 k=1 維持」は本決定で上書き。

---

## 2. 手法比較（union HV 中央値）

各問題の sweep 内では norm_params 共通＝HV比較可。問題間はHVスケールが異なるため**問題内**で比較すること。

| 手法・変種 | la21 (ref 1538,84.8) | la36 (ref 1669,157.6) |
|---|---|---|
| GA (center) | 4559.9 | 10366.2 |
| Memetic-LS (center) | 5019.9 | —※ |
| ILS-baseline | 5387.7 | 14093.7 |
| ILS+PR | 5387.7 | 14145.2 |
| ILS+repair | 5407.7 | 14211.8 |
| **Memetic+PR（center, k=1）** | **5332.2** | **15983.9** |
| Memetic+repair (center) | 5438.2 | —※ |
| ─ Memetic+PR **top_k=3**（採用） | **5438.2** | **16059.4** |
| ─ Memetic+PR **BI(best)** | 5388.2 | 掃引中止（18倍コスト, §5） |

※la36 sweep には memetic_ls / memetic_repair の掃引軸が無い。両者込みの正式な手法間比較は core_comparison_v3（norm_params別バッチ＝HV絶対値は本記録と非互換）。修論の根幹 memetic+PR/repair > LS は別途 p=0.002・10/10 で確定済。

### 感度 vs 手法間比較の規模感
手法間の差は **+13〜54%（la36）/ +7〜19%（la21）**。一方 top_k / BI の感度は **+0.5〜2%** ＝ **約1桁小さい**。

| 比較 | la21 ΔHV | la36 ΔHV |
|---|---|---|
| GA → Memetic+PR **[手法間]** | +772 (+16.9%) | +5618 (+54.2%) |
| ILS+PR → Memetic+PR **[手法間]** | (同水準) | +1839 (+13.0%) |
| Memetic-LS → Memetic+PR **[根幹]** | +312 (+6.2%) | — |
| center → top_k=3 *[感度]* | +106 (+2.0%, p=0.002) | +75 (+0.47%, ns) |
| center → BI *[感度]* | +56 (+1.1%) | （計算不可） |

→ top_k/BI を採否しても主結果（Memetic+PR ≫ GA/ILS, > LS）の序列は不変。感度は誤差レベル。

---

## 3. 感度分析サマリ（軸別・両問題）

- **有意プラス（提案手法）**: `pr_ls_top_k`（memetic, la21のみ有意）, `pr_step_strategy=best`（memetic, la21のみ有意）。→ §1の判断で top_k のみ採用。
- **ベースライン限定**: `mutpb`（GA: la36有意/la21閾値手前で 0.2志向, 0.05は悪化）, `pop_size`（GA: 30で悪化／memeticは頑健）。→ regime B で標準値固定。
- **全て ns**: `relink/repair trigger`, `kick_trigger_first`, `cxpb`, ILS の `pr_ls_top_k`・`max_strength`。
- **負方向**: `repair_strength`（>0で悪化）, `kick_prob`（>0.3 で負）, `perturb=swap`（壊滅）。

「掃引軸は両問題で概ね center 頑健」という全体像（kick_rtb_pr_noop）は維持。例外的に効くのが memetic の top_k（採用）。

---

## 補遺A: top_k のコスト構造（O(d²)非増幅・kに線形）

`path_relinking`（ils_scheduling.py:1104-1120）の top-k は:
1. 経路（path_intermediates）の構築 = O(d²) デコードを**1回だけ**実施（kに依存しない・共有）
2. `heapq.nsmallest(k, ...)` で上位 k 中間解を選択
3. 各に `local_search` を **k回**適用し最良を返す

→ 追加コストは「LSを (k−1) 回 余分に呼ぶ」だけで **k に線形**。外乱規模 d に比例して爆発する O(d²) 経路コストは**増幅しない**。実測 +26%（la36）は素直な固定オーバーヘッドで、問題規模に対しスケールする。BI（§5）が O(d²) を増幅して18倍になるのとは機構が異なる。

---

## 補遺B: pr_step_strategy=best (BI) の計算時間調査（2026-06-12 実測）

### 結論
**la36 で memetic_pr の `pr_step_strategy=best` は random(center) の約8倍重く、1試行あたり約90分（80〜100分）。** 本番（4問題×n=10-15）の既定にすると la36 系だけで計算破綻するため **random 維持・la36 BI 掃引は中止**。

### 単体計測（10世代）
- 条件: la36/la36_large, weights=[0.8,0.2], seed=7, kick_prob=0.3, pop_size=50, pr_ls_top_k=1
- 10世代 solver計測CPU = 120.7s → **12.1 s/gen** → 500世代換算 **約100分/試行**

| 戦略 | s/gen | 1試行(500gen) |
|---|---|---|
| center (random) 実測 | 1.27 | 634s ≈ 10.6分 |
| best 計測 | ~10〜12 | ~6000s ≈ **約90〜100分** |

→ best/random ≈ 約8倍。la36_large は外乱大→PR diffs多→O(diffs²)デコードが効き、best は経路上の全候補手をデコード評価するため大問題で爆発。

### la36 vs la21 の所要時間差（best・重み w10=[1.0,0.0]）
重み w10（makespan単独）が最も重く、問題サイズ差がそのまま時間差になる。

| 問題 | n | 平均CPU | レンジ |
|---|---|---|---|
| **la36_large** w10 | 3 | **12,778 s ≈ 3.55時間/試行** | 11,717 / 13,159 / 13,456 s |
| la21_delay147 w10 | 10 | **711 s ≈ 11.9分/試行** | 664〜767 s |

- **la36 は la21 の約18倍**（12,778 / 711 ≈ 18x）。1試行あたり **約12,000〜12,700秒（~3.4時間）も余分**。
- 重み依存も激烈: la21内でも w02_08=104s → w10_00=711s と**6.8倍**。makespan重視ほど PR経路(diffs)が伸び O(diffs²) が効く。la36_large は外乱大でこの効果が約18倍に増幅。
- 論文での扱い: 「best は小外乱(la21)で僅かに品質+(δ=+0.97)だが、large instance では O(d²) デコードが外乱規模で増大し ~18倍コスト。random を採用し scalability を確保」と記載。
