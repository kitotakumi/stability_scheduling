# 高安定 HV の閾値（P50）ロバストネス検証

**目的.** 主指標「高安定 HV」は、安定性 D の分割点を **P50 = 全手法・全 trial の Pareto 解の D をプールした中央値** に取る。この分割点は比較手法プールに相対的（＝手法セット依存で動く）なので、査読で「主指標がメソッド依存で定義される恣意的閾値では？」と突かれうる。そこで分割点を **P25 / P33 / P50 / P67 / P75** に掃引し、§4 の結論（H1・H2・スコアボードの二分構造）が閾値に頑健かを確認した。

`analyze_v3` 本体は書き換えず、前処理・`region_hv`・`_friedman_avg_rank` を import して定義を完全共有している（読み取り専用の診断）。

## スクリプトと出力

| スクリプト | 内容 | 出力 |
|---|---|---|
| [`highstab_threshold_sensitivity.py`](highstab_threshold_sensitivity.py) | per-scenario の手法順位＋主要ペア Wilcoxon p / Cliff δ を各閾値で | [`highstab_threshold_sensitivity_output.txt`](highstab_threshold_sensitivity_output.txt) |
| [`highstab_threshold_friedman.py`](highstab_threshold_friedman.py) | §4.4 の横断 **Friedman 平均順位**（`_friedman_avg_rank` と同一定義）を各閾値で再計算 | [`highstab_threshold_friedman_output.txt`](highstab_threshold_friedman_output.txt) |

再実行:

```
python experiments/core_comparison_v3/diagnostics/highstab_threshold_sensitivity.py
python experiments/core_comparison_v3/diagnostics/highstab_threshold_friedman.py
```

（既定 results_dir = `experiments/core_comparison_v3/results/main_v1`、所要 各2〜5分。出力は stdout。）

## 主要結果

### 1. パイプライン妥当性：P50 再計算が公表スコアボードを再現

Friedman 平均順位の P50 列（小さいほど良い）:

| 手法 | Friedman 平均順位 (P50) |
|---|---|
| ILS+repair | 2.62 |
| ILS+PR | 2.75 |
| Memetic+PR | 2.88 |
| Memetic+repair | 3.31 |
| ILS-baseline | 3.44 |
| **GA** | **6.38** |
| **Memetic-LS** | **6.62** |

原稿 §4.4 の記載「ILS 系と機構込み Memetic が首位群に密集（2.6〜3.4）／GA 6.4・Memetic-LS 6.6」と一致。→ 診断が analyze_v3 のスコアボードを正しく再現していることの確認。

### 2. 二分構造は全閾値で不変（Friedman 高度）

| 判定 | 結果 |
|---|---|
| 最下位2手法 | **全閾値で {GA, Memetic-LS}** 不変 |
| 上位5メンバー | **全閾値で {ILS-baseline, ILS+PR, ILS+repair, Memetic+PR, Memetic+repair}** 不変 |
| 完全順序 | 閾値で変動（**上位群の内部順位のみ**入れ替わり） |
| Friedman p | 全閾値で **< 1e-6**（Kendall W = 0.73〜0.90） |

各閾値の完全順序（良い順）:

```
P25: ILS+PR > Memetic+PR > ILS+repair > Memetic+repair > ILS-baseline > GA > Memetic-LS
P33: ILS+repair > Memetic+PR > ILS+PR > Memetic+repair > ILS-baseline > GA > Memetic-LS
P50: ILS+repair > ILS+PR > Memetic+PR > Memetic+repair > ILS-baseline > GA > Memetic-LS
P67: ILS+PR > Memetic+repair > Memetic+PR > ILS+repair > ILS-baseline > Memetic-LS > GA
P75: Memetic+PR > Memetic+repair > ILS+PR > ILS+repair > ILS-baseline > Memetic-LS > GA
```

> 注：高安定スコアボードの**首位は P50 だと ILS 系（ILS+repair）だが、閾値で交代する**（P75 は Memetic+PR）。これは僅差で密集する上位群内の揺れであり、本文 §4.4 の「閾値で動くのは上位群の内部順位のみ」が開示済み。図キャプションの「首位 ILS 系」は P50 の値。

### 3. H1・H2 の高安定比較は逆転しない（per-scenario 検定）

| 比較 | 閾値ロバスト性 |
|---|---|
| **H2**（Memetic+PR/+repair > Memetic-LS） | **全閾値 × 全8シナリオで p=0.001・δ=−1.0**（完全ロバスト） |
| **H1**（ILS-baseline vs Memetic-LS の完全分離） | 方向逆転は**どの閾値でもゼロ**。\|δ\|=1.0 完全分離は **P33〜P67 で全8シナリオ維持**、緩むのは両端の2点のみ（la36M-P75 で δ+0.20、la36L-P25 で +0.40）で、いずれも**向きは不変** |

## 結論

高安定 HV の P50 分割点は手法プール相対だが、**P25〜P75 のどこに取っても §4 の主結論は保たれる**：二分構造（上位群 vs 素の集団の最下位群）は Friedman 高度で不変、H1・H2 の高安定比較は逆転せず、変動するのは僅差の上位群内部順位のみ。→ 本文 §4.4「閾値頑健性」段落の主張は再現可能なスクリプト付きで裏付け済み。
