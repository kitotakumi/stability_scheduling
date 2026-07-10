# stability_scheduling — 安定性を考慮したジョブショップ再スケジューリング

予測リアクティブな**ジョブショップ再スケジューリング (JSSP rescheduling)** において、
**効率性（メイクスパン）**と**安定性（外乱前スケジュール $S_p$ からの変更量の小ささ）**を
同時に最適化する多目的最適化の研究コードです。

軌道ベース探索（**ILS**）・集団ベース探索（**Memetic GA**）・単純な **GA** を、
局所探索（N5 近傍）を揃えた統制条件で比較し、さらに解を $S_p$ へ引き戻す
**安定性誘導演算子（repair / path relinking）**を提案・評価します。

> **研究の主張**: 同一の安定性誘導演算子でも、それが埋め込まれる探索構造（集団 vs 軌道）に依存して
> 効果が**非対称**に現れる。軌道探索（ILS）は連続変形で高安定領域を自力充填する一方、
> 集団探索（Memetic）は交叉ゆえ充填が構造的に粗く、PR/repair による補完が大きく効く。
> 詳細は投稿論文 [`doc/apiems2026_manuscript.md`](doc/apiems2026_manuscript.md)（APIEMS 2026 投稿版）を参照。

---

## 動作環境

- Python 3.12
- 依存ライブラリ: `numpy`, `scipy`, `matplotlib`, `deap`

```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
# bash
source .venv/Scripts/activate

pip install numpy scipy matplotlib deap
```

---

## リポジトリ構成

### コアモジュール（リポジトリ直下）

| ファイル | 役割 |
| --- | --- |
| [`job_shop_scheduling.py`](job_shop_scheduling.py) | 問題定義（`problems/*.txt`）とシナリオ（`scenarios/*.json`）の読み込み・保持 |
| [`gantt_chart_operation.py`](gantt_chart_operation.py) | 遺伝子 ↔ ガントチャートのデコード／エンコード、外乱検出、リスケ対象の切り出し |
| [`evaluation.py`](evaluation.py) | 共通の評価関数（安定性＝順列偏差、メイクスパン、正規化、重み付き目的関数）。GA/ILS 間の**公平な比較**を担保 |
| [`genetic_operation.py`](genetic_operation.py) | 交叉（PPX）・突然変異（inversion）・選択（トーナメント）・個体生成 |
| [`ga_scheduling.py`](ga_scheduling.py) | GA ソルバー（ベースライン） |
| [`ils_scheduling.py`](ils_scheduling.py) | ILS ソルバー（N5 近傍局所探索、Taillard 加速、repair/PR キック） |
| [`memetic_scheduling.py`](memetic_scheduling.py) | Memetic GA ソルバー（GA 広域探索 × N5 局所探索 × repair/PR キック） |
| [`analysis.py`](analysis.py) | ガントチャート・散布図などの可視化 |

**探索手法の全体像（7 手法）**: `GA` / `ILS-baseline` / `ILS+repair` / `ILS+PR` /
`Memetic-LS` / `Memetic+repair` / `Memetic+PR`。
GA 遺伝子（GT 法のジョブ列）と ILS の machine-order 表現を相互変換し、両者で探索インフラを共用します。

### 問題・シナリオ

- [`problems/`](problems/) — OR-Library 標準形式の問題定義（`mt06`, `mt10`, `la21`, `la36`, `la40`, `ta21`）
- [`scenarios/`](scenarios/) — 初期ガントチャートと外乱後ガントチャートの組（`*.json`）。
  外乱規模の異なる `la36_small/middle/large`、`ta21_high` などを含む
- [`tools/`](tools/) — シナリオ生成（`generate_scenario.py`, `generate_multi_delay_scenario.py`）、
  headroom 測定（`measure_headroom.py`）などの補助スクリプト

### 実験

- [`experiments/core_comparison_v3/`](experiments/core_comparison_v3/) — **主実験**。
  7 手法 × 10 重み × 6 問題セット × n 試行の統制比較（`run_v3.py`）と解析（`analyze_v3.py`, 各種図生成）
- [`experiments/param_sweep_v1/`](experiments/param_sweep_v1/) — パラメータ感度分析（OAT）と確定デフォルト値の根拠（[`RESULTS.md`](experiments/param_sweep_v1/RESULTS.md)）
- [`experiments/experiment_utils.py`](experiments/experiment_utils.py) — 実験共通ユーティリティ（正規化パラメータ、各ソルバー実行ラッパ）
- `experiments/*/results/` は容量のため **git 管理外**。各 `run_*.py` から再現可能

### ドキュメント

- [`doc/apiems2026_manuscript.md`](doc/apiems2026_manuscript.md) — APIEMS 2026（釜山）投稿版原稿
- [`doc/research_document.md`](doc/research_document.md) — 研究の母艦ドキュメント
- [`doc/paper_apiems2026/`](doc/paper_apiems2026/) — 論文 docx 生成パイプライン（`make_docx.py` で再生成）
- `doc/references/` と参考文献 PDF は著作権のため **git 管理外**

---

## 使い方

### 主実験の実行

```bash
cd experiments/core_comparison_v3

# パイロット（10 試行、全問題・全重み・全手法）
python run_v3.py --n-trials 10 --n-jobs 4

# 本番（30 試行、results/main に保存）
python run_v3.py --n-trials 30 --output-dir results/main --n-jobs 4
```

`(problem × weight × method × trial)` をフラット化して `ProcessPoolExecutor` で並列実行します。
`1 ファイル = 1 run` で保存するため、途中停止・再開・部分実行に対応（同じ `--output-dir` を指定すれば resume）。

### 解析・図の生成

```bash
cd experiments/core_comparison_v3
python analyze_v3.py            # 集計・統計検定（HV, Wilcoxon, Cliff's δ 等）
python figures_v3.py            # 図の生成
```

### 単発の動作確認

```bash
python test_taillard_rerun.py   # mt10 で ILS を逐次実行し挙動確認
```

---

## 評価指標について

- **安定性** $D = \sum \omega \cdot |\text{init\_pos} - \text{cur\_pos}|$
  … 機械ごとの処理順序（順列）が $S_p$ からどれだけ動いたか（小さいほど安定）。
  既定は重みなし順列偏差（$\beta = 0$）。
- **効率性** … メイクスパン。
- 効率と安定性を $[0,1]^2$ に正規化し、重み付き和で単一目的化。重みを 10 点掃引して
  効率〜安定のトレードオフ全体を評価します。手法比較は主に union UEA の **Hypervolume (HV)** で行います。

---

## ライセンス / 引用

本リポジトリは研究用コードです。問題インスタンスは OR-Library 由来です。
成果を利用する場合は上記 APIEMS 2026 原稿を参照してください。
