# param_sweep_v1: パラメータ感度分析 設計

core_comparison_v3（主実験）の手法群について、各パラメータの感度を OAT（One-At-a-Time）で
測る。目的は **チューニング（良い値の把握）＋ 防御（ベースラインを不当に弱く設定していない
ことの証明）** であり、網羅的な全要因最適化ではない（修論/修士国際会議レベルの粒度）。

指標・統計検定は core_comparison_v3 の `analyze_v3.py` から **import** して再利用する
（重複実装しない）。記録フォーマットも run_v3 と同一（`uea_points` + `uea_points_t`）。

---

## 1. 方針

- **OATで十分**。全パラメータを中心値（center）に固定し、1つだけ振る。2D格子はやらない。
- 各パラメータは **触る手法だけ** で振る（爆発を防ぐ）。
- center 設定は手法をまたいで共有・1回だけ計算（各 axis の基準＝paired 比較の対照）。
- **計算予算の公平化**: PR-BI / repair強↑ / kick_prob↑ / trigger密↑ / top_k-LS は1反復あたりの
  コストを変える。**反復数でなく壁時計/評価回数を揃えて**比較する（記録済みの cpu_time・
  evaluations・TTTで担保）。

## 2. 問題・重み・試行

- **問題（2つ）**:
  - `la21 / la21_delay147`（15×10・汎用中規模・pilot 実績）
  - `la36 / la36_large`（15×15・遠い外乱・headroom62 で機構効果が明確・領域特化が出る）
  - ※サイズ多様＋効果明瞭のバランス。コストが厳しければ la21 を `mt10/mt10_delay60`（10×10・高速）に置換可。
- **重み（6点・安定性側の重み）**: `stab ∈ {0.0, 0.1, 0.2, 0.4, 0.6, 0.8}`
  → `weights = [[1.0,0.0],[0.9,0.1],[0.8,0.2],[0.6,0.4],[0.4,0.6],[0.2,0.8]]`
  （union HV 用に複数点。低安定側を密に。11点は主実験用、掃引はこの6点で十分）
- **試行**: 10（必要なら 15）。seed は core_v3 と同じ `trial*100+7`。

## 3. center（基準設定）

| 機構 | パラメータ | center |
|---|---|---|
| ILS 共通 | perturb / initial_strength / max_strength / strategy | insert / 2 / 5 / best |
| ILS+PR | relink_trigger / kick_trigger_first / pr_step_strategy / pr_ls_top_k | 10 / 400 / random / 1 |
| ILS+repair | repair_trigger / kick_trigger_first / repair_strength | 10 / 400 / 0(=経路長フル) |
| Memetic | kick_prob / repair_strength / pr_step_strategy / pr_ls_top_k | 0.3 / 0 / random / 1 |
| GA/Memetic | pop_size / cxpb / mutpb | 50 / 0.85 / 0.1 |
| 反復 | ils_max_iter / ga_ngen / memetic_ngen | 3000 / 500 / 500 |

## 4. 掃引軸（OAT）

各軸は center に1パラメータの override を載せたものを、対象手法だけで実行する。
**center 値も含む**（center は全軸共有・1回だけ実行）。

| 軸 | 対象手法 | center | 値 |
|---|---|---|---|
| **pr_step_strategy** | ils_pr, memetic_pr | random | random, best |
| **pr_ls_top_k** | ils_pr, memetic_pr | 1 | 1, 3, 5 |
| **repair_strength** | ils_repair, memetic_repair | 0 | 0, 2, 4, 8 |
| **kick_prob** | memetic_pr, memetic_repair | 0.3 | 0.1, 0.2, 0.3, 0.5, 0.7 |
| **kick_trigger_first** | ils_pr, ils_repair | 400 | 100, 200, 400, 600 |
| **trigger** | ils_pr(relink), ils_repair(repair) | 10 | 5, 10, 20, 40 |
| **pop_size** | ga, memetic_ls | 50 | 30, 50, 80 |
| **cxpb** | ga | 0.85 | 0.6, 0.85, 0.95 |
| **mutpb** | ga | 0.1 | 0.05, 0.1, 0.2 |
| **perturb** | ils_baseline | insert | insert, swap |
| **max_strength** | ils_baseline | 5 | 3, 5, 8 |

- **pr_ls_top_k**: 経路上スコア上位 k 中間解に LS をかけ最良を採る PR variant（新規実装、
  `ils_scheduling.path_relinking(ls_top_k=k)`）。「単純に有意差を確かめる」目的。k倍の LS で重いので
  必ず等予算（TTT/cpu）で読む。
- **repair_strength**: 鋸歯の depth 天井（cap）。0=経路長フル。random/巡回といった発火則は変えない。
- Tier: 提案機構（上6軸）が主。ベース機構（下5軸）は「GA/ILSを過小設定していない」防御。
  交叉/突然変異/選択の**方法**は掃かない（PPX 交叉 [Bierwirth 1996]＋inversion 突然変異を報告・引用のみ）。

## 5. 指標（analyze_v3 から import）

各 (問題 × 軸 × 値) について、対象手法ごとに:

1. **per-trial union UEA HV**（主・品質）: 6重みのUEAを trial 内で union → PF → HV。trial 間 中央値[IQR]。
   （`analyze_v3` の pareto_front / hypervolume / filter_baselines を使用）
2. **領域別 HV**（高安定 D≤P50 / 低安定 D>P50）: `analyze_v3.region_hv`。機構の領域特化を見る。
3. **TTT@95%（速度・下記）**。
4. **統計検定**: 各値 vs center を **Wilcoxon 符号順位（対応あり）+ Cliff's δ**
   （`analyze_v3.wilcoxon_paired` / `cliffs_delta`）。同一 seed で paired。

### 速度指標 — 何を一つ見るか

**自己参照 TTT@95%（per-trial union HV、trial 中央値）** を1つの定量速度基準とする。

- 定義: 各 trial が「自身の最終 union HV の 95%（start→final gain の95%）」に初到達する CPU 時間。
  その trial 間中央値。`analyze_v3._worker_trial_ttt` を per-trial union 点列に適用して算出。
- **なぜ自己参照**: 掃引は1手法のパラメータを振るので「自分の plateau にどれだけ速く着くか」を問う。
  （手法横断の common-target QRTD は主実験用。掃引内では自己参照が自然。）
- **なぜ95%**: 90% は ILS では即到達で弁別力が低い、99% は裾ノイズに敏感。95% が中庸。
- **必ず最終 union HV（品質）の隣で読む**: 速いが品質の低い解への収束を区別するため。
  各設定 →(union HV 中央値, TTT@95% 中央値) のペアで評価。TTT 単独で速さを語らない。

## 6. 出力構造

```
results/<out>/
├── config.json            # center, axes, configs(=method+param override), 問題/重み/試行
├── norm_params.json       # 問題×シナリオ共通正規化（core_v3 と同方式）
└── <problem>_<scenario>/
    └── raw/
        └── <method>__<tag>__<w_label>__t<trial>.json   # tag: "center" or "param=value"
```
- 同一 params は同一 tag → 同一ファイル＝自然に dedup（center は全軸で1回だけ実行）。
- 1ファイル = 1run、`uea_points`/`uea_points_t`/`history`/`convergence` を run_v3 と同形式で保存。

## 7. 使い方

```bash
# 実行（resume 可: 同じ --output-dir なら既存ファイルskip）
python run_sweep.py --n-trials 10 --n-jobs 8 --output-dir results/main

# 一部の軸だけ
python run_sweep.py --axes kick_prob pr_ls_top_k --n-trials 10

# 分析（analyze_v3 を import して指標・検定を計算）
python analyze_sweep.py --input-dir results/main
```

## 8. スコープ外（やらない／引用で済ます）

- 2D格子、kick_trigger_first の flat(none) 対照（反転は多因子なので単因子帰属しない）。
- 交叉/突然変異/選択の方法、strategy=best、RTB（確定済・引用）。
- 全問題・11重みでの掃引（主実験の役割）。
