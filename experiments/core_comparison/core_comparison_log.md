# コア比較実験ログ（実験1）

> evaluation_design.md §6 実験1「コア比較」の実行・分析ログ。
> raw データ収集は `run_core_comparison.py`。分析は別スクリプト `analyze_core.py`（未実装）。

---

## 1. 実験の目的

主張 A〜D を同時検証するための raw データを収集する。

| ID | 主張 | 必要データ |
|---|---|---|
| (A) | ILS は GA より速い | anytime curve（CPU time × best_score/HV） |
| (B) | ILS (+repair) の Pareto 覆域は GA を凌駕 | 全訪問点 → per-trial Pareto → union HV / EAF |
| (C) | repair 摂動は安定性側に Pareto を拡張 | base vs base+repair の差分 EAF / conditional HV |
| (D) | ILS は重みに頑健、GA は高 stab で degenerate | 重み×手法の改善成功率ヒートマップ |

## 2. 予算設計

**CPU 時間での打ち切りはしない**。ILS は `max_iterations=800`、GA は `ngen=500` まで自然収束まで走らせる。速度比較は anytime curve で任意時刻断面を事後抽出する（evaluation_design.md §4.1）。

## 3. 固定変数

| 変数 | 値 | 根拠 |
|---|---|---|
| ILS strategy | `'best'` | FI vs BI で有意差なし（2026-04-15） |
| active_schedule | `False` | N5 の理論保証が semi-active 前提 |
| stagnation_threshold | `None` | 悪化受理は不利（2026-04-14） |
| taillard_acceleration | `True` | 合成スコア下界フィルタで動作 |
| GA pop_size | `50` | 既存実装値 |
| 正規化 | GT 法ランダムサンプリングで共通推定 | `experiment_utils.compute_shared_norm_params` |

## 4. 独立変数（実行時に振る）

| 変数 | デフォルト | CLI |
|---|---|---|
| 問題セット | mt10, la21, la36, la40 の 4 問題 | `--problems` |
| weights | `[[0.85, 0.15]]` のみ | `--weights` |
| 手法 | `ga`, `ils_insert`, `ils_insert_repair` | `--methods` |
| trial 数 | 10 | `--trials` |

### 手法の選択肢

| キー | 中身 |
|---|---|
| `ga` | GA（pop_size=50, ngen=500） |
| `ils_insert` | ILS-baseline（insert 摂動） |
| `ils_insert_repair` | ILS + repair キック（insert 主摂動） |
| `ils_swap` | ILS-baseline（swap 摂動） |
| `ils_swap_repair` | ILS + repair キック（swap 主摂動） |

## 5. 使い方

```bash
# デフォルト実行: 4 問題 × weights=[0.85, 0.15] × 3 手法 × 10 trial
python run_core_comparison.py

# 問題・手法・重みを指定
python run_core_comparison.py \
    --problems mt10:mt10_delay60 la36:la36_delay148 \
    --methods ga ils_insert ils_insert_repair \
    --weights 0.85,0.15 \
    --trials 10

# 実行 + 分析を一気通貫で（--analyze フラグ）
python run_core_comparison.py --analyze

# 複数 weights 掃引（将来の多 weights Pareto 集約用）
python run_core_comparison.py \
    --weights 1.0,0.0 0.9,0.1 0.8,0.2 0.7,0.3 0.5,0.5 0.3,0.7 \
    --trials 10
```

## 6. 出力構造

```
results/core_<timestamp>/
├── config.json                               # 実行設定（再現用）
├── <problem>_<scenario>/
│   ├── results_<w_label>.json                # 全手法×全trialの履歴・最終値
│   └── summary_<w_label>.txt                 # 数値サマリ
└── cross_summary.txt                         # 横断サマリ
```

### `results_<w_label>.json` のスキーマ

```json
{
  "problem": "mt10", "scenario": "mt10_delay60",
  "weights": [0.85, 0.15], "init_makespan": 1080, "n_trials": 10,
  "ils_max_iter": 800, "ga_ngen": 500,
  "methods": {
    "ga": {
      "kind": "ga", "label": "GA",
      "baseline": [1080.0, 1.027],                    // 初期解相当の (ms, stab)
      "finals":   [{"trial":0,"seed":7,"makespan":...,"stability":...,"convergence":{...}}, ...],
      "anytime":  [[{"cpu_time":...,"best_ms":...,"best_st":...,"best_score":...}, ...], ...],
      "points":   [[[ms, st], ...], ...]               // 全世代×全個体の flat 点列（trialごと）
    },
    "ils_insert": {
      "kind": "ils", "label": "ILS-insert",
      "baseline": [1080, 0.0],                        // ILS は semi-active なので stab=0
      "finals":   [...],
      "anytime":  [[...], ...],
      "points":   [[[ls_ms, ls_st, accepted], ...], ...]   // 反復ごとのLS結果
    }
  }
}
```

- `baseline`: 手法固有の「初期解相当の点」。分析時に弱 dominance で除外する目印
  - ILS: `(init_ms, 0.0)` — semi-active decoding で stab=0
  - GA: `(ms_active, stab_active)` — active schedule decoding で stab が厳密 0 にならない
- `anytime`: 手法共通の時刻付き best 推移（anytime curve 用）
- `points`: 手法固有の訪問点列（Pareto/EAF 用）。GA は pop 全個体、ILS は LS 結果。

## 7. 分析: `analyze_core.py`

### 使い方
```bash
python analyze_core.py results/core_<timestamp>/
# オプション
python analyze_core.py results/core_<timestamp>/ \
    --snapshot-times 5 10 20 40 \
    --eaf-pairs ils_insert:ga ils_insert_repair:ils_insert
```

### 出力構成
```
<results_dir>/analysis/
├── per_problem/<problem>_<scenario>/<w_label>/
│   ├── anytime_hv_<w>.png              # anytime HV curve (full)
│   ├── anytime_region_hv_<w>.png       # anytime Region-restricted HV (low/mid/high 3 subplot)
│   ├── anytime_scalar_<w>.png          # anytime best weighted score
│   ├── final_pareto_<w>.png            # union Pareto overlay
│   ├── snapshot_pareto_T{5,10,20,40}s_<w>.png
│   ├── attainment_<w>.png              # 25/50/75% attainment surface
│   ├── individual_eaf_<w>.png          # N 手法の EAF heatmap 並列表示
│   ├── diff_eaf_<A>_vs_<B>_<w>.png     # 戦略 pair ごと
│   ├── hv_cmetric_<w>.txt              # union HV + C-metric 行列
│   ├── region_hv_<w>.txt               # Region-restricted HV (low/mid/high stab)
│   └── snapshot_stats_<w>.txt          # T 秒時点の数値表
└── cross_problem/
    ├── degeneracy_heatmap.png          # weights × methods × problems
    └── cross_summary.txt               # 全問題・全重みの HV/Region HV サマリ
```

### 指標一覧と主張対応

| 指標 | 主張 | 実装 |
|---|---|---|
| anytime HV curve (full) | (A) 速度 | ✅ |
| anytime Region-restricted HV (3 region) | (A)(B') 時系列×領域 | ✅ |
| anytime scalar curve | (A) deployment | ✅ |
| final Pareto overlay | (B) 品質 | ✅ |
| snapshot Pareto (T=5/10/20/40) | (A)(D) | ✅ |
| attainment surface (25/50/75%) | (B) 方向別 | ✅ |
| individual EAF | (D) | ✅ |
| 差分 EAF (pair) | (B)(C) | ✅ |
| union HV + C-metric | (B) | ✅ |
| Region-restricted HV | (B') 安定性側優位 | ✅ |
| snapshot stats table | (A) 要約数値 | ✅ |
| 改善成功率 heatmap | (D) degeneracy | ✅ |

### 設計のポイント

- **初期解除外**: 各手法が記録した `baseline` に弱 dominance される点（初期解相当）を全 Pareto/HV/EAF 計算から除外。GA の active schedule では stab が厳密 0 にならないため、method-specific baseline で正確に検知
  - ILS: `(init_ms, 0.0)` — semi-active decoding
  - GA: `(ms_active, stab_active)` — GA の original_individual を active schedule でデコードした値
- **EAF は pairwise**: CLI `--eaf-pairs A:B` または戦略デフォルト（ILS-insert vs GA, +repair vs baseline, +repair vs GA）
- **複数 weights の扱い**: 現状は weights ごとに個別分析（per_weight サブディレクトリ）。weights 横断集約はまだ未実装
- **Region-restricted HV**: 全手法の baseline-除外後 union Pareto から stab 軸 quartile を計算し、low_stab `[0, Q1]` / mid_stab `(Q1, Q3]` / high_stab `(Q3, stab_max]` の 3 領域を自動決定。参照点は `(init_ms, R_upper + margin)` で strict dominance を保証
- **anytime Region HV**: 領域境界は事前に固定（全時刻の union Pareto から算出）、時間で動かさない。各時刻 t で per-trial に領域別 HV を計算して trial 平均
- **HV 参照点**: 全手法 baseline-除外後の全訪問点の (ms_max, stab_max) にマージンを足した共通値

### 未実装（v2 候補）

- 複数 weights 横断集約（weights 空間全域の Pareto union）
- attainment surface 差分曲線

## 8. 実行履歴

| 日付 | 実行内容 | 結果ディレクトリ | 備考 |
|---|---|---|---|
| 2026-04-21 | 実装完了、smoke test のみ | - | Phase 1-4 実装 |

## 9. 未決事項

- [ ] `repair_trigger`, `repair_strength` の値: 現状 trigger=30, strength=2（`repair_perturb` 実験のデフォルト）。実験2（パラメータ掃引）で確定次第差し替え。
- [ ] HV 参照点: 複数 weights 集約 Pareto での参照点定義（evaluation_design.md §10）。
- [ ] 多 weights 掃引実行: 初回は weights=[0.85, 0.15] 単独。本格掃引は 6 水準 × 1000 run 規模。
