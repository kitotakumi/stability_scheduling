# 実験ログ（インデックス）

> 実験は目的別に3つのサブディレクトリに分離されています。各ディレクトリに実験スクリプト、結果ドキュメント、結果データが格納されています。

---

## ディレクトリ構成

```
experiments/
├── experiment_utils.py          (共通ユーティリティ)
├── experiment_log.md            (このファイル)
├── weight_sweep/
│   ├── run_weight_sweep.py      (実験スクリプト)
│   ├── weight_sweep_log.md      (結果・考察)
│   └── results/                 (実験結果データ)
├── ga_vs_ils/
│   ├── run_ga_vs_ils.py
│   ├── ga_vs_ils_log.md
│   └── results/
└── ils_analysis/
    ├── run_ils_analysis.py
    ├── ils_analysis_log.md
    └── results/
```

## 実験スクリプト

| スクリプト | 目的 | 実行例 |
|-----------|------|--------|
| `weight_sweep/run_weight_sweep.py` | 重みベクトルの最適値・耐性調査 | `python run_weight_sweep.py` |
| `ga_vs_ils/run_ga_vs_ils.py` | GA vs ILS 3手法の10試行比較 | `python run_ga_vs_ils.py --weights "0.95,0.05"` |
| `ils_analysis/run_ils_analysis.py` | ILS摂動手法・path_relinkモードの詳細分析 | `python run_ils_analysis.py` |
| `experiment_utils.py` | 共通ユーティリティ（問題設定、正規化、可視化） | (インポート用) |

## 結果ドキュメント

| ドキュメント | 内容 |
|-------------|------|
| [weight_sweep_log.md](weight_sweep/weight_sweep_log.md) | 重みスイープ結果。**ILS(insert)最適重み: w_stab=0.05** |
| [ga_vs_ils_log.md](ga_vs_ils/ga_vs_ils_log.md) | GA vs ILS比較。**ILS(insert) w=[0.95,0.05]が最良: MS平均1045.4** |
| [ils_analysis_log.md](ils_analysis/ils_analysis_log.md) | path_relinkモード分析。**Stab改善効果確認(swap+relink: 6.29→5.99)** |

## 統計方法

全実験共通: **改善成功試行(MS < 初期解1079)のみ**の統計値を報告。改善成功率を別途表示。

## 現時点の最良設定

- **手法**: ILS(insert)
- **重み**: w=[0.95, 0.05]
- **性能**: 平均MS=1045.4, 最良MS=1044, 8/10試行で1044到達, CPU≈14.6s
- **改善成功率**: 10/10 (100%)

## 主要な知見

1. **安定性をわずかに考慮するとMSも改善する** (w_stab=0.05でMS 1050→1047)
2. **ILS(insert)が全手法中最強** (1044到達、100%改善成功)
3. **ILS(swap)はMS=1051が構造的限界** (N5近傍の制約)
4. **path_relinkはStabを0.30〜1.0改善** (swap+relinkでStab=5.99に統一)
5. **全手法でw_stab≥0.4は破壊的** (初期解が目的関数上で最良と判定される)
6. **次の改善ポイント**: path_relinkの受理判定緩和 (MS非悪化+Stab改善で受理)
