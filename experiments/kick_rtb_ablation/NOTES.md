# PR (Path Relinking) 設計メモ — 再調査防止

最終更新: 2026-06-03

PR まわりで「同じ問題を何度も検討し直さない」ための所見メモ。詳細なコードは
`ils_scheduling.py::path_relinking` と `tools/probe_pr_infeasible.py` を参照。

## 1. FI（first-improvement）は計算時間の短縮にならない ❌

`path_relinking(step_strategy=...)` で BI('best') / FI('first') を切替可能。

- **検証**: la36_delay148, weight[0.8,0.2], n=6 pilot（`memetic_ls` / `memetic_pr`(BI) / `memetic_pr_fi`(FI)）。
- **結果**: FI ≈ BI **同品質**（UEA HV 6011 vs 5995, score/MS/領域HV すべて Cliff's delta negligible）。
  しかし **CPU はむしろ ~25% 遅い**（FI 中央値 ~4082s vs BI ~3257s）。
- **理由**: S_p 方向の経路はスカラーが悪化しがちで「改善する最初の swap」がほとんど
  出ない → FI も実質ほぼ全候補スキャン（= BI と同コスト）。さらに FI の非greedy な
  着地でキック後の N5 LS が重くなり、差し引きで遅くなる。
- **結論**: **時短目的で FI を使わない**。多様性が欲しい場面のオプションとしてのみ温存。

## 2. 経路の途中打ち切り（end_all_infeasible）はほとんど起きない

ある手で「S_ref 方向の direct-swap が全て infeasible（サイクル）」になると経路を
打ち切る分岐がある。

- **プローブ** (`tools/probe_pr_infeasible.py`, la36_small, N=50): **発火 0/50 (0%)**。
  経路長 initial_diffs も中央値 8 と短い（小外乱は reschedule 範囲が小さいため）。
- **対応**: 稀なので「起きたとき確実に経路を繋ぐ」突破ロジックを追加した
  （`_escape_infeasible`: 2手 direct-swap の組合せで feasible かつ S_ref に近づく状態を探す）。
  `path_relinking(escape_infeasible=True)` で有効。**memetic では有効化**、ILS は既定 False
  （既存挙動を保持）。稀なので O(diffs^2) の突破コストは平均性能にほぼ響かない。
- **未確認**: 大外乱（la36_large 等、経路が長い）では発火頻度が上がる可能性。必要なら
  プローブを大外乱で再実行して確認すること。

## 3. PR の計算時間は O(diffs²) で外乱規模に比例

PR は各手で「残り不一致数ぶんの候補」を毎回フル decode して評価するため、1 回の PR
コストは概ね **O(diffs²) 回の decode**。diffs（= S_cur と S_p の不一致数）は外乱の
reschedule 範囲に比例する。

- la36_small は diffs≈8 と小さく PR は安価。旧 la36_delay148（大外乱）で ~3000s/試行
  かかったのは diffs が大きかったため。

### 実装済みの時短（いずれも出力厳密不変・検証済み）
- **安定性計算の O(n)化**: `_rank_deviation` のループ内 `.index()` (O(L²)) を位置辞書で O(L) に。
  GA/ILS/memetic 全 evaluate に効く。長い外乱(la36_large)では evaluate のかなりを占めていた。
- **安定性差分化 (O(1)) — PR と N5局所探索**: 「親+1スワップ」型の近傍は親の安定性を1回
  計算し、各候補は動く2位置の項の差分だけ O(1) で更新する（`_stab_swap_delta`、注入口は
  `evaluate(stability=...)` と `_score_lower_bound(stability=...)`）。semi-active 限定
  （active はデコードで順序が変わるためフル評価にフォールバック）。
  - **PR**: 候補1個あたり evaluate を ~19% 削減（la36_large 540→439µs）。
  - **N5局所探索**: Taillard スクリーニングは全近傍の安定性を毎回計算しており（makespan は
    推定で O(1) なので安定性が律速＝doc/experiment_plan.md §で既述）、ここを差分化。
    スクリーニング1巡の安定性コストを **~80%削減**（la36_large 638→129µs、近傍が多いほど効果大）。
    **ILS・memetic の局所探索すべてに効く**。
  - 検証: PR 1962ケース / N5 77ケースで差分が厳密一致、local_search('best') が非taillard
    フル版と完全一致することも確認（出力不変）。

### decode 回数問題の解決：random-walk relinking（採用済み）
プロファイル(la36_large): evaluate の **約73% は build_gantt（デコード）**。安定性は最適化済み
なので残りは **decode の「回数」を減らす**問題で、これは **PR の各ステップで全候補を評価して
最良を選ぶ (best-selection) のをやめる**ことで解決した。

- **random-walk relinking を採用（既定）**: 各ステップで実行可能な direct swap を 1 つランダムに
  選ぶ（TS/PR, Peng et al. 2015 [本文 [6]] 流）。decode が **O(diffs²)→O(diffs)** に激減。
  pilot(la36_large, ngen=100, n=10): best-selection に対し **約3〜13倍高速**、UEA HV は
  **有意差なし(Wilcoxon p=0.72)**・高安定ゾーン同等。→ memetic・ILS とも既定を 'random' に切替
  （`step_strategy`/`pr_step_strategy` の既定='random'）。best-selection は `'best'` で sweep 比較用に残す。
- これにより **A2（Taillard下界の枝刈り）/ D1（未改変個体スキップ）は不要**。Taillard 推定は N5
  隣接 swap 専用で任意 transposition には使えず実装困難だったが、random-walk で目的を達成。
- FI（first-improvement）は時短にならず不採用（§1）。infeasible 突破は稀イベントの堅牢化（別目的）。
