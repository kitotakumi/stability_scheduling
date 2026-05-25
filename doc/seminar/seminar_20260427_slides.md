---
marp: true
theme: default
paginate: true
size: 16:9
math: katex
style: |
  section { font-size: 22px; }
  h1 { font-size: 38px; }
  h2 { font-size: 30px; }
  table { font-size: 20px; }
  pre { font-size: 18px; }
  blockquote { border-left: 4px solid #888; padding-left: 12px; color: #333; }
---

# ゼミ発表資料（2026-04-27 週）

### ILS vs GA の構造的説明と PR-repair 摂動の設計

kito / 2026-04-27

---

## 今週の要点

先週の指摘を受け、本研究の **構造面の説明** を整備した。

1. **GA vs ILS** — なぜ再スケジューリングで ILS が向くのか
2. **ILS の機構** — 何をどう繰り返しているか
3. **Path Relinking (PR)** — 「2 解を結ぶ経路」とは何か
4. **repair 摂動** — PR を ILS の摂動キックに転用する設計

加えて、**実験面でも基礎データを取得**:

- **GA / ILS-insert / ILS+repair の比較実験**（4 問題 × 6 重み × 10 trial）をざっくり走らせた
- **repair パラメータ（trigger, strength）の掃引** もざっくり実施し、当面の値を確定
- 本日は §5 で la21・la36 を中心に結果を共有

---

# §1. GA vs ILS

---

## 1.1 再スケジューリングの特殊性

通常の JSSP は「ゼロから解を作る」問題。

再スケジューリングは違う:

> **崩れる前の初期解 S₀ がすでに手元にある。**
> 新解 S* は S₀ に「近い」ほど望ましい（= 安定性）

⇒ **「強い prior (S₀) を活かす探索」** という性格を持つ

これが GA / ILS の向き不向きを決める

---

## 1.2 探索挙動の対比

![w:1000](ga_vs_ils_rescheduling.svg)

- **GA**: 交叉で子が親の中間に飛ぶ → S₀ の構造が壊れ、許容範囲（赤点線円）を飛び越える
- **ILS**: 小刻みな摂動 + 局所探索 → 許容範囲内に踏みとどまりつつ最適解方向へ

---

## 1.3 メカニズム比較

| 観点 | GA | ILS |
|---|---|---|
| 状態 | 集団 (population) | 1 本の `current` 解 |
| 主操作 | 交叉 + 突然変異 | 摂動 + 局所探索 |
| 構造の保存 | 2 親を切り貼り → 破壊的 | 連続変形 → 構造保持 |
| 強度の制御 | 集団全体に効く（読みにくい） | 摂動強度で **直接制御** |

---

## 1.4 帰結

- **GA**: 高 stab 重みで「S₀ に固着 (degeneracy)」or「遠くへ飛ぶ」の二極
  → 中間で踏みとどまれない
- **ILS**: 摂動強度 = 「どれだけ S₀ から離れるか」を直接制御できる

→ 主張 (A)（速度）と (D)（重み頑健性）の構造的根拠

---

# §2. ILS の機構

---

## 2.1 全体フロー

![h:380 center](ils_overall_framework.svg)

ILS は **3 ステップを反復するだけ**

---

## 2.1 擬似コード

```
current ← S₀
while 予算が残っている:
    perturbed ← 摂動(current)        # ① 局所最適から脱出
    local_opt ← 局所探索(perturbed)  # ② 近傍で改善し尽くす
    if 改善:
        current ← local_opt          # ③ 受理判定
```

ポイント: **「深掘り」と「脱出」が分離されている**

GA は交叉 1 個でこの 2 役を兼ねるので、操作の役割が曖昧

---

## 2.2 局所探索：N5 近傍

![h:340 center](N5neighborhood.png)

- **N5 近傍** (Nowicki & Smutnicki 1996): クリティカルパス上のブロック端点の隣接 swap のみ
- **閉路フリーが理論保証** = 必ず実行可能解
- 戦略は最良改善 ('best')。FI vs BI に差なしと先行実験で確認

---

## 2.3 摂動：脱出のキック

| 種類 | 操作 | 性格 |
|---|---|---|
| swap 摂動 | N5 swap を K 回連続適用 | 緩く壊す |
| insert 摂動 | 1 op を抜いて別位置へ挿入 | やや大きく崩す |

- 強度 K は停滞カウンタで段階的に増える（適応的）
- 摂動強度 K = **「安定性 vs 効率性のレバー」**

---

# §3. Path Relinking (PR)

---

## 3.1 PR の基本アイデア

PR = **「始点 s と目標 t を結ぶ経路」** を辿る手法

```
   s ●───────────────●───────────────● t
              ↑               ↑
          中間解 m₁        中間解 m₂
```

- 各 mᵢ は s の属性を t の属性へ少しずつ置き換えた解
- 経路上に **「s でも t でもない良解」** が混じる、というのが経験則

---

## 3.2 JSP での「経路」とは

解を `machine_orders`（機械ごとのジョブ列）と見ると、s と t の差は **「位置 i のジョブが違う」** という形で現れる

| 機械 m の位置 | 0 | 1 | 2 | 3 | 4 |
|---|---|---|---|---|---|
| 始点 s | J₁ | J₂ | J₃ | J₄ | J₅ |
| 目標 t | J₃ | J₁ | J₄ | J₂ | J₅ |
| 一致？ | ✗ | ✗ | ✗ | ✗ | ○ |

→ **不一致位置を 1 つずつ t に合わせていく** のが自然な経路

---

## 3.3 ムーブ戦略：direct swap

![h:340 center](direct_swap_focused.svg)

> 位置 i に t が要求するジョブ J を、現在列の中から探し、**位置 i のジョブと 1 ステップで swap する**

- 隣接 swap でなく **離れた位置同士でも一気に交換**
- 1 ムーブで 1 不一致位置を解消

---

## 3.4 本研究での観察

> **再スケジューリングでは、目標 t として「初期解 S₀」が常に手元にある。**

⇒ PR の「t に寄せる」操作を、そのまま **「安定性方向への引き寄せ」** に転用できる

これが次節 repair 摂動の出発点

---

# §4. repair 摂動

---

## 4.1 通常摂動との対比

| | 方向 | ねらい |
|---|---|---|
| 通常摂動 (swap / insert) | **ランダム** | 局所最適からの脱出 |
| **repair 摂動** | **S₀ に向かう** | 安定性方向の別の盆地へ脱出 |

- 通常: 「どこへ行くか分からないがとにかく動く」キック
- repair: 「S₀ という具体的な目的地に向かって動く」キック

---

## 4.2 構造

```
①  S₀ と現在解 S* の machine_orders を比較
        ↓
②  不一致位置を列挙し、K 個を選ぶ
        ↓
③  各々を direct swap で S₀ に合わせる    ← PR のムーブを 1 回分流用
        ↓
④  S₀ に少し寄った解を出発点として LS にかける
```

= **Mini-PR kick**（PR を完走せず、1 キック分だけ借りる最小コスト版）

---

## 4.3 なぜ効くのか / 設計上のポイント

**効く仮説**:
- 通常摂動 + LS は **元の盆地に戻りがち** → 安定性が改善しない
- repair 摂動は **出発点を S₀ 方向の別盆地に強制移動** → LS は別盆地の最適に収束 → MS をあまり犠牲にせず安定性が改善

**設計**:
- 通常摂動を **置き換えない**。停滞時のキックとして混在
- 制御は 2 パラメータ：`repair_trigger`、`repair_strength`
- 主張 (C)「repair は Pareto を安定性側へ拡張する」の検証対象

---

# §5. 実験結果（コア比較）

---

## 5.0 実験設定（要約）

| 因子 | 水準 |
|---|---|
| 問題 × 外乱 | mt10/la21/la36/la40 × delay |
| 重み | `[1.0,0]`〜`[0.3,0.7]` の 6 点 |
| 手法 | GA / ILS-insert / ILS-insert+repair |
| trial | 各 10 回（seed 固定） |
| 予算 | ILS 800 iter / GA 500 gen（自然収束）+ anytime 履歴 |

> 本スライドは **la21 delay147, weights=[0.85, 0.15]** を代表として図示。
> 4 問題横断の数値表も併記する。

---

## 5.1 速度 — anytime HV (la21)

![h:380 center](../../experiments/core_comparison/results/core_20260421_163911/analysis/per_problem/la21_la21_delay147/eff=0.85_stab=0.15/anytime_hv_eff=0.85_stab=0.15.png)

- ILS 系は **約 5 秒で最終 HV (≈1721) に到達**
- GA は 40 秒走らせても **HV 1584 止まり**（最終値: GA 1624 / ILS 1721）
- → 主張 (A): 同 CPU 時間での到達品質は ILS が圧倒

---

## 5.2 総合探索力 — union HV (la21 と la36)

| 問題 | 手法 | per-trial HV (mean±std) | union HV | \|Pareto\| |
|---|---|---|---|---|
| la21 | GA | 1123.34 ± **213.54** | 1624.76 | 7 |
| la21 | ILS-insert | 1713.18 ± 4.99 | **1721.18** | 5 |
| la21 | ILS+repair | 1697.68 ± 48.26 | 1720.34 | 5 |
| la36 | GA | 2327.96 ± 203.68 | 2622.87 | 10 |
| la36 | ILS-insert | 2360.34 ± 199.78 | **3035.73** | 11 |
| la36 | ILS+repair | 2304.44 ± 33.29 | 2449.86 | 5 |

- la21 は ILS 完全支配（C-metric 1.0）
- **la36 は様相が変わる**: GA(2622) > ILS+repair(2449) — 総合 HV で逆転
- → ただし「総合 HV ＝ 強さ」とは限らない。次スライドで領域を見る

---

## 5.3 la36 anytime HV — GA が repair を追い越す

![h:380 center](../../experiments/core_comparison/results/core_20260421_163911/analysis/per_problem/la36_la36_delay148/eff=0.85_stab=0.15/anytime_hv_eff=0.85_stab=0.15.png)

- ILS+repair は **5 秒で 2449 に到達後フラット**（安定性側の盆地で深く収束）
- GA は時間をかけて MS 方向に **2622 まで伸びる** → 総合 HV で repair を追い越す
- ILS-insert はバランス型で最終 3035 に到達

> GA は「広い領域に弱く分布」、repair は「狭い領域に強く集中」

---

## 5.4 領域別 HV — la36 anytime region HV

![h:380 center](../../experiments/core_comparison/results/core_20260421_163911/analysis/per_problem/la36_la36_delay148/eff=0.85_stab=0.15/anytime_region_hv_eff=0.85_stab=0.15.png)

| 領域 (la36, stab 範囲) | GA | ILS-insert | ILS+repair |
|---|---|---|---|
| **low_stab** [0, 6.02] 安定性重視 | **0.00** | 138.30 | **171.34** |
| mid_stab [6.02, 12.03] | **313.09** | 136.79 | 136.79 |
| high_stab [12.03, 18.05] MS 重視 | **159.49** | 81.97 | 0.00 |

> **得意領域がきれいに分離**: repair → 低 stab、GA → 中・高 stab
> GA は **低 stab 領域に 1 点も到達しない** = degeneracy ではなく「届かない」

---

## 5.5 視覚化 — 差分 EAF: ILS-insert vs GA (la36)

![h:340 center](../../experiments/core_comparison/results/core_20260421_163911/analysis/per_problem/la36_la36_delay148/eff=0.85_stab=0.15/diff_eaf_ils_insert_vs_ga_eff=0.85_stab=0.15.png)

- **左下（低 stab）= 青（ILS 優位）、右上（高 stab × 低 MS）= 赤（GA 優位）**
- 総合 HV では GA も善戦するが、**得意領域は明確に分離**
- 「どっちが強いか」ではなく「**どの領域を狙う問題か**」で手法を選ぶ話

---

## 5.6 視覚化 — 差分 EAF: ILS+repair vs ILS-insert (la36)

![h:340 center](../../experiments/core_comparison/results/core_20260421_163911/analysis/per_problem/la36_la36_delay148/eff=0.85_stab=0.15/diff_eaf_ils_insert_repair_vs_ils_insert_eff=0.85_stab=0.15.png)

- 青 = repair 優位、赤 = baseline ILS-insert 優位
- **下端（最も低 stab 側）に青の帯** → repair は baseline が届かない安定性深部を開拓
- 上側（中〜高 stab）は赤 → 効率性方向は baseline ILS の方が広く探索
- → **「repair は Pareto を安定性側へ拡張する」** の視覚的証拠

---

## 5.7 repair の効き — 安定性側の盆地探索

各問題の **低/中 stab 領域 HV**（repair vs baseline ILS）:

| 問題 | 領域 | ILS-insert | ILS+repair | 効果 |
|---|---|---|---|---|
| **la36** | low_stab | 138.30 | **171.34** | +24% |
| **la40** | mid_stab | 30.11 | **84.14** | **×2.8** |
| **mt10** | mid_stab | 36.06 | **49.31** | +37% |
| la21 | low_stab | 40.55 | 40.55 | 引き分け（既に到達済） |

la36 best stab 値（trial 平均）: GA **18.06** / ILS-insert 6.66 / **ILS+repair 4.82**

> repair は **baseline ILS が届かない安定性側の盆地** を新たに開拓
> 主張 (C)「Pareto を安定性側へ拡張」の支持証拠

---

## 5.8 結果まとめ

| 主張 | 結果 | 状態 |
|---|---|---|
| **(A) 速度** | ILS は T=5s で収束（la21）。GA は 40s 走らせても届かず | ✅ |
| **(B) Pareto 覆域** | la21 で ILS 完全支配。la36 では総合 HV で GA 善戦も、**領域別では低 stab で GA=0** | △ 領域別で見れば ILS 優位 |
| **(C) repair の安定側拡張** | 低・中 stab 領域 HV で +24%〜×2.8、best stab 値も最良 | ✅ |
| (D) 重み頑健性 | per-trial 分散 GA 200+ vs ILS+repair 33 で兆候あり | ⏳（重み別 degeneracy 解析は次週） |

> **得意領域の分離** が重要な findings：「総合 HV」一発で勝敗をつけない論立てが必要

---

# §6. まとめ

---

## 全体の構造マップ

```
   再スケジューリング = 「S₀ という強い prior 付きの探索」
            │
            ├─→ §1  ILS は摂動強度で破壊量を制御できる → ILS 採用
            ├─→ §2  ILS = N5 山登り + 摂動 の反復
            ├─→ §3  PR  = 2 解を結ぶ経路 + direct swap ムーブ
            └─→ §4  PR の direct swap を ILS の摂動に転用
                    = repair 摂動
                    → ILS に「S₀ への引き寄せ」を追加
```

---

## 議論したいこと

- §1 の構造的説明（**S₀ という prior の活かしやすさで GA / ILS を分ける**）の納得感

- §3 PR 説明と §4 repair 摂動説明の接続
  「経路の 1 キック分を借りる」というロジックが伝わるか

- §5 実験結果の解釈
  - **「得意領域の分離」**（GA → 中・高 stab、ILS+repair → 低 stab）を主張化する論立ての妥当性
  - la36 で総合 HV は GA が ILS+repair を上回るが、低 stab 領域では GA=0。**「総合 HV だけで優劣を決めない」** 評価設計を論文でどう打ち出すか
  - 速度比較の見せ方（anytime curve + snapshot 表）はゼミでも論文でも通せるか

---

## 付録: 用語ミニ辞典

| 用語 | 意味 |
|---|---|
| machine_orders | 機械ごとのジョブ列。本研究の解表現 |
| N5 近傍 | クリティカルブロック端点の隣接 swap のみ。閉路フリー保証 |
| direct swap | 離れた位置のジョブを 1 ステップで交換する PR の標準ムーブ |
| Path Relinking (PR) | 始点 s と目標 t を結ぶ経路上の中間解を拾う手法 |
| repair 摂動 (Mini-PR kick) | PR の direct swap を 1 キック分だけ ILS の摂動に転用 |
| degeneracy | GA が高 stab 重みで S₀ に固着し動けなくなる現象 |
