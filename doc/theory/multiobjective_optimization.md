# 多目的最適化と評価指標 — 文献ノート

> スケジューリング再最適化研究で使う概念の理論的背景を整理する。具体的な実験設計・指標の適用方針は [evaluation_design.md](../evaluation_design.md) を参照。

---

## 1. 2目的最適化の基礎

### 1.1 問題の構造

```
minimize  F(x) = (f₁(x), f₂(x))
subject to x ∈ Ω
```

本研究では f₁ = MS（メイクスパン）、f₂ = Stab（初期スケジュールからの順位偏差）。

### 1.2 Pareto 最適性

解 x が解 y を**支配 (dominate)** する ⟺ すべての i で fᵢ(x) ≤ fᵢ(y) かつ ある i で fᵢ(x) < fᵢ(y)。

どの解にも支配されない解の集合が **Pareto front**。2 目的問題では曲線として可視化できる。

---

## 2. 解法のアプローチ分類

多目的最適化問題の解法は大きく 3 種類に分類される。

| アプローチ | 説明 | 代表手法 |
|---|---|---|
| **スカラー化 (scalarization)** | 多目的を 1 目的に変換、N 重みで掃引 | Weighted sum, Tchebycheff, ε-constraint |
| **集団ベース多目的進化 (MOEA)** | 集団で front 全体を構成 | NSGA-II, SMS-EMOA |
| **分解ベース (decomposition)** | N subproblem に分解、subproblem 間で協調 | MOEA/D |

本研究はスカラー化アプローチ（Weighted sum sweep）を採用する。実運用ではスケジューラが単一の優先度重みで動くため、スカラー化は現実的かつ自然な選択である。

---

## 3. スカラー化手法の詳細

### 3.1 Weighted Sum (WS)

```
g_ws(x | w) = w₁·f₁(x) + w₂·f₂(x)
```

1950年代からの古典手法（Zadeh 1963）。スケジューリング応用研究で主流（Marler & Arora 2010）。

**凸限界**: 非凸 Pareto front の凹部に到達できない（Miettinen 1999）。WS の最適解は Pareto front の凸包上の点にしか対応しない。本研究では JSP の Pareto front 形状が事前に未知であるため、この限界は明示する必要がある（→ §6 Defense）。

### 3.2 Tchebycheff (TCH)

```
g_tch(x | w, z*) = max{ w₁·|f₁(x) - z*₁|, w₂·|f₂(x) - z*₂| }
```

z* = ideal point = (best f₁, best f₂)

**任意の Pareto 最適点に到達可能（凸でも非凸でも）**（Bowman 1976, Miettinen 1999）。WS より重みと front 上の点の対応が複雑であるが、凸限界がない点で理論的に優れる。本研究では補強実験として使う（時間余裕次第）。

### 3.3 MOEA/D との区別（重要）

MOEA/D（Zhang & Li 2007）の核心要素:
1. 重みベクトルの体系的生成（通常 50〜300 個）
2. **Neighborhood の定義**
3. **Subproblem 間の解共有**
4. **集団全体の協調進化**

本研究は **weighted sum sweep**（N 個の重みで WS を独立に解く）であり、MOEA/D ではない。共通点はスカラー化のみで、neighborhood も解共有も持たない。「MOEA/D-WS」と書くと方法論的に不正確なため、正確な呼称 **weighted sum scalarization sweep** を使う。

---

## 4. Pareto front の構築フレームワーク

### 4.1 2 つのフレームワーク

| フレームワーク | Pareto front の定義 | 性格 |
|---|---|---|
| Final population scenario | 最終集団（または最終解）の非劣解のみ | 伝統的（Zhang & Li 2007 原典）。運依存性が高い |
| **Unbounded External Archive (UEA)** | **探索過程の全 non-dominated 解** | 近年標準化。より realistic |
| Reduced UEA | UEA から pre-specified 数を選択 | UEA の subset selection |

### 4.2 文献的根拠

Ishibuchi らの研究（2020 以降）で UEA フレームワークが体系化された:

- 「最終集団は探索の初期に生成され破棄された他の解に dominated される解を含む」（Shu et al. 2022）→ final population は最適とは限らない
- 「UEA を使うと MOEA/D は state-of-the-art と competitive」（Tanabe & Ishibuchi 2018）
- 「UEA / reduced UEA シナリオは final population シナリオより実用的」（Tanabe & Ishibuchi 2020）

### 4.3 本研究での UEA 採用理由

1. 近年の標準的アプローチ
2. anytime 性能評価と整合（時間軸でのアーカイブ更新を見られる）
3. 探索軌跡保存設計と整合（ILS は LS 訪問点の全解を保存）
4. final population の運依存性を回避
5. GA との公平な比較（GA も探索過程の全 population を使える）

本研究での具体的な Pareto front 構築:
- **ILS**: LS 訪問点の全評価解
- **GA**: 全世代の全 population  
両手法で同一シナリオを採用することが「同一条件比較」の前提。

---

## 5. 評価指標

### 5.1 Hypervolume (HV)

参照点 r に対し、解集合 P が dominate する領域の体積:

```
HV(P, r) = volume( ∪_{p ∈ P} [p, r] )
```

- スカラー集約で手法比較に使いやすい（Zitzler et al. 2003）
- 参照点の設定が結果に影響するため、設定方針の明示が必要
- Pareto front 全体を 1 つの数値で表現できる

### 5.2 Region-restricted HV

Pareto 空間を領域分割し、各領域内で HV を計算（Knowles & Corne 2002 の考え方を応用）。stab 軸の quartile で low/mid/high に 3 分割し、特定領域での覆域を定量化できる。

「安定性方向での探索が優れている」という主張を weight に依存しない指標で定量化できる点が重要。

### 5.3 Empirical Attainment Function (EAF)

複数 trial の Pareto front の「α 確率で達成可能な領域」を可視化。差分 EAF で 2 手法の優劣を空間的に示せる。

定量化は Region-restricted HV と数学的に等価（`∫∫_R (α_A − α_B) dp = E[HV_R(A)] − E[HV_R(B)]`）。差分 EAF は視覚化に特化させる。

### 5.4 C-metric (Coverage)

```
C(A, B) = |B 内で A の何らかの解に dominate される点| / |B|
```

参照点不要。dominance を直接測定（Knowles & Corne 2002）。C(ILS, GA) が大きく C(GA, ILS) が小さければ ILS の優位を直接示せる。HV の補強指標として有効。

---

## 6. Weighted sum sweep + Pareto 評価の正当性

### 6.1 何が正当で何が正当でないか

**正当な部分**:
- 解集合に対する Pareto 評価指標（HV, EAF）の適用は、指標の数学的定義上問題ない（Zitzler et al. 2003）
- 同じ条件下で得られた解集合の比較は妥当

**正当でない部分**:
- 「Pareto 探索性能」を主張するのは飛躍
- WS は構造的に front 凸包しか触れない（Pareto front 全体の探索を目的としていない）

### 6.2 正確な主語

評価で測れるのは「**N-weight sweep の UEA 出力解集合の覆域**」であって、「Pareto front 探索能力」ではない。

```
❌ NG: 「WS-ILS は WS-GA より Pareto 探索性能で優れる」
✅ OK: 「重み付きスカラー化を N 個の重みで実行した UEA 解集合の Pareto 覆域は、ILS の方が GA より大きい」
```

### 6.3 数学的対応関係

```
WS scalar 最適化 (1 weight) の最適解 → Pareto front 凸包上の 1 点
N 個の weight で WS sweep   → Pareto front 凸包上の N 点近似
UEA scenario なら           → 軌跡から拾った非劣解集合 = 凸包近傍の点群
```

凸包外（凹部）は構造的に測れない → **凸限界を明示する**。

### 6.4 WS 凸限界問題への Defense

WS は非凸 Pareto front の凹部に到達できない。JSP の Pareto front 形状は事前に未知。

| Lv | 内容 | 強さ | コスト |
|---|---|---|---|
| 1 | 限界明記のみ | 弱 | 0 |
| 2 | + 文献根拠（Miettinen 1999） | 中 | 0 |
| 3 | + 1 問題で NSGA-II または TCH sweep で凸性 empirical 確認 | 強 | 半日〜1日 |
| 4 | + TCH 縮小実装で「scalar 化方式によらない優位性」検証 | 最強 | 3〜5日 |

採用方針: **Lv3 を最低限、Lv4 は時間余裕で**。

---

## 7. 本研究への接続

### 7.1 なぜスカラー化を基本とするか

本研究の対象は**再スケジューリングの実運用**である。実現場のスケジューラは「効率を 80%、安定性を 20% 重視する」といった単一の優先度ポリシーで動く。MOEA や多目的 EA のように Pareto front 全体を同時最適化することは、実用上も計算上もオーバースペックである。

したがって**アルゴリズムの基本動作はスカラー化（weighted sum）**で設計する。

### 7.2 なぜ Pareto 指標も使うか

単一スカラーでの評価は weight 設定に依存する。「weight [0.8, 0.2] では ILS が勝つが [0.5, 0.5] では拮抗する」という結果は、どの weight を基準とするかで結論が変わってしまう。

そこで**アルゴリズムの探索能力そのものを評価**するために、N 個の weight で実行した結果を集約し、Pareto 覆域（HV, C-metric 等）で比較する。これにより特定の weight 設定に依存しない「構造的な探索力の優位性」を主張できる。

### 7.3 なぜ MOEA/D に拡張しないか

MOEA/D は理論的に強力だが、本研究での採用は以下の理由でやりすぎである:

- アルゴリズム自体を multi-objective 化する必要が生じ、提案手法の設計が複雑になる
- 比較対象の GA も MOEA/D 化しないと公平でない
- 実運用で multi-objective な出力を扱う仕組みがない

**Weighted sum sweep（古典的スカラー化 + N 重みの集約）で十分**。WS は 1950 年代からの手法だが、スケジューリング研究でいまも主流であり（Marler & Arora 2010）、本研究が問う「どのアルゴリズムが重み付きスカラーの探索でより広い覆域を確保するか」という問いに対して適切な評価フレームワークを与える。

### 7.4 主張と指標の対応

| 主張 | 軸 | 主指標 |
|---|---|---|
| ILS は GA より構造的に高速 | (A) 速度 | per-weight anytime scalar / UEA HV curve |
| weight 別に ILS は GA より高品質 | (B-1) per-weight 質 | per-weight scalar 値 + 改善成功率 |
| UEA 解集合の Pareto 覆域で ILS が上 | (B-2) 統合 Pareto 質 | per-trial union UEA HV + Regional HV + C-metric |

(B-2) は WS sweep の出力解集合を Pareto 指標で評価するもの。「ILS のほうが N 重みの sweep 全体を通じてより広い覆域を確保している」という主張であり、「MOEA として優れている」という主張ではない。

---

## 8. 用語ミニ辞典

| 用語 | 意味 |
|---|---|
| WS | Weighted Sum scalarization |
| TCH | Tchebycheff scalarization |
| Weighted sum sweep | N 個の重みで WS を独立に解く古典手法 |
| MOEA/D | Multi-Objective EA based on Decomposition（neighborhood + 解共有を含む本格 MOEA）|
| NSGA-II | 集団ベース MOEA の代表 |
| HV | Hypervolume |
| Region-restricted HV | Pareto 空間を領域分割し領域内 HV を計算 |
| EAF | Empirical Attainment Function |
| C-metric | Coverage。手法間 dominance 直接測定 |
| UEA | Unbounded External Archive。探索過程の全非劣解を保存 |
| Final population scenario | 評価時に最終集団のみを使う伝統的フレームワーク |
| UEA scenario | 評価時に探索過程の全非劣解を使うフレームワーク |
| per-trial union UEA | trial 内で N weight の UEA を統合した非劣集合 |
| Ideal point (z*) | Pareto 空間での理想点 = (best f₁, best f₂) |
| Lucky punch problem | 分散の大きい手法が union 集約で有利になる現象 |
| Degeneracy（本研究文脈） | GA が高 stab 重みで初期解から動けなくなる現象 |
| Anytime curve | 時間 vs 指標の時系列プロット |
| ΔMS | initMS − finalMS（MS の改善幅）|
| Convexity limitation | WS が非凸 Pareto front の凹部に到達できない問題 |

---

## 9. 参考文献

### 多目的最適化基礎
- Miettinen, K. (1999). *Nonlinear Multiobjective Optimization*. Springer.
- Marler, R. T., & Arora, J. S. (2010). The weighted sum method for multi-objective optimization: new insights. *Structural and Multidisciplinary Optimization*, 41(6), 853–862.

### Scalarization
- Zadeh, L. (1963). Optimality and non-scalar-valued performance criteria. *IEEE Transactions on Automatic Control*, 8(1), 59–60.
- Bowman, V. J. (1976). On the relationship of the Tchebycheff norm and the efficient frontier of multiple-criteria objectives. *Lecture Notes in Economics and Mathematical Systems*, 130, 76–86.
- Steuer, R. E. (1986). *Multiple Criteria Optimization: Theory, Computation, and Application*. Wiley.

### MOEA/D
- Zhang, Q., & Li, H. (2007). MOEA/D: A multiobjective evolutionary algorithm based on decomposition. *IEEE Transactions on Evolutionary Computation*, 11(6), 712–731.

### Pareto 評価指標
- Knowles, J., & Corne, D. (2002). On metrics for comparing nondominated sets. *Proceedings of CEC 2002*, 711–716.
- Zitzler, E., Thiele, L., Laumanns, M., Fonseca, C. M., & da Fonseca, V. G. (2003). Performance assessment of multiobjective optimizers: an analysis and review. *IEEE Transactions on Evolutionary Computation*, 7(2), 117–132.

### UEA framework
- Ishibuchi, H., Pang, L. M., & Shang, K. (2020). A new framework of evolutionary multi-objective algorithms with an unbounded external archive. *Proceedings of ECAI 2020*.
- Tanabe, R., & Ishibuchi, H. (2020). An analysis of control parameters of MOEA/D under two different optimization scenarios. *Applied Soft Computing*, 70, 22–40.
- Shu, T., Nan, Y., & Ishibuchi, H. (2022). Effects of archive size on computation time and solution quality for the MOEA/D with unbounded external archive. *Applied Soft Computing*.

### JSSP + 安定性 + 再スケジューリング
- Rangsaritratsamee, R., Ferrell Jr, W. G., & Kurz, M. B. (2004). Dynamic rescheduling that simultaneously considers efficiency and stability. *Computers & Industrial Engineering*, 46(1), 1–15.
- Pfeiffer, A., Kádár, B., & Monostori, L. (2007). Stability-oriented evaluation of rescheduling strategies by using simulation. *Computers in Industry*, 58(7), 630–643.

### JSSP + ILS / TS / Path Relinking
- Nowicki, E., & Smutnicki, C. (1996). A fast taboo search algorithm for the job shop problem. *Management Science*, 42(6), 797–813.
- Peng, B., Lü, Z., & Cheng, T. C. E. (2015). A tabu search/path relinking algorithm to solve the job shop scheduling problem. *Computers & Operations Research*, 53, 154–164.
