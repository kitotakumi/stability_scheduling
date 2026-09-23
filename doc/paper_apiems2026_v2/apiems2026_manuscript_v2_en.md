# [APIEMS 2026 投稿用・v2 英語版] Asymmetric Effects of Stability-Inducing Operators across Trajectory and Population Search in Job-Shop Rescheduling

> **このファイルの位置づけ**
> [apiems2026_manuscript_v2.md](apiems2026_manuscript_v2.md)（日本語 v2 原稿）の英語版。内容・数値・強調・引用箇所は日本語原稿に従う。次の 3 点だけは APIEMS テンプレートの規定に合わせて日本語原稿と形式が異なる：(1) 引用は著者–年形式で、文献リストはアルファベット順（日本語原稿の [n] 番号は使わない）、(2) 第 1 レベル見出しは大文字、(3) 図表キャプションは "Figure n:" / "Table n:"。
> docx は `python make_docx_v2.py --lang en [--pages]` で生成する（出力 `APIEMS2026_manuscript_v2_en.docx`）。図は日本語版と同じ `figures/fig_v2_*_en.png`。
> **図の位置**：全幅図は Word のフロート表で、`![...]` 行の**直後の段落**があるページの上端に浮かぶ（`![...]` 行の位置は本文の順序ではなく、このアンカーを決めるためだけのもの）。図が乗り切らないページにアンカーが落ちると（別の図が既にあるページなど）、アンカー段落以降の本文が図と一緒に次ページへ押し出されて段が空く。空白が出たら `![...]` 行を前後の段落へ動かしてアンカーのページを変える。アンカーには**分割できる普通の段落**を選ぶ：表 2 キャプションのようにキャプションと表で塊になるブロックをアンカーにすると、残りの段に入りきらないときブロックごと次ページへ飛んで大きく空く。**1 ページ 1 枚に散らす**と段が埋まりやすい：図が乗るページは本文が図の高さぶん下へ送られるので、表 1 のような塊が段に収まるかどうかが変わる（図を後ろのページへ寄せると、図のないページで塊が段からあふれて空白になりやすい）。現在の配置：図 2 は §4.1 見出しの前（アンカー＝§4.1 見出し、p.5 上端）、図 3 は「We check whether this picture holds…」の前（アンカー＝その段落、p.6 上端）、図 4 は §4.3 の結びの段落の後ろ（アンカー＝§5 見出し、p.7 上端）で、日本語版と同じく図 2→p.5・図 3→p.6・図 4→p.7 に 1 枚ずつ乗る。本文を削るたびにアンカーがずれるので、再生成後は PDF 化して段の空白を確認する。

---

## Title / Authors / Abstract / Keywords

**Title**: Asymmetric Effects of Stability-Inducing Operators across Trajectory and Population Search in Job-Shop Rescheduling

**Abstract**

In predictive-reactive rescheduling, a high-quality pre-disruption schedule $S_p$ already exists, and a revised schedule must be both efficient (makespan) and stable (little change from $S_p$). A search method should therefore also be judged by how well it fills the vicinity of $S_p$ (hereafter, the high-stability region). In a controlled comparison with the local search aligned, trajectory-based search (ILS), which transforms $S_p$ continuously, outperformed population-based search (Memetic), which recombines solutions by crossover, in the hypervolume of the high-stability region in every trial of all eight scenarios. We then mounted stability-inducing operators that pull solutions back toward $S_p$ (PR, based on path relinking, and repair) on both structures, and their effect was asymmetric. In population-based search, the high-stability hypervolume improved in every trial of every scenario, reaching the level of trajectory-based search in seven of the eight scenarios, and with the operators it ranked first in the quality of the whole trade-off. Trajectory-based search, by contrast, fills this region by itself and had almost no room left for improvement. The best method changes with the evaluation criterion.

**Keywords**: Rescheduling; Job-shop scheduling; Stability; Iterated local search; Path relinking

---

## 1. INTRODUCTION

On manufacturing shop floors, disruptions such as operation delays and machine breakdowns occur frequently and make the original schedule difficult to execute. Responses fall into two broad classes: static approaches, which build in slack beforehand to raise tolerance, and dynamic approaches, which revise the schedule after a disruption occurs (Vieira et al., 2003). This study addresses the latter, specifically **predictive-reactive rescheduling** (Ouelhadj and Petrovic, 2009), which revises the executing schedule into a feasible one.

In rescheduling, the **efficiency** of the revised schedule trades off against the smallness of the change from the pre-disruption schedule (hereafter, the original schedule $S_p$), which we call **stability**. A large change incurs costs (hereafter, **schedule-change costs**): confusion on the shop floor, setup changeovers, rearrangement of materials and fixtures, reassignment of workers, and schedule changes that ripple into downstream processes and subcontractors. Stability is therefore an objective on a par with efficiency (Wu et al., 1993; Rangsaritratsamee et al., 2004), and this study formulates the problem as multi-objective optimization of efficiency and stability. Efficiency is measured by the **makespan (MS)** and stability by the **sequence deviation $D$** from $S_p$, i.e., how much the processing sequence on each machine has been reordered (smaller is more stable).

Rescheduling has a peculiarity that static scheduling lacks: **the solution that absorbs the delay while keeping the sequences of the original schedule $S_p$ is optimal for the stability objective, and it is given before the search even begins.** This solution sits at $D=0$ on the plane with $D$ on the horizontal axis and MS on the vertical (Figure 1). Because $S_p$ was optimized before the disruption, this solution is also good in efficiency, but it absorbs the delay as is, leaving room to shorten MS by resequencing. Where schedule-change costs cannot be ignored, the solution sought lies near $S_p$, in what this paper calls the **high-stability region**: a solution that keeps the change small yet still improves efficiency. In this problem, therefore, a method must be evaluated not only by the quality of the whole trade-off but also by **how well it fills the high-stability region**.

The job-shop scheduling problem (JSSP) addressed in this study is NP-hard (Garey et al., 1976), and at practical sizes metaheuristics are used rather than exact methods (Ouelhadj and Petrovic, 2009). Metaheuristics fall into two families (Blum and Roli, 2003): **trajectory-based search**, which advances by transforming a single solution little by little, and **population-based search**, which keeps several solutions and advances by recombining them. For multi-objective JSSPs, rescheduling included, population-based search built on genetic algorithms (GA) is the mainstream (Section 2). In this problem, however, where a high-quality initial solution $S_p$ is given, **might trajectory-based search, which transforms that solution little by little, be better suited to filling the high-stability region?** This question is the starting point of this study.

We test this in a controlled comparison, with the local search aligned, between iterated local search (ILS), representing trajectory-based search, and Memetic, a GA with local search built in, representing population-based search. In addition, we mount **stability-inducing operators** (hereafter, operators), which pull solutions back toward $S_p$, on both methods and examine how their effect changes with the search structure. The main findings are as follows.

- **Trajectory-based search (ILS) fills the high-stability region better than population-based search (Memetic).** Even with the same local search, population-based search fills it only coarsely and in some scenarios barely reaches it at all.
- **The operators lift population-based search to the level of trajectory-based search and add little to trajectory-based search itself.** What the operators fill is the part left unfilled, so trajectory-based search, which fills this region by itself, improves only in the few scenarios where some of it remains unfilled.
- **The best method changes with the criterion emphasized.** Operator-equipped Memetic is best in the quality of the whole trade-off, ILS is best in anytime performance (a good solution is available whenever the search is stopped), and the two are level in the quality of the high-stability region.

Figure 1 shows these findings schematically.

![Figure 1: Schematic of the conclusions of this study.](figures/fig_v2_concept_en.png)

The aim of this study is not a performance race against state-of-the-art methods but to isolate the relationship between search structure and operators by posing hypotheses and controlling the conditions.

---

## 2. RELATED WORK AND HYPOTHESES

Rescheduling studies divide into those that pursue efficiency alone and those that also require stability. That the former are the majority has been pointed out repeatedly (Rangsaritratsamee et al., 2004; Sun et al., 2026), and the latter, which this study presupposes, divides further into two lineages by how stability is secured.

**Lineage that embeds stability in the objective function (objective-embedding).** Attempts to optimize efficiency and stability simultaneously in rescheduling began with Wu et al. (1993), who formulated both as evaluation criteria on a single machine. For the JSSP, Rangsaritratsamee et al. (2004) proposed a memetic-type method combining a GA with local search, and Zhang et al. (2013) a hybrid of a GA and tabu search; for the flexible JSSP (FJSP), Fattahi and Fallahi (2010) likewise applied a memetic-type method. These three combine efficiency and stability into a single scalar objective through a weight. Under Pareto dominance, Valledor et al. (2022) solve a permutation flow shop with three objectives (makespan, total weighted tardiness, and start-time deviation) by an NSGA-II variant, and Luo et al. (2025) solve a dynamic FJSP with two objectives, efficiency and stability, by a parallel NSGA-II. All treat stability as **one term of the objective function**.

**Lineage that secures stability by restricting the search space (scope-limiting).** By contrast, Bean et al. (1991) proposed match-up scheduling, which reschedules only up to the point of returning to the original schedule, and Abumaizar and Svestka (1997) proposed AOR, which reschedules only the affected operations. Zakaria and Petrovic (2012) restrict the chromosome to an interval, and Sun et al. (2026) organize the range of operations allowed to be re-optimized into four levels. All have in common that stability is secured by **restricting the search space**.

**Remaining issues.** Given the two lineages above, three issues remain to our knowledge. **First, the difference between search structures has not been examined.** Across both lineages, the search-based methods concentrate on population-based search (Rangsaritratsamee et al., 2004; Fattahi and Fallahi, 2010; Zakaria and Petrovic, 2012; Zhang et al., 2013; Valledor et al., 2022; Luo et al., 2025; Sun et al., 2026). How the rescheduling-specific property that a high-quality original schedule $S_p$ already exists differentiates the two search structures (trajectory-based versus population-based) has not been analyzed by comparing them under identical conditions. **Second, no evaluation format reveals that difference.** Comparisons have taken the form of scalar values at a few weight settings (Rangsaritratsamee et al., 2004; Zhang et al., 2013), an efficiency measure reported alongside stability (Sun et al., 2026), or indicators over the whole Pareto front such as hypervolume (Valledor et al., 2022; Luo et al., 2025); none measures the quality restricted to the high-stability region. **Third, there is no means of moving a solution toward $S_p$.** The objective-embedding lineage weaves stability into the **evaluation** of a solution, and the scope-limiting lineage cuts the search **space** to guarantee it structurally (at the price of excluding from the search the efficiency–stability trade-off solutions outside the restriction). Neither has a **move** that brings a solution closer to $S_p$.

**Two hypotheses.** To address these issues, this study measures **the quality restricted to the high-stability region** separately from the quality of the whole front (Section 3.4) and poses two hypotheses with it as the response (Figure 1). Trajectory-based search (ILS), which applies continuous transformations starting from $S_p$, can fill the high-stability region in small steps while spreading outward from $S_p$, whereas population-based search (Memetic) advances by creating solutions that recombine the structures of two parents. Because recombination involves jumps that cannot be expressed as small changes from $S_p$, the filling of this region should be structurally coarse even with the same local search: **H1** (Figure 1(a)). If H1 holds, what population-based search lacks is the ability to fill the high-stability region. This study supplies it by designing the move toward $S_p$ as an operator (Section 3.3). The intermediate solutions on a path that **returns individuals scattered away from $S_p$ back to $S_p$ from the outside** should fill this gap and raise the quality of the high-stability region of population-based search to the level that trajectory-based search reaches by itself: **H2** (Figure 1(b)). Whether trajectory-based search, which already fills this region by spreading out from $S_p$, gains anything further is not obvious, so we do not predict the direction of the effect but mount the same operators on both structures and compare the two structures.

---

## 3. PROBLEM FORMULATION AND EXPERIMENTAL DESIGN

This section casts the hypotheses of Section 2 into testable form through the problem formulation (Section 3.1) and the design of the two factors and the responses (Sections 3.2–3.4).

### 3.1 Problem Definition

An operation delay leaves the operation set unchanged and can be handled by readjusting the processing sequences alone, so it is the disruption under which the premise of keeping the revised solution in the high-stability region holds most naturally. This study therefore restricts the disruption to a single operation delay (of length $\Delta$). After an operation delay occurs in an $n$-job, $m$-machine JSSP, we seek a revised schedule $S_q$ given the original schedule $S_p$. The shop floor continues execution by absorbing the delay without changing the machine sequences of $S_p$; we call this actual schedule the **right-shift solution $S_{RSR}$**. Operations that have started on $S_{RSR}$ before the rescheduling time $t_r$ (the time at which the delay is resolved) are **frozen**, and the operations from $t_r$ onward are the **optimization targets**. The decision variables are only the per-machine processing sequences of the target operations, and a solution is evaluated as the semi-active schedule obtained by fixing the frozen part at the front and pushing each operation to its earliest feasible start time at or after $t_r$.

**Stability measure.** Deviation measures fall into two families: (i) start-time deviation and (ii) sequence deviation. Sun et al. (2026), for example, distinguish the two explicitly and adopt the former (the sum of absolute start-time deviations) as their stability measure. This study adopts the latter, because the operations used in the search (the N5 neighborhood and the swap moves of the stability-inducing operators, Sections 3.2–3.3) are all permutation operations and because it is largely independent of MS.

$$
D(S_p, S_q) = \sum_{(i,j) \in \mathcal{O}_{\mathrm{opt}}} \left\lvert r_{i,j}^p - r_{i,j}^q \right\rvert \quad (1)
$$

Here $r_{i,j}$ is the processing rank of job $j$ on machine $i$, and $\mathcal{O}_{\mathrm{opt}}$ is the set of target operations (from $t_r$ onward). $D=0$ corresponds to the sequence-preserving right-shift solution $S_{RSR}$.

**Multi-objective formulation and scalarization.** We solve $\min_{S_q}\,(MS(S_q),\,D(S_p,S_q))$ by a weighted sum with weight $\lambda\in[0,1]$.

$$
F(S_q) = \lambda\,\hat D(S_p, S_q) + (1-\lambda)\,\widehat{MS}(S_q) \quad (2)
$$

Hatted symbols denote min–max normalized values, and a larger $\lambda$ places more weight on stability. This study sweeps $\lambda$ over several points (weighted-sum sweep) and runs an independent search at each point. The search at one point corresponds to the practical operation of rescheduling once with the weight fixed (Rangsaritratsamee et al., 2004). Consequently, the same experiment yields both the speed in practice, by following the search at each point over time, and the quality of the whole trade-off, by combining the solutions of all points (Section 3.4). A weighted sum cannot reach concave parts of a non-convex front (Marler and Arora, 2010), but since all methods share the same scalarization and the solutions in concave parts visited during the search are also recorded by the UEA (Section 3.4), we consider the effect on the structural comparison limited.

### 3.2 Factor 1: Search Structure — ILS and Memetic (H1)

To test H1 (Section 2), we place the **contrast between ILS and Memetic** at the core of the comparison, removing the confound of whether local search is present and contrasting only the search structure (trajectory-based versus population-based).

**Trajectory-based search (ILS).** ILS (Lourenço et al., 2019) is a framework that improves a single solution by iterating three steps: perturbing the current solution, improving the perturbed solution by local search, and deciding whether to accept the result. We adopt ILS among trajectory-based methods because it clearly separates intensification (local search) from escape (perturbation). This separation lets **the perturbation strength, how far the search moves from the current solution at once, be controlled directly as the number of insert moves**, and lets the stability-inducing operators (Section 3.3) be built in naturally as perturbations. The **initial solution** is $S_p$. The **local search**, shared by ILS and Memetic, is the N5 neighborhood, the standard for the JSSP (Nowicki and Smutnicki, 1996): candidates are restricted to adjacent swaps at the ends of critical blocks (moves with a prospect of improving the makespan), and acceptance is evaluated by $F(\lambda)$. The **perturbation** is an insert move whose strength (the number of insert moves) cycles in a sawtooth according to the degree of stagnation (VNS-type; Mladenović and Hansen, 1997). The **acceptance criterion** is better acceptance: the local optimum obtained after a perturbation replaces the current solution only when it strictly improves the best $F(\lambda)$.

**Population-based search (GA / Memetic).** The population-based search that serves as the counterpart for H1 is **Memetic** (Neri and Cotta, 2012), a GA with local search integrated into it, which is the standard configuration in prior work. The underlying GA is a standard one in the operation-based representation (Bierwirth, 1995): a chromosome is a sequence of operations, decoded into a schedule by the GT algorithm (Giffler and Thompson, 1960) and evaluated by $F(\lambda)$. The **initial population** likewise includes $S_p$ as one of its individuals. In each generation, parents are chosen by tournament selection with elitism, and offspring are created by PPX crossover (Bierwirth et al., 1996) and inversion mutation. Memetic applies the N5 above to each individual of this GA and writes the improved solution back to the chromosome (Lamarckian). The plain GA without local search serves as a reference baseline.

### 3.3 Factor 2: Stability-Inducing Operators — PR and Repair (H2)

**Path relinking (PR).** PR is a search method based on the idea that good intermediate solutions lie on the path connecting two high-quality solutions; it is generally combined with scatter search or the like and used to link solutions within an elite pool (Glover et al., 2000; Peng et al., 2015). We adapt it to the structure of rescheduling by **fixing the guiding solution to the single point $S_p$, the pre-disruption schedule**: the path runs from the current local optimum (initiating) toward $S_p$ (guiding). The distance between the two is measured by the number of operations whose ranks in Eq. (1) disagree, $d=\lvert\{(i,j)\in\mathcal{O}_{\mathrm{opt}} : r^p_{i,j}\neq r^q_{i,j}\}\rvert$. One step selects one disagreeing operation and swaps it with the operation currently at rank $r^p_{i,j}$ on the same machine, so that the selected operation comes to its rank in $S_p$. Repeating this reaches $D=0$ within $d$ steps, and the solution with the best $F(\lambda)$ on this path is returned. We fix the guiding solution to $S_p$ because (a) $S_p$ is the optimal endpoint of the stability objective, so PR is interpreted uniquely as a move toward that endpoint, and (b) on the way to $S_p$, solutions that raise stability while limiting the loss in MS can be expected to appear, which directly fills the high-stability region. The operation to swap is chosen at random from the feasible candidates, keeping the number of evaluations at $O(d)$. Best selection, which evaluates all candidates, costs $O(d^2)$; it gained only about +1% in overall HV while taking about 8 times longer under large disruptions, so we do not adopt it (preliminary experiments).

**Stability repair kick (hereafter, repair).** This operator is PR truncated midway. It applies the same swap move $k$ times, for a depth $k$ ($1\le k\le d$), to pull the solution toward $S_p$, that is, to "repair" the stability lost by drifting away from $S_p$, and then re-optimizes the point reached by local search without evaluating the intermediate solutions on the path. When $k$ equals $d$ the solution returns all the way to $S_p$, and when $k$ is small it stays near the current solution, so repeating with varying $k$ fills the range between the current solution and $S_p$ broadly.

**Mounting on both structures.** Both structures **share the core of the operators**, the swap move toward $S_p$; when an operator fires and how its result is returned follow each structure's standard way of building operators in (Table 1). In trajectory-based search (ILS), both operators act as perturbations: they fire upon stagnation, and the repair depth $k$ increases by 1 from 1 to $d$ at each firing and resets to 1 when the upper bound is reached or the best is improved (sawtooth). In population-based search (Memetic), they are built into the refinement of each individual: after the local search, the operator is applied to each individual stochastically (the repair depth $k$ is drawn uniformly from 1 to $d$), and the result, re-optimized by N5, is returned to the population unconditionally; acceptance is left to tournament selection. Hereafter, the configurations without operators are called **ILS-baseline** and **Memetic-baseline**, and each baseline together with its PR and repair versions is referred to as **the ILS variants** and **the Memetic variants** (three methods each).

### 3.4 Responses: Evaluation Metrics and Hypothesis Testing

We measure the hypervolume (HV) of the Pareto front (Zitzler and Thiele, 1999) under an unbounded external archive (**UEA**; Ishibuchi et al., 2020) that keeps the non-dominated solutions among all solutions each search generates **before they are subjected to selection**. The recording points are aligned across the two structures: solutions are recorded both immediately after every operation that moves a solution (the perturbation of trajectory-based search, the crossover and mutation of population-based search, and the operators of both) and after the local search. An HV that measures the whole front at once buries the difference in whether the high-stability region is filled. This study therefore measures, in addition to this HV, the quality restricted to this region and anytime performance, the availability of a good solution whenever the search is stopped:

- **Overall HV** (quality of the whole trade-off): the hypervolume over the whole region.
- **High-stability HV** (quality of the high-stability region): the hypervolume restricted to the high-stability region $D<P_\theta$. $P_\theta$ is the $\theta$-th percentile of $D$ over the Pareto solutions pooled over all methods and all trials; the value of $\theta$ and its sensitivity are treated in Section 4.1 (the shaded region in Figures 1 and 2).
- **AOC** (anytime performance; López-Ibáñez and Stützle, 2014): the time average of the per-weight HV-versus-log-time curve. Because the average is taken over log time, it weights the early rise of a search heavily.

For cross-scenario comparison, HV is computed after affine normalization to $[0,1]^2$ in each scenario, with reference point $(1.1,1.1)$. Solutions that keep the machine sequences of $S_p$ ($D=0$, such as $S_{RSR}$) are trivial solutions that every method obtains without searching, so they are excluded from the HV computation. Overall HV and high-stability HV are computed after merging, in each trial, the solutions of all points of the weight sweep. AOC, conversely, takes **the search at each weight as its unit**: the value averaged over wall-clock log-time at each weight is then averaged over all weights for each trial (the time window is common to all methods at each weight).

**Statistics.** Method comparisons within each scenario are evaluated by the Mann–Whitney U test and the effect size Cliff's $\delta$ (Derrac et al., 2011), and all tests are two-sided. $|\delta|$=1.0 means complete separation: every trial of one method exceeds every trial of the other. The magnitude of a difference is reported as the ratio or relative difference of medians.

**Correspondence between hypotheses and tests.** Hypotheses are posed on high-stability HV alone. H1 is tested by comparing ILS-baseline and Memetic-baseline, which share the local search, and is regarded as supported if ILS is significantly ahead in all eight scenarios. H2 is tested in the same form on the gain of operator-equipped Memetic over Memetic-baseline, and the level the gain reaches is checked by comparison with ILS-baseline. Multiple comparisons are handled by Holm correction, with the same comparison repeated across the eight scenarios forming one family.

---

## 4. COMPUTATIONAL EXPERIMENTS

This section first gives the experimental setup (Section 4.1) and then presents the results metric by metric.

![Figure 2: Actual Pareto fronts for the ta21 pair (non-dominated solutions of the trial with the median overall HV). Repair is omitted because it gives almost the same front as PR in the high-stability region.](figures/fig_v2_front_en.png)

### 4.1 Experimental Setup

We use eight scenarios obtained by imposing a completion delay of a single operation ($\Delta$=60–148, about 0.9–1.5 times the processing time of the affected operation) on five benchmark instances (mt10, la21, la36, la40, ta21): la36S/la36M/la36L (27/54/73%), ta21S/ta21L (32/82%), mt10 (72%), la21 (35%), and la40 (32%). The percentage in parentheses is the **rescheduling ratio** $\rho=n_{res}/\text{ops}$, the fraction of all operations that become re-optimization targets $\mathcal{O}_{opt}$ under the disruption; the eight scenarios span 27–82%. For each instance the original schedule $S_p$ is generated by a 500-generation GA. The **la36 ladder** and the **ta21 pair** vary only the magnitude of the disruption stepwise on the same instance with the same $S_p$. The complete definition of each scenario, raw per-trial values, and tests are provided in the public repository (https://github.com/kitotakumi/stability_scheduling). The experimental settings are listed in Table 1.

**Table 1.** Experimental settings (common to all eight scenarios)

| Item | Setting |
| --- | --- |
| Weight sweep | 10 points, $\lambda$ = 0, 0.1, …, 0.9 ($\lambda$=1.0 is excluded because it degenerates to the trivial solution $S_{RSR}$) |
| Trials | 10 per (scenario × method) |
| Computational budget | 3000 iterations for every ILS variant, 500 generations for every GA/Memetic variant |
| Search | GA/Memetic variants: $cx_{pb}$=0.85, $mut_{pb}$=0.1, pop=50 ($S_p$ plus 49 randomly generated individuals). ILS variants: insert perturbation of 2–5 moves, cycled in a sawtooth |
| Operators | Memetic variants: fire with probability 0.3 per individual, and PR returns the best of the top 3 intermediate solutions on the path. ILS variants: fire first after 400 iterations without improvement and then every 10, and PR returns the single best solution on the path |
| Environment | AMD Ryzen 5 7530U; Python 3.12 (NumPy, DEAP, SciPy). All random seeds fixed |

**Validity checks.** The boundary of the high-stability region is set at $\theta$=50, so $P_{50}$ is the median of $D$ over the Pareto solutions pooled across all methods and trials. The boundary is defined from the pool of compared methods, but varying $\theta$ over 33–67 leaves the conclusions of H1 and H2 unchanged (complete separation in every scenario, $p<$0.001). Although Memetic+PR takes 4.8–9.4 times as long as ILS in wall-clock time, the median run of every method reaches 99% of its final HV within 70% of its budget, so differences in overall HV are not due to computation time. The method parameters were fixed at standard values, and the 11 axes that determine behavior, including those of the base search structures, were varied one at a time above and below the central value in two scenarios; the sensitivity (+0.5–2% in overall HV) is an order of magnitude smaller than the differences among methods (+7–54%).

### 4.2 High-Stability HV: ILS Fills the Region Alone, the Population Fills It with Operators (H1, H2)

Memetic-baseline fails to fill the region in two ways (Figure 2). (a) In ta21S it has not a single non-dominated solution in the high-stability region, and needs $D$=16 to match the MS that ILS attains at $D$=2. (b) In ta21L it does enter the region but is inferior to ILS in MS, and the two are level only at the efficiency end. In either case, once the operators are mounted, the fronts of the two structures overlap in the high-stability region, and in five of the eight scenarios the non-dominated solutions of ILS+PR and Memetic+PR coincide exactly as point sets.

![Figure 3: Interaction plot of search structure × stability-inducing operator (high-stability HV; medians and interquartile ranges; n=10). Symbols give the significance level of the two-sided Mann–Whitney U test against the configuration without operators, in the same color as the corresponding line (\* $p<$0.05, \*\* $p<$0.01, \*\*\* $p<$0.001; no correction for multiple comparisons).](figures/fig_v2_interaction_en.png)

We check whether this picture holds in all eight scenarios (Figure 3). The gap between the points without operators corresponds to H1, and the slope of the Memetic line to H2.

- **H1: ILS-baseline exceeds Memetic-baseline in all eight scenarios with complete separation** ($p<$0.001, $|\delta|$=1.0). In three scenarios (la36S, ta21S, la40), Memetic-baseline has almost no non-dominated solution in the high-stability region (0 points in 9–10 of the 10 trials; Figure 2(a)), and only ILS fills the region. In the remaining five scenarios, ILS is 2.0–4.5 times higher. Significance survives Holm correction in every scenario, and H1 is supported.
- **H2: The operators raise the high-stability HV of Memetic in every trial of all eight scenarios and bring it to or above the level of ILS-baseline in seven of the eight.** The gain is a complete separation in every scenario ($p<$0.001, $|\delta|$=1.0): in the three scenarios where Memetic-baseline had almost no solution in the high-stability region, solutions appear, and in the remaining five the high-stability HV becomes 2.1–4.7 times larger. Compared with ILS-baseline, the level reached shows no significant difference in six scenarios (in five of them the medians differ by at most 1%), is 4–5% higher in ta21L, and 5–10% lower in la40 (Holm-adjusted $p_{\text{adj}}<$0.015). H2 is supported except for the one scenario that falls short. Whether the gain is an effect of perturbation in general or one specific to the direction toward $S_p$ was checked by a control experiment in two scenarios (mt10, la36S) that matches the strength and randomizes only the direction: a random direction reproduces only about 70% of the gain, falls short of the operators in high-stability HV with complete separation ($p<$0.001), and needs 1.7–2.4 times as long to reach the same level.
- **Adding the operators to ILS yields almost no gain.** Operator-equipped ILS exceeds its baseline in only two of the eight scenarios (la36L, ta21L). For both PR and repair, $p$=0.013 in la36L and $p<$0.001 in ta21L, but after Holm correction only ta21L remains ($p_{\text{adj}}<$0.002; la36L is 0.091). Even in ta21L the gain is small, the median rising from 0.029 to 0.031 (although every trial with an operator exceeds every trial of the baseline, $|\delta|$=1.0). In the remaining six scenarios there is no difference.

**When the operators help.** That the same operators act asymmetrically across the structures reflects less a difference between the structures themselves than a difference in whether each structure reaches the attainable level by itself. High-stability HV has a level at which several methods arrive and stop: in every scenario, three to five of the seven methods lie within 2% of the best value (Figure 3). Whether mounted on ILS or on Memetic, the operators raise high-stability HV if the structure has not reached this level, and yield no gain if it has. Because the effect changes with whether the level was reached even within the ILS variants, where the firing design is the same, the asymmetry cannot be explained by the firing design differing between the structures (Table 1).

### 4.3 Overall HV and AOC: The Best Method Changes with the Criterion Emphasized

We first look at the anytime overall-HV curves of two representative scenarios, which show both the level reached and how quickly it is reached (Figure 4).

In ta21S, operator-equipped Memetic reaches the same level as ILS (1.17), but it takes 6.6–7.8 s to get there against 2.5 s for ILS. In la36L, Memetic overtakes ILS at 7–10 s and climbs to 0.94, whereas ILS levels off at 0.80. The endpoint of the curve and the speed of the ramp-up point to different methods as the best.


**Table 2.** Cross-scenario summary of the three metrics. Each cell gives "average rank / ARPD% (relative deviation from the best method) / number of top-group scenarios out of 8 (scenarios in which the method shows no significant difference from the best method of that scenario; uncorrected)"; the first two are averages over the eight scenarios (lower is better); for the last, higher is better. Because the eight scenarios are not independent of one another (Section 4.1), the cross-scenario ranks are an exploratory summary of rank consistency.

| Method | Overall HV | High-stability HV | AOC |
| --- | --- | --- | --- |
| ILS-baseline | 4.44 / 10 / 1 | 3.44 / 2 / 6 | 2.75 / 4 / 6 |
| ILS+repair | 3.56 / 9 / 2 | 2.62 / 0 / 8 | 2.62 / 5 / 6 |
| ILS+PR | 3.75 / 10 / 3 | 2.75 / 0 / 8 | 2.50 / 4 / 6 |
| GA | 6.94 / 25 / 0 | 6.38 / 71 / 0 | 7.00 / 48 / 0 |
| Memetic-baseline | 4.81 / 5 / 2 | 6.62 / 78 / 0 | 3.50 / 18 / 2 |
| Memetic+repair | 2.50 / 1 / 6 | 3.31 / 2 / 6 | 4.00 / 22 / 1 |
| Memetic+PR | 2.00 / 0 / 7 | 2.88 / 1 / 6 | 5.62 / 27 / 1 |

**Overall HV.** The best level is reached by operator-equipped Memetic, which also occupies the top group in Table 2. As in Section 4.2, the operators lift only Memetic: they raise its overall HV significantly in six of the eight scenarios (+2.8–11.7%), whereas for ILS the change stays within ±0.5% in every scenario. Which structure wins changes with the scenario: in la36S, ta21S, la40, and la21 (hereafter group A) ILS-baseline is significantly ahead of Memetic-baseline, whereas in la36M, la36L, mt10, and ta21L (hereafter group B) Memetic-baseline is ahead instead (in one of the four the difference is not significant). The ILS variants are level with the best only in group A; in group B they fall 5–38% short. The difference in the high-stability region in Section 4.2 is therefore not a reflection of ILS being superior overall. The same split appears in the minimum MS reached over the whole sweep: in group A all methods except GA reach the same value, whereas in la36M, la36L, and mt10 of group B the Memetic variants are 1.2–2.2% lower (significant for all three pairs with matching operators, $p<$0.01; in ta21L the gap is within 0.6%).

**AOC.** The three ILS variants are level with the best, lying within 2% of it in six scenarios (Table 2; in the remaining la36S and mt10, Memetic-baseline is best and the ILS variants fall 10–22% short). The fast start of the ILS variants can be attributed to their trajectory, which improves the single good solution $S_p$ in small steps (Figure 4). The operators do not impair this fast start: in ILS, where they fire only upon stagnation, the three variants stay within 3% of one another in every scenario, with no significant difference. In Memetic, by contrast, they come at a cost: PR traces the path all the way to $S_p$ and re-optimizes the intermediate solutions, so each call is expensive, and because it fires per individual it lowers AOC significantly in five of the eight scenarios (by 7–26%).

**PR or repair.** For ILS the choice hardly matters: the two differ by at most 2% on all three metrics, and the difference is significant only in two scenarios for overall HV, in opposite directions there. For Memetic the answer depends on the criterion. PR is slightly ahead in final quality: the significant differences are few (2/8 scenarios for overall HV, 3/8 for high-stability HV), but all of them favor PR. In AOC the order reverses, and repair significantly exceeds PR in 6/8 scenarios.

![Figure 4: Anytime curves of overall HV, measured on the points visited up to time $t$ (medians over trials). AOC is the average of the per-weight curves and does not equal the area under the weight-merged curves shown here.](figures/fig_v2_anytime_en.png)

---

## 5. CONCLUSIONS

In rescheduling, the original schedule $S_p$ from before the disruption is in hand before the search begins. With the aim of securing stability by making use of it, this study examined the relationship between search structure and stability-inducing operators through a controlled comparison over 8 scenarios × 7 methods × n=10 trials, with the local search, the objective function, the supply of $S_p$, and the budget all aligned. The main findings are as follows.

1. **H1: Trajectory-based search (ILS) fills the high-stability region better than population-based search (Memetic).** Trajectory-based search was ahead in high-stability HV in every trial of all eight scenarios: in three of them population-based search barely reaches the region at all, and in the remaining five the gap is a factor of 2.0–4.5.
2. **H2 and the cross-structure comparison: the operators lift population-based search to the level of trajectory-based search and add little to trajectory-based search itself.** High-stability HV has a level at which several methods arrive and stop. Population-based search does not reach this level by itself, but with the operators mounted its high-stability HV rises in every trial and attains that level in seven of the eight scenarios. Trajectory-based search, which reaches the level by itself, gains little from the operators; they improve it only in the two scenarios where it falls short of the level on its own. Whether the operators help is thus decided not by the difference between the structures but by whether the structure reaches this level by itself.
3. **Practical implication: the best method changes with the criterion emphasized.** Memetic with stability-inducing operators excels in the quality of the whole trade-off (overall HV), both ILS and operator-equipped Memetic in the quality of the high-stability region (high-stability HV), and the ILS variants in anytime performance (AOC). Hence one chooses operator-equipped Memetic to pursue the quality of the whole trade-off when computation time is affordable, and ILS to produce a revised plan with few changes quickly (when efficiency alone matters, the Memetic variants are at least as good). Of the operators mounted on Memetic, PR is slightly ahead in final quality and repair in anytime performance (Section 4.3).

The central contribution of this study is to establish an evaluation format that judges a rescheduling method by **the quality restricted to the high-stability region**, and, on that basis, to show in a controlled comparison with the local search aligned that population-based search, the mainstream choice, does not fill this region fully, and that supplying the move toward $S_p$ as an **operator** makes up the shortfall. The value of an operator is decided by whether the structure that mounts it already fills this region by itself. A report that an operator is effective cannot be generalized without reference to the structure it was mounted on.

**Limitations.** (i) Comparisons whose $p$ lies at the edge of the significance level are not settled at n=10. (ii) Validity under start-time deviation has not been examined. (iii) The disruptions are limited to operation delays. (iv) The effect of the quality of $S_p$ on the results has not been examined. (v) Each search structure is represented by a single method. In addition to re-verification with these conditions varied, future work includes integration of the scope-limiting approach with the operators, verification of a scheme that builds stability into the crossover itself, and re-verification with Pareto-native methods such as NSGA-II.

---

## REFERENCES

Abumaizar, R. J. and Svestka, J. A. (1997) Rescheduling job shops under random disruptions. *International Journal of Production Research*, 35, 2065-2082.

Bean, J. C., Birge, J. R., Mittenthal, J., and Noon, C. E. (1991) Matchup scheduling with multiple resources, release dates and disruptions. *Operations Research*, 39, 470-483.

Bierwirth, C. (1995) A generalized permutation approach to job shop scheduling with genetic algorithms. *OR Spektrum*, 17, 87-92.

Bierwirth, C., Mattfeld, D. C., and Kopfer, H. (1996) On permutation representations for scheduling problems. In *Parallel Problem Solving from Nature—PPSN IV*, LNCS 1141, Springer, 310-318.

Blum, C. and Roli, A. (2003) Metaheuristics in combinatorial optimization: Overview and conceptual comparison. *ACM Computing Surveys*, 35, 268-308.

Derrac, J., García, S., Molina, D., and Herrera, F. (2011) A practical tutorial on the use of nonparametric statistical tests as a methodology for comparing evolutionary and swarm intelligence algorithms. *Swarm and Evolutionary Computation*, 1, 3-18.

Fattahi, P. and Fallahi, A. (2010) Dynamic scheduling in flexible job shop systems by considering simultaneously efficiency and stability. *CIRP Journal of Manufacturing Science and Technology*, 2, 114-123.

Garey, M. R., Johnson, D. S., and Sethi, R. (1976) The complexity of flowshop and jobshop scheduling. *Mathematics of Operations Research*, 1, 117-129.

Giffler, B. and Thompson, G. L. (1960) Algorithms for solving production-scheduling problems. *Operations Research*, 8, 487-503.

Glover, F., Laguna, M., and Martí, R. (2000) Fundamentals of scatter search and path relinking. *Control and Cybernetics*, 29, 653-684.

Ishibuchi, H., Pang, L. M., and Shang, K. (2020) A new framework of evolutionary multi-objective algorithms with an unbounded external archive. In *Proceedings of the 24th European Conference on Artificial Intelligence (ECAI 2020)*, IOS Press, 283-290.

López-Ibáñez, M. and Stützle, T. (2014) Automatically improving the anytime behaviour of optimisation algorithms. *European Journal of Operational Research*, 235, 569-582.

Lourenço, H. R., Martin, O. C., and Stützle, T. (2019) Iterated local search: Framework and applications. In Gendreau, M. and Potvin, J.-Y. (eds), *Handbook of Metaheuristics*, 3rd ed., Springer, 129-168.

Luo, J., El Baz, D., Xue, R., Hu, J., and Shi, L. (2025) A fully parallel multi-objective genetic algorithm for optimization of flexible shop floor production performance and schedule stability under dynamic environments. *Annals of Operations Research*, 351, 489-524.

Marler, R. T. and Arora, J. S. (2010) The weighted sum method for multi-objective optimization: New insights. *Structural and Multidisciplinary Optimization*, 41, 853-862.

Mladenović, N. and Hansen, P. (1997) Variable neighborhood search. *Computers & Operations Research*, 24, 1097-1100.

Neri, F. and Cotta, C. (2012) Memetic algorithms and memetic computing optimization: A literature review. *Swarm and Evolutionary Computation*, 2, 1-14.

Nowicki, E. and Smutnicki, C. (1996) A fast taboo search algorithm for the job shop problem. *Management Science*, 42, 797-813.

Ouelhadj, D. and Petrovic, S. (2009) A survey of dynamic scheduling in manufacturing systems. *Journal of Scheduling*, 12, 417-431.

Peng, B., Lü, Z., and Cheng, T. C. E. (2015) A tabu search/path relinking algorithm to solve the job shop scheduling problem. *Computers & Operations Research*, 53, 154-164.

Rangsaritratsamee, R., Ferrell Jr, W. G., and Kurz, M. B. (2004) Dynamic rescheduling that simultaneously considers efficiency and stability. *Computers & Industrial Engineering*, 46, 1-15.

Sun, R., Cheng, G., Ding, Q., and Zhao, X. (2026) Impact of optimization scope on solution quality and stability in dynamic flexible job shop rescheduling. *Computers & Industrial Engineering*, 215, Article 111943.

Valledor, P., Gómez, A., Puente, J., and Fernández, I. (2022) Solving rescheduling problems in dynamic permutation flow shop environments with multiple objectives using the hybrid dynamic non-dominated sorting genetic II algorithm. *Mathematics*, 10, 2395.

Vieira, G. E., Herrmann, J. W., and Lin, E. (2003) Rescheduling manufacturing systems: A framework of strategies, policies, and methods. *Journal of Scheduling*, 6, 39-62.

Wu, S. D., Storer, R. H., and Chang, P.-C. (1993) One-machine rescheduling heuristics with efficiency and stability as criteria. *Computers & Operations Research*, 20, 1-14.

Zakaria, Z. and Petrovic, S. (2012) Genetic algorithms for match-up rescheduling of the flexible manufacturing systems. *Computers & Industrial Engineering*, 62, 670-686.

Zhang, L., Gao, L., and Li, X. (2013) A hybrid genetic algorithm and tabu search for a multi-objective dynamic job shop scheduling problem. *International Journal of Production Research*, 51, 3516-3531.

Zitzler, E. and Thiele, L. (1999) Multiobjective evolutionary algorithms: A comparative case study and the strength Pareto approach. *IEEE Transactions on Evolutionary Computation*, 3, 257-271.
