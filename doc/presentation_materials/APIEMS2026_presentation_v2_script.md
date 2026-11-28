# APIEMS 2026 発表原稿（英語版 v2）

対象のスライドは `APIEMS2026_presentation_v2.pptx` の本編（1〜25 枚目）です。
全体で約 1,243 語あり，1 分 120 語で読むと約 10 分 50 秒かかります（図を指す合図 1 つにつき 2 秒を含む）。
各スライドの見出しの秒数は同じ速さで数えた目安で，累計はそこまでの合計です。

**通しで計るときの区切り**

| 区切り | 累計の目安 |
|---|---|
| 10 枚目：1 章（仮説）の終わり | 4:25 |
| 15 枚目：2 章（比較手法と検定）の終わり | 7:10 |
| 21 枚目：3 章（追加の評価）の終わり | 9:55 |
| 24 枚目：最後（限界と今後の課題） | 10:50 |

**読み方**

| 書き方 | 読み方 |
|---|---|
| S p | エス・ピー（初期スケジュール $S_p$） |
| ILS | アイ・エル・エス |
| Memetic | ミメティック（/mɪˈmɛtɪk/） |
| HV | エイチ・ヴィー |
| PR / PR-k | ピー・アール／ピー・アール・ケー |
| N5 | エヌ・ファイブ |
| ta21L | ティー・エー・トゥエンティワン・エル |

〔 〕は読まない合図で，そこで図や箱を指します。

---

## 1. Title（約 15 秒・累計 0:15）

Hello, everyone. My name is Takumi Kito, from Waseda University.

Today, I'd like to present our study, Trajectory-Based and Population-Based Search with Path Relinking for Job-Shop Rescheduling under Sequence-Deviation Stability.

## 2. Outline（約 5 秒・累計 0:20）

This talk has four parts. Let me start with the background.

## 3. Disruptions and Rescheduling（約 20 秒・累計 0:40）

Disruptions such as machine breakdowns and operation delays can stop a schedule from running as planned.

In this study, when a disruption occurs, we revise the pre-disruption schedule. We call it the initial schedule, S p. This approach is called predictive-reactive rescheduling.

## 4. Rescheduling and Stability（約 30 秒・累計 1:10）

〔ガント〕Changing the job order on a machine after a delay, as in this Gantt chart, can shorten the makespan.

〔右〕But on the shop floor, it causes problems like setup changeovers and reassigning workers.

So rescheduling is a bi-objective problem. We minimize the makespan for efficiency, and the change from S p for stability.

## 5. Pareto Front and Hypervolume (HV)（約 40 秒・累計 1:50）

With two objectives, there is no single best solution. Both objectives are minimized, so the lower left is better.

A solution is non-dominated if no other solution is better in both objectives. In the figure, nothing lies to its lower left. These solutions form the Pareto front.

We measure the quality of a front by the hypervolume, HV. It is the area dominated by the front, up to a reference point. A larger HV means a better front.

## 6. Focus and Evaluation Metric: High-Stability HV（約 30 秒・累計 2:20）

When schedule changes are costly, we need highly stable solutions, that is, solutions close to S p.

〔網掛け〕So we focus on the high-stability region. It is the half of all methods' non-dominated solutions that are closest to S p.

Our main metric is the high-stability HV, the HV within this region.

## 7. Starting Point: A Peculiarity of Rescheduling（約 45 秒・累計 3:05）

Our starting point is a special feature of rescheduling. The initial schedule S p is available without any search. Its change is zero, the best stability, and its makespan was already optimized before the disruption.

We think this S p may strongly affect which search structure is better, especially in high-stability HV. Yet this point has received little attention.

〔下の 2 つの箱〕Metaheuristics have two search structures. Trajectory-based search, such as ILS, changes one solution little by little. Population-based search, such as GA, keeps several solutions and recombines them.

## 8. Related Work: Two Approaches to Securing Stability（約 20 秒・累計 3:25）

Prior work secures stability in two ways. One embeds stability in the objective, as a term to minimize. The other limits the rescheduling scope, for example to the operations affected by the disruption.

In both approaches, the methods are mostly GA-based, population-based search.

## 9. Three Issues Remaining in Related Work（約 30 秒・累計 3:55）

This leaves three issues.

First, trajectory-based and population-based search have not been compared. In particular, the effect of S p on them has not been analyzed.

Second, methods are compared by scalar values at a few weights, or by the HV of the whole front. So highly stable solutions are not evaluated in detail.

Third, there is no operator that creates highly stable solutions.

## 10. Hypotheses（約 30 秒・累計 4:25）

We set two hypotheses.

Hypothesis 1: trajectory-based search has a larger high-stability HV than population-based search. This is because crossover jumps far from S p, while trajectory-based search changes S p little by little.

Hypothesis 2: stability-inducing operators raise the high-stability HV of population-based search. They move solutions toward S p, and yield small-change solutions on the way.

## 11. Outline (Part 2)（約 5 秒・累計 4:30）

Next, the problem setting and experimental design.

## 12. Problem Setting（約 40 秒・累計 5:10）

We consider a job-shop scheduling problem in which one operation is delayed.

The decision variables are the sequences on each machine after the rescheduling time.

Stability is measured by the sequence deviation D, the sum of rank differences from S p.

〔右の例〕In this example, J2 and J4 each move two ranks, so D is four.

We combine the makespan and D into a weighted sum F, sweep the weight lambda, and evaluate all non-dominated solutions by HV.

## 13. Trajectory-Based vs. Population-Based Search（約 40 秒・累計 5:50）

We compare one method from each structure.

〔左〕For trajectory-based search, we use iterated local search, ILS. Starting from S p, it repeats perturbation and local search, and accepts a new solution if F improves.

〔右〕For population-based search, we use a memetic algorithm, a GA with local search on each individual.

To isolate the effect of structure, both share the same local search, N5, the same objective F, and the same S p.

## 14. Proposed: Stability-Inducing Operators via Path Relinking (PR)（約 50 秒・累計 6:40）

Our stability-inducing operators use path relinking, PR. Usually, PR links two good solutions. We instead link the current solution with S p, so the operators move solutions toward S p.

〔右の例〕In this example on one machine, the current sequence has D equal to four. Swapping J4 and J3 gives D equal to two, and swapping J3 and J2 reaches S p.

〔左の図〕PR returns the solution with the smallest F on this path. PR-k is a lighter version to reduce computation time. It skips evaluating the path and just returns the solution after k swaps.

## 15. Compared Methods and Hypothesis Tests（約 30 秒・累計 7:10）

For each of ILS and Memetic, we use three versions: without operators, with PR-k, and with PR. That makes six methods.

We test both hypotheses on high-stability HV. Hypothesis 1 is supported if ILS significantly exceeds Memetic in all scenarios. Hypothesis 2 is supported if Memetic with PR-k and Memetic with PR each significantly exceed Memetic in all scenarios.

## 16. Outline (Part 3)（約 5 秒・累計 7:10）

Now, the computational experiments.

## 17. Experimental Setup（約 10 秒・累計 7:25）

We built eight scenarios by delaying a single operation in five benchmark instances, and ran each method ten times at each of the ten weights.

## 18. High-Stability HV: Pareto Fronts in Scenario ta21L（約 35 秒・累計 8:00）

First, the Pareto fronts in scenario ta21L. The star marks S p, and the shaded area is the high-stability region.

〔左〕In panel a, without operators, Memetic's whole front is worse than that of ILS, and it has almost no solutions in the high-stability region.

〔右〕In panel b, with PR, Memetic's front catches up with that of ILS, while ILS improves only slightly.

## 19. High-Stability HV: All 8 Scenarios (Hypothesis Tests)（約 45 秒・累計 8:45）

This figure shows the high-stability HV in all eight scenarios, with the best value scaled to one. Orange is ILS, and green is Memetic.

〔H1 の矢印〕For Hypothesis 1, look at the gap at None. Without operators, every trial of ILS exceeds every trial of Memetic, in all eight scenarios. So Hypothesis 1 is supported.

〔H2 の矢印〕For Hypothesis 2, look at how the green line rises. With the operators, Memetic improves in every scenario, again in every trial. So Hypothesis 2 is also supported.

## 20. High-Stability HV: Effect of the Operators（約 30 秒・累計 9:15）

With the operators, Memetic rises from less than half of the best value to ninety percent or more, reaching the level of ILS in seven of the eight scenarios.

ILS is already near the best value in six of the eight scenarios, and improves only in the other two.

So the operators help only where the search without them falls short of the best value.

## 21. Additional Evaluation: Overall HV over Time（約 40 秒・累計 9:55）

〔左の図〕Apart from the hypotheses, this figure shows the overall HV, the quality of the whole front, over computation time.

In ta21L, ILS rises fastest. Memetic lags, and the operators slow it further, because they add computation time.

In terms of the final value, however, the operators lift Memetic to the top, while ILS barely changes.

Most other scenarios show the same tendency. So the best method depends on whether speed or the final value matters.

## 22. Outline (Part 4)（約 5 秒・累計 10:00）

Finally, the conclusions.

## 23. Summary（約 35 秒・累計 10:35）

To summarize.

First, Hypothesis 1 was supported. In high-stability HV, trajectory-based search, ILS, far exceeds population-based search, Memetic.

Second, Hypothesis 2 was supported. Stability-inducing operators raise the high-stability HV of population-based search to the level of trajectory-based search.

Third, on choosing a method. For whole-front quality, use population-based search with the operators. For high-stability quality, use trajectory-based search, or population-based search with the operators. For short computation times, use trajectory-based search.

## 24. Limitations and Future Work（約 15 秒・累計 10:50）

These conclusions hold under the conditions of this experiment, such as single-operation delays and one method for each structure. Re-examining them under other conditions is future work.

Thank you for your attention.
