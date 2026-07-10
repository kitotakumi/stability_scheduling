# -*- coding: utf-8 -*-
"""APIEMS 2026 English manuscript content.

Faithful translation of apiems2026_manuscript.md (Japanese draft reviewed by
the author), tightened at the phrasing level only: every claim, number,
citation, emphasis, and caveat of the Japanese source is preserved.
Citations follow the template's author-year style; references alphabetical.
"""

_REFS = [
    ('ref', 'Abumaizar, R. J. and Svestka, J. A. (1997) Rescheduling job shops under random disruptions. *International Journal of Production Research*, **35**, 2065-2082.'),
    ('ref', 'Bean, J. C., Birge, J. R., Mittenthal, J., and Noon, C. E. (1991) Matchup scheduling with multiple resources, release dates and disruptions. *Operations Research*, **39**, 470-483.'),
    ('ref', 'Bierwirth, C. (1995) A generalized permutation approach to job shop scheduling with genetic algorithms. *OR Spektrum*, **17**, 87-92.'),
    ('ref', 'Bierwirth, C., Mattfeld, D. C., and Kopfer, H. (1996) On permutation representations for scheduling problems. *Parallel Problem Solving from Nature—PPSN IV*, LNCS 1141, Springer, 310-318.'),
    ('ref', 'Blum, C. and Roli, A. (2003) Metaheuristics in combinatorial optimization: overview and conceptual comparison. *ACM Computing Surveys*, **35**, 268-308.'),
    ('ref', 'Giffler, B. and Thompson, G. L. (1960) Algorithms for solving production-scheduling problems. *Operations Research*, **8**, 487-503.'),
    ('ref', 'Glover, F., Laguna, M., and Martí, R. (2000) Fundamentals of scatter search and path relinking. *Control and Cybernetics*, **29**, 653-684.'),
    ('ref', 'Ishibuchi, H., Pang, L. M., and Shang, K. (2020) A new framework of evolutionary multi-objective algorithms with an unbounded external archive. *Proceedings of the 24th European Conference on Artificial Intelligence (ECAI 2020)*, IOS Press, 283-290.'),
    ('ref', 'Katragjini, K., Vallada, E., and Ruiz, R. (2013) Flow shop rescheduling under different types of disruption. *International Journal of Production Research*, **51**, 780-797.'),
    ('ref', 'López-Ibáñez, M. and Stützle, T. (2014) Automatically improving the anytime behaviour of optimisation algorithms. *European Journal of Operational Research*, **235**, 569-582.'),
    ('ref', 'Lourenço, H. R., Martin, O. C., and Stützle, T. (2019) Iterated local search: framework and applications. In Gendreau, M. and Potvin, J.-Y. (eds.), *Handbook of Metaheuristics*, 3rd ed., Springer, 129-168.'),
    ('ref', 'Marler, R. T. and Arora, J. S. (2010) The weighted sum method for multi-objective optimization: new insights. *Structural and Multidisciplinary Optimization*, **41**, 853-862.'),
    ('ref', 'Mladenović, N. and Hansen, P. (1997) Variable neighborhood search. *Computers & Operations Research*, **24**, 1097-1100.'),
    ('ref', 'Neri, F. and Cotta, C. (2012) Memetic algorithms and memetic computing optimization: A literature review. *Swarm and Evolutionary Computation*, **2**, 1-14.'),
    ('ref', 'Nowicki, E. and Smutnicki, C. (1996) A fast taboo search algorithm for the job shop problem. *Management Science*, **42**, 797-813.'),
    ('ref', 'Ouelhadj, D. and Petrovic, S. (2009) A survey of dynamic scheduling in manufacturing systems. *Journal of Scheduling*, **12**, 417-431.'),
    ('ref', 'Peng, B., Lü, Z., and Cheng, T. C. E. (2015) A tabu search/path relinking algorithm to solve the job shop scheduling problem. *Computers & Operations Research*, **53**, 154-164.'),
    ('ref', 'Rangsaritratsamee, R., Ferrell Jr, W. G., and Kurz, M. B. (2004) Dynamic rescheduling that simultaneously considers efficiency and stability. *Computers & Industrial Engineering*, **46**, 1-15.'),
    ('ref', 'Sörensen, K. (2015) Metaheuristics—the metaphor exposed. *International Transactions in Operational Research*, **22**, 3-18.'),
    ('ref', 'Sun, R., Cheng, G., Ding, Q., and Zhao, X. (2026) Impact of optimization scope on solution quality and stability in dynamic flexible job shop rescheduling. *Computers & Industrial Engineering*, **215**, Article 111943.'),
    ('ref', 'Wu, S. D., Storer, R. H., and Chang, P.-C. (1993) One-machine rescheduling heuristics with efficiency and stability as criteria. *Computers & Operations Research*, **20**, 1-14.'),
    ('ref', 'Zakaria, Z. and Petrovic, S. (2012) Genetic algorithms for match-up rescheduling of the flexible manufacturing systems. *Computers & Industrial Engineering*, **62**, 670-686.'),
    ('ref', 'Zhang, L., Gao, L., and Li, X. (2013) A hybrid genetic algorithm and tabu search for a multi-objective dynamic job shop scheduling problem. *International Journal of Production Research*, **51**, 3516-3531.'),
]

BLOCKS = [
    ('title', 'Asymmetric Effects of Stability-Inducing Operators across '
              'Trajectory and Population Search in Job-Shop Rescheduling'),
    ('authors', ['Takumi Kito']),
    ('affil', ['Department of Industrial and Management Systems Engineering, '
               'Graduate School of Creative Science and Engineering, Waseda '
               'University, Tokyo, Japan',
               'Tel: (+81) 80-4756-3741, Email: kito@toki.waseda.jp']),
    ('abstract',
     'In predictive-reactive rescheduling, a high-quality preschedule $S_p$ already exists '
     'before the disruption, and revised schedules must be both efficient (makespan) and '
     'stable (small deviation from $S_p$). Filling the vicinity of $S_p$ (the '
     'high-stability region) thus becomes a performance axis as important as global '
     'search quality. Through controlled comparisons sharing an identical N5 local search, '
     'we show that a trajectory-based method (ILS) fills this vicinity by itself through '
     'continuous transformation, whereas a population-based method (memetic algorithm) '
     'remains structurally coarse there because of crossover (H1); ILS thereby '
     'wins high-stability coverage in all eight scenarios with complete separation '
     '(|$δ$|=1.0). We further propose a '
     'rescheduling-specific adaptation of path relinking whose guiding solution is fixed '
     'to $S_p$ (PR), and its one-step variant (repair), as stability-inducing operators '
     'that pull solutions toward $S_p$. Experiments over eight scenarios, seven methods, and ten trials show '
     'that the same operator acts asymmetrically depending on the host structure—more '
     'than doubling the population\'s high-stability coverage and lifting it to first '
     'place in overall quality, while the trajectory host is already saturated (H2)—and '
     'quantify a complementary structure in which the best method switches with the '
     'evaluation metric.'),
    ('keywords', 'Keywords:',
     'Rescheduling; Job-shop scheduling; Stability; Iterated local search; '
     'Path relinking'),

    # ================= 1 =================
    ('h1', '1. INTRODUCTION'),
    ('p', 'Manufacturing shop floors frequently face disruptions—operation delays, '
          'machine breakdowns—that make the original schedule infeasible. Among the '
          'static (built-in tolerance) and dynamic (after-the-fact revision) responses, '
          'we address the latter, specifically **predictive-reactive rescheduling** '
          '(Ouelhadj and Petrovic, 2009), which repairs the executing schedule into a '
          'feasible one after a disruption. '
          'We focus on **operation delays**, handled by resequencing alone (machine '
          'assignments kept), for which the premise of keeping the revised schedule '
          'near the original holds most clearly.'),
    ('p', 'The efficiency of the revised schedule (makespan, MS) and the amount of '
          'change from the pre-disruption schedule $S_p$ (stability) trade off. Large '
          'changes incur shop-floor confusion, setup and fixture changeovers, material '
          're-kitting, worker re-assignment, and rippling schedule changes to downstream '
          'or outsourced operations, so stability is an objective on par with MS '
          '(Wu et al., 1993; Rangsaritratsamee et al., 2004); we formulate '
          '**multi-objective optimization of efficiency and stability**.'),
    ('p', 'The essential peculiarity of rescheduling is that **a high-quality '
          'pre-disruption schedule $S_p$ already exists as an initial solution, and the '
          'revised schedule must deviate little from it**. Optimal solutions are likely '
          'distributed near $S_p$, so the merit of a search method acquires a new axis: '
          'not mere global search ability, but **how well it fills the vicinity of $S_p$ '
          '(the high-stability region)**.'),
    ('p', 'Our objectives are threefold: (1) a comparative analysis of how the '
          'existence of a high-quality initial solution $S_p$ affects the search '
          'behavior of single-solution and population-based methods; (2) the design '
          'of stability-inducing operators (PR and repair) that pull solutions '
          'toward $S_p$, and of their interaction with the host '
          'search structure; (3) a multi-perspective evaluation methodology '
          'integrating speed, Pareto coverage, and performance by stability band. '
          'Our focus is the structural analysis of the mechanism–host interaction '
          'rather than a performance race against the state of the art '
          '(Sörensen, 2015).'),

    # ================= 2 =================
    ('h1', '2. RELATED WORK AND POSITIONING'),
    ('p', '**Joint consideration of efficiency and stability.** The JSSP is NP-hard and '
          'metaheuristics are the mainstream (Ouelhadj and Petrovic, 2009). Joint '
          'optimization of efficiency and stability in rescheduling was pioneered by Wu '
          'et al. (1993) and pursued for the JSSP by Rangsaritratsamee et al. (2004) '
          '(hybrid GA) and Zhang et al. (2013) (GA + tabu search), and for flow shops by '
          'Katragjini et al. (2013). These works mostly bundle the two objectives '
          'into one weighted-sum scalar (Rangsaritratsamee et al., 2004; Zhang et al., '
          '2013), where stability enters not as an independent mechanism but '
          'passively, as a single evaluation term of the objective. '
          'Of the two metaheuristic families—'
          'trajectory (single-solution) and population-based (Blum and Roli, 2003)—'
          'the solvers here are **predominantly population-based**, anchored in GAs.'),
    ('p', '**Embedding stability as a mechanism (scope-limiting approaches).** A '
          'separate lineage guarantees stability structurally by **limiting the '
          'rescheduling scope**: match-up scheduling (Bean et al., 1991), AOR, '
          'which reschedules only the affected operations (Abumaizar and Svestka, '
          '1997), Zakaria and Petrovic (2012), who restrict the chromosome to '
          'an interval, and Sun et al. (2026), who formalize the scope into a '
          'four-level hierarchy. Common to all is '
          'guaranteeing stability by **restricting the search space**.'),
    ('p', '**Remaining issues and positioning.** Three issues remain. First, existing '
          'rescheduling studies are predominantly GA-based (population) and lack a '
          'controlled contrast with a trajectory-based method; to our knowledge, how '
          'the rescheduling-specific existence of a high-quality initial '
          'solution $S_p$ affects search-structure behavior (trajectory vs. '
          'population) has not been analyzed head-on; we do so '
          'via a controlled contrast between ILS and a memetic algorithm sharing the '
          'same local search (N5). Second, neither existing means of securing '
          'stability steers the search toward stability actively: they fall into '
          'scope limiting, which excludes trade-off solutions outside the '
          'restriction, and objective embedding, in which the search itself does not '
          'pursue stability, leaving high-stability filling to the raw ability of the '
          'base search. We propose operators (PR, repair) that '
          'actively pull solutions toward $S_p$; securing stability as an operator '
          'rather than by restricting the space is our core design claim against the '
          'scope-limiting family. Third, existing '
          'evaluations are mostly scalar comparisons at a few fixed weights, lacking a '
          'multi-axis methodology covering Pareto coverage, convergence speed, and '
          'stability bands; we evaluate with union HV, high-stability HV, and AOC '
          '(Section 3.4).'),
    ('p', 'We pose two hypotheses. **H1 (suitability)**: a population-based method '
          '(memetic) that largely recombines solutions by crossover scatters '
          'offspring away from $S_p$ and fills the high-stability region (near $S_p$) '
          'only coarsely, whereas a trajectory-based method (ILS) that fills the '
          'neighborhood by continuous transformation from $S_p$ covers it efficiently. '
          '**H2 (complementarity)**: operators pulling solutions toward $S_p$ (PR, '
          'repair) complement precisely the neighborhood filling that populations are '
          'structurally poor at, so their effect appears **asymmetrically** with the '
          'host structure (large for the population, marginal for the trajectory). '
          'Section 3 presents the methods; Section 4 tests both hypotheses.'),

    # ================= 3 =================
    ('h1', '3. PROBLEM SETTING AND PROPOSED METHODS'),
    ('h2', '3.1 Problem Definition'),
    ('p', 'After a single operation delay (length $Δ$) in an $n$-job, $m$-machine '
          'JSSP, we seek a revised schedule $S_q$ against the original $S_p$. '
          'Feasibility is first restored by the **right-shift repaired solution '
          '$S_{RSR}$**, which keeps the machine sequences of $S_p$. Operations started '
          'before the rescheduling time $t_r$ (when the delay is resolved) are '
          '**frozen**; those from $t_r$ onward are the **optimization targets**. The '
          'decision variables are only the per-machine sequences of the target '
          'operations; a solution becomes an executable schedule by fixing the frozen '
          'part and shifting each operation to its earliest feasible start after '
          '$t_r$ under the given machine sequences.'),
    ('p', '**Stability metric.** Deviation measures fall into two families: (i) '
          'start-time deviation (temporal stability) and (ii) processing-sequence '
          '(permutation) deviation (sequence stability). Sun et al. (2026), for example, '
          'distinguish the two explicitly and adopt the former (the sum of absolute '
          'start-time deviations, ADST). We adopt the latter because our search '
          'mechanisms (the direct swaps of PR/repair, the N5 neighborhood) are all '
          'permutation operations and it is highly independent of MS.'),
    ('eq', 'EQ1', '(1)'),
    ('p_noindent',
     'Here $r_{i,j}$ is the processing rank of job $j$ on machine $i$, and $O_{opt}$ is '
     'the set of target operations (from $t_r$ onward). $D$=0 corresponds to the '
     'sequence-preserving right-shift solution $S_{RSR}$. Temporal stability based on '
     'start times is outside this metric\'s scope—an explicit limitation.'),
    ('p', '**Multi-objective formulation and scalarization.** We minimize '
          '$(MS(S_q), D(S_p, S_q))$ jointly via a weighted sum with weight $λ∈[0,1]$.'),
    ('eq', 'EQ2', '(2)'),
    ('p_noindent',
     'Hatted symbols denote min–max normalized values. We sweep the weight over multiple '
     'points, merge the solutions, and evaluate Pareto coverage (Section 3.4). The '
     'premise for method differences to appear is that room remains to improve MS from '
     '$S_{RSR}$ by sequence re-optimization (non-degeneracy).'),
    ('p', '**Why weighted scalarization rather than Pareto-native methods.** First, '
          '**consistency with operational practice**: the efficiency–stability priority '
          'is specified in advance as a weight by the decision maker (the standard '
          'rescheduling framework (Rangsaritratsamee et al., 2004)); '
          'each weight yields a solution, aligning with the per-weight anytime '
          'HV($t$) (AOC, Section 3.4) that measures preference-specific ramp-up. '
          'Second, **a common basis for controlled comparison**: only by having ILS '
          'and the memetic share the same scalar objective $F(λ)$ can we mount the '
          'same mechanism on both structures unmodified and isolate structural '
          'differences (methods premised on population-specific non-dominated '
          'sorting, e.g., NSGA-II, do not fit this control). To avoid dependence on '
          'any single weight, we sweep $λ$ and merge all non-dominated solutions '
          'under UEA (Ishibuchi et al., 2020). The weighted sum misses concave parts '
          'of the front (Marler and Arora, 2010), but because all methods are '
          'evaluated under identical conditions, the structural discussion is '
          'unaffected (re-verification with Pareto-native methods is future work).'),
    ('h2', '3.2 ILS and Its Rationale (H1)'),
    ('p', 'In rescheduling—where the initial solution $S_p$ is high-quality and '
          'optima lie near it—a trajectory (single-solution) method that '
          'continuously transforms from $S_p$ should fill the high-stability region '
          'better (H1). We therefore center the comparison on **ILS vs. Memetic-LS '
          'with the local search (N5) aligned**, removing the local-search confound '
          'and contrasting only the search structure.'),
    ('p', 'Among single-solution methods we adopt ILS (Lourenço et al., 2019) '
          'because exploitation (local search) and escape (perturbation) are '
          'separated, the perturbation strength directly controls the distance from '
          '$S_p$, and the stability-inducing mechanism (Section 3.3) inserts '
          'naturally as a perturbation. The local search is the standard **N5 '
          'neighborhood (Nowicki and Smutnicki, 1996)** (candidates restricted to '
          'swaps of adjacent jobs at critical-block ends—only moves with a prospect '
          'of makespan improvement), shared by ILS and Memetic-LS. Its unit of '
          'search—an adjacent swap at a critical-block end—is a minimal structural '
          'change, so it fills the high-stability region in fine increments of '
          'deviation from $S_p$, fitting our stability-focused setting. The perturbation '
          'is an insert move whose strength (= distance from $S_p$) cycles in a '
          'sawtooth with stagnation (VNS-type (Mladenović and Hansen, 1997)); best '
          'and current update only when $F(λ)$ strictly improves. The candidate set '
          'is makespan-motivated and $λ$-independent, but moves are accepted by '
          '$F(λ)$, so the local optima shift toward efficiency or stability with $λ$.'),
    ('p', '**Population-based methods (GA / Memetic-LS).** The population control '
          'encodes individuals in the operation-based representation (Bierwirth, '
          '1995) with a standard GA (PPX crossover (Bierwirth et al., 1996), '
          'inversion mutation, tournament selection with elitism), applying the same '
          'N5 to each individual in a Lamarckian manner to form **Memetic-LS** (GA + '
          'local-search integration (Neri and Cotta, 2012)); the plain GA is a '
          'reference baseline. Sharing N5 with ILS, its difference reduces solely to '
          'the search structure (single trajectory vs. population + crossover). '
          '**Crossover is a destructive operation splicing two parents, so offspring '
          'readily jump away from the vicinity of $S_p$; even Memetic-LS with N5 '
          'therefore fills the high-stability region only coarsely—the crux of H1.**'),
    ('h2', '3.3 Stability-Inducing Operators (PR and repair) and H2'),
    ('p', '**Path relinking (PR).** PR exploits the idea that good intermediate '
          'solutions lie on the path connecting two high-quality solutions; it is '
          'commonly combined with scatter search, linking solutions within an elite '
          'pool (Glover et al., 2000; Peng et al., 2015). Our rescheduling-specific '
          'feature is that **the guiding solution is fixed to the single point $S_p$, '
          'the pre-disruption schedule**: from the current local optimum (initiating) '
          'toward $S_p$ (guiding), the path is traced by shrinking the disagreements '
          'one by one with direct swaps, returning the best solution on the path. '
          'Fixing to $S_p$ (a) makes PR uniquely interpretable as a "directed move '
          'toward the stability anchor," $S_p$ being the optimal endpoint of the '
          'stability objective, and (b) lines the intermediates between $S_{cur}$, a '
          'local optimum of $F(λ)$, and $S_p$ up along the MS–stability trade-off, '
          'contributing directly to Pareto filling. Each step picks one feasible '
          'disagreement swap at random, keeping evaluations at $O(d)$ ($d$ = number '
          'of disagreements; no significant HV difference from best-selection '
          '$O(d^2)$ in preliminary experiments).'),
    ('p', '**Stability repair kick (repair).** This transfers PR\'s "one-step approach '
          'toward $S_p$" to the perturbation kick of ILS—a mini-PR truncated midway. '
          'Upon stagnation, a few direct swaps pull the solution toward $S_p$ '
          '("repairing" the stability lost by drifting), and the local search restarts. '
          'The depth also cycles in a sawtooth, covering the stability side of the '
          'front in a band.'),
    ('p', '**H2 (complementarity hypothesis).** PR and repair both pull solutions '
          'toward $S_p$, complementing exactly the high-stability filling that '
          'population-based search is poor at; ILS has already filled the '
          'neighborhood by itself (H1), leaving little headroom. The mechanism lies '
          'in solution redundancy and path length: GA-derived solutions are not '
          'MS-optimized as rigorously as N5 achieves—leaving redundant sequences '
          'whose stability can improve without hurting MS—and sit far from $S_p$ '
          'with long PR paths, while ILS clings to the vicinity of $S_p$ with short '
          'paths. Hence we predict **an asymmetry: large effects on the population, '
          'marginal on the trajectory**.'),
    ('p', 'Figure 1 schematizes the three structures\' behavior on the $(MS, D)$ '
          'plane—population scattering (H1), trajectory neighborhood filling (H1), and '
          'mechanism pull-back (H2). We study seven methods: ILS-baseline / ILS+repair '
          '/ ILS+PR / GA / Memetic-LS / Memetic+PR / Memetic+repair.'),
    ('h2', '3.4 Evaluation Framework'),
    ('p', 'Scalar comparison at one weight makes conclusions weight-dependent and '
          'cannot capture how much of the efficiency–stability trade-off was covered. '
          'We therefore sweep $λ$ (weighted-sum sweep) and measure Pareto coverage '
          'under **UEA** (Ishibuchi et al., 2020), which archives all non-dominated '
          'solutions visited. Rescheduling also values stable solutions near $S_p$ '
          'over the efficiency extreme, and "whether a good solution is available '
          'whenever computation stops" matters operationally. Three metrics answer '
          'the three questions—**overall quality, filling with stable solutions, '
          'speed**: **union HV (overall quality)** = hypervolume over the whole '
          'region; **high-stability HV (primary)** = hypervolume restricted to $D$ < '
          'P50 (near $S_p$), P50 being the median $D$ of Pareto solutions pooled over '
          'all methods and trials; **AOC (anytime performance (López-Ibáñez and '
          'Stützle, 2014))** = the time average of the HV-versus-log-time curve.'),
    ('fig', 'fig_concept_en.png',
     'Figure 1: Schematic search behavior on the (MS, D) plane. (a) population '
     'scatters away from $S_p$ via crossover; (b) trajectory (ILS) fills the '
     'vicinity of $S_p$; (c) PR/repair pulls scattered solutions back toward '
     '$S_p$.',
     'full', 0.80),
    ('p', 'For cross-scenario comparison, HV is computed after affine normalization to '
          '$[0,1]^2$ per scenario with reference point $(1.1,1.1)$. The anytime '
          'HV($t$) for AOC is measured on the wall clock, with the integration window '
          '**common to all methods** and the same logarithmic-width normalization '
          '(apples-to-apples). AOC '
          'averages the anytime HV($t$) over the 10-weight sweep.'),
    ('p', '**Statistics.** Within each scenario we use the one-sided Wilcoxon '
          'signed-rank test (paired by trial; one-sided since H1 and H2 fix the '
          'direction in advance) plus Cliff\'s $δ$. When |$δ$|=1.0 (complete '
          'separation), the one-sided Wilcoxon with n=10 reaches its lower bound '
          '$p$ ≈ 0.001, so "$p$=0.001" below signifies complete directional '
          'consistency (magnitudes are reported via $δ$ and ratios). Cross-scenario '
          'tendencies are summarized by Friedman tests, average ranks, and '
          'Kendall\'s $W$; since the eight scenarios derive from five instances and '
          'the la36 ladder and ta21 pair vary only the disruption within one '
          'instance, these cross statistics are an **exploratory summary of rank '
          'consistency**, not independent-sample tests. Magnitudes are also given '
          'as ARPD% ((best − $x$)/best × 100; lower is better).'),

    # ================= 4 =================
    ('h1', '4. COMPUTATIONAL EXPERIMENTS'),
    ('p', 'This chapter establishes H1 (Section 4.2, the trajectory\'s suitability for '
          'the high-stability region) and H2 (Section 4.3, the host-dependent '
          'asymmetric effect of the same mechanism) through controlled pairwise '
          'comparisons, then surveys the implications with an overall scoreboard '
          '(Section 4.4). The central finding is that implementing the stability lever '
          'as an operator allowed **the host-structure × stability-mechanism '
          'interaction to be isolated under control**; as a consequence we also '
          'observe a complementary structure in which the best method changes with '
          'the evaluation axis (Section 4.4).'),
    ('h2', '4.1 Experimental Setup'),
    ('p', 'Benchmarks: mt10, la21, la36, la40, ta21. Disruptions total 8 scenarios: '
          'la36S/la36M/la36L (27/54/73%), ta21S/ta21L (32/82%), mt10 (72%), la21 '
          '(35%), la40 (32%), where the percentage is the **rescheduling ratio** '
          '$ρ=n_{res}/ops$—the fraction of all operations that became re-optimization '
          'targets $O_{opt}$. We adopt $ρ$ as the control axis because it appears to '
          'be the main determinant of effective difficulty here (see the end of '
          'Section 4.4). In particular, the **la36 ladder** (27/54/73%) and the **ta21 '
          'pair** (32/82%) vary only the rescheduling ratio stepwise within the same '
          'instance and the same $S_p$, isolating the $ρ$-dependence of method '
          'differences without confounds. Each disruption is a completion delay of '
          'one operation; $Δ$ is about 0.9–1.5 times the affected operation\'s '
          'processing time ($Δ$=60–148), and $t_r$ is the delay resolution '
          '(= delayed completion) time (Section 3.1). Complete per-scenario '
          'definitions of the affected operation, $Δ$, $t_r$, and how $ρ$ was '
          'controlled (via the disrupted operation\'s position in $S_p$) are in the '
          'public repository (below).'),
    ('p', '**Weights and trials.** $λ$ is swept over 10 points (0–0.9, step 0.1; the '
          'pure-stability endpoint $λ$=1.0 is excluded since the optimum degenerates '
          'to the trivial $S_{RSR}$ ($D$=0)); 10 trials per (scenario × method). The '
          'unit of analysis is the trial: one trial is the whole 10-weight sweep—the '
          'non-dominated set merging each weight\'s visited solutions under UEA. HV '
          'thus yields one value per trial, and all tests use n=10.'),
    ('p', '**Computational budget.** ILS iteration cap 3000; GA/memetic 500 '
          'generations (both to natural convergence). Iterations and generations '
          'differ in unit, so nominal values are aligned within each family and '
          'cross-method speed is compared by wall-clock AOC. Wall-clock time differs '
          'across families (Memetic+PR is 5–8× ILS, owing to path decoding toward '
          '$S_p$ in our Python implementation), but this is a mechanism cost, not a '
          'truncation artifact: a convergence check over all seven methods shows the '
          'median run reaching 99% of its final HV within half its budget.'),
    ('p', '**Initial solution and GA settings.** $S_p$ is a high-quality active '
          'schedule generated by GA-500 (decoded by the GT algorithm (Giffler and '
          'Thompson, 1960); it satisfies the non-degeneracy condition of Section 3.1). '
          'All methods receive $S_p$ as a common start (ILS as the initial solution; '
          'GA/memetic within the initial population). The GA backbone ($cx_{pb}$=0.85, '
          '$mut_{pb}$=0.1, pop=50) is fixed at standard values, not an independent '
          'variable.'),
    ('p', '**Environment and reproducibility.** AMD Ryzen 5 7530U; Python 3.12 '
          '(NumPy, DEAP, SciPy). All seeds fixed. Code and scenario definitions: '
          'https://github.com/kitotakumi/stability_scheduling.'),
    ('h2', '4.2 Result 1: Trajectory (ILS) vs. Population (Memetic) (H1)'),
    ('p', 'To avoid mechanism confounds, we compare ILS-baseline vs. Memetic-LS with '
          'the local search aligned (**both carry the identical N5**, so differences '
          'stem from the search structure, not the presence of local search). The '
          'high-stability gap is therefore attributable to '
          'the fact that even a competent population with the same local search '
          'cannot structurally fill the vicinity of $S_p$ because of crossover. '
          'Figure 2 shows the results.'),
    ('p_noindent',
     '• **Union HV: comparable** (ILS 5 wins, Memetic 3). Memetic wins precisely on '
     'the scenarios with large improvement headroom at mid-to-high rescheduling '
     'ratios (la36M, mt10, la36L; 54–73%), the winner switching with the ratio '
     '(Section 4.4).'),
    ('p_noindent',
     '• **High-stability HV (primary): ILS wins in all 8 scenarios with complete '
     'separation** ($p$=0.001, |$δ$|=1.0). In the three low-ratio scenarios (la36S, ta21S, la40), '
     'Memetic-LS places not a single solution in the high-stability region—only ILS '
     'covers the vicinity of $S_p$; in the remaining five, ILS is 2–4.5×. With '
     '|$δ$|=1.0, significance survives Holm correction over the 8-scenario family '
     'everywhere.'),
    ('p_noindent',
     '• **AOC: ILS significantly ahead in 6/8** (the exceptions la36S and mt10 have '
     'small re-optimization subproblems where the population\'s early coverage pays '
     'off).'),
    ('fig', 'fig_claim1_en.png',
     'Figure 2: H1—(a) union HV is comparable, but (b) high-stability HV favors ILS '
     'in all 8 scenarios, and (c) AOC favors ILS in 6/8 (ILS-baseline vs. Memetic-LS, '
     'medians; \\*p<.05 \\*\\*p<.01 \\*\\*\\*p<.001).',
     'full'),
    ('p', '**Structural cause (visit-density difference map, Figure 3).** The '
          'per-method normalized visit-density difference on an $(MS,D)$ grid shows '
          'ILS concentrating on the low-$D$ front band while Memetic disperses into '
          'the high-$D$ unstable region. The population does visit low $D$, but '
          'destructive crossover keeps it from refining MS there as thoroughly as '
          'the continuously transforming ILS, even with the same N5; at equal low '
          '$D$ its solutions lose to ILS in MS and fail to fill the region in the '
          'Pareto sense—the structural cause of the high-stability HV gap. Holding '
          'many solutions far from $S_p$ also lengthens the paths, creating exactly '
          'the headroom the mechanisms (PR/repair) can fill by pulling back—picked '
          'up in H2 (Section 4.3).'),
    ('h2', '4.3 Result 2: Asymmetric Effects of the PR/repair Operators (H2)'),
    ('p', 'We examine the high-stability HV gain from adding the mechanisms to each '
          'baseline, by host (Figure 4).'),
    ('p_noindent',
     '• **Population (Memetic): substantial improvement** (more than 2× in all 8 '
     'scenarios, $p$=0.001, |$δ$|=1.0). The mechanisms directly fill the previously '
     'unreachable vicinity of $S_p$; the population **catches up with ILS in the '
     'high-stability region and overtakes it in overall quality (union HV)** '
     '(Section 4.4).'),
    ('p_noindent',
     '• **Trajectory (ILS): mostly saturated.** In most scenarios the baseline has '
     'already filled the high-stability region, tying at the ceiling. The '
     'mechanisms are significant **only in the highest-ratio band: la36L (73%) and '
     'ta21L (82%)** (raw $p$=0.016, 0.001; after Holm correction over the '
     '8-scenario ILS family only ta21L remains significant, $p_{adj}$≈0.008; the '
     'gain is small—the ta21L median rises 0.029 → 0.031—but consistently positive '
     'across all trials). The effect is thus not absent: ILS leaves little headroom '
     'by filling the neighborhood itself, and a small margin remains only when the '
     'disruption is so large that the filling cannot keep up.'),
    ('p', '**Mechanistic cause (PR path statistics, Figure 5).** The path '
          'statistics directly corroborate the asymmetry: **Memetic has large path '
          'lengths $d_0$ and finds improving solutions on roughly 30–65% of '
          'paths**, whereas **ILS has short paths and a discovery rate near 0% '
          'everywhere** (0.4% even on ta21L). This reflects more than trial '
          'opportunities: ILS has already filled the vicinity of $S_p$—the short '
          'paths are themselves a symptom—so no headroom remains on the segments '
          'PR traverses, whereas in Memetic many unoptimized intermediates remain '
          'between the scattered solutions and $S_p$; the directed move thus comes '
          'up empty on ILS. The slight la36L/ta21L effect owes to the post-kick '
          'local search re-optimizing pockets that ILS\'s own filling left '
          'unfilled, not to the directing itself.'),
    ('fig', 'fig_density_en.png',
     'Figure 3: Structural cause of H1—normalized visit-density difference maps '
     '(red: ILS-baseline denser; blue: Memetic-LS denser; sign·√|Δ|); orange: ILS '
     'PF; green: Memetic PF. Four representative scenarios; the rest are in the '
     'repository.',
     'full'),
    ('fig', 'fig_claim2_en.png',
     'Figure 4: H2—the operators (PR, repair) (a) more than double the population\'s '
     'high-stability HV in every scenario, but (b) are mostly saturated on the '
     'trajectory host (high-stability HV medians vs. baseline; \\*p<.05 \\*\\*p<.01 '
     '\\*\\*\\*p<.001).',
     'full'),
    ('p', '**PR or repair—host-dependent usage.** Whether PR and repair need '
          'distinguishing—on quality (union/high-stability HV) and anytime '
          'performance (AOC)—depends on the host. On **ILS** (small $d$) they are '
          'nearly indistinguishable: quality is saturated with either (significant '
          'only on la36L/ta21L), and—the mechanism fires only upon stagnation (not '
          'in the early phase log-time AOC weights heavily) with short, cheap $O(d)$ '
          'paths—the AOC differences among ILS-baseline/+repair/+PR are '
          'non-significant in all 8 scenarios (effectively tied); the choice is '
          'immaterial. On **Memetic** (large $d$) the division is clear: **PR is '
          'slightly better in quality**, while **repair beats PR in AOC in all 8 '
          'scenarios** ($δ$ up to +1.00)—PR traverses the long path to $S_p$ before '
          'returning the best intermediate, ramping up slowly, whereas repair '
          'truncates after a few moves and updates the incumbent immediately. Hence: '
          'Memetic+PR when final quality is paramount, repair when the budget is '
          'tight and anytime performance matters.'),
    ('fig', 'fig_mech_pr_en.png',
     'Figure 5: Mechanistic cause of H2—(a) PR path length $d_0$ (mean disagreements '
     'to $S_p$) and (b) improvement discovery rate (share of calls finding a '
     'solution better than the initiating one). Memetic: long paths, many '
     'improvements; ILS: short paths, almost none.',
     'full'),
    ('h2', '4.4 Overall Scoreboard and Integration of Results'),
    ('p', 'We survey the structures established by H1 and H2 on the **overall '
          'scoreboard** of 7 methods × 8 scenarios × 3 metrics (Figure 6), '
          'corroborating the complementary structure announced at the outset: the '
          'best method switches with the metric—the reshuffling of green (best) '
          'cells across the metrics is the visual evidence. Friedman average ranks '
          'separate the methods clearly on all three metrics, with medium-to-large '
          'cross-scenario rank consistency (Kendall\'s $W$=0.59/0.81/0.63, '
          '$p$<0.001; per Section 3.4, an exploratory summary given the correlated '
          'scenarios).'),
    ('p', '**Reading the metrics.** Reading the green (best) distribution metric by '
          'metric makes the complementary structure concrete (parenthetical figures '
          'are Friedman average ranks, lower is better). On **union HV (quality)**, '
          'the operator-equipped populations form the top group (Memetic+PR 2.0, '
          'Memetic+repair 2.5), the three ILS variants and the plain Memetic-LS sit '
          'in the middle (3.6–4.8), and only GA trails (6.9); the method that tied ILS '
          'in Section 4.2 was the plain population (Memetic-LS), and the '
          'mechanism-equipped Memetic+PR rises above it to lead—robust to '
          'leave-one-out (unchanged whichever scenario is removed), so the narrow '
          'margin is not driven by any single scenario. On **high-stability HV '
          '(primary)**, the ILS family and the operator-equipped Memetic cluster at '
          'the top (2.6–3.4), while only the operator-less populations collapse with '
          'ARPD ≈ 70–78%, failing to reach the vicinity of $S_p$ (GA 6.4, '
          'Memetic-LS 6.6)—exactly what H1 (the trajectory fills it natively) and H2 '
          '(the operators complement the population\'s coarseness so it catches up) '
          'predict. On **AOC (anytime)**, the three ILS variants lead (2.5–2.8), the '
          'plain Memetic-LS and Memetic+repair are in the middle (3.5, 4.0), and the '
          'slow-warming Memetic+PR and GA trail (5.6, 7.0). The best method changes '
          'with the evaluation axis (Memetic+PR for overall quality; the ILS family '
          'for stability emphasis and speed), and the two structures complement the '
          'different demands of rescheduling.'),
    ('p', '**[Exploratory observation] Union-HV winner and the rescheduling ratio.** '
          'The union HV winner corresponds to $ρ$, splitting at around 50%: ILS '
          'below, Memetic at mid-to-high (on the same la36 ladder 27/54/73%, ILS → '
          'Memetic → Memetic); only $ρ$, which incorporates the fixed-skeleton '
          'proportion, captures this split—the number of movable operations or the '
          'problem size does not. Our reading: the stability term makes $S_p$ '
          '($D$=0) an attractor. At small $ρ$ the fixed skeleton dominates the '
          'solution, so no good local optimum far from $S_p$ can arise however the '
          'movable part is reordered; good solutions are thus confined to the $S_p$ '
          'neighborhood and ILS\'s neighborhood filling suffices; as $ρ$ '
          'grows and the efficiency extreme recedes, the single trajectory struggles '
          'to escape the basin while Memetic, retaining crossover-scattered '
          'individuals, reaches it (isomorphic to H1)—$ρ$ thus proxies the '
          'difficulty of escaping the attractor. However, ta21L (82%) is an '
          'exception (≈tie, $p$=0.053), and confounding by $S_p$ quality and '
          'dependence on the permutation-deviation representation remain, so this '
          'stays exploratory; the primary high-stability dominance and mechanism '
          'asymmetry are established independently by direct tests on all 8 '
          'scenarios.'),
    ('p', '**Divergent vs. convergent.** ILS (spreading outward from $S_p$) and '
          'Memetic+PR (pulling a scattered population toward $S_p$) reach similar '
          'final Pareto quality from opposite directions, but ILS holds a good '
          'incumbent early while Memetic+PR ramps up after a warm-up (overtaking '
          'later when headroom exists); AOC aggregates this crossover on a log-time '
          'axis, reflecting early performance—hence the ILS family\'s clear lead '
          'there.'),

    # ================= 5 =================
    ('h1', '5. CONCLUSIONS'),
    ('p', 'Targeting stability-aware JSSP rescheduling, this study compared '
          'trajectory vs. population search structures, proposed the '
          'stability-inducing operators (PR, repair), and built a multi-perspective '
          'evaluation methodology, verified over 8 scenarios × 7 methods × n=10. The '
          'claims are threefold.'),
    ('p_noindent',
     '1. **The trajectory-based method (ILS) efficiently fills the high-stability '
     'region (H1).** Comparable to the population in overall '
     'quality, it wins in all 8 scenarios with complete separation over the '
     'counterpart sharing the identical N5 in the primary high-stability region '
     '($p$=0.001, |$δ$|=1.0; '
     '2–4.5× in five scenarios, the population reaching the region not at all in '
     'the other three) and leads in anytime performance in 6/8—derived from the '
     'search structure, not the presence of local search.'),
    ('p_noindent',
     '2. **The effects of PR and repair appear asymmetrically depending on the host '
     'structure (H2).** They more than double the population\'s high-stability HV, '
     'lifting it to the ILS level, whereas the trajectory, having already filled '
     'the neighborhood, is mostly saturated. The asymmetry is not fixed, however: '
     'at extremely high rescheduling ratios where ILS\'s filling cannot keep up '
     '(ta21L, 82%), a significant mechanism effect remains even on the trajectory.'),
    ('p_noindent',
     '3. **Complementary structure of trajectory and population.** The best method '
     'changes with the metric, and the two structures complement the different '
     'demands of rescheduling. The practical prescription is clear—Memetic+PR when '
     'final quality is paramount, Memetic+repair when the budget is tight and '
     'anytime performance matters, the ILS family when stability and responsiveness '
     'are the priority (Section 4.3).'),
    ('p', 'Implementing the stability lever as an "operator" made the same mechanism '
          'portable to both hosts, and **isolating its host-dependent asymmetry under '
          'control is the central contribution of this study**.'),
    ('p', '**Limitations.** (i) Results rest on n=10 (the saturated main '
          'conclusions are robust; borderline cases are another matter). (ii) The '
          'rescheduling-ratio–union-HV-winner correspondence involves confounding '
          'by $S_p$ quality and representation dependence, and stays exploratory. '
          '(iii) Stability is measured only by permutation deviation; validity '
          'under start-time (temporal) deviation is unverified (Section 3.1).'),
    ('p', '**Future work.** Re-verification under start-time deviation (temporal '
          'stability); extension to disruptions altering machine assignments (e.g., '
          'breakdowns); integrating the scope lever with the operator lever (the '
          'inducing operators within the affected range); comparison with an '
          '$S_p$-biased crossover as an alternative population-side prescription (a '
          'direct test of the crossover-destructiveness reading of H1); and '
          're-verification with Pareto-native methods (e.g., NSGA-II).'),
    ('fig', 'fig_scoreboard_en.png',
     'Figure 6: Overall scoreboard—(a) union HV, (b) high-stability HV, (c) AOC over '
     '7 methods × 8 scenarios. Cells: RPD% relative to the best of each scenario '
     '(green = best, 0%); each panel sorted by its own Friedman average rank.',
     'full'),

    ('h1', 'REFERENCES'),
] + _REFS
