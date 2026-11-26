# -*- coding: utf-8 -*-
"""英語版の発表スライド（build_pptx_en.py）と図（make_talk_figures_en.py）の対訳表。

SLIDES：スライド関数のコードにある日本語の文字列定数 → 英語。キーは JIMA v2 の日本語版に出る文字列そのもの
  （v2 が v1 の文言を置き換えたスライドは，置き換えた後の文字列）。「\n」「\v」の働きはキーと同じで，
  英語の行の分け方に合わせて置き直す。タプルのキーはタプルごと置き換える（行の数を変える場合）。
SLIDES_BY_FN：同じ日本語を，スライド関数ごとに訳し分ける場合。
FIGURES：図の中の日本語（matplotlib の Text に入る文字列そのもの）→ 英語。FIGURES_BY_FN は図の関数ごとの訳し分け。
KEEP：コードにあっても画面に出ない日本語（入れ子の関数の docstring など）。訳さずに残す。

用語は APIEMS 2026 の英語原稿（../paper_apiems2026_v2/apiems2026_manuscript_v2_en.md）に合わせる：
  単点探索 = trajectory-based search，多点探索 = population-based search，安定性誘導演算子 = stability-inducing
  operator，高安定領域 = high-stability region，高安定 HV = high-stability HV，全域 HV = overall HV，
  順序偏差 = sequence deviation，ILS 系 = ILS variants，最良に並ぶ = level with the best，完全分離 = no overlap。
  仮説 1・2 は札の幅に収まるよう H1・H2 と書く。
"""
import re

# 日本語（かな・漢字・全角の記号）を含むかの判定
CJK = re.compile(r"[　-〿぀-ヿ㐀-䶿一-鿿豈-﫿＀-￯]")
# 手法名などの全角の記号だけを含む文字列（ILS＋PR など）は，半角に直せばそのまま英語として使う
_FW = str.maketrans({"＋": "+", "（": "(", "）": ")", "＝": "=", "〜": "–", "～": "–", "　": " "})


def normalize(s):
    """全角の記号を半角に直して日本語が残らなければ，その文字列を返す（残れば None）。"""
    t = s.translate(_FW)
    return None if CJK.search(t) else t


SEC1 = "1. Background and Hypotheses"
SEC2 = "2. Problem Setting and Experimental Design"
SEC3 = "3. Computational Experiments"
SEC4 = "4. Conclusions"

SLIDES = {
    # ---------------------------------------------------------------- 共通（フッター・記号）
    "1. 研究背景と仮説": SEC1,
    "2. 問題設定と実験計画": SEC2,
    "3. 数値実験": SEC3,
    "4. 結論": SEC4,
    "予備": "Backup",
    "［予備］": "[Backup] ",
    # 箇条書き（_bul）の見出しのぶら下げ：「**見出し**：」は英語では「**見出し**: 」
    r"\*\*(.+?)\*\*：": r"\*\*(.+?)\*\*: ",
    "：": ": ",
    "　－ ": "   – ",
    "　": "   ",
    "[20]　": "[20]  ",
    "番目": "",

    # ---------------------------------------------------------------- 表紙
    "処理順序の変更量を考慮した再スケジューリングにおける":
        "Trajectory- and Population-Based Search with\vPath Relinking for Job-Shop Rescheduling",
    "path relinking の適用と探索構造の比較評価": "under Sequence-Deviation Stability",
    "早稲田大学": "Waseda University",
    "○鬼頭 拓海，谷水 義隆，渡邉 るりこ": "Takumi Kito, Yoshitaka Tanimizu, and Ruriko Watanabe",
    "日本経営工学会 2026年秋季大会　2026年10月": "APIEMS 2026   Busan, Korea   November 29 – December 2, 2026",

    # ---------------------------------------------------------------- 外乱と再スケジューリング
    "外乱と再スケジューリング": "Disruptions and Rescheduling",
    "外乱が起きたら，事前に作成した初期スケジュール $S_p$ を修正する（予測リアクティブ再スケジューリング）":
        "When a disruption occurs, the initial schedule $S_p$ prepared in advance is revised "
        "(predictive-reactive rescheduling)",
    "外乱": "Disruption",
    "製造現場では，機械故障や\n作業遅延などにより，\nスケジュール通りに作業を\n実行できないことがある":
        # machine breakdowns・operation delays が行をまたがないよう，区切りのよい所で改行する
        "On the shop floor,\nmachine breakdowns,\noperation delays, and\nsimilar events can prevent\n"
        "operations from running\nas scheduled",
    "動的アプローチ": "Dynamic approach",
    "外乱の発生後に，\nスケジュールを修正する": "Revise the schedule after a disruption occurs",
    "静的アプローチ": "Static approach",
    "外乱に対してロバストな\nスケジュールを\n事前に作成する": "Build a schedule robust to disruptions in advance",
    "完全リアクティブ": "Completely reactive",
    "事前のスケジュールを持たず，\nディスパッチングルールなどで\n作業をその場で決める":
        "Keep no schedule in advance; decide operations on the spot, e.g., by dispatching rules",
    "本研究": "This study",
    "予測リアクティブ再スケジューリング": "Predictive-reactive rescheduling",
    ("外乱が起きたら，外乱前の", "スケジュール（**初期スケジュール $S_p$**）", "を修正する"):
        ("When a disruption occurs, revise the", "pre-disruption schedule", "(**initial schedule $S_p$**)"),

    # ---------------------------------------------------------------- 再スケジューリングと安定性
    "再スケジューリングと安定性": "Rescheduling and Stability",
    "スケジュールの変更は現場にコストを生むため，生産効率と安定性の二目的最適化として扱う":
        "Schedule changes incur costs on the shop floor, so rescheduling is treated as bi-objective "
        "optimization of efficiency and stability",
    "例：": "Example: ",
    "**M2 の J1 が 3 遅延（赤）** → 時刻 $t_r$ 以降の処理順序を変更":
        "**J1 on M2 is delayed by 3 (red)** → the sequences from time $t_r$ onward can be revised",
    "$S_p$ の処理順序を保つ": "Keep the order of $S_p$",
    "総所要時間 19": "Makespan 19",
    "変更なし": "No change",
    "M3 の順序を入れ替える": "Swap the order on M3",
    "総所要時間 16": "Makespan 16",
    "順序を変更": "Reordered",
    "開始済み（凍結）": "Started (frozen)",
    "再スケジューリングの対象": "To be rescheduled",
    "処理順序の変更は，現場にコストを生む": "Sequence changes incur costs",
    "（スケジュール変更コスト）": "(schedule-change costs)",
    "段取り替え": "Setup changeovers",
    "資材・治具の再手配": "Rearranging\nmaterials & fixtures",
    "作業者の再配置": "Reassigning workers",
    "下流工程・外注先への\n予定変更の波及": "Ripples downstream\n& to subcontractors",
    "二目的最適化[[3,4]]": "Bi-objective optimization[[3,4]]",
    "生産効率": "Efficiency",
    "総所要時間を\n小さく": "Minimize\nthe makespan",
    "安定性": "Stability",
    "初期スケジュール\nからの変更を小さく": "Minimize change\nfrom $S_p$",

    # ---------------------------------------------------------------- Pareto フロントと HV
    "Pareto フロントと hypervolume（HV）": "Pareto Front and Hypervolume (HV)",
    "両目的とも最小化であるため Pareto フロントは左下にあるほど良く，本研究はその品質を HV で測る":
        "Since both objectives are minimized, a Pareto front closer to the lower left is better; "
        "this study measures its quality by HV",
    "● 二目的最適化では，解が 1 つに定まらない．\n両方の目的で同時に上回る解が他にない解を\n"
    "**非劣解**（図では，左下に他の解がない点），\nその集合を **Pareto フロント**と呼ぶ":
        "● In bi-objective optimization, no single solution is best. A solution is **non-dominated** if no "
        "other is better in both objectives (in the figure, none lies to its lower left); their set is the "
        "**Pareto front**",
    "→ 両目的とも最小化であるため，\nPareto フロントは**左下にあるほど良い**":
        "→ Since both objectives are minimized, a Pareto front **closer to the lower left is better**",
    "● **hypervolume（HV）**[[5]]：非劣解が支配する\n領域（参照点まで）の面積．\n"
    "値が大きいほど Pareto フロントの品質が高い":
        "● **Hypervolume (HV)**[[5]]: the area dominated by the non-dominated solutions (up to a "
        "reference point). A larger value means a higher-quality Pareto front",
    "→ 本研究は，Pareto フロントの品質を HV で測る": "→ This study measures Pareto-front quality by HV",

    # ---------------------------------------------------------------- 着目点と評価指標
    "本研究の着目点と評価指標：高安定 HV": "Focus and Evaluation Metric: High-Stability HV",
    "スケジュール変更コストが無視できない場面では高安定領域の解が求められるため，"
    "本研究は高安定領域に限った HV（高安定 HV）で手法を評価する":
        "Where schedule-change costs cannot be ignored, solutions in the high-stability region are needed, "
        "so this study evaluates methods by the HV restricted to that region (high-stability HV)",
    "● スケジュール変更コストが無視できない\n場面で求められるのは，**安定性の高い解**，\n"
    "すなわち Pareto フロント上の変更量が少ない解":
        "● Where schedule-change costs cannot be ignored, what is needed is a **highly stable solution**, "
        "that is, a Pareto-front solution with little change from $S_p$",
    "→ 本研究は，**高安定領域**（全手法の非劣解のうち，\n変更量が中央値より小さい範囲）の HV に\n"
    "着目する．この HV を**高安定 HV** と呼び，\nこれで手法を評価する":
        "→ This study focuses on the HV of the **high-stability region** (where the change is below the "
        "median among the non-dominated solutions of all methods). We call it the **high-stability HV** "
        "and evaluate methods by it",

    # ---------------------------------------------------------------- 研究の出発点
    "本研究の出発点：再スケジューリングの特殊性": "Starting Point: A Peculiarity of Rescheduling",
    "$S_p$ が探索前に得られているこの問題では，この特性が探索構造の優劣を左右するのではないか":
        "In this problem $S_p$ is available before the search, and this property may decide which "
        "search structure is better",
    ("★ 再スケジューリングの特殊性：", "初期スケジュール $S_p$（処理順序を保って",
     "遅延を吸収した解）は，**探索せずに得られる**．", "変更量 0 で安定性は最良，$MS$ も外乱前に最適化済"):
        ("★ A peculiarity of rescheduling:",
         "$S_p$ (sequences kept, delay absorbed) is **obtained without any search**. Its change is 0 "
         "(best stability), and its $MS$ was optimized before the disruption"),
    "→ この $S_p$ の存在が，**探索構造の優劣，特に**": "→ Does this $S_p$ **strongly affect which search**",
    "**高安定 HV に大きな影響を与えるのではないか**": "**structure is better, especially in high-stability HV?**",
    "メタヒューリスティクスの探索構造": "Search structures of metaheuristics",
    "単点探索": "Trajectory-based",
    "1 つの解を少しずつ\n変更しながら進む": "Transforms a single\nsolution little by little",
    "例：TS，SA，ILS": "e.g., TS, SA, ILS",
    "多点探索": "Population-based",
    "複数の解を保持し，\n互いに組み替えて進む": "Keeps several solutions\nand recombines them",
    "例：GA，NSGA-II，Memetic": "e.g., GA, NSGA-II, Memetic",

    # ---------------------------------------------------------------- 先行研究（カードの中身は RELATED）
    "先行研究：安定性を確保する 2 つのアプローチ": "Related Work: Two Approaches to Securing Stability",
    "安定性の確保は 2 つのアプローチに分かれるが，どちらもメタヒューリスティクスによる解法は多点探索に集中している":
        "Approaches to securing stability fall into two groups, and both rely mainly on population-based "
        "metaheuristics",
    "どちらのアプローチでも，メタヒューリスティクスによる解法は GA を基軸とする**多点探索に集中**":
        "In both approaches, the metaheuristics used **concentrate on population-based search built on GAs**",

    # ---------------------------------------------------------------- 3 つの課題
    "先行研究に残された 3 つの課題": "Three Issues Remaining in Related Work",
    "先行研究には，探索構造の比較・安定性を重視した評価・安定性の高い解を作り出す操作が欠けている":
        "Related work lacks a comparison of search structures, an evaluation focused on stability, and "
        "an operation that creates highly stable solutions",
    "探索構造の差が\n検証されていない": "Search structures\nnot compared",
    "$S_p$ の存在が単点探索と多点探索の優劣に与える影響は，\n分析されていない":
        "How the presence of $S_p$ affects the relative merits of\n"
        "trajectory-based and population-based search is not analyzed",
    "安定性を重視した品質が\n適切に評価されていない": "Stability-focused quality\nnot properly evaluated",
    "比較は少数の重みでのスカラー値や，Pareto フロント全体の\nHV で行われ，安定性の高い解の品質は\n"
    "詳細に分析されていない":
        "Comparisons use scalar values at a few weights or the HV of\n"
        "the whole Pareto front; the quality of highly stable solutions\nis not analyzed in detail",
    "安定性の高い解を\n作り出す操作がない": "No operation creating\nhighly stable solutions",
    "**目的関数に組み込む**：安定性は解の評価にだけ入る\n**範囲を限定する**：安定性は探索空間の制限にだけ入る\n"
    "→ どちらも，安定性の高い解を作り出す操作を持たない":
        "**Embedding in the objective**: stability enters only the evaluation\n"
        "**Limiting the scope**: stability enters only as a search-space limit\n"
        "→ Neither has an operation that creates highly stable solutions",

    # ---------------------------------------------------------------- 仮説
    "本研究の仮説": "Hypotheses",
    "課題 1・3 に対して 2 つの仮説を立て，どちらも高安定 HV（課題 2 への対応）で判定する":
        "Two hypotheses address Issues 1 and 3; both are tested by high-stability HV (which addresses Issue 2)",
    "仮説 1": "H1",
    "仮説 2": "H2",
    "課題 1": "Issue 1",
    "課題 3": "Issue 3",
    "交叉で 2 親の染色体を組み替える**多点探索**は，\n$S_p$ からの小さな変更では表せない飛躍を伴う．\n"
    "そのため，$S_p$ を初期解として少しずつ変更する\n**単点探索**のほうが，**高安定 HV が大きい**":
        "Crossover in **population-based search** makes jumps rather than small changes from $S_p$, so "
        "**trajectory-based search**, changing $S_p$ little by little, attains a **larger high-stability HV**",
    "多点探索に，解を $S_p$ に近づける移動\n（**安定性誘導演算子**）を組み込めば，\n"
    "その経路上に変更量の小さい解が現れるため，\n**多点探索の高安定 HV は向上する**":
        "Equipping **population-based search** with a move toward $S_p$ (a **stability-inducing operator**) "
        "yields small-change solutions on the way, so **its high‑stability HV rises**",   # ‑：ハイフンで折り返さない
    "主眼は最先端手法との性能競争ではなく，条件を統制して探索構造と演算子の関係を切り分けること":
        "The aim is not to beat state-of-the-art methods but to isolate the structure–operator relationship "
        "under controlled conditions",

    # ---------------------------------------------------------------- 問題設定
    "問題設定": "Problem Setting",
    "決定変数は，遅延後に最適化対象となる作業の，機械ごとの処理順序":
        "The decision variables are the per-machine sequences of the operations that become optimization "
        "targets after the delay",
    "$n$ ジョブ・$m$ 機械のジョブショップ問題（JSP）に，\n単一作業の遅延（遅延量 $Δ$）が発生":
        "In an $n$-job, $m$-machine job-shop scheduling\nproblem (JSSP), a single operation is delayed (by $Δ$)",
    "再スケジューリング時刻 $t_r$ より前に開始した作業は\n凍結し，以降の作業集合 $O_{opt}$ の"
    "**機械ごとの処理順序**\nを決定変数とする":
        "Operations started before the rescheduling time $t_r$\nare frozen; the **per-machine sequences** of "
        "the\nremaining operations $O_{opt}$ are the decision variables",
    "安定性は**処理順序の変更量**で測る（順序偏差 $D$）":
        "Stability: **change in sequences** (sequence deviation $D$)",
    "$r_{i,j}$：機械 $i$ におけるジョブ $j$ の処理順位": "$r_{i,j}$: processing rank of job $j$ on machine $i$",
    "二目的（$MS$, $D$）は重み付き和 $F$ にまとめ，重み $λ$ を\n生産効率重視から安定性重視まで掃引し，得られた\n"
    "全非劣解の HV で評価する":
        "($MS$, $D$) are combined into a weighted sum $F$; the\nweight $λ$ is swept from efficiency to "
        "stability, and all\nresulting non-dominated solutions are evaluated by HV",
    "例：ある機械の処理順序": "Example: sequence on one machine",
    "初期スケジュール $S_p$ は $D$ = 0": "The initial schedule $S_p$ has $D$ = 0",

    # ---------------------------------------------------------------- 安定性誘導演算子（v2 の置き換え後の文言）
    "提案手法：path relinking（PR）に基づいた安定性誘導演算子":
        "Proposed: Stability-Inducing Operators via Path Relinking (PR)",
    "PR で現在解と初期スケジュール $S_p$ を結び，現在解から $S_p$ へ swap でたどる経路上の解を返す操作にする":
        "PR links the current solution with the initial schedule $S_p$; the operator returns solutions on "
        "the path traced from the current solution to $S_p$ by swaps",
    "通常の PR": "Standard PR",
    "探索中に得た良い解どうしを結び，その間に現れるより良い解を探す[[18,19]]":
        "Links good solutions found during the search and looks for better ones between them[[18,19]]",
    "**現在解と $S_p$ を結ぶ** → 解を $S_p$ に近づける操作になる":
        "**Links the current solution with $S_p$** → an operation that brings solutions toward $S_p$",
    "例：ある機械の処理順序を $S_p$ に近づける": "Example: one machine's sequence toward $S_p$",
    "現在解": "Current",
    "1 回目": "Swap 1",
    "2 回目": "Swap 2",
    "swap：J4 と J3 を入れ替え": "swap J4 and J3",
    "swap：J3 と J2 を入れ替え（$S_p$ に到達）": "swap J3 and J2 ($S_p$ reached)",
    "赤枠：$S_p$ と順位が異なる作業（その数が $d$）\n1 回の swap で 1 作業以上が $S_p$ の順位に戻り，$d$ が減る":
        "Red frame: operations not at their $S_p$ rank ($d$ in total)\n"
        "Each swap returns ≥ 1 operation to its $S_p$ rank; $d$ falls",
    "組み込み方　": "How they are added   ",
    "停滞時に摂動として発動　": "as a perturbation upon stagnation   ",
    "各個体に確率的に改善操作として発動": "stochastically per individual",

    # ---------------------------------------------------------------- 比較する手法と仮説の判定
    "比較する手法と仮説の判定": "Compared Methods and Hypothesis Tests",
    "仮説 1 は ILS と Memetic，仮説 2 は Memetic と演算子を組み込んだ Memetic の高安定 HV を比較して判定する":
        "H1 is tested by comparing the high-stability HV of ILS and Memetic, H2 by comparing Memetic "
        "with Memetic + operators",
    "探索構造": "Search structure",
    "安定性誘導演算子なし": "No operator",
    "PR-$k$ を組み込む": "With PR-$k$",
    "PR を組み込む": "With PR",
    "参考の GA（局所探索なし）も含めた 7 手法で比較":
        "7 methods in total, including a plain GA (no local search) as a reference",
    "単点探索のほうが，多点探索より高安定 HV が大きい":
        "Trajectory-based search has a larger high-stability HV than population-based search",
    "ILS が Memetic を全シナリオで有意に上回れば支持": "supported if ILS significantly exceeds Memetic in all scenarios",
    "安定性誘導演算子を組み込めば，多点探索の高安定 HV は向上する":
        "Stability-inducing operators raise the high-stability HV of population-based search",
    "Memetic＋PR-$k$ と Memetic＋PR がそれぞれ Memetic を全シナリオで有意に上回れば支持":
        "supported if Memetic+PR-$k$ and Memetic+PR each significantly exceed Memetic in all scenarios",
    "判定：": "Criterion: ",
    "検定：": "Test: ",
    "両側 Mann–Whitney の $U$ 検定．全シナリオにわたる同じ比較は Holm 法で補正":
        "two-sided Mann–Whitney $U$ test; the same comparison across all scenarios is Holm-corrected",
    "効果量：": "Effect size: ",
    "Cliff's $δ$．$|δ|$ = 1.0 は，一方の全試行が他方の全試行を上回る完全分離":
        "Cliff's $δ$; $|δ|$ = 1.0 means no overlap (every trial of one method exceeds every trial of the other)",

    # ---------------------------------------------------------------- 実験設定
    "実験設定": "Experimental Setup",
    "5 つのベンチマーク問題に単一作業の遅延を与え，8 シナリオを構成":
        "Eight scenarios are built by delaying a single operation in five benchmark instances",
    "項目": "Item",
    "設定": "Setting",
    "問題（$n$ × $m$）": "Instance ($n$ × $m$)",
    "mt10（10×10），la21（15×10），la36・la40（15×15），ta21（20×20）":
        "mt10 (10×10), la21 (15×10), la36 and la40 (15×15), ta21 (20×20)",
    "単一作業の完了遅延（$Δ$ = 60〜148，対象作業の処理時間の約 0.9〜1.5 倍）":
        "Completion delay of one operation ($Δ$ = 60–148, ≈ 0.9–1.5× its processing time)",
    "シナリオ": "Scenarios",
    "8 つ：la36S・la36M・la36L，ta21S・ta21L，mt10，la21，la40\nla36 と ta21 は同一問題・同一 $S_p$ のまま"
    "遅延の発生時刻を変え，\n対象作業が少ない順に S・M・L":
        "8: la36S, la36M, la36L, ta21S, ta21L, mt10, la21, la40\n"
        "la36 and ta21 keep the same instance and $S_p$ and vary when the delay occurs;\n"
        "S, M, L in increasing number of target operations",
    "初期スケジュール $S_p$": "Initial schedule $S_p$",
    "500 世代の GA で生成": "Generated by a 500-generation GA",
    "重み $λ$": "Weights $λ$",
    "0, 0.1, …, 0.9 の 10 点": "10 points: 0, 0.1, …, 0.9",
    "計算予算": "Computational budget",
    "ILS 系 3000 反復，GA・Memetic 系 500 世代": "ILS variants: 3000 iterations; GA and Memetic variants: 500 generations",
    "試行": "Trials",
    "各手法・各重みで 10 試行（全乱数シード固定）": "10 per method and weight (all random seeds fixed)",
    "**妥当性**：高安定領域の境界・パラメータを変えても，結論は変わらない":
        "**Validity**: the conclusions are unchanged when the high-stability boundary or the parameters vary",
    "手法のパラメータと妥当性の確認は予備スライドに掲載": "Method parameters and validity checks are in the backup slides",

    # ---------------------------------------------------------------- 高安定 HV：ta21L の Pareto フロント
    "高安定 HV の比較：代表シナリオ ta21L の Pareto フロント": "High-Stability HV: Pareto Fronts in Scenario ta21L",
    "ta21L では，演算子なしの Memetic は高安定領域にほとんど入らず，PR を組み込むと演算子なしの ILS と重なる":
        "In ta21L, Memetic without operators barely enters the high-stability region; with PR, "
        "it overlaps ILS without operators",
    "(a) 演算子なし：Memetic は高安定領域にほとんど入らず，入った解も $MS$ で ILS に劣る":
        "(a) Without operators: Memetic barely enters the high-stability region and trails ILS in $MS$ there",
    "(b) PR を組み込む：Memetic も高安定領域に広がり，演算子なしの ILS（薄い橙）と重なる":
        "(b) With PR: Memetic also spreads into the region and overlaps ILS without operators (light orange)",

    # ---------------------------------------------------------------- 高安定 HV：全 8 シナリオ（v2 の置き換え後の文言）
    "高安定 HV の比較：全 8 シナリオ（仮説の判定）": "High-Stability HV: All 8 Scenarios (Hypothesis Tests)",
    "仮説 1・仮説 2 とも，全 8 シナリオで支持された": "Both H1 and H2 are supported in all 8 scenarios",
    "**仮説 1 を支持**：演算子なしで ILS > Memetic（全シナリオで $p_{adj}$ < 0.001，$|δ|$ = 1.0）":
        "**H1 supported**: without operators, ILS > Memetic (all scenarios: $p_{adj}$ < 0.001, $|δ|$ = 1.0)",
    "**仮説 2 を支持**：演算子を組み込んだ Memetic > Memetic（全シナリオで $p_{adj}$ < 0.001，$|δ|$ = 1.0）":
        "**H2 supported**: Memetic + operators > Memetic (all scenarios: $p_{adj}$ < 0.001, $|δ|$ = 1.0)",
    "※ 最良値：各シナリオでの，比較手法の中央値のうち最大の値":
        "Best value: the largest median among the compared methods in each scenario",

    # ---------------------------------------------------------------- 高安定 HV：演算子の効果
    "高安定 HV の比較：演算子の効果": "High-Stability HV: Effect of the Operators",
    "演算子を組み込むと両構造とも最良値の付近に達する．ILS は組み込む前から最良値の付近にある":
        "With the operators, both structures come close to the best value; ILS is already close to it "
        "without them",
    "**Memetic**：演算子を組み込むと，最良値の 0〜48% から 90〜100% まで大幅に向上":
        "**Memetic**: the operators lift it sharply, from 0–48% to 90–100% of the best value",
    "**ILS**：演算子なしで，8 シナリオ中 6 つがすでに最良値の 98.5% 以上．\n"
    "最良値に届いていなかった la36L・ta21L だけ，演算子で向上（94% → 100%）．\nHolm 補正後も有意なのは ta21L":
        "**ILS**: without operators, 6 of the 8 scenarios are already at 98.5% of the best value or more.\n"
        "Only la36L and ta21L, which fell short of the best value, improve with the operators (94% → 100%).\n"
        "Of these, only ta21L remains significant after Holm correction",
    "→ 演算子の効果は，**探索構造だけでは最良値に届かない場合に，その差を埋める形で現れる**":
        "→ The operators take effect **where the structure alone falls short of the best value, closing the gap**",

    # ---------------------------------------------------------------- 追加の評価
    "追加の評価：全域 HV と AUC": "Additional Evaluation: Overall HV and AUC",
    "仮説の判定とは別に，同じ試行を，フロント全体の品質と anytime 性能でも評価する":
        "Apart from the hypothesis tests, the same trials are also evaluated by whole-front quality "
        "and anytime performance",
    "**縦軸：全域 HV**": "**Vertical axis: overall HV**",
    "Pareto フロント全体の品質": "Quality of the whole Pareto front",
    "**横軸：計算時間（対数）**": "**Horizontal axis: computation time (log)**",
    "立ち上がりの速さを重視する": "Emphasizes how quickly HV rises",
    "**曲線の下の面積：AUC（anytime 性能）**[[20]]": "**Area under the curve:**\n**AUC (anytime performance)**[[20]]",
    "重み $λ$ を 1 つ選んで探索したとき，\nどの時点で計算を打ち切っても，\n大きい HV が得られるか":
        "Whether a search with one weight $λ$\nattains a large HV whenever it is stopped",
    "重みごとに求め，全ての重みで平均する": "Computed per weight and averaged over all weights",

    # ---------------------------------------------------------------- AUC と全域 HV
    "AUC と全域 HV の比較": "AUC and Overall HV",
    "最良の手法は評価指標によって異なる：AUC は ILS，全域 HV は Memetic＋演算子":
        "The best method differs by metric: ILS in AUC, Memetic + operators in overall HV",
    "**ta21L** では，ILS が最も早く立ち上がる．\nMemetic は遅れ，演算子でさらに遅れる":
        "In **ta21L**, ILS rises the fastest; Memetic lags,\nand the operators delay it further",
    "**全 8 シナリオ**では，ILS 系は 6，Memetic は 2，\nMemetic＋演算子は 1 シナリオで最良に並ぶ":
        "**Scenarios (of 8)** level with the best:\nILS variants 6, Memetic 2, Memetic + operators 1",
    "全域 HV": "Overall HV",
    "**ta21L** では，演算子で Memetic の最終値が\n上がり，最も高くなる．\nILS は演算子でほぼ変わらない":
        "In **ta21L**, the operators raise Memetic's final\nvalue to the highest; ILS barely changes",
    "**全 8 シナリオ**では，Memetic＋演算子は 7，\nILS 系は 4，Memetic は 1 シナリオで最良に並ぶ":
        "**Scenarios (of 8)** level with the best:\nMemetic + operators 7, ILS variants 4, Memetic 1",
    "最良に並ぶ：中央値が最良手法の中央値から 2% 以内": "Level with the best: median within 2% of the best median",

    # ---------------------------------------------------------------- まとめ
    "まとめ": "Summary",
    "高安定 HV は単点探索が上回り，多点探索も安定性誘導演算子を組み込めば同程度に達する":
        "Trajectory-based search leads in high-stability HV, and population-based search reaches a similar "
        "level when equipped with stability-inducing operators",
    "仮説 1 を支持": "H1\nsupported",
    "高安定 HV は**単点探索（ILS）**が多点探索（Memetic）を大きく上回る":
        "In high-stability HV, **trajectory-based search (ILS)**\vfar exceeds population-based search (Memetic)",
    "仮説 2 を支持": "H2\nsupported",
    "安定性誘導演算子は，**多点探索の高安定 HV を単点探索と同程度まで向上**させる":
        "Stability-inducing operators **raise the high-stability HV of**\v"
        "**population-based search to the level of trajectory-based search**",
    "手法の選び方": "Choosing\na method",
    "Pareto フロント全体の品質を重視する現場では **多点探索＋安定性誘導演算子**\n"
    "高安定領域の品質を重視する現場では **単点探索** か **多点探索＋安定性誘導演算子**\n"
    "短い計算時間で解を得る必要がある現場では **単点探索**":
        "For whole-front quality: **population-based search + operators**\n"
        "For high-stability quality: **trajectory-based** or **population-based + operators**\n"
        "For short computation times: **trajectory-based search**",

    # ---------------------------------------------------------------- 限界と今後の課題
    "限界と今後の課題": "Limitations and Future Work",
    "結論は本実験の条件の下で得たもので，条件を変えた再検証が今後の課題":
        "The conclusions hold under the conditions of this experiment; re-examining them under other "
        "conditions is future work",
    "限界": "Limitations",
    "(i) $p$ が有意水準に近い比較は，$n$ = 10 では決着していない":
        "(i) Comparisons with $p$ near the significance level are not settled at $n$ = 10",
    "(ii) 安定性を開始時刻偏差で測る場合の妥当性は検証していない":
        "(ii) Validity when stability is measured by start-time deviation is not examined",
    "(iii) 外乱は単一作業の遅延に限る": "(iii) Disruptions are limited to single-operation delays",
    "(iv) 初期スケジュール $S_p$ の最適化の程度が結果に与える影響は検証していない":
        "(iv) The effect of how well the initial schedule $S_p$ is optimized is not examined",
    "(v) 各探索構造の代表は 1 手法ずつである": "(v) Each search structure is represented by a single method",
    "今後の課題": "Future work",
    "上記の条件を変えた再検証": "Re-examination with the above conditions varied",
    "再スケジュールの範囲を限定するアプローチと安定性誘導演算子の統合":
        "Combining the scope-limiting approach with stability-inducing operators",
    "安定性を交叉自体に持たせる方式の検証": "Building stability into the crossover itself",
    "Pareto 支配に基づく手法（NSGA-II 等）での再検証": "Re-examination with Pareto-based methods such as NSGA-II",
    "データ：各シナリオの定義と，試行ごとの値・検定は公開リポジトリに収録\x0b"
    "https://github.com/kitotakumi/stability_scheduling":
        "Data: scenario definitions, per-trial values, and tests are in the public repository\v"
        "https://github.com/kitotakumi/stability_scheduling",

    # ---------------------------------------------------------------- 参考文献・予備の扉
    "参考文献": "References",
    "予備スライド": "Backup Slides",

    # ---------------------------------------------------------------- ［予備］問題設定の根拠
    "［予備］問題設定の根拠": "[Backup] Rationale for the Problem Setting",
    "**外乱を単一の作業遅延に限る理由**：作業遅延は作業集合を変えず，処理順序の再調整だけで対処できる．"
    "修正スケジュールを高安定領域に保つという前提が，最も素直に成り立つ外乱である":
        "**Why the disruption is a single operation delay**: an operation delay leaves the operation set "
        "unchanged and can be handled by readjusting the sequences alone, so it is the disruption under which "
        "the premise of keeping the revised schedule in the high‑stability region holds most naturally",   # ‑：ハイフンで折り返さない
    "**再スケジューリング時刻 $t_r$**：遅延の解消時刻．遅延の発生から $t_r$ までは，"
    "機械上の順序を $S_p$ から変えずに遅延を吸収して実行を続ける（right-shift）．"
    "この間に開始した作業はもう変更できないため凍結し，$t_r$ 以降の作業の機械ごとの処理順序だけを決定変数とする":
        "**Rescheduling time $t_r$**: the time at which the delay is resolved. From the delay until $t_r$, "
        "execution continues by absorbing the delay without changing the machine sequences of $S_p$ "
        "(right-shift). Operations started in this interval can no longer be changed and are frozen; only the "
        "per-machine sequences of the operations from $t_r$ onward are decision variables",
    "**安定性を順序偏差で測る理由**：変更量の測り方は，開始時刻偏差と順序偏差の二系統に大別される"
    "（Sun ら (2026) は前者の開始時刻の絶対偏差和を採る）．本研究は，探索で用いる操作"
    "（N5 近傍と安定性誘導演算子の swap 移動）がいずれも順列操作であること，"
    "および $MS$ との独立性が高いことから，順序偏差を採る":
        "**Why stability is measured by sequence deviation**: measures of change fall into start-time deviation "
        "and sequence deviation (Sun et al. (2026) adopt the former, the sum of absolute start-time "
        "deviations). We adopt sequence deviation because the moves used in the search (the N5 neighborhood "
        "and the swap moves of the operators) are all permutation moves and because it is largely "
        "independent of $MS$",

    # ---------------------------------------------------------------- ［予備］重み付き和によるスカラー化
    "［予備］重み付き和によるスカラー化": "[Backup] Weighted-Sum Scalarization",
    "$\\hat{D}$，$\\widehat{MS}$ は min-max 正規化値．$λ$ = 0, 0.1, …, 0.9 の 10 点で探索を独立に行う"
    "（$λ$ = 1.0 は $D$ = 0 の自明解に縮退するため除く）":
        "$\\hat{D}$ and $\\widehat{MS}$ are min–max normalized. Independent searches at the 10 points "
        "$λ$ = 0, 0.1, …, 0.9 ($λ$ = 1.0 is excluded: it degenerates to the trivial solution at $D$ = 0)",
    "**運用との対応**：生産効率と安定性の優先度は，意思決定者が事前に重みとして与える．"
    "1 点ぶんの探索は，重みが定まった状態で再スケジューリングを 1 回行う実運用に対応する":
        "**Correspondence to practice**: the priority between efficiency and stability is given in advance by "
        "the decision maker as a weight. The search at one point corresponds to rescheduling once in practice "
        "with the weight fixed",
    "**先行研究との対応**：Rangsaritratsamee ら (2004)・Zhang ら (2013)・Fattahi & Fallahi (2010) も，"
    "生産効率と安定性を重みで 1 つのスカラーにまとめる":
        "**Correspondence to prior work**: Rangsaritratsamee et al. (2004), Zhang et al. (2013), and "
        "Fattahi & Fallahi (2010) also combine efficiency and stability into one scalar by a weight",
    "**統制比較の共通基盤**：単点探索と多点探索に同じ目的 $F$ を共有させ，同じ安定性誘導演算子を"
    "変更なしで組み込める → 構造間で条件を揃えて比較できる":
        "**Common basis for a controlled comparison**: trajectory-based and population-based search share the "
        "same objective $F$, and the same stability-inducing operators can be added without modification → "
        "the structures are compared under aligned conditions",
    "**評価との対応**：各重みの探索を時間で追えば立ち上がりの速さ（AUC）が，"
    "全重みの解を合わせれば Pareto フロント全体の HV が，同じ実験から測れる":
        "**Correspondence to evaluation**: following the search at each weight over time gives how quickly HV "
        "rises (AUC), and merging the solutions of all weights gives the HV of the whole Pareto front, both "
        "from the same experiment",
    "**限界**：重み付き和は非凸な Pareto フロントの凹部に届かない．ただし全手法が同じスカラー化を共有し，"
    "探索中に訪問した凹部の解も記録するため，構造差の比較への影響は限定的．"
    "Pareto 支配に基づく手法（NSGA-II 等）での再検証は今後の課題":
        "**Limitation**: a weighted sum cannot reach concave parts of a non-convex Pareto front. Since all "
        "methods share the same scalarization and solutions in concave parts visited during the search are "
        "also recorded, the effect on the structural comparison is limited. Testing Pareto-based "
        "methods such as NSGA-II is future work",

    # ---------------------------------------------------------------- ［予備］なぜ ILS と Memetic か
    "［予備］なぜ ILS と Memetic か": "[Backup] Why ILS and Memetic?",
    "単点探索の代表：ILS（反復局所探索）": "Trajectory-based: ILS (iterated local search)",
    "摂動 → 局所探索 → 受理判定を反復し，1 つの解を改善する枠組み":
        "A framework that improves a single solution by iterating perturbation → local search → acceptance",
    "**局所探索と摂動が独立した段に分かれている**ため，摂動の強度（現在解から一度に"
    "どれだけ離すか）を明示的に設定でき，安定性誘導演算子も局所探索・受理判定を"
    "変えずに摂動の段へ組み込める":
        "**Local search and perturbation are separate stages**, so the perturbation strength (how far the "
        "search moves at once) can be set explicitly, and the operators can be placed in the perturbation "
        "stage without altering local search or acceptance",
    "他の単点探索は局所最適からの脱出を別の形で行う：SA は悪化解の確率的な受理，"
    "タブーサーチは再訪の禁止":
        "Other trajectory-based methods escape local optima differently: SA accepts worse solutions "
        "probabilistically, tabu search forbids revisits",
    "多点探索の代表：Memetic（GA＋局所探索）": "Population-based: Memetic (GA + local search)",
    "先行研究（Rangsaritratsamee ら 2004，Fattahi & Fallahi 2010）で用いられる，"
    "**GA に局所探索を組み込んだ標準的な構成**":
        "Used in prior work (Rangsaritratsamee et al. 2004; Fattahi & Fallahi 2010): "
        "**the standard configuration of a GA with local search built in**",
    "局所探索をもたない素の GA は参考として比較する": "A plain GA without local search is compared as a reference",

    # ---------------------------------------------------------------- ［予備］安定性誘導演算子の設計
    "［予備］安定性誘導演算子の設計": "[Backup] Design of the Stability-Inducing Operators",
    "**誘導解を $S_p$ に固定する理由**：(a) $S_p$ は順序偏差 $D$ を最小にする解なので，"
    "PR を「$D$ を最小にする解へ向かう移動」と一意に解釈できる．"
    "(b) $S_p$ へ向かう経路上には $D$ が小さく $MS$ の劣化も小さい解が現れると期待でき，"
    "それが高安定領域の非劣解になる":
        "**Why the guiding solution is fixed to $S_p$**: (a) $S_p$ minimizes the sequence deviation $D$, so PR "
        "has a single interpretation as a move toward the solution minimizing $D$. (b) On the path to $S_p$, "
        "solutions with small $D$ and only a small loss in $MS$ can be expected, and they become the "
        "non-dominated solutions of the high-stability region",
    "**PR-$k$ で $k$ を変える理由**：$k$ が $d$ に等しければ $S_p$ まで戻り，小さければ現在解の近くに留まる．"
    "$k$ を変えて繰り返すことで，現在解と $S_p$ の間の広い範囲に解を得る":
        "**Why $k$ varies in PR-$k$**: when $k$ equals $d$ the solution returns all the way to $S_p$, and when "
        "$k$ is small it stays near the current solution, so repeating with varying $k$ yields solutions over "
        "a broad range between the current solution and $S_p$",
    "**組み込み方**：両構造に「$S_p$ に近づける swap 移動」という核を共有させ，発動の契機と返し方は"
    "各構造の標準的な組み込み方に合わせる．ILS は停滞時に摂動として発動し，$k$ を 1 から $d$ まで "
    "1 ずつ増やして，上限に達するか best が改善したら 1 に戻す．Memetic は局所探索の後に個体ごとに"
    "確率的に適用し（$k$ は 1〜$d$ から一様に選ぶ），N5 で再最適化して無条件に集団へ戻す"
    "（採否はトーナメント選択に委ねる）":
        "**How they are added**: both structures share the core, the swap move toward $S_p$, while when the "
        "operator fires and how its result is returned follow each structure's standard way. ILS fires it as "
        "a perturbation upon stagnation, increasing $k$ by 1 from 1 to $d$ and resetting it to 1 when the "
        "upper bound is reached or the best improves. Memetic applies it stochastically to each individual "
        "after local search ($k$ drawn uniformly from 1 to $d$), re-optimizes by N5, and returns the result to "
        "the population unconditionally (acceptance is left to tournament selection)",
    "**入れ替える作業の選び方**：実行可能な候補からランダムに選び，評価回数を $O(d)$ に抑える．"
    "全候補を評価する best 選択は $O(d^2)$ で，予備実験では全域 HV が約 +1% 増えるのに対し，"
    "最大で約 8 倍の時間を要したため採らない":
        "**How the operation to swap is chosen**: at random among the feasible candidates, keeping the number "
        "of evaluations at $O(d)$. Best selection, which evaluates all candidates, costs $O(d²)$; in "
        "preliminary experiments it gained only about +1% in overall HV while taking up to about 8 times "
        "longer, so it is not adopted",

    # ---------------------------------------------------------------- ［予備］評価指標の詳細
    "［予備］評価指標の詳細": "[Backup] Details of the Metrics",
    "高安定 HV": "High-stability HV",
    "対象の解": "Solutions",
    "全ての重み $λ$ の解を合わせる": "Merged over all weights $λ$",
    "同左": "Merged over all weights $λ$",
    "重み $λ$ ごと": "Per weight $λ$",
    "$D$ の範囲": "Range of $D$",
    "全域": "All",
    "$D$ < $P_{θ}$（$θ$ = 50）": "$D$ < $P_{θ}$ ($θ$ = 50)",
    "値": "Value",
    "計算終了時の HV": "HV at the end",
    "HV の推移を対数時間で平均し，\n全ての重みで平均": "HV curve averaged over log time,\nthen over all weights",
    "**共通**：各シナリオで [0, 1]² に正規化し，参照点 (1.1, 1.1) で算出する．$D$ = 0 の解（$S_p$）は除く":
        "**Common**: normalized to [0, 1]² in each scenario and computed with the reference point (1.1, 1.1); "
        "solutions with $D$ = 0 ($S_p$) are excluded",
    "**$P_{θ}$**：全手法・全試行の非劣解の $D$ の $θ$ パーセンタイル":
        "**$P_{θ}$**: the $θ$-th percentile of $D$ over the non-dominated solutions of all methods and trials",
    "**AUC**：時間窓は各重みで全手法共通": "**AUC**: the time window is common to all methods at each weight",
    "本編の全域 HV の推移の図は全ての重みの解を合わせた曲線なので，AUC はその面積とは一致しない":
        "The overall-HV curves in the main slides merge the solutions of all weights, so AUC does not equal "
        "their area",

    # ---------------------------------------------------------------- ［予備］手法のパラメータ
    "［予備］手法のパラメータ": "[Backup] Method Parameters",
    "GA・Memetic 系": "GA/Memetic variants",
    "operation-based 表現，GT 法でデコード．エリート保存付きトーナメント選択，PPX 交叉（交叉率 0.85），"
    "逆位突然変異（突然変異率 0.1）．集団 50（$S_p$ を 1 個体含め，残り 49 個体は無作為生成）":
        "Operation-based representation decoded by the GT algorithm; tournament selection with elitism; "
        "PPX crossover (rate 0.85); inversion mutation (rate 0.1); population 50 ($S_p$ plus 49 random "
        "individuals)",
    "Memetic の局所探索": "Memetic local search",
    "各個体に N5 局所探索を適用し，改善解を染色体に書き戻す（Lamarckian 型）":
        "N5 local search on each individual; the improved solution is written back to the chromosome (Lamarckian)",
    "ILS 系": "ILS variants",
    "初期解は $S_p$．insert 摂動 2〜5 手を停滞度に応じて巡回．局所探索は N5 近傍"
    "（クリティカルブロック端の隣接 swap）．best の $F$ を厳密に改善したときのみ現在解を更新":
        "Initial solution $S_p$; insert perturbation of 2–5 moves, cycled by the degree of stagnation; N5 local "
        "search (adjacent swaps at critical-block ends); the current solution is updated only when the best "
        "$F$ strictly improves",
    "演算子（Memetic 系）": "Operators (Memetic)",
    "各個体に確率 0.3 で発動．PR は経路上の上位 3 中間解を局所探索して最良を返す":
        "Fire with probability 0.3 per individual; PR applies local search to the top 3 intermediate solutions "
        "on the path and returns the best",
    "演算子（ILS 系）": "Operators (ILS)",
    "無改善 400 反復で初回，以降 10 反復ごとに発動．PR は経路上の最良 1 解を返す":
        "Fire first after 400 iterations without improvement, then every 10; PR returns the single best "
        "solution on the path",
    "記録": "Recording",
    "選択の前に生成した全解から非劣解を保存（両構造で記録点を揃える）":
        "Non-dominated solutions among all solutions generated before selection are archived (same recording "
        "points in both structures)",
    "環境": "Environment",
    "AMD Ryzen 5 7530U，Python 3.12（NumPy・DEAP・SciPy）": "AMD Ryzen 5 7530U; Python 3.12 (NumPy, DEAP, SciPy)",

    # ---------------------------------------------------------------- ［予備］結果の補足
    "［予備］結果の補足": "[Backup] Supplementary Results",
    "**PR か PR-$k$ か**：ILS ではどちらを組み込んでも 3 指標とも差は 2% 以内．Memetic では全域 HV・高安定 HV は "
    "PR がわずかに上回り（有意差は各 2 シナリオ），AUC は 1 回の計算が軽い PR-$k$ が 8 シナリオ中 6 つで有意に上回る":
        "**PR or PR-$k$**: in ILS, the two differ by at most 2% on all three metrics. In Memetic, PR is slightly "
        "ahead in overall and high-stability HV (significant in 2 scenarios each), while in AUC PR-$k$, whose "
        "calls are cheaper, is significantly ahead in 6 of the 8 scenarios",
    "**ILS で演算子により有意に向上した 2 シナリオ**：la36L・ta21L で $p$ < 0.05．Holm 補正後に残るのは ta21L のみ"
    "（la36L は $p_{adj}$ = 0.091）．中央値の向上は 6.3〜6.5%":
        "**The 2 scenarios where the operators significantly improved ILS**: $p$ < 0.05 in la36L and ta21L. "
        "After Holm correction only ta21L remains (la36L: $p_{adj}$ = 0.091). The median gain is 6.3–6.5%",
    "最良値に届いていなかった 2 シナリオだけで効果が現れる → 非対称は組み込み方の違いでは説明できない":
        "The effect appears only in the 2 scenarios short of the best value → the asymmetry cannot be "
        "explained by how the operators are added",

    # ---------------------------------------------------------------- ［予備］全域 HV と AUC（代表 2 シナリオ）
    "全域 HV と AUC：代表 2 シナリオの推移": "Overall HV and AUC: Two Representative Scenarios",
    "立ち上がりは ILS が速く，到達した全域 HV は演算子を組み込んだ Memetic が同等以上":
        "ILS rises faster, while Memetic + operators reaches an overall HV at least as high",
    "(a) ta21S：演算子を組み込んだ Memetic も ILS と同じ全域 HV に達するが，立ち上がりが遅い":
        "(a) ta21S: Memetic + operators reaches the same overall HV as ILS but rises more slowly",
    "(b) la36L：Memetic 系は途中で ILS 系を追い越す．到達値は演算子を組み込んだ Memetic が最大":
        "(b) la36L: the Memetic variants overtake the ILS variants midway; Memetic + operators ends highest",

    # ---------------------------------------------------------------- ［予備］結論の妥当性
    "［予備］結論の妥当性": "[Backup] Validity of the Conclusions",
    "**高安定領域の境界**：$θ$ を 50 から 33〜67 に変えても，仮説 1・仮説 2 の結論は変わらない（全シナリオで完全分離）":
        "**Boundary of the high-stability region**: varying $θ$ from 50 to 33–67 leaves the conclusions of H1 "
        "and H2 unchanged (no overlap in every scenario)",
    "**計算時間**：壁時計は Memetic＋PR が ILS の 4.8〜9.4 倍．ただし中央値の試行は，全手法とも予算の 7 割以内で"
    "最終の全域 HV の 99% に達する → 全域 HV の差は計算予算の不足によらない":
        "**Computation time**: in wall-clock time Memetic+PR takes 4.8–9.4 times as long as ILS, but the median "
        "run of every method reaches 99% of its final overall HV within 70% of its budget → differences in "
        "overall HV are not due to an insufficient budget",
    "**パラメータ感度**：手法パラメータは標準値で固定した．挙動を定める 11 軸（ベースの探索構造側を含む）を "
    "2 シナリオで一要因ずつ上下に振ったが，全域 HV の変化は +0.5〜2% で，手法間の差（+7〜54%）より一桁小さい":
        "**Parameter sensitivity**: the method parameters were fixed at standard values. The 11 axes that "
        "determine behavior (including those of the base search structures) were varied one at a time above "
        "and below in 2 scenarios; the change in overall HV, +0.5–2%, is an order of magnitude smaller than "
        "the differences among methods (+7–54%)",
    "**ランダム方向の対照実験**：Memetic に組み込んだ演算子による向上が，摂動一般の効果か，"
    "$S_p$ に近づける方向に固有の効果かを確かめるため，強度を揃えて方向だけをランダム化した対照を "
    "2 シナリオ（mt10・la36S）で実行した":
        "**Random-direction control**: to check whether the gain from the operators in Memetic is an effect of "
        "perturbation in general or specific to the direction toward $S_p$, a control matching the strength "
        "and randomizing only the direction was run in 2 scenarios (mt10, la36S)",
    "ランダム方向は向上幅の 7 割前後しか再現せず，高安定 HV では演算子に完全分離で及ばない":
        "A random direction reproduces only about 70% of the gain and falls short of the operators in "
        "high-stability HV with no overlap",
    "同じ高安定 HV に達するには，1.7〜2.4 倍の時間を要する":
        "It needs 1.7–2.4 times as long to reach the same high-stability HV",
}

SLIDES_BY_FN = {
    # 表の行の見出しは列の幅に収まる短い形にする
    "s_design": {"単点探索（ILS）": "Trajectory (ILS)", "多点探索（Memetic）": "Population (Memetic)"},
}

# 画面に出ない日本語（入れ子の関数の docstring・エラーの文言）
KEEP = {
    "kind: root（前提）／main（採る側）／alt（採らない側）",
}

# ---------------------------------------------------------------- 1 章の章扉の項目・先行研究のカード
CHAPTERS = [
    ("1", "Background and Hypotheses",
     "Disruptions and rescheduling · Pareto front and HV · High-stability region ·\v"
     "High-stability HV · Peculiarity of rescheduling · Related work · Issues · Hypotheses"),
    ("2", "Problem Setting and Experimental Design",
     "Problem setting · Search structures · Stability-inducing operators · Hypothesis tests"),
    ("3", "Computational Experiments",
     "Experimental setup · High-stability HV · Additional evaluation (overall HV and AUC)"),
    ("4", "Conclusions", "Summary · Limitations and future work"),
]

RELATED = [
    ("Embedding stability in the objective",
     [("Wu et al. (1993)", "Single machine; efficiency and stability as criteria"),
      ("Rangsaritratsamee et al. (2004)", "JSSP; GA + local search"),
      ("Zhang et al. (2013)", "JSSP; GA + tabu search"),
      ("Fattahi & Fallahi (2010)", "FJSP; GA + local search"),
      ("Valledor et al. (2022), Luo et al. (2025)", "Pareto dominance with NSGA-II variants")],
     "Stability is only a term of the objective"),
    ("Limiting the rescheduling scope",
     [("Bean et al. (1991)", "Reschedule only up to the match-up point"),
      ("Abumaizar & Svestka (1997)", "Reschedule only the affected operations"),
      ("Zakaria & Petrovic (2012)", "Restrict the GA chromosome to an interval"),
      ("Sun et al. (2026)", "Four levels of re-optimization scope")],
     "Stability via a restricted search space"),
]

# ---------------------------------------------------------------- 図
FIGURES = {
    "模式図": "Schematic",
    "Pareto フロント": "Pareto front",
    "min　$S_p$ からの変更量（安定性）": "min   Change from $S_p$ (stability)",
    "min　総所要時間 $MS$（生産効率）": "min   Makespan $MS$ (efficiency)",
    "min　順序偏差 $D$（安定性）": "min   Sequence deviation $D$ (stability)",
    "高安定領域": "High-stability\nregion",
    "（変更量が\n中央値より小）": "(change below\nthe median)",
    "参照点": "Reference point",
    "高安定 HV": "High-stability HV",
    "全域 HV": "Overall HV",
    "（高安定 HV を含む）": "(including high-stability HV)",
    "探索せずに得られる": "Obtained without search",
    "単点探索（ILS）": "Trajectory-based (ILS)",
    "多点探索（Memetic）": "Population-based (Memetic)",
    "安定性誘導演算子：$S_p$ へ近づける": "Operator: toward $S_p$",
    "順序偏差 $D$": "Sequence deviation $D$",
    "現在解": "Current solution",
    "が返す解": "output",
    "（経路上で $F$ 最小）": "(min $F$ on path)",
    "（$k$ 回の swap で打ち切り）": "(after $k$ swaps)",
    "$S_p$ へ swap でたどる": "Swaps toward $S_p$",
    "探索A：立ち上がりが速い → AUC 大": "Search A: rises fast → large AUC",
    "探索B：最終値は高いが遅い": "Search B: higher final value, but slow",
    "計算時間（対数）": "Computation time (log)",
    "探索A の AUC（曲線の下の面積）": "AUC of search A (area under the curve)",
    "経過時間 [s]（対数）": "Elapsed time [s] (log)",
    "演算子なし": "No operator",
    "良い": "Better",
    "初期スケジュール $S_p$": "Initial schedule $S_p$",
    "(a) 演算子なし": "(a) Without operators",
    "(b) PR を組み込む": "(b) With PR",
    "総所要時間 $MS$": "Makespan $MS$",
    "演算子なし（a）": "Without operators (a)",
    "なし": "None",
    "高安定 HV（最良値＝1）": "High-stability HV (best = 1)",
    "高安定 HV\n（最良値＝1）": "High-stability HV\n(best = 1)",
}

FIGURES_BY_FN = {
    # 探索構造のスライドの小さな模式図は 2 in 幅なので，軸の名前は記号だけにする
    "fig_struct": {"順序偏差 $D$": "$D$"},
}
