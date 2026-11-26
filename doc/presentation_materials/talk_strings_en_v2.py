# -*- coding: utf-8 -*-
"""英語版 v2（build_pptx_en_v2.py）の言い回し。英語版（talk_strings_en.py）の訳のうち，本文を句に縮めるものだけを
「英語版の訳 → v2 の訳」で置き換える（キーを英語にしておくと，日本語の文を書き写さずに済む）。

方針（英語の発表としての組み直し）:
  - 本文は句で書き，文は口で言う。1 項目は 1〜2 行，項目は 3 つ程度まで
  - 句にしても，何の・何からの・どの条件か（有意か・最終値か・演算子の有無など）は削らない。削ると読めない行は
    短い節で書く
  - 結論を先に置き，理由を後ろに付ける（「〜なので〜」の形を避ける）
  - 図が示していることを本文で言い直さない
  - 題名・図・表・予備スライドは英語版と同じ
"""

REPLACE = {
    # ---------------------------------------------------------------- 外乱と再スケジューリング
    "On the shop floor,\nmachine breakdowns,\noperation delays, and\nsimilar events can prevent\n"
    "operations from running\nas scheduled":
        "Machine breakdowns,\noperation delays, etc.\n→ the schedule cannot\nrun as planned",
    "Revise the schedule after a disruption occurs": "Revise the schedule\nafter a disruption",
    "Build a schedule robust to disruptions in advance": "Build a robust schedule\nin advance",
    "Keep no schedule in advance; decide operations on the spot, e.g., by dispatching rules":
        "No schedule in advance;\ndecide on the spot\n(e.g., dispatching rules)",
    ("When a disruption occurs, revise the", "pre-disruption schedule", "(**initial schedule $S_p$**)"):
        ("Revise the pre-disruption schedule", "(**initial schedule $S_p$**)"),

    # ---------------------------------------------------------------- 研究の出発点
    ("★ A peculiarity of rescheduling:",
     "$S_p$ (sequences kept, delay absorbed) is **obtained without any search**. Its change is 0 "
     "(best stability), and its $MS$ was optimized before the disruption"):
        ("★ $S_p$ is available **without any search**",
         "– change 0 (the best stability)",
         "– $MS$ already optimized before the disruption"),
    "→ Does this $S_p$ **strongly affect which search**": "→ Does having $S_p$ in advance **change which search**",
    "Transforms a single\nsolution little by little": "One solution,\nchanged little by little",
    "Keeps several solutions\nand recombines them": "Several solutions,\nrecombined",

    # ---------------------------------------------------------------- 先行研究
    "In both approaches, the metaheuristics used **concentrate on population-based search built on GAs**":
        "Both approaches mostly use **GA-based, population-based search**",

    # ---------------------------------------------------------------- 3 つの課題
    "How the presence of $S_p$ affects the relative merits of\n"
    "trajectory-based and population-based search is not analyzed":
        "Effect of $S_p$ on trajectory- vs. population-based search:\n**not analyzed**",
    "Comparisons use scalar values at a few weights or the HV of\n"
    "the whole Pareto front; the quality of highly stable solutions\nis not analyzed in detail":
        "Compared by scalar values at a few weights\vor by HV of the whole front\n"
        "→ highly stable solutions **not evaluated in detail**",
    "**Embedding in the objective**: stability enters only the evaluation\n"
    "**Limiting the scope**: stability enters only as a search-space limit\n"
    "→ Neither has an operation that creates highly stable solutions":
        "**Embedding in the objective**: stability only in the evaluation\n"
        "**Limiting the scope**: stability only via a restricted search space\n"
        "→ **No operation creates highly stable solutions**",

    # ---------------------------------------------------------------- 仮説（主張を太字で，理由は「– 」の行で添える。
    # 理由の行の字下げと文字の大きさは build_pptx_en_v2.py の _indent_reasons で付ける）
    "Crossover in **population-based search** makes jumps rather than small changes from $S_p$, so "
    "**trajectory-based search**, changing $S_p$ little by little, attains a **larger high-stability HV**":
        "**Trajectory-based search has a larger**\v**high-stability HV than population-based**\n"
        "– crossover jumps far from $S_p$; trajectory-based\vsearch changes $S_p$ little by little",
    "Equipping **population-based search** with a move toward $S_p$ (a **stability-inducing operator**) "
    "yields small-change solutions on the way, so **its high‑stability HV rises**":
        "**Stability-inducing operators raise the**\v**high-stability HV of population-based search**\n"
        "– they move solutions toward $S_p$, yielding\vsmall-change solutions on the way",

    # ---------------------------------------------------------------- 問題設定
    "In an $n$-job, $m$-machine job-shop scheduling\nproblem (JSSP), a single operation is delayed (by $Δ$)":
        "$n$-job, $m$-machine job-shop scheduling problem\n(JSSP); one operation delayed by $Δ$",
    "Operations started before the rescheduling time $t_r$\nare frozen; the **per-machine sequences** of "
    "the\nremaining operations $O_{opt}$ are the decision variables":
        "Operations started before $t_r$: **frozen**\n"
        "Decision variables: **per-machine sequences** after $t_r$",
    "($MS$, $D$) are combined into a weighted sum $F$; the\nweight $λ$ is swept from efficiency to "
    "stability, and all\nresulting non-dominated solutions are evaluated by HV":
        "Weighted sum $F$ of $MS$ and $D$; sweep the weight $λ$\n"
        "→ evaluate all non-dominated solutions by HV",

    # ---------------------------------------------------------------- 安定性誘導演算子
    "Links good solutions found during the search and looks for better ones between them[[18,19]]":
        "Link two good solutions; search for better ones between them[[18,19]]",
    "**Links the current solution with $S_p$** → an operation that brings solutions toward $S_p$":
        "**Link the current solution with $S_p$** → moves solutions toward $S_p$",

    # ---------------------------------------------------------------- 高安定 HV：ta21L の Pareto フロント
    "(a) Without operators: Memetic barely enters the high-stability region and trails ILS in $MS$ there":
        "(a) No operator: Memetic barely enters the region; worse $MS$ than ILS there",
    "(b) With PR: Memetic also spreads into the region and overlaps ILS without operators (light orange)":
        "(b) With PR: Memetic fills the region, overlapping ILS from (a) (light orange)",

    # ---------------------------------------------------------------- 高安定 HV：演算子の効果
    "**Memetic**: the operators lift it sharply, from 0–48% to 90–100% of the best value":
        "**Memetic**: with the operators, 0–48% → 90–100% of the best value",
    "**ILS**: without operators, 6 of the 8 scenarios are already at 98.5% of the best value or more.\n"
    "Only la36L and ta21L, which fell short of the best value, improve with the operators (94% → 100%).\n"
    "Of these, only ta21L remains significant after Holm correction":
        "**ILS**: already ≥ 98.5% of the best value in 6 of 8 scenarios\n"
        "Gains only in la36L and ta21L (94% → 100%); after Holm, significant only in ta21L",
    "→ The operators take effect **where the structure alone falls short of the best value, closing the gap**":
        "→ Operators help **only where the structure alone falls short of the best value**",

    # ---------------------------------------------------------------- 追加の評価
    "Whether a search with one weight $λ$\nattains a large HV whenever it is stopped":
        "Large HV whenever the search is stopped\n(per weight $λ$)",

    # ---------------------------------------------------------------- AUC と全域 HV
    "In **ta21L**, ILS rises the fastest; Memetic lags,\nand the operators delay it further":
        "In **ta21L**, ILS rises fastest; Memetic lags,\nslowed further by the operators",
    "In **ta21L**, the operators raise Memetic's final\nvalue to the highest; ILS barely changes":
        "In **ta21L**, the operators lift Memetic's final\nvalue to the top; ILS barely changes",
}


def apply(slides):
    """英語版の対訳表 slides（日本語 → 英語）の値を REPLACE で置き換える。置き換え元が対訳表にない
    （英語版の訳が変わった）ときは止める。"""
    hit = set()
    for ja, en in slides.items():
        if en in REPLACE:
            slides[ja] = REPLACE[en]
            hit.add(en)
    missing = [en for en in REPLACE if en not in hit]
    if missing:
        raise RuntimeError("talk_strings_en_v2: 英語版の対訳表に見つからない訳: " + repr(missing))
