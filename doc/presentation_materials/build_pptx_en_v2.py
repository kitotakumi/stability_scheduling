# -*- coding: utf-8 -*-
"""APIEMS 2026（釜山）発表の英語版 v2：英語の発表として組み直した版を生成する。

英語版（build_pptx_en.py）は日本語版の文をそのまま英訳しているので，本文が長い文になっている。
v2 は英語の発表の作りに合わせ，本文を句に縮めて文字を減らす（文は口で言う）。構成・題名・図・表・予備スライドは
英語版と同じ。英語版の出力（APIEMS2026_presentation.pptx）は変えない。

英語版からの変更:
  - 本文の文を句にする（言い回しは talk_strings_en_v2.py。英語版の訳を v2 の訳に置き換えてから組む）
  - Pareto フロントと HV・着目点と評価指標：段落の文を，「● 見出し：句」の項目に組み直す（このファイルの関数）
  - 高安定 HV：演算子の効果：ILS の項目が短くなった分，「→」の行を上げる
  - 仮説：主張を太字で書き，理由は字下げした一段小さい「– 」の行で添える

前提:  python make_talk_figures_en.py を先に実行（assets/ に英語版の図を生成）。
再生成:  python build_pptx_en_v2.py          （pptx と同名の PDF も書き出す．PDF 不要なら --no-pdf）
"""
import os

import talk_strings_en as S
import talk_strings_en_v2 as S2

# 英語版は読み込むときに対訳を当てるので，その前に対訳表を v2 の言い回しに置き換える
S2.apply(S.SLIDES)

import build_pptx_en as E   # noqa: E402
from build_pptx import ML, SW, TOP, FS_BODY   # noqa: E402
from pptx.util import Inches, Pt   # noqa: E402

B, V2 = E.B, E.V2
B.OUT = os.path.join(B.HERE, "APIEMS2026_presentation_v2.pptx")


def s_pareto():
    """Pareto フロントと HV（1 章）。図と配置は英語版と同じで，本文を定義の項目（非劣解・Pareto フロント・HV）と
    「→ 帰結」に分ける。"""
    s = B.slide_frame("Pareto Front and Hypervolume (HV)",
                      "Since both objectives are minimized, a Pareto front closer to the lower left is better; "
                      "this study measures its quality by HV", S.SEC1)
    _w, h = B.pic(s, "concept_pareto_v2.png", ML - 0.05, TOP, w=5.8)
    tx = ML + 5.9
    tw = SW - ML - tx
    tb, tf = B.add_textbox(s, tx, TOP + 0.1, tw, 2.2)
    B._paras(tf, [
        ("● No single best solution in bi-objective optimization", FS_BODY, 0),
        ("● **Non-dominated**: no other solution is better in both objectives (none to its lower left)",
         FS_BODY, 8),
        ("● **Pareto front**: the set of non-dominated solutions", FS_BODY, 8),
        ("→ Both objectives minimized: **lower left is better**", FS_BODY, 14),
    ])
    B._divider(s, tx, TOP + 2.45, tw)
    tb, tf = B.add_textbox(s, tx, TOP + 2.8, tw, h - 2.8)
    B._paras(tf, [
        ("● **Hypervolume (HV)**[[5]]: area dominated by the front, up to a reference point", FS_BODY, 0),
        ("→ **Larger HV = better front**: our quality measure", FS_BODY, 14),
    ])
    B._cite_note(s, [5])


def s_focus():
    """着目点と評価指標（1 章）。図と配置は英語版と同じで，本文を「求められる解 → 高安定領域 → 高安定 HV」の
    3 項目にする。"""
    s = B.slide_frame("Focus and Evaluation Metric: High-Stability HV",
                      "Where schedule-change costs cannot be ignored, solutions in the high-stability region are "
                      "needed, so this study evaluates methods by the HV restricted to that region "
                      "(high-stability HV)", S.SEC1)
    _w, h = B.pic(s, "concept_hv_v2.png", ML - 0.05, TOP, w=5.8)
    tx = ML + 5.9
    tw = SW - ML - tx
    tb, tf = B.add_textbox(s, tx, TOP + 0.1, tw, h - 0.1)
    B._paras(tf, [
        ("● When schedule changes are costly, **highly stable** solutions (little change from $S_p$) are needed",
         FS_BODY, 0),
        ("● **High-stability region**: the half of all methods' non‑dominated solutions closest to $S_p$",
         FS_BODY, 8),   # ‑：ハイフンで折り返さない
        ("→ **High-stability HV**: HV in this region (main metric)", FS_BODY, 14),
    ])


def _raise_arrow(slide):
    """高安定 HV：演算子の効果。ILS の項目が v2 で 2 行になったので，「→」の行を英語版（3 行の項目の下）から
    0.32 in 上げ，他のスライドの「→」と同じ間隔で箇条書きに続ける。"""
    for sh in slide.shapes:
        if sh.has_text_frame and sh.text_frame.text.startswith("→ Operators help"):
            sh.top -= Inches(0.32)
            return
    raise RuntimeError("_raise_arrow: 「→」の行が見つからない")


REASON_FS = 16       # 仮説の理由の行の文字の大きさ（主張の 18 pt より一段小さく，図形の中の文字と同じ）
REASON_INDENT = 14   # 理由の行の字下げ（pt）


def _indent_reasons(slide):
    """仮説：「– 」で始まる理由の段落を字下げし（2 行目は「– 」の後ろにそろえる），主張より一段小さい文字にする。"""
    n = 0
    for sh in slide.shapes:
        if not sh.has_text_frame:
            continue
        for p in sh.text_frame.paragraphs:
            if not (p.runs and p.runs[0].text.startswith("– ")):
                continue
            for r in p.runs:   # 添字（文字の 0.72 倍）も同じ比で縮める
                if r.font.size is not None:
                    r.font.size = Pt(r.font.size.pt * REASON_FS / FS_BODY)
            hang = B._text_pt("– ", REASON_FS)
            pPr = p._p.get_or_add_pPr()
            pPr.set("marL", str(int(Pt(REASON_INDENT + hang))))
            pPr.set("indent", str(-int(Pt(hang))))
            p.space_before = Pt(6)
            n += 1
    if n != 2:
        raise RuntimeError(f"_indent_reasons: 理由の段落が 2 つ見つからない（{n} 個）")


V2.s_pareto = s_pareto   # V2 の _s_pareto_and_focus が V2 の名前で呼ぶ
V2.s_focus = s_focus
B.s_band = E._after(B.s_band, _raise_arrow)
B.s_hypotheses = E._after(B.s_hypotheses, _indent_reasons)

if __name__ == "__main__":
    E.main()
