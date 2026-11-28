# -*- coding: utf-8 -*-
"""APIEMS 2026（釜山）発表の英語版 v2：英語の発表として組み直した版を生成する。

英語版（build_pptx_en.py）は日本語版の文をそのまま英訳しているので，本文が長い文になっている。
v2 は英語の発表の作りに合わせ，本文を句に縮めて文字を減らす（文は口で言う）。図・表は英語版と同じで，本編には
AUC を説明するスライドを置かない。英語版の出力（APIEMS2026_presentation.pptx）は変えない。

英語版からの変更:
  - 本文の文を句にする（言い回しは talk_strings_en_v2.py。英語版の訳を v2 の訳に置き換えてから組む）
  - Pareto フロントと HV・着目点と評価指標：段落の文を，「● 見出し：句」の項目に組み直す（このファイルの関数）
  - 探索構造：ILS を選んだ理由の「the operators」を「PR」にする
  - 比較する手法と仮説の判定・実験設定：本編では参考の GA（局所探索なし）を挙げない（GA の結果は予備にだけ
    出てくる）。手法の表の下の GA の注を置かず，計算予算は ILS 系と Memetic 系だけを書く
  - 高安定 HV：演算子の効果：ILS の向上の Holm 補正後の有意性を，本文でなく図の下の注に書く
  - 追加の評価：AUC を説明するスライドを置かず，全域 HV の推移のスライド 1 枚にする。ta21L の推移から立ち上がりの
    速さと最終値を読み，「ほかのシナリオも大半が同じ傾向」の 1 行を置く（AUC による全 8 シナリオの集計は予備の
    集計表。このファイルの関数）。3 章の章扉の項目もこれに合わせる
  - 予備の評価指標の詳細：AUC の項目で文献 [20] を引く（本編では AUC を説明しないので，AUC の文献はここで引く）
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
from build_pptx import ML, SW, TOP, BOT, FS_BODY, FS_HEAD, FS_NOTE   # noqa: E402
from pptx.enum.text import MSO_ANCHOR   # noqa: E402
from pptx.util import Pt   # noqa: E402

B, V2 = E.B, E.V2
B.OUT = os.path.join(B.HERE, "APIEMS2026_presentation_v2.pptx")


def s_pareto():
    """Pareto フロントと HV（1 章）。図と配置は英語版と同じで，本文を定義の項目（非劣解・Pareto フロント・HV）と
    「→ 帰結」に分ける。両目的とも最小化で左下が良いことは図（軸の min と Better の矢印）が示し，二目的では
    最良の解が 1 つに決まらないことは口で言うので書かない。"""
    s = B.slide_frame("Pareto Front and Hypervolume (HV)",
                      "Since both objectives are minimized, a Pareto front closer to the lower left is better; "
                      "this study measures its quality by HV", S.SEC1)
    _w, h = B.pic(s, "concept_pareto_v2.png", ML - 0.05, TOP, w=5.8)
    tx = ML + 5.9
    tw = SW - ML - tx
    tb, tf = B.add_textbox(s, tx, TOP + 0.1, tw, 1.3)
    B._paras(tf, [
        ("● **Non-dominated**: no other solution is better in both objectives (none to its lower left)",
         FS_BODY, 0),
        ("● **Pareto front**: the set of non-dominated solutions", FS_BODY, 8),
    ])
    B._divider(s, tx, TOP + 1.52, tw)
    tb, tf = B.add_textbox(s, tx, TOP + 1.87, tw, h - 1.87)
    B._paras(tf, [
        ("● **Hypervolume (HV)**[[5]]: area dominated by the front, up to a reference point", FS_BODY, 0),
        ("→ **Larger HV = better front**", FS_BODY, 14),
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
        ("● When schedule changes are costly, **highly stable** solutions (close to $S_p$) are needed",
         FS_BODY, 0),
        ("● **High-stability region**: the half of all methods' non‑dominated solutions closest to $S_p$",
         FS_BODY, 8),   # ‑：ハイフンで折り返さない
        ("→ **High-stability HV**: HV in this region (main metric)", FS_BODY, 14),
    ])


def s_anytime():
    """追加の評価（3 章）。全域 HV の推移の図は英語版と同じで，右の列は ta21L（左の図）の読み取りを，立ち上がりの
    速さと最終値の 1 項目ずつで書き，最後に「ほかのシナリオも大半が同じ傾向」の 1 行を置く。AUC は本編で説明しない
    （AUC による全 8 シナリオの集計と AUC の定義は予備）。"""
    s = B.slide_frame("Additional Evaluation: Overall HV over Time",
                      "The best method differs by aspect: ILS in the speed of rise, Memetic + operators in the final "
                      "overall HV", S.SEC3)
    fw = 5.5
    B.pic(s, "anytime_ta21L_v2.png", ML, TOP, w=fw)
    rx = ML + 5.9
    rw = SW - ML - rx
    B._text(s, rx, TOP, rw, 0.42, "Speed of rise", FS_HEAD, color=B.BLUE, bold=True, anchor=MSO_ANCHOR.MIDDLE)
    B._bul(s, rx, TOP + 0.5, rw, ["In **ta21L**, ILS rises fastest; Memetic lags,\nslowed further by the operators"],
           size=FS_BODY, gap=6)
    B._divider(s, rx, TOP + 1.3, rw)
    B._text(s, rx, TOP + 1.47, rw, 0.42, "Final value", FS_HEAD, color=B.BLUE, bold=True, anchor=MSO_ANCHOR.MIDDLE)
    B._bul(s, rx, TOP + 1.97, rw,
           ["In **ta21L**, the operators lift Memetic's final\nvalue to the top; ILS barely changes"],
           size=FS_BODY, gap=6)
    B._divider(s, rx, TOP + 2.77, rw)
    B._text(s, rx, TOP + 2.94, rw, 0.42, "→ Most other scenarios show **the same tendency**", FS_BODY,
            color=B.INK, emph=B.BLUE, anchor=MSO_ANCHOR.MIDDLE)
    B._text(s, ML, BOT - 0.36, fw, 0.36, "Results for all 8 scenarios are in the backup slides", FS_NOTE,
            color=B.MUTE)


def _holm_note(slide):
    """高安定 HV：演算子の効果。ILS の向上の Holm 補正後の有意性を，本文でなく図の下の注に書く。"""
    B._text(slide, ML, BOT - 0.36, SW - 2 * ML, 0.36,
            "After Holm correction, the improvement of ILS is significant only in ta21L", FS_NOTE, color=B.MUTE)


AUC_ITEM = ("AUC", ": the time window is common to all methods at each weight")


def _cite_auc(slide):
    """予備の評価指標の詳細：AUC の項目の見出しの直後に文献 [20] を引き，スライドの下に文献を添える
    （本編では AUC を説明しないので，AUC の文献はここで引く）。"""
    head, body = AUC_ITEM
    hits = 0
    for sh in slide.shapes:
        if not sh.has_text_frame:
            continue
        for p in sh.text_frame.paragraphs:
            runs = p.runs
            if len(runs) >= 3 and runs[1].text == head and runs[2].text == body:
                cite = B._cite_run(p, [20])   # 段落の末尾に付くので，見出しの直後へ移す
                runs[1]._r.addnext(cite._r)
                hits += 1
    if hits != 1:
        raise RuntimeError(f"_cite_auc: AUC の項目が 1 つでない（{hits} 個）")
    B._cite_note(slide, [20])


CH3_ITEMS = ("Additional evaluation (overall HV and AUC)", "Additional evaluation (overall HV over time)")


def _chapters_v2():
    """章扉の 3 章の項目の追加の評価を，本編のスライドの題名に合わせる。"""
    old, new = CH3_ITEMS
    chapters = [(n, title, sub.replace(old, new)) for n, title, sub in B.CHAPTERS]
    if chapters == B.CHAPTERS:
        raise RuntimeError(f"_chapters_v2: 「{old}」が章扉の項目にない")
    return chapters


WHY_ILS = ("so the operators fit into it", "so PR fits into it")


def _why_ils_pr(slide):
    """探索構造：ILS を選んだ理由の「the operators」を「PR」にする（同じスライドの交叉・突然変異も演算子なので，
    演算子だけでは安定性誘導演算子を指すと読めない）。"""
    old, new = WHY_ILS
    hits = 0
    for sh in slide.shapes:
        if not sh.has_text_frame:
            continue
        for p in sh.text_frame.paragraphs:
            for r in p.runs:
                if old in r.text:
                    r.text = r.text.replace(old, new)
                    hits += 1
    if hits != 1:
        raise RuntimeError(f"_why_ils_pr: 「{old}」が 1 か所でない（{hits} か所）")


GA_NOTE = "7 methods in total, including a plain GA (no local search) as a reference"


def _drop_ga_note(slide):
    """比較する手法と仮説の判定：手法の表の下の参考の GA の注を置かない。"""
    hits = [sh for sh in slide.shapes if sh.has_text_frame and sh.text_frame.text == GA_NOTE]
    if len(hits) != 1:
        raise RuntimeError(f"_drop_ga_note: GA の注が 1 つでない（{len(hits)} 個）")
    el = hits[0]._element
    el.getparent().remove(el)


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
B.s_structures = E._after(B.s_structures, _why_ils_pr)
B.s_design = E._after(B.s_design, _drop_ga_note)
B.s_extra = lambda: None   # AUC を説明するスライドは置かない（AUC の定義と文献は予備の評価指標の詳細）
B.s_band = E._after(B.s_band, _holm_note)
B.s_anytime = s_anytime
B.s_hypotheses = E._after(B.s_hypotheses, _indent_reasons)
B.s_backup_metrics = E._after(B.s_backup_metrics, _cite_auc)
B.CHAPTERS = _chapters_v2()

if __name__ == "__main__":
    E.main()
