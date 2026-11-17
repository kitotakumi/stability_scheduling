# -*- coding: utf-8 -*-
"""APIEMS 2026（釜山）発表の英語版 pptx を生成する。

構成・図の配置・文字の大きさは JIMA2026 秋 v2（build_pptx_v2.py）と同じにし，文字だけを英語にする。
v1・v2 のスライド関数をそのまま使うので，v1・v2 の組み方を直せば英語版にもそのまま反映される。
v1・v2 の出力は変えない。

仕組み:
  - スライド関数のコードにある日本語の文字列定数を，組む前に talk_strings_en.py の対訳で英語に置き換える
    （関数の __code__ の定数を差し替える）。文字列が組まれる前に置き換わるので，行数から箱の高さを決める
    処理なども英語の文で動く。v2 が v1 の文言を置き換えたスライド（安定性誘導演算子・高安定 HV の全 8 シナリオ）は，
    v2 の置き換えを当ててから訳す
  - 章扉・先行研究のカード（モジュールの定数）は talk_strings_en.py の CHAPTERS・RELATED に替え，
    出典の短い形は「ら」を「et al.」にする
  - 文字は Segoe UI，言語は en-US。図は make_talk_figures_en.py が作る *_en.png を貼る
  - 英語では組み方を変えるスライドは，このファイルで関数を差し替える（探索構造：流れ図の箱を広げる，
    予備の全域 HV と AUC の集計表：表を広げる）か，組んだ後に図形を直す（外乱と再スケジューリング：
    「This study」の札を広げる，研究の出発点：カードの本文の左端を見出しの文字にそろえる，
    予備のなぜ ILS と Memetic か：カードの枠を縦に伸ばす）
  - 対訳のない日本語が残れば，組む前（文字列定数）と組んだ後（スライドの文字・ノート）に一覧を出して止める

前提:  python make_talk_figures_en.py を先に実行（assets/ に英語版の図を生成）。
再生成:  python build_pptx_en.py          （pptx と同名の PDF も書き出す．PDF 不要なら --no-pdf）
"""
import inspect
import os
import re
import sys
import types

from fontTools.ttLib import TTFont
from PIL import ImageFont
from pptx.util import Inches, Pt

import build_pptx as B
import build_pptx_v2 as V2   # import すると v2 の差し替えが B に当たる
import talk_strings_en as S
from build_pptx import (ML, CW, SW, TOP, FS_BODY, FS_HEAD, FS_NOTE, BLUE, MUTE, INK, WHITE, CARDLINE,
                        ORANGE, GREEN, MSO_ANCHOR)

B.OUT = os.path.join(B.HERE, "APIEMS2026_presentation.pptx")
FONT = "Segoe UI"
FONT_DIR = os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "Fonts")

# ================================================================ 文字列定数の置き換え
MISSING = []   # 対訳のない日本語：(関数名, 文字列)


def _tr_str(s, where, pre):
    for old, new in pre:   # v2 が v1 の文言を置き換えたスライドは，置き換えてから訳す
        s = s.replace(old, new)
    if not S.CJK.search(s) or s in S.KEEP:
        return s
    t = S.SLIDES_BY_FN.get(where, {}).get(s, S.SLIDES.get(s))
    if t is None:
        t = S.normalize(s)
    if t is None:
        MISSING.append((where, s))
        return s
    return t


def _tr_const(c, where, pre):
    if isinstance(c, str):
        return _tr_str(c, where, pre)
    if isinstance(c, tuple):
        if c in S.SLIDES:   # 行の数を変える場合はタプルごと置き換える
            return S.SLIDES[c]
        return tuple(_tr_const(e, where, pre) for e in c)
    if isinstance(c, types.CodeType):   # 入れ子の関数・ラムダ
        return c.replace(co_consts=tuple(_tr_const(e, where, pre) for e in c.co_consts))
    return c


def translate(fn, pre=()):
    """関数 fn のコードの文字列定数を英語にする（docstring は除く）。関数そのものを書き換えるので，
    B と V2 のどちらから呼ばれても英語で組まれる。pre は訳す前に当てる（日本語 → 日本語の）置き換え。"""
    co = fn.__code__
    consts = [c if (i == 0 and fn.__doc__ is not None and c == fn.__doc__) else _tr_const(c, fn.__name__, pre)
              for i, c in enumerate(co.co_consts)]
    fn.__code__ = co.replace(co_consts=tuple(consts))
    return fn


# v2 が文言を置き換えた 2 枚は，v2 の置き換え（pairs）を当てた v1 の関数を訳して使う
_op = inspect.getclosurevars(V2._s_operators_v2).nonlocals
_hs = inspect.getclosurevars(V2._s_highstab_v2).nonlocals
B.s_operators = translate(_op["fn"], _op["pairs"])
B.s_highstab = translate(_hs["fn"], _hs["pairs"])
for _fn in [B.title_slide_talk, B.s_intro, B.s_background, V2.s_pareto, V2.s_focus, B.s_special,
            B.s_related, B.s_gaps, B.s_hypotheses, B.s_problem, B.s_design, V2.s_setup, V2.s_front,
            V2.s_band, B.s_extra, V2.s_anytime, V2.s_conclusion, B.s_limits, B.s_references,
            B.backup_divider, B.s_backup_problem, B.s_backup_scalar, B.s_backup_why, B.s_backup_operator,
            B.s_backup_metrics, B.s_backup_params, V2._s_backup_results_v1, V2._s_anytime_v1,
            V2._as_backup, B.s_backup_validity, B._bul, B._cite_note]:
    translate(_fn)

# 章扉は v1 の関数で組む（v2 は日本語の 1 章の項目を 2 行に分けるが，英語は対訳の \v で改行する）
B.CHAPTERS = S.CHAPTERS
B.chapter = V2._chapter_v1
# 先行研究のカード：文献番号は日本語のカードの同じ位置の著者から引き継ぐ
B.RELATED_NUM = {en: B.RELATED_NUM[ja]
                 for (_, ja_refs, _), (_, en_refs, _) in zip(B.RELATED, S.RELATED)
                 for (ja, _), (en, _) in zip(ja_refs, en_refs)}
B.RELATED = S.RELATED
B.REFS = {n: (short.replace(" ら (", " et al. ("), bib) for n, (short, bib) in B.REFS.items()}

# ================================================================ 文字（Segoe UI・英語）
B.JP = FONT
B._set_font.__defaults__ = tuple(FONT if d == "Meiryo" else d for d in B._set_font.__defaults__)
_run_ja = B._run


def _run(paragraph, text, size, color=INK, bold=False, italic=False, baseline=None, font=None):
    """v1 の _run を Segoe UI・en-US で組む。対訳の「\n」「\v」は段落内の改行にする（run は改行を持てないため）。"""
    r = None
    for i, part in enumerate(re.split(r"[\n\v]", text)):
        if i:
            paragraph.add_line_break()
        r = _run_ja(paragraph, part, size, color=color, bold=bold, italic=italic, baseline=baseline, font=FONT)
        r._r.get_or_add_rPr().set("lang", "en-US")
    return r


B._run = _run

_PIL = {False: ImageFont.truetype(os.path.join(FONT_DIR, "segoeui.ttf"), 1000),
        True: ImageFont.truetype(os.path.join(FONT_DIR, "segoeuib.ttf"), 1000)}
_SYM = ImageFont.truetype(os.path.join(FONT_DIR, "seguisym.ttf"), 1000)
_HAS = set(TTFont(os.path.join(FONT_DIR, "segoeui.ttf")).getBestCmap())


def _text_pt(text, size, bold=False):
    """Segoe UI で組んだときの文字列の幅（pt）。ぶら下げの幅に使う。Segoe UI にない記号（★ など）は，
    PowerPoint が代わりに使う Segoe UI Symbol で測る。"""
    text = text.replace("\u3000", "  ")
    w, i = 0.0, 0
    while i < len(text):
        has = ord(text[i]) in _HAS
        j = i
        while j < len(text) and (ord(text[j]) in _HAS) == has:
            j += 1
        w += (_PIL[bold] if has else _SYM).getlength(text[i:j])
        i = j
    return w / 1000 * size


B._text_pt = _text_pt

# Segoe UI は字面が行の中央より下に寄る（大文字の中心で文字の大きさの約 0.06 倍，小文字の多い行ではもっと下）。
# 上下中央揃えの文字は，下の余白を足して字面を上げる（v1 の Meiryo は逆に字面が上に寄るので，上の余白で下げている）
OPTICAL_RAISE = 0.08


def _optical_center(prs):
    """上下中央揃えの文字（図形・テキストボックス・表のセル）を，字面が箱の中央に来るよう上げる。題名は除く。"""
    def raise_(obj, tf):
        sizes = [r.font.size.pt for p in tf.paragraphs for r in p.runs if r.font.size is not None]
        if sizes:
            obj.margin_bottom = obj.margin_bottom + Pt(2 * OPTICAL_RAISE * max(sizes))

    for slide in prs.slides:
        for sh in slide.shapes:
            if sh.name == "title":
                continue
            if sh.has_text_frame and sh.text_frame.vertical_anchor == MSO_ANCHOR.MIDDLE:
                raise_(sh.text_frame, sh.text_frame)
            if sh.has_table:
                for row in sh.table.rows:
                    for cell in row.cells:
                        if cell.vertical_anchor == MSO_ANCHOR.MIDDLE:
                            raise_(cell, cell.text_frame)


B._optical_center = _optical_center

# ================================================================ 図（英語版）
EN_FIGS = {"concept_pareto_v2.png", "concept_hv_v2.png", "concept_sp_v2.png", "concept_h1.png", "concept_h2.png",
           "concept_op.png", "struct_ils.png", "struct_mem.png", "metric_auc.png", "front_ta21L_v2.png",
           "highstab_all8_v2.png", "highstab_4_v2.png", "anytime_ta21L_v2.png", "anytime.png"}
_asset_v2 = B.asset   # v1 の図の名前 → v2 の図（V2_ASSETS）を含む


def _asset(name):
    path = _asset_v2(name)
    base = os.path.basename(path)
    return os.path.join(os.path.dirname(path), base[:-4] + "_en.png") if base in EN_FIGS else path


B.asset = _asset


# ================================================================ 英語で組み方を変えるスライド

def s_structures():
    """v1 の探索構造のスライドの英語版。英語の手順名が 1 行に収まるよう，流れ図の箱を v1 の 3.25 in から 3.45 in に
    広げ，構造の模式図はわずかに右へ寄せる（戻りの矢印と重ならないように）。それ以外の組み方は v1 と同じ。"""
    s = B.slide_frame("Trajectory-Based vs. Population-Based Search",
                      "With local search and the other conditions held identical, we compare how solutions are "
                      "made: changed little by little, or recombined by crossover", S.SEC2)
    top, fh_ = TOP, 4.82
    gap = 0.3
    cw = (CW - gap) / 2
    fw = 3.45
    cols = [
        (ORANGE, "Trajectory-based (ILS)",
         [("Initial solution: $S_p$", None), ("Local search: N5", "shared"),
          ("Perturbation: insert moves", ORANGE),
          ("Local search: N5", "shared"), ("Accept if $F$ improves", None)],
         2, 4, "Iterate",
         "Why chosen: perturbation is a stage separate from\nlocal search, so the operators fit into it"),
        (GREEN, "Population-based (Memetic)",
         [("Initial population: $S_p$ + random", None), ("Selection", None),
          ("Crossover of 2 parents", GREEN),
          ("Mutation", None), ("Local search: N5 (each individual)", "shared")],
         1, 4, "Next\ngeneration",
         "Why chosen: the standard configuration in prior work,\na GA with local search built in"),
    ]
    for k, (col, head, steps, lf, lt, loop, why) in enumerate(cols):
        x = ML + k * (cw + gap)
        B._rrect(s, x, top, cw, fh_, WHITE, line=CARDLINE, radius=0.05)
        B._rect(s, x, top, cw, 0.52, col)
        tb, tf = B.add_textbox(s, x + 0.2, top, cw - 0.4, 0.52, anchor=MSO_ANCHOR.MIDDLE)
        B._run(tf.paragraphs[0], head, FS_HEAD, color=WHITE, bold=True)
        B._cite_run(tf.paragraphs[0], [15] if k == 0 else [16], WHITE)
        B._flow(s, x + 0.25, top + 0.78, fw, steps, bh=0.5, gap=0.18, loop_from=lt, loop_to=lf,
                loop_label=loop, fs=B.FS_IN, label_w=1.1, label_fs=FS_NOTE)
        B.pic(s, "struct_ils.png" if k == 0 else "struct_mem.png", x + cw - 1.74, top + 0.6, w=1.68)
        # 2 行目は「Why chosen: 」の後ろにそろえる（ぶら下げ）
        tb, tf = B.add_textbox(s, x + 0.25, top + 4.1, cw - 0.5, 0.62)
        p = tf.paragraphs[0]
        p.line_spacing = 1.05
        hang = int(Pt(B._text_pt("Why chosen: ", FS_NOTE)))
        pPr = p._p.get_or_add_pPr()
        pPr.set("marL", str(hang))
        pPr.set("indent", str(-hang))
        for j, line in enumerate(why.split("\n")):
            if j:
                p.add_line_break()
            B._add_runs(p, line, FS_NOTE, color=MUTE)
    B._boxtext(s, ML, top + fh_ + 0.2, CW, 0.6,
               [("To isolate the effect of structure, both share the local search (N5[[17]]), the objective $F$, and $S_p$",
                 FS_BODY, INK, True)], fill=B.FILL, line=BLUE, radius=0.12)
    B._cite_note(s, [15, 16, 17])


def s_union_auc():
    """v1 の全域 HV と AUC の集計表（予備）の英語版。表の見出し（Overall HV・High-stability HV）が 1 行に収まるよう，
    表の幅を v1 の 5.75 in から 6.6 in に広げ，右の読み取りの列をその分詰める。"""
    s = B.slide_frame("Overall HV and AUC: All 8 Scenarios",
                      "The best method differs by metric: Memetic + operators in overall HV, ILS in AUC", S.SEC3)
    tw = 6.6
    B._text(s, ML, TOP, tw, 0.42, "Scenarios level with the best (of 8)", FS_HEAD, color=BLUE, bold=True,
            anchor=MSO_ANCHOR.MIDDLE)
    trh = 0.57
    B.render_table(s, [
        ["Method", "Overall HV", "High-stability HV", "AUC"],
        ["ILS", "4", "6", "**6**"],
        ["ILS+PR-$k$", "4", "**8**", "**6**"],
        ["ILS+PR", "4", "**8**", "**6**"],
        ["GA", "0", "0", "0"],
        ["Memetic", "1", "0", "2"],
        ["Memetic+PR-$k$", "**7**", "**7**", "1"],
        ["Memetic+PR", "**7**", "**7**", "0"],
    ], ML, TOP + 0.5, tw, col_w=[2.0, 1.55, 2.25, 0.8], size=FS_BODY, row_h=trh)
    B._text(s, ML, TOP + 0.5 + 8 * trh + 0.08, tw, 0.36,
            "Level with the best: median within 2% of the best median", FS_NOTE, color=MUTE)
    rx = ML + tw + 0.45
    rw = SW - ML - rx
    B._text(s, rx, TOP, rw, 0.42, "Overall HV", FS_HEAD, color=BLUE, bold=True, anchor=MSO_ANCHOR.MIDDLE)
    B._bul(s, rx, TOP + 0.5, rw, [
        "Memetic + operators: level with\nthe best in 7",
        "ILS variants: level with the best in 4",
        "The operators raise Memetic's overall\nHV but barely change that of ILS",
    ], size=FS_BODY, gap=6)
    B._divider(s, rx, TOP + 2.45, rw)
    B._text(s, rx, TOP + 2.65, rw, 0.42, "AUC", FS_HEAD, color=BLUE, bold=True, anchor=MSO_ANCHOR.MIDDLE)
    B._bul(s, rx, TOP + 3.15, rw, [
        "ILS variants: level with the best in 6",
        "In Memetic, the operators fire per\nindividual and add computation time;\n"
        "PR lowers AUC by 7–26% in 5 scenarios",
    ], size=FS_BODY, gap=6)


def _after(fn, fix):
    """スライド関数 fn で組んだ後に，そのスライドへ fix を当てる。"""
    def wrapped():
        fn()
        fix(B.prs.slides[-1])
    wrapped.__name__ = fn.__name__
    return wrapped


def _widen(text, dw):
    """文字が text の図形を，右端を保ったまま左へ dw in 広げる（英語の札が 1 行に収まるように）。"""
    def fix(slide):
        for sh in slide.shapes:
            if sh.has_text_frame and sh.text_frame.text == text:
                sh.left -= Inches(dw)
                sh.width += Inches(dw)
                return
        raise RuntimeError(f"_widen: 「{text}」の図形が見つからない")
    return fix


def _taller(top, h, new_h):
    """上端 top・高さ h の図形（カードの枠）を高さ new_h にする（英語の箇条書きが枠に収まるように）。"""
    def fix(slide):
        hit = [sh for sh in slide.shapes
               if abs(sh.top - Inches(top)) < Inches(0.01) and abs(sh.height - Inches(h)) < Inches(0.01)]
        if not hit:
            raise RuntimeError(f"_taller: 上端 {top}・高さ {h} の図形が見つからない")
        for sh in hit:
            sh.height = Inches(new_h)
    return fix


def _align_card_body(slide):
    """研究の出発点：単点探索・多点探索のカードの本文の左端を，見出しの帯の文字の左端にそろえる。v1 は本文の枠を
    カードの枠から 0.05 in に置くので，見出しの帯の文字より 0.07 in 左に出る（日本語版は例の行を 1 行に収めるため
    この幅が要るが，英語は行が短いので見出しと同じ幅に収まる）。"""
    cy = TOP + 3.0 + 0.42   # カードの上端（v1 と同じ）．本文の枠は見出しの帯の下の cy + 0.43 から
    body = [sh for sh in slide.shapes
            if sh.has_text_frame and abs(sh.top - Inches(cy + 0.43)) < Inches(0.005)]
    if len(body) != 2:
        raise RuntimeError(f"_align_card_body: カードの本文の枠が 2 つ見つからない（{len(body)} 個）")
    for sh in body:   # 見出しの帯の文字と同じ左端（カードの枠から 0.12 in）・同じ幅にする
        sh.left += Inches(0.07)
        sh.width -= Inches(0.14)


B.s_structures = s_structures
V2._s_union_auc_v1 = s_union_auc   # 予備（V2.s_backup_results が _as_backup で組む）
B.s_intro = _after(B.s_intro, _widen("This study", 0.25))
B.s_special = _after(B.s_special, _align_card_body)
B.s_backup_why = _after(B.s_backup_why, _taller(1.45, 3.3, 3.9))   # 予備：なぜ ILS と Memetic か


# ================================================================ 組む・確かめる

def _japanese_left(prs):
    """組んだスライドの文字・表・ノートに残った日本語を (スライド番号, 文字) で返す。"""
    left = []
    for i, slide in enumerate(prs.slides, 1):
        texts = []
        for sh in slide.shapes:
            if sh.has_text_frame:
                texts.append(sh.text_frame.text)
            if sh.has_table:
                texts += [c.text_frame.text for row in sh.table.rows for c in row.cells]
        if slide.has_notes_slide:
            texts.append(slide.notes_slide.notes_text_frame.text)
        left += [(i, t) for t in texts if S.CJK.search(t)]
    return left


def main():
    if MISSING:
        print("対訳のない日本語（talk_strings_en.py の SLIDES に足す）:")
        for where, s in dict.fromkeys(MISSING):
            print(f"  [{where}] {s!r}")
        sys.exit(1)
    B.build()
    left = _japanese_left(B.prs)
    if left:
        print("スライドに日本語が残っている:")
        for i, t in left:
            print(f"  [{i}] {t!r}")
        sys.exit(1)
    if "--no-pdf" not in sys.argv:
        B.export_pdf(B.OUT)


if __name__ == "__main__":
    main()
