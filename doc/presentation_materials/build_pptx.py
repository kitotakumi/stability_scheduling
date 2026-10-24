# -*- coding: utf-8 -*-
"""学会発表（JIMA2026 秋・15 分）用 pptx を生成する（深い青テーマ）。

構成は仮説の提示までを JIMA 予稿、その後を APIEMS v2 原稿（実験計画 → 指標ごとの結果 → 結論）に合わせる。

デザイン方針:
  - テーマカラーは「深い青」の単色系。装飾を足さず余白で見せる。
  - 表紙・章扉は深い青の背景に白文字。本文は白背景。
  - 本文スライドは「タイトル → 単色の細い区切り線 → コンテンツ」。コンテンツは区切り線の下の
    同じ高さ（TOP）から上揃えで書き出す。各スライドの主張の 1 文はノートに置き（話すときの台本），
    画面では本文や結果の読み取りが同じことを言う。
  - 文字の大きさは役割ごとに 1 つに揃える（FS_TITLE・FS_BODY・FS_HEAD・FS_IN・FS_NOTE）。
  - 図は seminar/ の原図をそのまま使用（SVGは render_assets.py で忠実にPNG化）。

前提:  python make_talk_figures.py を先に実行（assets/ に図を生成）。
再生成:  python build_pptx.py          （pptx と同名の PDF も書き出す．PDF 不要なら --no-pdf）
"""
import os
import re as _re
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE
from pptx.oxml.ns import qn
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------- パス
HERE = os.path.dirname(os.path.abspath(__file__))
DOC = os.path.dirname(HERE)
SEM = os.path.join(DOC, "seminar")
ASSETS = os.path.join(HERE, "assets")
os.makedirs(ASSETS, exist_ok=True)
OUT = os.path.join(HERE, "JIMA2026fall_発表.pptx")

def sem(name):  return os.path.join(SEM, name)
def asset(name): return os.path.join(ASSETS, name)

# ---------------------------------------------------------------- 配色（深い青の単色系）
INK    = RGBColor(0x26, 0x30, 0x3A)   # 本文
BLUE   = RGBColor(0x16, 0x38, 0x5C)   # 主役：深い青（見出し・線・強調・表ヘッダ）
COVER  = RGBColor(0x12, 0x31, 0x50)   # 表紙・章扉の背景
BLUESOFT = RGBColor(0x3E, 0x61, 0x8C) # 章番号などの控えめな青
LTBLUE = RGBColor(0xAF, 0xC2, 0xD8)   # 濃紺背景上のサブ文字
EN_BLUE = RGBColor(0x8C, 0xA3, 0xC0)  # 濃紺背景上の英字
MUTE   = RGBColor(0x6E, 0x7A, 0x88)   # 脚注・キャプション
FILL   = RGBColor(0xED, 0xF2, 0xF7)   # 淡い青（コールアウト・表の交互行）
ROWALT = RGBColor(0xF6, 0xF9, 0xFB)
WHITE  = RGBColor(0xFF, 0xFF, 0xFF)
HAIR   = RGBColor(0xD6, 0xDE, 0xE7)   # ごく薄い罫線

JP = "Meiryo"

# ---------------------------------------------------------------- レイアウト定数(inch)
SW, SH = 13.333, 7.5
ML = 0.62
CW = SW - 2 * ML
LINE_Y = 1.06
TOP = 1.32            # 本文の書き出し（題名の区切り線の下）．全スライドでこの高さから上揃えで始める
CONTENT_BOTTOM = 7.02
BOT = CONTENT_BOTTOM - 0.1   # 中身（箱・図）の下端を揃える線．書き出し（TOP）と下端の両方を全スライドで揃える

# 文字の大きさ（pt）．役割ごとに 1 つに揃える（スライドごとに変えると統一感が崩れる）
FS_TITLE = 28   # 題名
FS_BODY = 18    # 本文（文章・箇条書き・表）
FS_HEAD = 18    # 図形の見出し（カード・枠・札．太字）
FS_IN = 16      # 図形の中の文字（流れ図・ガント・例の箱・カードの中身）
FS_NOTE = 15    # 注記・出典・図の注

prs = Presentation()
prs.slide_width = Inches(SW)
prs.slide_height = Inches(SH)
BLANK = prs.slide_layouts[6]
_page = {"n": 0}

# ================================================================ テキスト（数式・太字）

def _set_font(run, size, color=INK, bold=False, italic=False, font=JP):
    f = run.font
    f.name = font
    f.size = Pt(size)
    f.bold = bold
    f.italic = italic
    f.color.rgb = color
    rPr = run._r.get_or_add_rPr()
    ea = rPr.find(qn('a:ea'))
    if ea is None:
        ea = rPr.makeelement(qn('a:ea'), {}); rPr.append(ea)
    ea.set('typeface', font)


def _run(paragraph, text, size, color=INK, bold=False, italic=False,
         baseline=None, font=JP):
    r = paragraph.add_run(); r.text = text
    _set_font(r, size, color=color, bold=bold, italic=italic, font=font)
    # 言語を日本語にする。既定の en-US のままだと PowerPoint が禁則処理と句読点のぶら下げを
    # 掛けず，「，」が行頭に来る
    r._r.get_or_add_rPr().set('lang', 'ja-JP')
    if baseline is not None:
        r._r.get_or_add_rPr().set('baseline', str(int(baseline)))
    return r


_SYM = [
    ("\\left|", "|"), ("\\right|", "|"), ("\\left(", "("), ("\\right)", ")"),
    ("\\lambda", "λ"), ("\\rho", "ρ"), ("\\delta", "δ"), ("\\Delta", "Δ"),
    ("\\beta", "β"), ("\\tau", "τ"), ("\\in", "∈"), ("\\sum", "Σ"),
    ("\\times", "×"), ("\\approx", "≈"), ("\\cdot", "·"), ("\\min", "min"),
    ("\\,", " "), ("\\;", " "), ("\\!", ""), ("\\ ", " "), ("\\{", "{"), ("\\}", "}"),
]


def _latex_clean(s):
    s = _re.sub(r'\\widehat\{([^{}]*)\}', r'\1', s)
    s = _re.sub(r'\\hat\{([^{}]*)\}', r'\1', s)
    s = _re.sub(r'\\hat\s+(\w)', r'\1', s)
    s = _re.sub(r'\\mathcal\{([^{}]*)\}', r'\1', s)
    s = _re.sub(r'\\mathrm\{([^{}]*)\}', r'\1', s)
    s = _re.sub(r'\\text\{([^{}]*)\}', r'\1', s)
    for a, b in _SYM:
        s = s.replace(a, b)
    return s


def _emit_mathexpr(paragraph, expr, size, color, bold):
    s = _latex_clean(expr)
    i, n, buf = 0, len(s), ""
    def flush():
        nonlocal buf
        if buf:
            _run(paragraph, buf, size, color=color, bold=bold, italic=True); buf = ""
    while i < n:
        c = s[i]
        if c in "_^":
            flush(); i += 1
            if i < n and s[i] == "{":
                j = s.find("}", i)
                grp = s[i + 1:j] if j != -1 else s[i + 1:]
                i = (j + 1) if j != -1 else n
            else:
                grp = s[i] if i < n else ""; i += 1
            grp = grp.strip()
            if grp:
                base = -22000 if c == "_" else 30000
                _run(paragraph, grp, size * 0.72, color=color, bold=bold,
                     italic=True, baseline=base)
        else:
            buf += c; i += 1
    flush()


def _emit_math(paragraph, seg, size, color, bold, emph):
    use = emph if (bold and emph is not None) else color
    for i, part in enumerate(seg.split("$")):
        if i % 2 == 0:
            if part:
                _run(paragraph, part, size, color=use, bold=bold)
        elif part:
            _emit_mathexpr(paragraph, part, size, use, bold)


def _add_runs(paragraph, text, size, color=INK, bold=False, emph=None):
    """**太字** と $数式$ を run に分解。emph 指定時は太字だけその色にする。"""
    state = bold
    for seg in text.split("**"):
        if seg:
            _emit_math(paragraph, seg, size, color, state, emph)
        state = not state


def add_textbox(slide, x, y, w, h, anchor=MSO_ANCHOR.TOP):
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    tf.margin_left = tf.margin_right = Inches(0.02)
    tf.margin_top = tf.margin_bottom = Inches(0.01)
    return tb, tf


def _hline(slide, x, y, w, color=BLUE, weight=1.25):
    ln = slide.shapes.add_connector(2, Inches(x), Inches(y), Inches(x + w), Inches(y))
    ln.line.color.rgb = color
    ln.line.width = Pt(weight)
    ln.shadow.inherit = False
    return ln


def _rect(slide, x, y, w, h, fill, line=None):
    sp = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
    if fill is None:
        sp.fill.background()
    else:
        sp.fill.solid(); sp.fill.fore_color.rgb = fill
    if line is None:
        sp.line.fill.background()
    else:
        sp.line.color.rgb = line; sp.line.width = Pt(0.75)
    sp.shadow.inherit = False
    return sp


# ---- 作図用（角丸ボックス・コネクタ・カード） ----
CARDLINE = RGBColor(0xC9, 0xD6, 0xE5)

def _rrect(slide, x, y, w, h, fill, line=None, radius=0.10, lw=1.1):
    sp = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE,
                                Inches(x), Inches(y), Inches(w), Inches(h))
    if fill is None:
        sp.fill.background()
    else:
        sp.fill.solid(); sp.fill.fore_color.rgb = fill
    if line is None:
        sp.line.fill.background()
    else:
        sp.line.color.rgb = line; sp.line.width = Pt(lw)
    sp.shadow.inherit = False
    try:
        sp.adjustments[0] = radius
    except Exception:
        pass
    return sp


def _boxtext(slide, x, y, w, h, lines, fill=BLUE, line=None, radius=0.10):
    sp = _rrect(slide, x, y, w, h, fill, line, radius)
    tf = sp.text_frame; tf.word_wrap = True
    tf.margin_left = tf.margin_right = Inches(0.06)
    tf.margin_top = tf.margin_bottom = Inches(0.02)
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    for i, (t, s, c, b) in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = PP_ALIGN.CENTER; p.line_spacing = 1.0
        _add_runs(p, t, s, color=c, bold=b, emph=c)
    return sp


def _connector(slide, x1, y1, x2, y2, color=BLUESOFT, wt=1.4, arrow=True, kind=2):
    cn = slide.shapes.add_connector(kind, Inches(x1), Inches(y1), Inches(x2), Inches(y2))
    cn.line.color.rgb = color; cn.line.width = Pt(wt); cn.shadow.inherit = False
    if arrow:
        ln = cn.line._get_or_add_ln()
        ln.append(ln.makeelement(qn('a:tailEnd'),
                                 {'type': 'triangle', 'w': 'med', 'len': 'med'}))
    return cn


def _card(slide, x, y, w, h, title, body, badge=None, accent=BLUE):
    _rrect(slide, x, y, w, h, WHITE, line=CARDLINE, radius=0.05, lw=1.1)
    tb, tf = add_textbox(slide, x + 0.2, y + 0.2, w - 0.4, 0.85)
    tf.word_wrap = True
    p = tf.paragraphs[0]; p.line_spacing = 1.02
    if badge:
        _run(p, badge + " ", 18, color=accent, bold=True)
    _add_runs(p, title, 14, color=accent, bold=True, emph=accent)
    tb, tf = add_textbox(slide, x + 0.2, y + (1.12 if badge else 0.78), w - 0.4,
                         h - (1.25 if badge else 0.9))
    tf.word_wrap = True
    p = tf.paragraphs[0]; p.line_spacing = 1.1
    _add_runs(p, body, 12.5, color=INK, emph=accent)

# ================================================================ スライド枠（本文）

def _title_zone(slide, title):
    tb, tf = add_textbox(slide, ML, 0.34, CW, 0.66, anchor=MSO_ANCHOR.MIDDLE)
    tb.name = "title"   # 題名は区切り線との間隔で位置を決めているので，_optical_center で動かさない
    _add_runs(tf.paragraphs[0], title, FS_TITLE, color=BLUE, bold=True)
    _hline(slide, ML, LINE_Y, CW, color=BLUE, weight=1.25)


def _plain(text):
    """ノート用の平文。**太字** と $数式$ の記号を外す（$S_p$ → Sp）。"""
    text = text.replace("**", "")
    return _re.sub(r"\$([^$]*)\$",
                   lambda m: _re.sub(r"[_^{}]", "", _latex_clean(m.group(1))), text)


def _footer(slide, section):
    _page["n"] += 1
    if section:
        tb, tf = add_textbox(slide, ML, 7.08, 9.0, 0.30)
        _run(tf.paragraphs[0], section, 9, color=MUTE)
    tb, tf = add_textbox(slide, SW - ML - 1.2, 7.08, 1.2, 0.30)
    p = tf.paragraphs[0]; p.alignment = PP_ALIGN.RIGHT
    _run(p, str(_page["n"]), 10, color=MUTE)

# ================================================================ ブロック描画（フロー）
GAP = 0.15

def _img_size(path):
    with Image.open(path) as im:
        return im.width, im.height


def render_bullets(slide, items, x, y, w, spacing=1.0):
    """items: (level, text)。level 2 は控えめ（MUTE・強調なし）の脚注的行。
    spacing は行間の追加倍率（1.0=既定、>1 で全体をゆったり）。"""
    tb, tf = add_textbox(slide, x, y, w, 0.4)
    first, lines = True, 0
    for level, text in items:
        p = tf.paragraphs[0] if first else tf.add_paragraph()
        first = False
        p.line_spacing = 1.06 * spacing
        if level == 0:
            p.space_before = Pt(7 * spacing)
            _run(p, "▪ ", 12, color=BLUE, bold=True)
            _add_runs(p, text, 15.5, color=INK, emph=BLUE)
            indent = 0.0
        elif level == 1:
            p.space_before = Pt(3 * spacing)
            p.level = 1
            _run(p, "－ ", 12, color=MUTE)
            _add_runs(p, text, 14, color=INK, emph=BLUE)
            indent = 0.42
        else:  # level 2: 控えめな脚注行（強調色を使わない）
            p.space_before = Pt(4 * spacing)
            _add_runs(p, text, 12, color=MUTE, emph=MUTE)
            indent = 0.0
        cap = max(8, (w - indent) / 0.163)
        wlen = sum(2 if ord(c) > 0x2000 else 1 for c in text)
        lines += max(1, -(-wlen // int(cap)))
    return (lines * 0.29 + len(items) * 0.11 + 0.05) * (spacing if spacing > 1 else 1)


def render_image(slide, path, x, y, w, target_h=None, target_w=None, caption=None,
                 align="center", border=False):
    iw, ih = _img_size(path)
    ar = iw / ih
    if target_h is not None:
        h = target_h; wd = h * ar
        if wd > w:
            wd = w; h = wd / ar
    else:
        wd = (target_w if target_w else w); h = wd / ar
    ix = x + (w - wd) / 2 if align == "center" else (x if align == "left" else x + (w - wd))
    pic = slide.shapes.add_picture(path, Inches(ix), Inches(y), Inches(wd), Inches(h))
    if border:
        pic.line.color.rgb = HAIR; pic.line.width = Pt(0.75)
    total = h
    if caption:
        tb, tf = add_textbox(slide, x, y + h + 0.03, w, 0.30)
        p = tf.paragraphs[0]; p.alignment = PP_ALIGN.CENTER
        _add_runs(p, caption, 10.5, color=MUTE)
        total += 0.32
    return total


def _cell_border(cell, edges=("bottom",), color="D6DEE7", w_pt=0.75):
    tcPr = cell._tc.get_or_add_tcPr()
    tag = {"left": "a:lnL", "right": "a:lnR", "top": "a:lnT", "bottom": "a:lnB"}
    for e in edges:
        el = tcPr.find(qn(tag[e]))
        if el is not None:
            tcPr.remove(el)
        ln = tcPr.makeelement(qn(tag[e]), {'w': str(int(w_pt * 12700)), 'cap': 'flat'})
        fill = ln.makeelement(qn('a:solidFill'), {})
        clr = fill.makeelement(qn('a:srgbClr'), {'val': color})
        fill.append(clr); ln.append(fill); tcPr.append(ln)


def render_table(slide, rows, x, y, w, col_w=None, size=12.5, header=True, align=None, row_h=0.33):
    nr, nc = len(rows), len(rows[0])
    gshape = slide.shapes.add_table(nr, nc, Inches(x), Inches(y), Inches(w), Inches(0.4 * nr))
    tbl = gshape.table
    tblPr = tbl._tbl.tblPr
    tblPr.set('firstRow', '0'); tblPr.set('bandRow', '0')
    for sid in list(tblPr.findall(qn('a:tableStyleId'))):
        tblPr.remove(sid)
    sid = tblPr.makeelement(qn('a:tableStyleId'), {})
    sid.text = "{5940675A-B579-460E-94D1-54222C63F5DA}"
    tblPr.append(sid)
    if col_w:
        tot = sum(col_w)
        for i, c in enumerate(tbl.columns):
            c.width = Inches(w * col_w[i] / tot)
    for ri, row in enumerate(rows):
        tbl.rows[ri].height = Inches(row_h)
        is_head = header and ri == 0
        for ci, val in enumerate(row):
            cell = tbl.cell(ri, ci)
            cell.margin_left = Inches(0.09); cell.margin_right = Inches(0.06)
            cell.margin_top = Inches(0.02); cell.margin_bottom = Inches(0.02)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            cell.fill.solid()
            cell.fill.fore_color.rgb = BLUE if is_head else (WHITE if ri % 2 == 1 else ROWALT)
            p = cell.text_frame.paragraphs[0]
            cell.text_frame.word_wrap = True
            if align and align[ci] == 'c':
                p.alignment = PP_ALIGN.CENTER
            elif align and align[ci] == 'l':
                p.alignment = PP_ALIGN.LEFT
            elif ci == 0:
                p.alignment = PP_ALIGN.LEFT
            else:
                p.alignment = PP_ALIGN.CENTER
            for li, line in enumerate(str(val).split("\n")):
                q = p if li == 0 else cell.text_frame.add_paragraph()
                q.alignment = p.alignment
                _add_runs(q, line, size, color=(WHITE if is_head else INK),
                          bold=is_head, emph=(WHITE if is_head else BLUE))
            if not is_head:
                _cell_border(cell, edges=("bottom",))
    return 0.33 * nr + 0.05


def render_note(slide, text, x, y, w):
    cap = max(8, (w - 0.3) / 0.152)
    wlen = sum(2 if ord(c) > 0x2000 else 1 for c in text)
    lines = max(1, -(-wlen // int(cap)))
    h = lines * 0.27 + 0.22
    _rect(slide, x, y, w, h, FILL)
    _rect(slide, x, y, 0.06, h, BLUE)          # 左アクセントバー
    tb, tf = add_textbox(slide, x + 0.22, y, w - 0.34, h, anchor=MSO_ANCHOR.MIDDLE)
    tf.word_wrap = True
    p = tf.paragraphs[0]; p.line_spacing = 1.04
    _add_runs(p, text, 13, color=INK, emph=BLUE)
    return h


def eq_png(name, latex, fontsize=22):
    path = asset(f"eq_{name}.png")
    fig = plt.figure(figsize=(0.01, 0.01))
    fig.text(0, 0, f"${latex}$", fontsize=fontsize, color="#16385C")
    fig.savefig(path, dpi=200, bbox_inches="tight", pad_inches=0.08, facecolor="white")
    plt.close(fig)
    return path


def render_equation(slide, latex, x, y, w, number=None, name="eq", h=0.6):
    path = eq_png(name, latex)
    iw, ih = _img_size(path); ar = iw / ih
    wd = min(w * 0.8, h * ar); hh = wd / ar
    slide.shapes.add_picture(path, Inches(x + (w - wd) / 2), Inches(y), Inches(wd), Inches(hh))
    if number:
        tb, tf = add_textbox(slide, x + w - 0.7, y + hh / 2 - 0.15, 0.7, 0.3)
        p = tf.paragraphs[0]; p.alignment = PP_ALIGN.RIGHT
        _run(p, number, FS_NOTE, color=INK)
    return hh + 0.12

# ================================================================ フロー

def content_slide(title, key, section, blocks):
    slide = slide_frame(title, key, section)
    _render_blocks(slide, blocks, ML, TOP, CW)
    return slide


def _render_blocks(slide, blocks, x, y, w):
    cur = y
    for b in blocks:
        if 'y' in b:
            cur = b['y']
        k = b['k']
        if k == 'spacer':
            cur += b.get('h', 0.2); continue
        if k == 'bullets':
            hh = render_bullets(slide, b['items'], x, cur, b.get('w', w),
                                spacing=b.get('spacing', 1.0))
        elif k == 'image':
            hh = render_image(slide, b['path'], x, cur, b.get('w', w),
                              target_h=b.get('h'), target_w=b.get('tw'),
                              caption=b.get('caption'), align=b.get('align', 'center'),
                              border=b.get('border', False))
        elif k == 'table':
            hh = render_table(slide, b['rows'], x, cur, b.get('w', w),
                              col_w=b.get('col_w'), size=b.get('size', 12.5),
                              align=b.get('align'), header=b.get('header', True))
        elif k == 'note':
            hh = render_note(slide, b['text'], x, cur, b.get('w', w))
        elif k == 'draw':
            hh = b['fn'](slide, x, cur, b.get('w', w))
        elif k == 'eq':
            hh = render_equation(slide, b['latex'], x, cur, b.get('w', w),
                                 number=b.get('number'), name=b.get('name', 'eq'),
                                 h=b.get('h', 0.6))
        elif k == 'row':
            ratio = b.get('ratio', [1] * len(b['cols']))
            tot = sum(ratio); gap = b.get('gap', 0.34)
            avail = w - gap * (len(b['cols']) - 1)
            hmax, cx = 0, x
            for col, rr in zip(b['cols'], ratio):
                cwid = avail * rr / tot
                hmax = max(hmax, _render_blocks(slide, col, cx, cur, cwid))
                cx += cwid + gap
            hh = hmax
        else:
            hh = 0
        cur += hh + GAP
    return cur - y

# ================================================================ 表紙・章扉

def title_slide():
    slide = prs.slides.add_slide(BLANK)
    _rect(slide, 0, 0, SW, SH, COVER)
    # 上部の細いアクセント
    _hline(slide, 0.95, 2.28, 2.6, color=BLUESOFT, weight=2.0)
    tb, tf = add_textbox(slide, 0.95, 2.45, SW - 1.9, 1.9)
    p = tf.paragraphs[0]; p.line_spacing = 1.12
    _run(p, "ジョブショップ再スケジューリングにおける", 31, color=WHITE, bold=True)
    p2 = tf.add_paragraph(); p2.line_spacing = 1.12
    _run(p2, "安定性誘導演算子の非対称効果", 31, color=WHITE, bold=True)
    p3 = tf.add_paragraph(); p3.space_before = Pt(10)
    _run(p3, "— 軌道探索と集団探索の横断比較 —", 17, color=LTBLUE, bold=True)
    # 英題
    tb, tf = add_textbox(slide, 0.97, 4.55, SW - 1.9, 0.7)
    p = tf.paragraphs[0]; p.line_spacing = 1.05
    _run(p, "Asymmetric Effects of Stability-Inducing Operators across "
            "Trajectory and Population Search in Job-Shop Rescheduling",
         12.5, color=EN_BLUE, italic=True)
    # 著者
    _hline(slide, 0.97, 5.55, 3.4, color=BLUESOFT, weight=1.0)
    tb, tf = add_textbox(slide, 0.97, 5.68, SW - 1.9, 1.3)
    rows = [("早稲田大学 創造理工学部 経営システム工学科", 14.5, LTBLUE, False),
            ("鬼頭 拓海", 18, WHITE, True),
            ("ゼミ発表　2026-07-21　｜　投稿先：APIEMS 2026（釜山）", 12, EN_BLUE, False)]
    for i, (txt, sz, col, bold) in enumerate(rows):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.space_before = Pt(5 if i else 0)
        _run(p, txt, sz, color=col, bold=bold)
    return slide


def divider_slide(no, title, subtitle=None):
    slide = prs.slides.add_slide(BLANK)
    _rect(slide, 0, 0, SW, SH, COVER)
    tb, tf = add_textbox(slide, 1.15, 2.35, 1.7, 1.5, anchor=MSO_ANCHOR.MIDDLE)
    _run(tf.paragraphs[0], no, 78, color=BLUESOFT, bold=True)
    tb, tf = add_textbox(slide, 2.75, 2.35, SW - 3.6, 1.5, anchor=MSO_ANCHOR.MIDDLE)
    p = tf.paragraphs[0]
    _run(p, title, 30, color=WHITE, bold=True)
    if subtitle:
        p2 = tf.add_paragraph(); p2.space_before = Pt(10)
        _run(p2, subtitle, 14, color=LTBLUE)
    _hline(slide, 2.78, 4.28, 5.2, color=BLUESOFT, weight=1.25)
    return slide


CHAPTERS = [("1", "研究背景と仮説", "外乱と再スケジューリング・高安定領域・高安定 HV・特殊性・先行研究・課題・仮説"),
            ("2", "問題設定と実験計画", "問題設定・探索構造・安定性誘導演算子・仮説の判定"),
            ("3", "数値実験", "実験設定・高安定 HV・追加の評価（全域 HV と AUC）"),
            ("4", "結論", "まとめ・限界と今後の課題")]


def chapter(no):
    """章扉。4 章すべてを並べ，当該の章だけを白い太字と淡い帯で目立たせる。
    スライド番号を本文と通しにするため番号を進め，右下に小さく出す。"""
    slide = prs.slides.add_slide(BLANK)
    _rect(slide, 0, 0, SW, SH, COVER)
    rh, gap = 0.95, 0.2
    y0 = (SH - (len(CHAPTERS) * rh + (len(CHAPTERS) - 1) * gap)) / 2
    x0, w = 2.2, SW - 4.4
    for i, (n, title, sub) in enumerate(CHAPTERS):
        y = y0 + i * (rh + gap)
        cur = (n == no)
        if cur:
            band = _rrect(slide, x0 - 0.3, y - 0.05, w + 0.6, rh + 0.1, BLUESOFT, radius=0.12)
            _alpha(band, 0.45)
        tb, tf = add_textbox(slide, x0, y, 1.0, rh, anchor=MSO_ANCHOR.MIDDLE)
        _run(tf.paragraphs[0], n, 40 if cur else 30, color=WHITE if cur else BLUESOFT, bold=True)
        tb, tf = add_textbox(slide, x0 + 1.1, y, w - 1.1, rh, anchor=MSO_ANCHOR.MIDDLE)
        p = tf.paragraphs[0]
        _run(p, title, 26 if cur else 20, color=WHITE if cur else BLUESOFT, bold=cur)
        if cur and sub:
            p2 = tf.add_paragraph(); p2.space_before = Pt(4)
            _run(p2, sub, 13, color=LTBLUE)
    _page["n"] += 1
    tb, tf = add_textbox(slide, SW - ML - 1.2, 7.08, 1.2, 0.30)
    p = tf.paragraphs[0]; p.alignment = PP_ALIGN.RIGHT
    _run(p, str(_page["n"]), 10, color=LTBLUE)
    return slide


# ================================================================ 発表用の作図

def _oval(slide, cx, cy, d, fill, line=None):
    sp = slide.shapes.add_shape(MSO_SHAPE.OVAL, Inches(cx - d / 2), Inches(cy - d / 2),
                                Inches(d), Inches(d))
    sp.fill.solid(); sp.fill.fore_color.rgb = fill
    if line is None:
        sp.line.fill.background()
    else:
        sp.line.color.rgb = line; sp.line.width = Pt(0.75)
    sp.shadow.inherit = False
    return sp

ORANGE = RGBColor(0xE0, 0x6C, 0x00)   # 単点探索（ILS）＝図の橙
GREEN  = RGBColor(0x0F, 0x5C, 0x28)   # 多点探索（Memetic）＝図の緑
BROWN  = RGBColor(0x2E, 0x1A, 0x0E)   # 安定性誘導演算子＝図の茶
VIOLET = RGBColor(0x6A, 0x4C, 0x93)   # 仮説 2 の札と枠（青の仮説 1 と見分けるための色）
RED    = RGBColor(0xB0, 0x3A, 0x2E)
SAND   = RGBColor(0xF5, 0xEB, 0xD0)   # 高安定領域の帯
SANDLN = RGBColor(0x7A, 0x5C, 0x12)


def _divider(slide, x, y, w):
    """話のまとまりの間に引く，ごく薄い横の区切り線。"""
    _rect(slide, x, y, w, 0.012, HAIR)


def _text(slide, x, y, w, h, text, size, color=INK, bold=False, align=PP_ALIGN.LEFT,
          anchor=MSO_ANCHOR.TOP, ls=1.05, emph=None):
    tb, tf = add_textbox(slide, x, y, w, h, anchor=anchor)
    for i, line in enumerate(text.split("\n")):     # 「\n」は段落の区切り，「\v」は段落内の改行
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        p.line_spacing = ls
        for k, seg in enumerate(line.split("\v")):
            if k:
                p.add_line_break()
            _add_runs(p, seg, size, color=color, bold=bold, emph=emph or color)
    return tb


def _text_pt(text, size, bold=False):
    """Meiryo で組んだときの文字列の幅（pt）。箇条書きのぶら下げを見出しの幅に合わせるのに使う。
    フォントのファイルがない環境では，全角 1 字＝1 em・半角 1 字＝0.6 em で見積もる。"""
    try:
        from PIL import ImageFont
        name = "meiryob.ttc" if bold else "meiryo.ttc"
        f = ImageFont.truetype(os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "Fonts", name), 1000)
        return f.getlength(text) / 1000 * size
    except OSError:
        return sum(1.0 if ord(c) > 0x2000 else 0.6 for c in text) * size


def _bul(slide, x, y, w, items, size=15, gap=8, color=INK, emph=BLUE, mark="▪ "):
    """items: 文字列 or (level, 文字列)。level 1 は字下げした小さめの行。
    level 0 の文字列の「\n」は同じ項目の中の改行で，2 行目以降は行頭記号の後ろ（「**ILS**：」のような
    見出しで始まる項目では，見出しの後ろ）にそろえる。"""
    tb, tf = add_textbox(slide, x, y, w, 0.5)
    for i, it in enumerate(items):
        lv, t = it if isinstance(it, tuple) else (0, it)
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.line_spacing = 1.08
        p.space_before = Pt(0 if i == 0 else (gap if lv == 0 else 3))
        if lv == 0:
            _run(p, mark, size * 0.8, color=BLUE, bold=True)
            lines = t.split("\n")
            if len(lines) > 1:   # ぶら下げ：行頭記号（文字の 0.8 倍の太字）と見出しの幅だけ 2 行目以降を下げる
                hang = _text_pt(mark, size * 0.8, bold=True)
                head = _re.match(r"\*\*(.+?)\*\*：", lines[0])
                if head:
                    hang += _text_pt(head.group(1), size, bold=True) + _text_pt("：", size)
                hang = int(Pt(hang))
                pPr = p._p.get_or_add_pPr()
                pPr.set('marL', str(hang))
                pPr.set('indent', str(-hang))
            for j, line in enumerate(lines):
                if j:
                    p.add_line_break()
                _add_runs(p, line, size, color=color, emph=emph)
        else:
            _run(p, "　－ ", size * 0.85, color=MUTE)
            _add_runs(p, t, size * 0.9, color=color, emph=emph)
    return tb


def slide_frame(title, key, section):
    """key はスライドの主張の 1 文。画面には出さず，ノートに置いて話すときの台本にする
    （画面では本文や結果の読み取りが同じことを言う）。"""
    slide = prs.slides.add_slide(BLANK)
    _title_zone(slide, title)
    _footer(slide, section)
    if key:
        slide.notes_slide.notes_text_frame.text = _plain(key)
    return slide


def pic(slide, name, x, y, w=None, h=None):
    path = asset(name)
    iw, ih = _img_size(path)
    if w is None:
        w = h * iw / ih
    if h is None:
        h = w * ih / iw
    slide.shapes.add_picture(path, Inches(x), Inches(y), Inches(w), Inches(h))
    return w, h


# ---------------------------------------------------------------- 表紙

def title_slide_talk():
    slide = prs.slides.add_slide(BLANK)
    _rect(slide, 0, 0, SW, SH, COVER)
    _hline(slide, 0.95, 2.05, 2.6, color=BLUESOFT, weight=2.0)
    # 題名は上下 2 本の線の間で上下中央に置く
    tb, tf = add_textbox(slide, 0.95, 2.05, SW - 1.9, 4.75 - 2.05, anchor=MSO_ANCHOR.MIDDLE)
    for i, t in enumerate(["処理順序の変更量を考慮した再スケジューリングにおける",
                           "path relinking の適用と探索構造の比較評価"]):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.line_spacing = 1.15
        _run(p, t, 30, color=WHITE, bold=True)
    _hline(slide, 0.97, 4.75, 3.4, color=BLUESOFT, weight=1.0)
    tb, tf = add_textbox(slide, 0.97, 4.9, SW - 1.9, 1.6)
    rows = [("早稲田大学", 15, LTBLUE, False),
            ("○鬼頭 拓海，谷水 義隆，渡邉 るりこ", 19, WHITE, True),
            ("日本経営工学会 2026年秋季大会　2026年10月", 12.5, EN_BLUE, False)]
    for i, (txt, sz, col, bold) in enumerate(rows):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.space_before = Pt(6 if i else 0)
        _run(p, txt, sz, color=col, bold=bold)
    _page["n"] = 1   # 表紙を 1 枚目として数え，以降はスライド番号＝通し番号


# ---------------------------------------------------------------- 2. 背景

JOBC = {"J1": RGBColor(0x3E, 0x61, 0x8C), "J2": RGBColor(0xD9, 0x8E, 0x45),
        "J3": RGBColor(0x6F, 0x9E, 0x7C)}
DELAY_FILL = RGBColor(0xF6, 0xD5, 0xD1)
# 3 ジョブ × 3 機械の模式例（経路 J1: M1→M2→M3，J2: M2→M3→M1，J3: M3→M1→M2）。
# M2 の J1 が 3 だけ遅延し，時刻 9 に終わる＝再スケジューリング時刻 t_r = 9。
# t_r より前に開始した作業は凍結し，未着手の 4 作業（M2 の J3，M3 の J1・J2，M1 の J2）を決め直す。
#   案A：S_p の処理順序を保ったまま後ろへずらす（総所要時間 19）
#   案B：M3 で J1 と J2 の順序を入れ替える（総所要時間 16）
GANTT_T, GANTT_TR = 20.0, 9
GANTT_SP = {"M1": [("J1", 0, 3), ("J3", 4, 3), ("J2", 12, 4)],
            "M2": [("J2", 0, 3), ("J1", 3, 3), ("J3", 7, 3)],
            "M3": [("J3", 0, 4), ("J1", 6, 3), ("J2", 9, 3)]}
GANTT_A = {"M1": [("J1", 0, 3), ("J3", 4, 3), ("J2", 15, 4)],
           "M2": [("J2", 0, 3), ("J1", 3, 6), ("J3", 9, 3)],
           "M3": [("J3", 0, 4), ("J1", 9, 3), ("J2", 12, 3)]}
GANTT_B = {"M1": [("J1", 0, 3), ("J3", 4, 3), ("J2", 12, 4)],
           "M2": [("J2", 0, 3), ("J1", 3, 6), ("J3", 9, 3)],
           "M3": [("J3", 0, 4), ("J2", 9, 3), ("J1", 12, 3)]}


def _alpha(shape, alpha):
    """塗りの不透明度（0〜1）を設定する。"""
    clr = shape.fill._xPr.find(qn('a:solidFill')).find(qn('a:srgbClr'))
    a = clr.makeelement(qn('a:alpha'), {'val': str(int(alpha * 100000))})
    clr.append(a)


def _vline(slide, x, y0, y1, color=INK, weight=1.5, dash=True):
    ln = slide.shapes.add_connector(1, Inches(x), Inches(y0), Inches(x), Inches(y1))
    ln.line.color.rgb = color
    ln.line.width = Pt(weight)
    if dash:
        from pptx.enum.dml import MSO_LINE
        ln.line.dash_style = MSO_LINE.ROUND_DOT if dash == "dot" else MSO_LINE.DASH
    ln.shadow.inherit = False
    return ln


def _gantt(slide, x, y, w, ops, T=GANTT_T, rh=0.36, rg=0.12, fs=10.5, delay=None, moved=(),
           label_w=0.42, tr=None, marks=(), note_fs=10):
    """機械ごとの作業バー。delay=(機械, ジョブ, 元の長さ) なら遅延分を赤で塗る。
    moved は順序が変わった (機械, ジョブ) で赤枠で囲む。tr を与えると，その時刻に破線を引き，
    それより前（開始済み＝凍結）を白く薄める。marks=[(時刻, ラベル)] は灰の点線で時刻を示し，
    ラベルは note_fs の大きさで線の上に置く。"""
    gx, gw = x + label_w, w - label_w
    for r, (m, lst) in enumerate(ops.items()):
        yy = y + r * (rh + rg)
        _text(slide, x, yy, label_w, rh, m, fs, color=MUTE, anchor=MSO_ANCHOR.MIDDLE)
        for j, st, du in lst:
            bx = gx + gw * st / T
            if delay and (m, j) == delay[:2]:
                base = delay[2]
                _boxtext(slide, bx + 0.015, yy, gw * base / T - 0.03, rh, [(j, fs, WHITE, True)],
                         fill=JOBC[j], radius=0.08)
                _rrect(slide, bx + gw * base / T + 0.015, yy, gw * (du - base) / T - 0.03, rh,
                       DELAY_FILL, line=RED, radius=0.08, lw=1.2)
                continue
            line = RED if (m, j) in moved else None
            sp = _boxtext(slide, bx + 0.015, yy, gw * du / T - 0.03, rh, [(j, fs, WHITE, True)],
                          fill=JOBC[j], line=line, radius=0.08)
            if line is not None:
                sp.line.width = Pt(2.4)
    bottom = y + 3 * (rh + rg) - rg
    _hline(slide, gx, bottom + 0.04, gw, color=HAIR, weight=1.0)
    if tr is not None:
        xt = gx + gw * tr / T
        veil = _rect(slide, gx - 0.02, y - 0.04, xt - gx + 0.02, bottom - y + 0.08, WHITE)
        _alpha(veil, 0.55)
        _vline(slide, xt, y - 0.12, bottom + 0.08, color=INK, weight=1.6)
    for t, lab in marks:
        xm = gx + gw * t / T
        _vline(slide, xm, y - 0.12, bottom + 0.08, color=MUTE, weight=1.4, dash="dot")
        if lab:
            _text(slide, xm - 0.3, y - 0.4, 0.6, 0.3, lab, note_fs, color=MUTE,
                  align=PP_ALIGN.CENTER)
    return bottom + 0.05 - y


def s_intro():
    s = slide_frame("外乱と再スケジューリング",
                    "外乱が起きたら，事前に作成した初期スケジュール $S_p$ を修正する（予測リアクティブ再スケジューリング）",
                    "1. 研究背景と仮説")
    # 横フロー．主経路（外乱 → 動的 → 予測リアクティブ＝本研究）を上段に一直線に置き，
    # 各分岐で採らない側（静的／完全リアクティブ）は下段に落とす．根（外乱）は両段にまたがる
    y1 = TOP + 0.25                          # 上段の箱の上辺に「本研究」の札が乗るぶんだけ下げる
    rg = 0.5                                 # 段間
    h1 = (BOT - y1 - rg) * 0.53              # 上段・下段の高さ（下段の下端を BOT に揃える）
    h2 = BOT - y1 - rg - h1
    y2 = y1 + h1 + rg
    c1, c2 = y1 + h1 / 2, y2 + h2 / 2        # 各段の中心
    # 幅は各列の最長の行（区切りのよい所で改行した 1 行）が収まる分だけ取り，残りを外乱の箱に回す
    gap, w1, w2 = 0.28, 3.25, 4.8
    w0 = CW - 2 * gap - w1 - w2
    x0 = ML
    x1 = x0 + w0 + gap
    x2 = x1 + w1 + gap

    def node(x, y, w, h, head, body, kind):
        """kind: root（前提）／main（採る側）／alt（採らない側）"""
        if kind == "alt":
            _rrect(s, x, y, w, h, WHITE, line=CARDLINE, radius=0.06, lw=1.1)
            hc, bc = MUTE, MUTE
        else:
            _rrect(s, x, y, w, h, FILL, line=BLUE if kind == "main" else CARDLINE,
                   radius=0.06, lw=1.8 if kind == "main" else 1.1)
            hc, bc = BLUE, INK
        tb, tf = add_textbox(s, x + 0.2, y, w - 0.4, h, anchor=MSO_ANCHOR.MIDDLE)
        p = tf.paragraphs[0]
        _run(p, head, FS_HEAD, color=hc, bold=True)
        # 「\n」は段落内の改行（箱の幅で語の途中から折り返さないよう，区切りのよい所で改行する）
        q = tf.add_paragraph()
        q.space_before = Pt(6)
        q.line_spacing = 1.1
        for j, line in enumerate(body.split("\n")):
            if j:
                q.add_line_break()
            _add_runs(q, line, FS_BODY, color=bc, emph=bc)

    # 外乱の箱は 2 段の高さいっぱいには伸ばさず，両段の矢印の高さ（c1〜c2）を少し上下に越える分だけ取る
    node(x0, c1 - 0.6, w0, c2 - c1 + 1.2, "外乱",
         "製造現場では，機械故障や\n作業遅延などにより，\nスケジュール通りに作業を\n実行できないことがある",
         "root")
    node(x1, y1, w1, h1, "動的アプローチ", "外乱の発生後に，\nスケジュールを修正する", "main")
    node(x1, y2, w1, h2, "静的アプローチ",
         "外乱に対してロバストな\nスケジュールを\n事前に作成する", "alt")
    node(x2, y2, w2, h2, "完全リアクティブ",
         "事前のスケジュールを持たず，\nディスパッチングルールなどで\n作業をその場で決める", "alt")
    # 予測リアクティブ（本研究）．S_p の存在が以後のすべての前提
    # 本研究の箱は淡い青の地に太い紺の枠（濃い塗りは重心を上に寄せるので使わない）
    _rrect(s, x2, y1, w2, h1, FILL, line=BLUE, radius=0.06, lw=2.8)
    _boxtext(s, x2 + w2 - 1.3, y1 - 0.19, 1.1, 0.38, [("本研究", FS_NOTE, WHITE, True)],
             fill=BLUESOFT, radius=0.4)
    tb, tf = add_textbox(s, x2 + 0.2, y1, w2 - 0.4, h1, anchor=MSO_ANCHOR.MIDDLE)
    p = tf.paragraphs[0]
    _run(p, "予測リアクティブ再スケジューリング", FS_HEAD, color=BLUE, bold=True)
    q = tf.add_paragraph()
    q.space_before = Pt(6)
    q.line_spacing = 1.1
    for j, line in enumerate(["外乱が起きたら，外乱前の",
                              "スケジュール（**初期スケジュール $S_p$**）",
                              "を修正する"]):
        if j:
            q.add_line_break()
        _add_runs(q, line, FS_BODY, color=INK, emph=BLUE)
    # 矢印．採る側は青・太，採らない側は灰・細．動的 → 完全リアクティブは列間を下りる肘線
    e = 0.03
    _connector(s, x0 + w0 + e, c1, x1 - e, c1, color=BLUE, wt=2.4, kind=1)
    _connector(s, x0 + w0 + e, c2, x1 - e, c2, color=MUTE, wt=1.6, kind=1)
    # 完全リアクティブへの肘線は動的の箱の下寄りから出し，予測リアクティブへの矢印の先と重ねない
    _connector(s, x1 + w1 + e, y1 + h1 * 0.82, x2 - e, c2, color=MUTE, wt=1.6, kind=2)
    _connector(s, x1 + w1 + e, c1, x2 - e, c1, color=BLUE, wt=2.6, kind=1)


def s_background():
    s = slide_frame("再スケジューリングと安定性",
                    "スケジュールの変更は現場にコストを生むため，生産効率と安定性の二目的最適化として扱う",
                    "1. 研究背景と仮説")
    # 左：例（2 通りの修正を上下に）
    T = GANTT_T
    lx, lw_ = ML, 6.3
    tb, tf = add_textbox(s, lx, TOP, CW, 0.4, anchor=MSO_ANCHOR.MIDDLE)
    p = tf.paragraphs[0]
    _run(p, "例：", FS_IN, color=INK, bold=True)
    _add_runs(p, "**M2 の J1 が 3 遅延（赤）** → 時刻 $t_r$ 以降の処理順序を変更",
              FS_IN, color=INK, emph=RED)
    cases = [("$S_p$ の処理順序を保つ", GANTT_A, (), "総所要時間 19", "変更なし", 1),
             ("M3 の順序を入れ替える", GANTT_B, {("M3", "J1"), ("M3", "J2")},
              "総所要時間 16", "順序を変更", 0)]
    c0, ch, cg = TOP + 0.5, 2.5, 0.14
    for k, (head, ops, moved, ms, chg, good) in enumerate(cases):
        cy = c0 + k * (ch + cg)
        _rrect(s, lx, cy, lw_, ch, WHITE, line=CARDLINE, radius=0.04)
        _text(s, lx + 0.2, cy + 0.1, lw_ - 3.6, 0.42, head, FS_IN, color=INK, bold=True,
              anchor=MSO_ANCHOR.MIDDLE)
        # 良い側は青．順序を変えた案の「順序を変更」はコストの原因なので赤
        for i, txt in enumerate([ms, chg]):
            fill = BLUE if i == good else (RED if (i == 1 and moved) else FILL)
            hot = fill != FILL
            _boxtext(s, lx + lw_ - 3.38 + i * 1.66, cy + 0.1, 1.6, 0.42,
                     [(txt, FS_NOTE, WHITE if hot else INK, hot)], fill=fill, radius=0.3)
        g_x, g_w, g_y = lx + 0.15, lw_ - 0.3, cy + 1.0
        # 時刻 19 の点線の数字は下の案にだけ付ける（上の案は「総所要時間 19」の札があり，
        # 数字を付けると「再スケジューリングの対象」の文字に接してしまう）
        _gantt(s, g_x, g_y, g_w, ops, rh=0.4, rg=0.1, fs=FS_IN, delay=("M2", "J1", 3),
               moved=moved, label_w=0.5, tr=GANTT_TR, marks=[(19, "19" if k else "")],
               note_fs=FS_NOTE)
        xt = g_x + 0.5 + (g_w - 0.5) * GANTT_TR / T
        _text(s, xt - 0.3, g_y - 0.42, 0.6, 0.3, "$t_r$", FS_IN, color=INK, bold=True,
              align=PP_ALIGN.CENTER)
        if k == 0:
            _text(s, g_x + 0.5, g_y - 0.4, xt - g_x - 0.85, 0.3, "開始済み（凍結）", FS_NOTE,
                  color=MUTE, align=PP_ALIGN.RIGHT)
            _text(s, xt + 0.15, g_y - 0.4, 2.75, 0.3, "再スケジューリングの対象", FS_NOTE, color=MUTE)

    # 右：変更のコスト → 二目的最適化
    rx = lx + lw_ + 0.4
    rw = SW - ML - rx
    # 右の列は左のカード列と上下を揃える．二目的の箱は詰めて，コストの例 4 つに高さを回す
    top_r, bot_r = c0, c0 + 2 * ch + cg
    kh = 2.75
    # 左の「順序を変更」から右のコストへつなぐ
    _connector(s, lx + lw_ + 0.05, c0 + ch + cg + 0.31, rx - 0.05, top_r + kh / 2 + 0.2,
               color=RED, wt=2.0, kind=1)
    _rrect(s, rx, top_r, rw, kh, RGBColor(0xFB, 0xEE, 0xEC), line=RED, radius=0.05, lw=1.4)
    tb, tf = add_textbox(s, rx + 0.22, top_r + 0.12, rw - 0.44, 0.85, anchor=MSO_ANCHOR.MIDDLE)
    p = tf.paragraphs[0]
    p.line_spacing = 1.05
    _run(p, "処理順序の変更は，現場にコストを生む", FS_HEAD, color=RED, bold=True)
    q = tf.add_paragraph()
    q.line_spacing = 1.05
    _run(q, "（スケジュール変更コスト）", FS_IN, color=RED, bold=True)
    costs = ["段取り替え", "資材・治具の再手配", "作業者の再配置", "下流工程・外注先への\n予定変更の波及"]
    cw2 = (rw - 0.44 - 0.15) / 2
    for i, c in enumerate(costs):
        cx = rx + 0.22 + (i % 2) * (cw2 + 0.15)
        cy = top_r + 1.05 + (i // 2) * 0.82
        _boxtext(s, cx, cy, cw2, 0.72, [(c, FS_IN, INK, True)], fill=WHITE, line=CARDLINE,
                 radius=0.1)
    by = top_r + kh + 0.4
    bh = bot_r - by
    _connector(s, rx + rw / 2, top_r + kh + 0.05, rx + rw / 2, by - 0.05, color=BLUE, wt=3.0,
               kind=1)
    _rrect(s, rx, by, rw, bh, WHITE, line=BLUE, radius=0.05, lw=1.6)
    _text(s, rx, by + 0.06, rw, 0.45, "二目的最適化", FS_HEAD, color=BLUE, bold=True,
          align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
    # 箱の幅で語の途中から折り返さないよう，区切りのよい所で改行する
    objs = [("生産効率", "総所要時間を\n小さく"), ("安定性", "初期スケジュール\nからの変更を小さく")]
    ow = (rw - 0.44 - 0.45) / 2
    for i, (head, body) in enumerate(objs):
        ox = rx + 0.22 + i * (ow + 0.45)
        oy = by + 0.6
        oh = bh - 0.6 - 0.15
        _rrect(s, ox, oy, ow, oh, FILL, line=CARDLINE, radius=0.08)
        _boxtext(s, ox, oy, ow, 0.42, [(head, FS_IN, WHITE, True)], fill=BLUE, radius=0.12)
        _text(s, ox + 0.08, oy + 0.42, ow - 0.16, oh - 0.42, body, FS_IN, color=INK,
              align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE, ls=1.05)
        if i == 0:
            _text(s, ox + ow, oy, 0.45, oh, "⇄", 20, color=BLUE, bold=True,
                  align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)


# ---------------------------------------------------------------- 3. 特殊性

def _paras(tf, paras):
    """(文字列, 文字の大きさ, 段落前の間隔 pt) を順に段落にする。「\n」は段落内の改行。"""
    for i, (t, sz, sb) in enumerate(paras):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.line_spacing = 1.08
        p.space_before = Pt(sb)
        for j, line in enumerate(t.split("\n")):
            if j:
                p.add_line_break()
            _add_runs(p, line, sz, color=INK, emph=BLUE)


def s_focus():
    """着目点と評価指標（1 章）。高安定領域が問われる理由は変更コストであって S_p の品質ではないので，
    特殊性（S_p が探索前にある）とは別のスライドにする。高安定領域の品質をどう測るかを同じ図で示し，
    仮説（課題 2 への対応）はこの指標で判定する。1 章で渡すのは「何を測るか」だけで，P_θ・D = 0 の除外・
    重みの合併などの算出の細部は実験設定の表に置く。全域 HV の名前は図の中で付ける。"""
    s = slide_frame("本研究の着目点と評価指標：高安定 HV",
                    "スケジュール変更コストが無視できない場面では高安定領域の解が求められるため，"
                    "本研究は高安定領域に限った HV（高安定 HV）で手法を評価する",
                    "1. 研究背景と仮説")
    # 図は特殊性のスライドと同じ平面・同じ位置・同じ幅で置く
    w, h = pic(s, "concept_hv.png", ML - 0.05, TOP, w=5.8)
    tx = ML + 5.9
    tw = SW - ML - tx
    # 本文は図の上端の高さから上揃えで書き出す。「● 事実 → 本研究の選択」の 2 組を，組の中は詰め，
    # 組の間を空けて並べる。語の途中で折り返さないよう，区切りのよい所で改行する
    # 組の間には区切り線を引く
    tb, tf = add_textbox(s, tx, TOP + 0.1, tw, 1.6)
    _paras(tf, [
        ("● スケジュール変更コストが無視できない\n場面で求められるのは，**安定性の高い解**，\n"
         "すなわち Pareto フロント上の**高安定領域**の解", FS_BODY, 0),
        ("→ 本研究は，この**高安定領域の品質**に着目する", FS_BODY, 14),
    ])
    _divider(s, tx, TOP + 1.95, tw)
    tb, tf = add_textbox(s, tx, TOP + 2.3, tw, h - 2.3)
    _paras(tf, [
        ("● 本研究は，Pareto フロントの品質を\n**hypervolume（HV）**で測る．\n"
         "HV は非劣解が支配する領域（参照点まで）の\n面積で，大きいほど品質が高い", FS_BODY, 0),
        ("→ 高安定領域に限った HV を**高安定 HV** と呼び，\nこれで手法を評価する", FS_BODY, 14),
    ])


def s_special():
    """研究の出発点。着目点（高安定領域）と評価指標のスライドの後・先行研究の前に置き，S_p が探索前に得られるという特殊性が
    探索構造の優劣を左右するのではないか，という問いを立てる。どちらの構造が良いかは仮説 1 の役目なので
    ここでは言わない。★ 特殊性から → で問いを立て，問いで初めて使う「探索構造」（単点探索・多点探索）の
    定義と例をその下にカードで置く。
    図は着目点と評価指標のスライドと同じ平面・同じ配置にし，★（初期解 S_p）はこの図で初めて描く。"""
    s = slide_frame("本研究の出発点：再スケジューリングの特殊性",
                    "$S_p$ が探索前に得られているこの問題では，"
                    "この特性が探索構造の優劣を左右するのではないか",
                    "1. 研究背景と仮説")
    w, h = pic(s, "concept_sp.png", ML - 0.05, TOP, w=5.8)
    tx = ML + 5.9
    tw = SW - ML - tx
    # 右の列：本文（★・→）を上から書き出し，その下に「メタヒューリスティクスの探索構造」の見出しと
    # 単点・多点のカードを並べる。カードの下端は図の下端とほぼ揃う。
    # 語の途中で折り返さないよう，区切りのよい所で改行する
    tb, tf = add_textbox(s, tx, TOP + 0.1, tw, 2.6)
    # 特殊性（見出しの後で段落内改行）
    p = tf.paragraphs[0]
    p.line_spacing = 1.08
    for j, line in enumerate(["★ 再スケジューリングの特殊性：",
                              "初期解 $S_p$（処理順序を保って遅延を",
                              "吸収した解）は，**探索せずに得られる**．",
                              "変更量 0 で安定性は最良，$MS$ も外乱前に最適化済"]):
        if j:
            p.add_line_break()
        _add_runs(p, line, FS_BODY, color=INK, emph=BLUE)
    # 問い
    p = tf.add_paragraph()
    p.line_spacing = 1.08
    p.space_before = Pt(14)
    for j, line in enumerate(["→ この $S_p$ の存在が，**探索構造の優劣，特に**",
                              "**高安定 HV に大きな影響を与えるのではないか**"]):
        if j:
            p.add_line_break()
        _add_runs(p, line, FS_BODY, color=INK, emph=BLUE)
    ly = TOP + 2.8                # 見出し＋カードは本文の直後に続ける（問いとの間を空けすぎない）
    _divider(s, tx, ly - 0.25, tw)   # 問いと，探索構造の定義の間
    # 問いで使った「探索構造」の定義と例をカードで並べる。見出しの作りは探索構造のスライド（2 章）と同じで，
    # 色は図の単点＝橙・多点＝緑。多点探索の例は，本研究が扱う交叉を持つ GA 系に限る（先行研究の解法も
    # GA 系で，仮説 1 の機序も交叉による。ACO・PSO は「互いに組み替える」に当たらないので挙げない）。
    # ILS・Memetic は本研究の代表手法。単点・多点は Blum & Roli によるメタヒューリスティクスの分類なので，
    # 見出しでそれを明示する
    tb, tf = add_textbox(s, tx, ly, tw, 0.36, anchor=MSO_ANCHOR.MIDDLE)
    p = tf.paragraphs[0]
    _run(p, "メタヒューリスティクスの探索構造", FS_IN, color=INK)
    _run(p, "（Blum & Roli, 2003）", FS_NOTE, color=MUTE)
    # カードは列の幅いっぱいに 2 枚並べる（例の行が 1 行に収まるよう，間と内側の余白を詰める）
    cy, ch, cg = ly + 0.42, 1.55, 0.1
    cw = (tw - cg) / 2
    for k, (name, col, body, ex) in enumerate([
            # 本文はカードの幅で語の途中から折り返さないよう，区切りのよい所で改行する
            ("単点探索", ORANGE, "1 つの解を少しずつ\n変更しながら進む", "例：TS，SA，ILS"),
            ("多点探索", GREEN, "複数の解を保持し，\n互いに組み替えて進む", "例：GA，NSGA-II，Memetic")]):
        x = tx + k * (cw + cg)
        _rrect(s, x, cy, cw, ch, WHITE, line=CARDLINE, radius=0.05)
        _rect(s, x, cy, cw, 0.45, col)
        _text(s, x + 0.12, cy, cw - 0.24, 0.45, name, FS_HEAD, color=WHITE, bold=True,
              anchor=MSO_ANCHOR.MIDDLE)
        tb, tf = add_textbox(s, x + 0.05, cy + 0.52, cw - 0.1, ch - 0.58)
        for i, (t, sz) in enumerate([(line, FS_IN) for line in body.split("\n")] + [(ex, FS_NOTE)]):
            q = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
            q.line_spacing = 1.05
            _run(q, t, sz, color=INK)


# 先行研究：(著者, 内容)
RELATED = [
    ("安定性を目的関数に組み込むアプローチ",
     [("Wu ら (1993)", "単一機械で効率と安定性を評価基準に定式化"),
      ("Rangsaritratsamee ら (2004)", "JSP．GA＋局所探索"),
      ("Zhang ら (2013)", "JSP．GA＋タブーサーチ"),
      ("Fattahi & Fallahi (2010)", "FJSP．GA＋局所探索"),
      ("Valledor ら (2022)，Luo ら (2025)", "NSGA-II 系で Pareto 支配")],
     "安定性は解の評価（目的関数の一項）にのみ反映"),
    ("再スケジュールの対象範囲を限定するアプローチ",
     [("Bean ら (1991)", "復帰時点までを再スケジュール（match-up）"),
      ("Abumaizar & Svestka (1997)", "影響を受ける作業だけを再スケジュール"),
      ("Zakaria & Petrovic (2012)", "GA の染色体を区間に限定"),
      ("Sun ら (2026)", "再最適化の範囲を 4 段階に整理")],
     "探索空間の制限で安定性を確保する"),
]


def s_related():
    s = slide_frame("先行研究：安定性を確保する 2 つのアプローチ",
                    "安定性の確保は 2 つのアプローチに分かれるが，"
                    "どちらもメタヒューリスティクスによる解法は多点探索に集中している",
                    "1. 研究背景と仮説")
    gap = 0.3
    cw = (CW - gap) / 2
    # 単点探索・多点探索の定義は前の出発点のスライドにあるので，ここは下端の 1 文だけにする
    # 書き出しは TOP，下端の 1 文は本文の下限（CONTENT_BOTTOM）に揃える
    y, h = TOP, 5.0
    for k, (head, refs, common) in enumerate(RELATED):
        x = ML + k * (cw + gap)
        _rrect(s, x, y, cw, h, WHITE, line=CARDLINE, radius=0.04)
        _rect(s, x, y, cw, 0.58, BLUE)
        _text(s, x + 0.15, y, cw - 0.3, 0.58, head, FS_HEAD, color=WHITE, bold=True,
              anchor=MSO_ANCHOR.MIDDLE)
        tb, tf = add_textbox(s, x + 0.25, y + 0.8, cw - 0.5, 3.4)
        for i, (who, what) in enumerate(refs):
            p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
            p.space_before = Pt(0 if i == 0 else 9)
            p.line_spacing = 1.0
            _run(p, who, FS_IN, color=BLUE, bold=True)
            p2 = tf.add_paragraph()
            p2.line_spacing = 1.0
            _run(p2, "　" + what, FS_IN, color=INK)
        _boxtext(s, x + 0.25, y + h - 0.7, cw - 0.5, 0.52, [(common, FS_IN, BLUE, True)],
                 fill=FILL, radius=0.2)
    _text(s, ML, CONTENT_BOTTOM - 0.5, CW, 0.5,
          "どちらのアプローチでも，メタヒューリスティクスによる解法は GA を基軸とする**多点探索に集中**",
          FS_BODY, color=INK, align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE, emph=BLUE)


# ---------------------------------------------------------------- 5. 3 つの課題

def s_gaps():
    s = slide_frame("先行研究に残された 3 つの課題",
                    "先行研究には，探索構造の比較・高安定領域での評価・安定性の高い解を作り出す操作が欠けている",
                    "1. 研究背景と仮説")
    # 課題 3 は操作の欠如として述べる：既存は安定性を「評価」か「探索空間の制限」で扱い，
    # 安定性の高い解を作り出す操作は持たない。作り出す手段は S_p に近づけることに限らないので，
    # 本研究の演算子の仕組み（S_p に近づける）はここでは言わない。課題 1 や仮説 1 にも寄りかからない。
    # 課題 1 は，出発点のスライドの問い（S_p の存在が探索構造の優劣を左右するか）が
    # 確かめられていないことを言う。多点探索への集中は直前の先行研究のスライドにあるので言い直さない
    # 本文は行ごとに区切りのよい所で改行する（語の途中で折り返さないように）。行の高さは行数から決める
    rows = [
        ("探索構造の差が\n検証されていない",
         "$S_p$ の存在が単点探索と多点探索の優劣に与える影響は，\n分析されていない"),
        ("高安定領域の品質が\n評価されていない",
         "比較は少数の重みでのスカラー値や，Pareto フロント全体の\n"
         "HV で行われ，高安定領域で手法がどう異なるかは\n分析されていない"),
        ("安定性の高い解を\n作り出す操作がない",
         "**目的関数に組み込む**：安定性は解の評価にだけ入る\n"
         "**範囲を限定する**：安定性は探索空間の制限にだけ入る\n"
         "→ どちらも，安定性の高い解を作り出す操作を持たない"),
    ]
    y, rg = TOP + 0.05, 0.18
    hw = 4.0
    rhs = [0.325 * (body.count("\n") + 1) + 0.3 for _, body in rows]
    pad = (BOT - y - sum(rhs) - rg * (len(rows) - 1)) / len(rows)   # 下端を BOT に揃える
    for i, (head, body) in enumerate(rows):
        rh = rhs[i] + pad
        _rrect(s, ML, y, CW, rh, WHITE, line=CARDLINE, radius=0.05)
        _boxtext(s, ML + 0.22, y + (rh - 0.56) / 2, 0.56, 0.56, [(str(i + 1), FS_HEAD, WHITE, True)],
                 fill=BLUE, radius=0.5)
        _text(s, ML + 1.0, y + 0.08, hw - 0.9, rh - 0.16, head, FS_HEAD, color=BLUE, bold=True,
              anchor=MSO_ANCHOR.MIDDLE, ls=1.1)
        _rect(s, ML + hw + 0.2, y + 0.18, 0.015, rh - 0.36, HAIR)
        _text(s, ML + hw + 0.45, y + 0.06, CW - hw - 0.7, rh - 0.12, body, FS_BODY, color=INK,
              anchor=MSO_ANCHOR.MIDDLE, ls=1.08, emph=BLUE)
        y += rh + rg


# ---------------------------------------------------------------- 6. 仮説

def s_hypotheses():
    s = slide_frame("本研究の仮説",
                    "課題 1・3 に対して 2 つの仮説を立て，どちらも高安定 HV（課題 2 への対応）で判定する",
                    "1. 研究背景と仮説")
    gap = 0.2
    cw = (CW - gap) / 2
    # 機序と主張を別の文にする（「多点探索は…伴うため，単点探索のほうが」だと主題が宙に浮く）
    hyps = [("仮説 1", "課題 1",
             "交叉で 2 親の染色体を組み替える**多点探索**は，\n"
             "$S_p$ からの小さな変更では表せない飛躍を伴う．\n"
             "そのため，$S_p$ を初期解として少しずつ変更する\n"
             "**単点探索**のほうが，**高安定 HV が大きい**", "concept_h1.png"),
            ("仮説 2", "課題 3",
             "多点探索に，解を $S_p$ に近づける移動\n"
             "（**安定性誘導演算子**）を組み込めば，\n"
             "その経路上に変更量の小さい解が現れるため，\n"
             "**多点探索の高安定 HV は向上する**", "concept_h2.png")]
    # 図を大きく取るため，札は箱の上辺に重ね，箱は 4 行ぶんの高さにする。札の色は判定のスライドと同じ
    # 札は箱の上に置く（上辺に重ねると本文と詰まって見える）。色は 2 つとも同じ青
    for k, (tag, src, text, img) in enumerate(hyps):
        x = ML + k * (cw + gap)
        # 本文の箱を一番見せたいので，札は小さく詰め，その分を本文の箱のゆとりに回す
        _boxtext(s, x + 0.02, TOP, 1.05, 0.36, [(tag, FS_IN, WHITE, True)], fill=BLUE, radius=0.4)
        by, bh = TOP + 0.42, 1.85
        _rrect(s, x, by, cw, bh, FILL, line=BLUE, radius=0.06, lw=1.5)
        _text(s, x + 0.2, by + 0.1, cw - 0.4, bh - 0.2, text, FS_BODY, color=INK, ls=1.12,
              anchor=MSO_ANCHOR.MIDDLE, emph=BLUE)
        ih = 6.62 - (by + bh + 0.08)
        iw, _h = _img_size(asset(img))
        w_img = min(cw - 0.04, ih * iw / _h)
        pic(s, img, x + (cw - w_img) / 2, by + bh + 0.08, w=w_img)
    # 研究の位置づけ（判定のスライドから移した。仮説を立てた場所で言うほうが据わりがよい）
    # 本文の下限（CONTENT_BOTTOM）は守る。図をこれ以上大きくするなら，この一文を外して口頭にする
    _text(s, ML, 6.68, CW, 0.32,
          "主眼は最先端手法との性能競争ではなく，条件を統制して探索構造と演算子の関係を切り分けること",
          FS_NOTE, color=MUTE, align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)


# ---------------------------------------------------------------- 7. 問題設定

def s_problem():
    s = slide_frame("問題設定",
                    "決定変数は，遅延後に最適化対象となる作業の，機械ごとの処理順序",
                    "2. 問題設定と実験計画")
    # 左の箇条書きは区切りのよい所で改行する（2 行目以降は行頭記号の後ろにそろう）
    _bul(s, ML, TOP + 0.1, 6.6, [
        "$n$ ジョブ・$m$ 機械のジョブショップ問題（JSP）に，\n単一作業の遅延（遅延量 $Δ$）が発生",
        "再スケジューリング時刻 $t_r$ より前に開始した作業は\n"
        "凍結し，以降の作業集合 $O_{opt}$ の**機械ごとの処理順序**\nを決定変数とする",
    ], size=FS_BODY, gap=12)
    # 問題・安定性の測り方・評価の仕方の間に区切り線を引く
    _divider(s, ML, 3.4, 6.6)
    _bul(s, ML, 3.64, 6.6, [
        "安定性は**処理順序の変更量**で測る（順序偏差 $D$）",
    ], size=FS_BODY)
    # 左の段も右の枠と同じく下端（BOT）まで使う
    render_equation(s, r"D(S_p,S_q)=\sum_{(i,j)\in O_{opt}}\left|r^p_{i,j}-r^q_{i,j}\right|",
                    ML, 4.2, 6.6, number="(1)", name="talk_D", h=0.75)
    _text(s, ML + 0.3, 5.05, 6.3, 0.32, "$r_{i,j}$：機械 $i$ におけるジョブ $j$ の処理順位", FS_NOTE,
          color=MUTE)
    _divider(s, ML, BOT - 1.35, 6.6)
    _bul(s, ML, BOT - 1.15, 6.6, [
        "二目的（$MS$, $D$）は重み付き和 $F$ にまとめ，重み $λ$ を\n"
        "生産効率重視から安定性重視まで掃引し，得られた\n全非劣解で評価する",
    ], size=FS_BODY)
    # 右：順序偏差の例（上端は左の箇条書きと揃える）
    rx, rw = 7.75, 4.95
    _rrect(s, rx, TOP, rw, BOT - TOP, RGBColor(0xF7, 0xFA, 0xFC), line=CARDLINE, radius=0.04)
    _text(s, rx + 0.25, TOP + 0.12, rw - 0.5, 0.42, "例：ある機械の処理順序", FS_HEAD, color=BLUE,
          bold=True)
    jc = {"J1": RGBColor(0x3E, 0x61, 0x8C), "J2": RGBColor(0xD9, 0x8E, 0x45),
          "J3": RGBColor(0x6F, 0x9E, 0x7C), "J4": RGBColor(0x8A, 0x86, 0xA8)}
    # 1 つ離れた位置の入れ替え（2 番目と 4 番目）にして，D が入れ替えた数でなく順位の差の和で
    # あることが見えるようにする。S_p は演算子のスライド（13）の誘導解と同じ並びにそろえる
    seqs = [("$S_p$", ["J1", "J2", "J3", "J4"], TOP + 1.35), ("$S_q$", ["J1", "J4", "J3", "J2"], TOP + 3.0)]
    bw, bh, bx0 = 0.82, 0.6, rx + 1.0
    pos = {}
    for lab, seq, yy in seqs:
        _text(s, rx + 0.25, yy, 0.7, bh, lab, FS_HEAD, color=INK, bold=True, anchor=MSO_ANCHOR.MIDDLE)
        for r, j in enumerate(seq):
            bx = bx0 + r * (bw + 0.12)
            _boxtext(s, bx, yy, bw, bh, [(j, FS_IN, WHITE, True)], fill=jc[j], radius=0.1)
            _text(s, bx - 0.05, yy - 0.37 if lab == "$S_p$" else yy + bh + 0.04, bw + 0.1, 0.32,
                  f"{r + 1}番目", FS_NOTE, color=MUTE, align=PP_ALIGN.CENTER)
            pos[(lab, j)] = (bx + bw / 2, yy, yy + bh)
    for j in ["J1", "J2", "J3", "J4"]:
        x1, _a, y1 = pos[("$S_p$", j)]
        x2, y2, _b = pos[("$S_q$", j)]
        moved = x1 != x2
        _connector(s, x1, y1 + 0.02, x2, y2 - 0.02, color=RED if moved else HAIR,
                   wt=2.0 if moved else 1.2, arrow=moved, kind=1)
    _text(s, rx + 0.25, TOP + 4.25, rw - 0.5, 0.5, "$D$ = |2−4| ＋ |4−2| = **4**", 20, color=INK,
          align=PP_ALIGN.CENTER, emph=RED)
    _text(s, rx + 0.25, TOP + 4.9, rw - 0.5, 0.4, "初期解 $S_p$ は $D$ = 0", FS_IN, color=MUTE,
          align=PP_ALIGN.CENTER)


# ---------------------------------------------------------------- 8. 比較する探索構造

SHARED_FILL = RGBColor(0xDD, 0xE6, 0xEF)   # 揃えた局所探索の箱（FILL より少し濃い）
TINT = {ORANGE: RGBColor(0xFB, 0xE4, 0xCC),  # 解を作る操作の箱の地色（ベタ塗りより薄く）
        GREEN: RGBColor(0xD7, 0xE9, 0xDC)}


def _flow(slide, x, y, w, steps, bh=0.52, gap=0.3, loop_label=None, loop_from=None, loop_to=None,
          fs=12.5, label_w=1.3, label_fs=11):
    """縦に並ぶ処理ボックスを矢印でつなぐ。steps: (文字列, 強調色 or None)。
    loop_from/loop_to の段のあいだに右側の戻り矢印を描き，loop_label を label_fs の大きさで添える。"""
    ys = []
    for i, (t, accent) in enumerate(steps):
        yy = y + i * (bh + gap)
        ys.append(yy)
        if accent == "shared":
            _boxtext(slide, x, yy, w, bh, [(t, fs, INK, False)], fill=SHARED_FILL, line=CARDLINE,
                     radius=0.12)
        elif accent is not None:
            sp = _boxtext(slide, x, yy, w, bh, [(t, fs, accent, True)],
                          fill=TINT.get(accent, FILL), line=accent, radius=0.12)
            sp.line.width = Pt(1.6)
        else:
            _boxtext(slide, x, yy, w, bh, [(t, fs, INK, False)], fill=WHITE, line=CARDLINE,
                     radius=0.12)
        if i:
            _connector(slide, x + w / 2, yy - gap + 0.02, x + w / 2, yy - 0.02, color=MUTE,
                       wt=1.4, kind=1)
    if loop_from is not None:
        rx = x + w + 0.28
        y1 = ys[loop_from] + bh / 2
        y0 = ys[loop_to] + bh / 2
        _connector(slide, x + w + 0.02, y1, rx, y1, color=MUTE, wt=1.4, arrow=False, kind=1)
        _connector(slide, rx, y1, rx, y0, color=MUTE, wt=1.4, arrow=False, kind=1)
        _connector(slide, rx, y0, x + w + 0.04, y0, color=MUTE, wt=1.4, kind=1)
        if loop_label:
            _text(slide, rx + 0.08, (y0 + y1) / 2 - 0.2, label_w, 0.4, loop_label, label_fs,
                  color=MUTE, anchor=MSO_ANCHOR.MIDDLE)
    return ys[-1] + bh - y


def s_structures():
    s = slide_frame("比較する探索構造：単点探索と多点探索",
                    "局所探索などの条件を揃え，解の作り方（少しずつ変更するか，交叉で組み替えるか）を比較する",
                    "2. 問題設定と実験計画")
    # 2 つの探索構造を枠で囲んで横に並べ，それぞれ手順を縦の流れ図で描く。
    # 色付きの箱＝解を作る操作（ここが 2 構造の差），薄い青＝揃えた局所探索。
    # 仮説 1 の図は仮説のスライドにあるので置かず，その幅を流れ図に回す。選定理由は流れ図の下に注記の大きさで添える
    top, fh_ = TOP, 4.82
    gap = 0.3
    cw = (CW - gap) / 2
    cols = [
        (ORANGE, "単点探索（ILS）",
         [("初期解：$S_p$", None), ("局所探索：N5 近傍", "shared"),
          ("摂動：数手の insert 移動", ORANGE),
          ("局所探索：N5 近傍", "shared"), ("受理判定：$F$ が改善すれば更新", None)],
         2, 4, "反復",
         "選定理由：局所探索と摂動が別の段にあり，\n演算子を摂動に組み込める"),
        (GREEN, "多点探索（Memetic）",
         [("初期集団：$S_p$＋無作為な個体", None), ("選択", None),
          ("交叉：2 親を組み替え", GREEN),
          ("突然変異", None), ("局所探索：N5 近傍（各個体）", "shared")],
         1, 4, "世代交代",
         "選定理由：先行研究で標準的な，\nGA に局所探索を組み込んだ構成"),
    ]
    for k, (col, head, steps, lf, lt, loop, why) in enumerate(cols):
        x = ML + k * (cw + gap)
        _rrect(s, x, top, cw, fh_, WHITE, line=CARDLINE, radius=0.05)
        _rect(s, x, top, cw, 0.52, col)
        tb, tf = add_textbox(s, x + 0.2, top, cw - 0.4, 0.52, anchor=MSO_ANCHOR.MIDDLE)
        _run(tf.paragraphs[0], head, FS_HEAD, color=WHITE, bold=True)
        # 流れ図は左に寄せて幅を詰め，枠の右上に構造の小さな模式図を置く
        _flow(s, x + 0.25, top + 0.78, 3.25, steps, bh=0.5, gap=0.18,
              loop_from=lt, loop_to=lf, loop_label=loop, fs=FS_IN, label_w=1.1, label_fs=FS_NOTE)
        pic(s, "struct_ils.png" if k == 0 else "struct_mem.png", x + cw - 1.8, top + 0.6, w=1.68)
        # 2 行目は「選定理由：」の後ろにそろえる（ぶら下げ）
        tb, tf = add_textbox(s, x + 0.25, top + 4.1, cw - 0.5, 0.62)
        p = tf.paragraphs[0]
        p.line_spacing = 1.05
        hang = int(Pt(_text_pt("選定理由：", FS_NOTE)))
        pPr = p._p.get_or_add_pPr()
        pPr.set('marL', str(hang))
        pPr.set('indent', str(-hang))
        for j, line in enumerate(why.split("\n")):
            if j:
                p.add_line_break()
            _add_runs(p, line, FS_NOTE, color=MUTE)
    _boxtext(s, ML, top + fh_ + 0.2, CW, 0.6,
             [("探索構造の差だけを比較するため，局所探索（N5 近傍）・目的関数 $F$・初期解 $S_p$ を両者で揃える",
               FS_BODY, WHITE, True)],
             fill=COVER, radius=0.12)


# ---------------------------------------------------------------- 9. 安定性誘導演算子

def s_operators():
    # 手法名 path relinking はここで出し，略称 PR もここで定義する（PR-k は図の札で定義する）
    s = slide_frame("提案手法：path relinking（PR）による安定性誘導演算子",
                    "PR の誘導解を初期スケジュール $S_p$ に固定し，現在解から $S_p$ へ swap でたどる経路上の解を返す操作にする",
                    "2. 問題設定と実験計画")
    # ---- 上：通常の PR との違い（提案はここ）。全幅の 2 行で，図と例はその下に並べる
    for i, (head, col, body) in enumerate([
            ("通常の PR", MUTE, "探索中に得た良い解どうしを結び，その間を探索する"),
            ("本研究", BLUE, "誘導解を $S_p$ に**固定** → 解を $S_p$ に近づける操作になる")]):
        # 札（塗りの箱）は提案側の「本研究」だけにし，通常の PR は同じ位置に文字だけで置く
        yy = TOP + i * 0.46
        if i:
            _boxtext(s, ML, yy + 0.04, 1.35, 0.4, [(head, FS_NOTE, WHITE, True)], fill=col, radius=0.3)
        else:
            _text(s, ML, yy, 1.35, 0.48, head, FS_NOTE, color=col, bold=True, align=PP_ALIGN.CENTER,
                  anchor=MSO_ANCHOR.MIDDLE)
        _text(s, ML + 1.5, yy, CW - 1.5, 0.48, body, FS_BODY, color=INK if i else MUTE,
              anchor=MSO_ANCHOR.MIDDLE, emph=BLUE)
    mid = TOP + 0.98
    # ---- 左：(D, MS) 平面での動き。PR・PR-k が返す解は図の中に注記する
    fw = 5.9
    _w, fh = pic(s, "concept_op.png", ML - 0.05, mid, w=fw)
    # ---- 右：ある機械の処理順序を S_p に近づける例（swap 1 回で D が減る）
    lx = ML + fw + 0.3
    lw_ = SW - ML - lx
    ch = 3.9
    _rrect(s, lx, mid, lw_, ch, RGBColor(0xF7, 0xFA, 0xFC), line=CARDLINE, radius=0.04)
    _text(s, lx + 0.2, mid + 0.08, lw_ - 0.4, 0.4, "例：ある機械の処理順序を $S_p$ に近づける", FS_IN,
          color=BLUE, bold=True, anchor=MSO_ANCHOR.MIDDLE)
    jc = {"J1": RGBColor(0x3E, 0x61, 0x8C), "J2": RGBColor(0xD9, 0x8E, 0x45),
          "J3": RGBColor(0x6F, 0x9E, 0x7C), "J4": RGBColor(0x8A, 0x86, 0xA8)}
    SP = ["J1", "J2", "J3", "J4"]
    bw, bh, bg = 0.66, 0.46, 0.08
    bx0 = lx + 1.75

    def seq_row(yy, lab, seq, dlab, lab_color=INK):
        _text(s, lx + 0.2, yy, 1.55, bh, lab, FS_IN, color=lab_color, bold=True,
              anchor=MSO_ANCHOR.MIDDLE)
        for c, j in enumerate(seq):
            bx = bx0 + c * (bw + bg)
            diff = j != SP[c]
            sp = _boxtext(s, bx, yy, bw, bh, [(j, FS_IN, WHITE, True)], fill=jc[j], radius=0.1,
                          line=RED if diff else None)
            if diff:
                sp.line.width = Pt(2.4)
        _text(s, bx0 + 4 * (bw + bg) + 0.06, yy, 0.95, bh, dlab, FS_IN, color=INK,
              anchor=MSO_ANCHOR.MIDDLE)

    # 目標（誘導解 S_p）を最初に示し，現在解と並べて差異を見せる
    _rrect(s, lx + 0.12, mid + 0.52, lw_ - 0.24, 0.58, RGBColor(0xCF, 0xDC, 0xEA), radius=0.1)
    seq_row(mid + 0.58, "$S_p$（誘導解）", SP, "$D$ = 0", lab_color=BLUE)
    path = [("現在解", ["J1", "J4", "J2", "J3"], "$D$ = 4", None),
            ("1 回目", ["J1", "J3", "J2", "J4"], "$D$ = 2", "swap：J4 と J3 を入れ替え"),
            ("2 回目", ["J1", "J2", "J3", "J4"], "$D$ = 0", "swap：J3 と J2 を入れ替え（$S_p$ に到達）")]
    y0r, gy = mid + 1.2, 0.34
    for r, (lab, seq, dlab, note) in enumerate(path):
        yy = y0r + r * (bh + gy)
        if note:
            # 1 行に収まるよう，ラベルの列の途中から書き出す
            _text(s, lx + 1.2, yy - gy, lw_ - 1.3, gy, "↓ " + note, FS_NOTE, color=RED,
                  anchor=MSO_ANCHOR.MIDDLE)
        seq_row(yy, lab, seq, dlab)
    _text(s, lx + 0.2, mid + ch - 0.62, lw_ - 0.4, 0.56,
          "赤枠：$S_p$ と順位が異なる作業（その数が $d$）\n"
          "1 回の swap で 1 作業が $S_p$ の順位に戻り，$D$ が減る", FS_NOTE, color=MUTE, ls=1.05,
          anchor=MSO_ANCHOR.MIDDLE)
    # ---- 下：組み込み方（全幅の 1 行）
    by_ = mid + ch + 0.2
    _rrect(s, ML, by_, CW, 0.52, FILL, line=CARDLINE, radius=0.12)
    tb, tf = add_textbox(s, ML + 0.25, by_, CW - 0.5, 0.52, anchor=MSO_ANCHOR.MIDDLE)
    p = tf.paragraphs[0]
    _run(p, "組み込み方　", FS_BODY, color=BLUE, bold=True)
    for who, col, how in [("ILS", ORANGE, "停滞時に摂動として発動　"),
                          ("Memetic", GREEN, "各個体に確率的に改善操作として発動")]:
        _run(p, who + "：", FS_BODY, color=col, bold=True)
        _run(p, how, FS_BODY, color=INK)


# ---------------------------------------------------------------- 10. 実験計画

def s_design():
    s = slide_frame("比較する手法と仮説の判定",
                    "仮説 1 は ILS と Memetic，仮説 2 は Memetic と演算子を組み込んだ Memetic の高安定 HV を比較して判定する",
                    "2. 問題設定と実験計画")
    # 比較する手法：探索構造 × 安定性誘導演算子。どの比較がどの仮説かを，表の上の枠と
    # 下の札の色で対応させる（仮説 1＝青・演算子なしの列，仮説 2＝紫・Memetic の行）。
    # 「主眼は性能競争でない」の一文は仮説のスライドへ移した
    H1, H2 = BLUE, VIOLET
    ty, trh, tw = TOP + 0.05, 0.6, CW
    colw = [3.0, 3.2, 2.9, 2.9]
    render_table(s, [["探索構造", "安定性誘導演算子なし", "PR-$k$ を組み込む", "PR を組み込む"],
                     ["単点探索（ILS）", "ILS", "ILS＋PR-$k$", "ILS＋PR"],
                     ["多点探索（Memetic）", "Memetic", "Memetic＋PR-$k$", "Memetic＋PR"]],
                 ML, ty, tw, col_w=colw, size=FS_BODY, align=['l', 'c', 'c', 'c'], row_h=trh)
    tbl = s.shapes[-1].table
    for r, col in [(1, ORANGE), (2, GREEN)]:
        for c in range(4):
            for p in tbl.cell(r, c).text_frame.paragraphs:
                for run in p.runs:
                    run.font.color.rgb = col
                    run.font.bold = True
    cx = [ML]
    for w_ in colw:
        cx.append(cx[-1] + w_ * tw / sum(colw))
    ry = [ty + i * trh for i in range(4)]

    def frame(c0, c1, r0, r1, color, dash=False, lw=2.2):
        sp = _rrect(s, cx[c0] + 0.05, ry[r0] + 0.05, cx[c1 + 1] - cx[c0] - 0.1,
                    ry[r1 + 1] - ry[r0] - 0.1, None, line=color, radius=0.08, lw=lw)
        if dash:
            from pptx.enum.dml import MSO_LINE
            sp.line.dash_style = MSO_LINE.DASH
        return sp

    frame(1, 3, 2, 2, H2)                      # 仮説 2：Memetic と Memetic＋演算子
    frame(1, 1, 1, 2, H1)                       # 仮説 1：ILS と Memetic
    # 枠の色は下の札と同じにし，表の中には文字を足さない（GA は実験設定のスライドで触れる）
    # 仮説ごとに，比較する手法と支持とみなす条件を書き下す（札の色は表の枠と同じ）
    hyps = [("仮説 1", H1, "単点探索のほうが，多点探索より高安定 HV が大きい",
             "ILS が Memetic を全シナリオで有意に上回れば支持"),
            ("仮説 2", H2, "安定性誘導演算子を組み込めば，多点探索の高安定 HV は向上する",
             "Memetic＋PR-$k$ と Memetic＋PR がそれぞれ Memetic を全シナリオで有意に上回れば支持")]
    tx = ML + 1.5
    txw = SW - ML - tx
    y = ty + 3 * trh + 0.5   # 表は見出しを含めて 3 行
    for tag, col, claim, rule in hyps:
        _boxtext(s, ML, y + 0.04, 1.3, 0.44, [(tag, FS_HEAD, WHITE, True)], fill=col, radius=0.3)
        _text(s, tx, y, txw, 0.52, "**" + claim + "**", FS_BODY, color=INK,
              anchor=MSO_ANCHOR.MIDDLE, emph=INK)
        # 判定は仮説の文の補足なので，図形の中の文字と同じ大きさにする（1 行に収める）
        tb, tf = add_textbox(s, tx, y + 0.54, txw, 0.4, anchor=MSO_ANCHOR.MIDDLE)
        p = tf.paragraphs[0]
        _run(p, "判定：", FS_IN, color=BLUE, bold=True)
        _add_runs(p, rule, FS_IN, color=INK)
        y += 1.3
        _divider(s, ML, y - 0.15, CW)
    # 検定（下端に揃える）
    jy = BOT - 0.66
    tb, tf = add_textbox(s, ML, jy, CW, 0.6)
    for k, (head, body) in enumerate([
            ("検定：", "両側 Mann–Whitney の $U$ 検定．全シナリオにわたる同じ比較は Holm 法で補正"),
            ("効果量：", "Cliff's $δ$．$|δ|$ = 1.0 は，一方の全試行が他方の全試行を上回る完全分離")]):
        q = tf.paragraphs[0] if k == 0 else tf.add_paragraph()
        q.space_before = Pt(0 if k == 0 else 2)
        _run(q, head, FS_NOTE, color=MUTE, bold=True)
        _add_runs(q, body, FS_NOTE, color=MUTE)


# ---------------------------------------------------------------- 10. 実験設定

def s_setup():
    """本編は結果のスライドを読むのに要る点だけにする（シナリオ名・試行数・高安定領域の境界）。
    設定の一覧は予備（実験設定の詳細），妥当性の数値は予備（頑健性と計算時間・内的妥当性）に置く。"""
    s = slide_frame("実験設定",
                    "5 つのベンチマーク問題に単一作業の遅延を与え，8 シナリオを構成",
                    "3. 数値実験")
    _bul(s, ML, TOP + 0.1, CW, [
        "**シナリオ**：5 問題で単一作業の遅延の発生時刻を変えて 8 つ（la36・ta21 は S・M・L）．各 10 試行",
        "**比較手法**：ILS・Memetic と，それぞれに PR-$k$／PR を組み込んだ 6 手法＋参考の GA",
        "**重み $λ$**：生産効率重視から安定性重視まで 10 点で探索し，全ての重みの解を合わせて評価",
        "**高安定領域**：全手法の非劣解のうち，順序偏差 $D$ が中央値より小さい範囲",
        "**妥当性**：高安定領域の境界・計算予算・パラメータを変えても，結論は変わらない",
    ], size=FS_BODY, gap=12)
    _text(s, ML, 4.1, CW, 0.36, "設定の詳細と妥当性の確認は予備スライドに掲載", FS_NOTE, color=MUTE)


def s_backup_setup():
    s = slide_frame("［予備］実験設定の詳細", None, "予備")
    rows = [
        ["項目", "設定"],
        ["比較手法", "ILS・Memetic と，それぞれに PR-$k$／PR を組み込んだ 6 手法＋参考の GA（局所探索なし）"],
        ["問題・外乱", "mt10・la21・la36・la40・ta21 に単一作業の完了遅延（$Δ$ = 60〜148，対象作業の処理時間の約 0.9〜1.5 倍）"],
        ["シナリオ", "8 つ：la36S・la36M・la36L，ta21S・ta21L，mt10，la21，la40\n"
                    "la36 と ta21 は同一問題・同一 $S_p$ のまま遅延の発生時刻を変え，対象作業が少ない順に S・M・L"],
        ["初期スケジュール $S_p$", "500 世代の GA で生成"],
        ["重み $λ$", "0, 0.1, …, 0.9 の 10 点（$λ$ = 1.0 は $D$ = 0 の自明解に縮退するため除外）"],
        ["計算予算", "ILS 系 3000 反復，GA・Memetic 系 500 世代"],
        ["高安定領域の境界", "$D$ < $P_{θ}$．$P_{θ}$ は全手法・全試行の非劣解の $D$ の $θ$ パーセンタイルで，"
                          "$θ$ = 50（中央値）"],
        ["HV の算出", "全ての重み $λ$ の解を合わせて算出．$D$ = 0 の解（$S_p$）は全手法が探索せずに得るため除く"],
        ["試行", "各条件 10 試行（全乱数シード固定）"],
    ]
    # 探索と演算子のパラメータ・計算環境は予備（手法のパラメータ）に，妥当性の数値は予備（頑健性と計算時間・
    # 内的妥当性）にあるので，ここには置かない
    render_table(s, rows, ML, 1.5, CW, col_w=[2.3, 9.79], size=13, align=['l', 'l'])


# ---------------------------------------------------------------- 11〜14. 結果

def s_front():
    s = slide_frame("高安定 HV の比較：代表 2 シナリオの非劣解",
                    "ta21S では，演算子なしの Memetic は高安定領域に非劣解を持たない",
                    "3. 数値実験")
    w = 12.0
    _w, fh = pic(s, "front_ta21.png", ML + (CW - w) / 2, TOP, w=w)
    _bul(s, ML, TOP + fh + 0.25, CW, [
        "(a) ta21S：Memetic は高安定領域に非劣解を持たない",
        "(b) ta21L：Memetic は高安定領域に入るが，$MS$ で ILS に劣る",
        "どちらのシナリオでも，演算子を組み込むと両構造の Pareto フロント（点線）は高安定領域で重なる",
    ], size=FS_BODY, gap=8)


def _highstab_guide_px(path):
    """highstab_all8.png の最初のパネルから，「なし」と「PR-k」の x，値 1 と値 0 の y を画素で拾う。
    読み方のガイドを図の再生成後も合わせるための自己較正。見つからなければ None。"""
    try:
        import numpy as np
        im = np.asarray(Image.open(path).convert("RGB")).astype(int)
        H, W, _ = im.shape

        def near(c, tol=40):
            return np.abs(im - np.array(c)).sum(axis=2) < tol

        x_hi, y_lo, y_hi = int(0.27 * W), int(0.12 * H), int(0.50 * H)

        def clusters(mask):
            ys, xs = np.nonzero(mask[y_lo:y_hi, :x_hi])
            if len(xs) == 0:
                return []
            cols = np.bincount(xs, minlength=x_hi)
            dense = np.nonzero(cols > 0.6 * cols.max())[0]
            out = []
            for g in np.split(dense, np.nonzero(np.diff(dense) > 5)[0] + 1):
                m = (xs >= g[0]) & (xs <= g[-1])
                out.append((float(xs[m].mean()), float(ys[m].mean()) + y_lo))
            return out

        org = clusters(near((0xE0, 0x6C, 0x00)))      # ILS の橙の丸
        grn = clusters(near((0x0F, 0x5C, 0x28)))      # Memetic の緑の四角
        xn, y1 = org[0]
        xk = [c for c in grn if c[0] > xn + 0.02 * W][0][0]
        gray = near((0x80, 0x80, 0x80), 60) | near((0x55, 0x60, 0x6B), 60)
        rows = np.nonzero(gray[:int(0.52 * H), :x_hi].sum(axis=1) > 0.15 * W)[0]
        y0 = float(rows.mean())
        return xn, xk, y1, y0, W
    except Exception:
        return None


def s_highstab():
    s = slide_frame("高安定 HV の比較：全 8 シナリオ（仮説の判定）",
                    "仮説 1・仮説 2 とも，全 8 シナリオで支持された",
                    "3. 数値実験")
    w = 12.0
    px0, py0 = ML + (CW - w) / 2, TOP
    _w, fh = pic(s, "highstab_all8.png", px0, py0, w=w)
    # 読み方のガイドを最初のパネル（la36S）に重ねる。位置は図の画素から自己較正する
    # （現在の図では 3000 px 幅で，なし x=331，PR-k x=557，値 1 → y=258，値 0 → y=443）。
    # 仮説 1＝演算子なしでの両構造の差（青の縦の両矢印），仮説 2＝Memetic の上昇（紫の矢印）
    g = _highstab_guide_px(asset("highstab_all8.png")) or (331, 557, 258, 443, 3000)
    k = w / g[4]
    xn, xk = px0 + g[0] * k, px0 + g[1] * k
    y1, y0 = py0 + g[2] * k, py0 + g[3] * k

    def tag(x, y, label, col):
        _boxtext(s, x - 0.36, y - 0.13, 0.72, 0.26, [(label, 11, col, True)], fill=WHITE,
                 line=col, radius=0.06)

    bx = xn - 0.13
    ln = _connector(s, bx, y1 + 0.08, bx, y0 - 0.08, color=BLUE, wt=1.5, kind=1)
    l_ = ln.line._get_or_add_ln()
    l_.append(l_.makeelement(qn('a:headEnd'), {'type': 'triangle', 'w': 'med', 'len': 'med'}))
    tag(bx + 0.08, (y1 + y0) / 2, "仮説 1", BLUE)
    _connector(s, xn + 0.21, y0 - 0.03, xk - 0.12, y1 + 0.27, color=VIOLET, wt=1.6, kind=1)
    tag(xk + 0.09, y0 - 0.13, "仮説 2", VIOLET)
    # 判定の結果（「支持」）は箇条書きの見出しで言う
    by = py0 + fh + 0.55   # 図と下端の間で釣り合う高さに置く
    _bul(s, ML, by, CW, [
        "**仮説 1 を支持**：演算子なしで ILS > Memetic（全シナリオで $p$ < 0.001，$|δ|$ = 1.0）",
    ], size=FS_BODY, gap=7, emph=BLUE)
    # 演算子でどこまで向上するか（両構造とも最良値の付近に達する）は，次のスライドで同じパネルを 4 つに絞って述べる
    _bul(s, ML, by + 0.55, CW, [
        "**仮説 2 を支持**：演算子を組み込んだ Memetic > Memetic（全シナリオで $p$ < 0.001，$|δ|$ = 1.0）",
    ], size=FS_BODY, gap=7, emph=VIOLET)
    # 「最良値」は縦軸の名前（最良値 = 1）としてこの図で初めて出るので，定義はここに 1 度だけ書く
    _text(s, ML, py0 + fh + 0.04, CW, 0.32, "※ 最良値：各シナリオでの，比較手法の中央値のうち最大の値", FS_NOTE,
          color=MUTE)


def s_band():
    """演算子の効果。前のスライドと同じパネルを 4 つ（ILS が演算子なしで最良値にある la36S・ta21S と，
    届いていない la36L・ta21L）に絞って大きく見せ，演算子を組み込むと両構造とも最良値の付近に達し，
    ILS は組み込む前からそこにある，ということだけを述べる。仮説のガイドは付けず，目印は la36L・ta21L の
    ILS の演算子なしの点を囲む丸だけ（図の中に描く。箇条書きの「○」と対応）。箇条書きは Memetic → ILS の順。
    数は公開 CSV と同じ統計（最良値を 1 とした中央値，PR-k と PR の両方を含む範囲，有意は補正なし）。"""
    s = slide_frame("高安定 HV の比較：演算子の効果",
                    "演算子を組み込むと両構造とも最良値の付近に達する．ILS は組み込む前から最良値の付近にある",
                    "3. 数値実験")
    w = 12.0
    _w, fh = pic(s, "highstab_4.png", ML + (CW - w) / 2, TOP, w=w)
    cy = TOP + fh
    _bul(s, ML, cy + 0.5, CW, [
        "**Memetic**：演算子を組み込むと，最良値の 0〜48% から 90〜100% まで大幅に向上",
        "**ILS**：演算子なしで，8 シナリオ中 6 つがすでに最良値の 98.5% 以上．\n"
        "最良値に届いていなかった la36L・ta21L（**○**）だけ，演算子で有意に向上（94% → 100%）",
    ], size=FS_BODY, gap=8)
    # 「→」は箇条書きの直後に，他のスライドの「→」と同じ間隔で続ける
    _text(s, ML, cy + 1.78, CW, 0.42,
          "→ 演算子の効果は，**探索構造だけでは最良値に届かない場合に，その差を埋める形で現れる**",
          FS_BODY, color=INK, emph=BLUE, anchor=MSO_ANCHOR.MIDDLE)


def s_extra():
    """追加の評価。仮説の判定は高安定 HV だけで行い，全域 HV と AUC は同じ試行を別の観点から見る
    追加の評価として，結果を示す直前に置く（実験を追加したわけではない）。全域 HV の面積の図は 1 章
    （評価指標のスライド）にあるので，ここでは全域 HV の推移の図 1 つで両方を示す：縦軸が全域 HV，
    推移の下の面積が AUC。AUC を重みごとに求めて全重みで平均することは予備スライド（評価指標の詳細）で扱う。"""
    s = slide_frame("追加の評価：全域 HV と AUC",
                    "仮説の判定とは別に，同じ試行を，フロント全体の品質と anytime 性能でも評価する",
                    "3. 数値実験")
    fw = 6.3
    _w, fh = pic(s, "metric_auc.png", ML, TOP, w=fw)
    tx = ML + fw + 0.5
    tb, tf = add_textbox(s, tx, TOP + 0.1, SW - ML - tx, fh - 0.1)
    # 縦軸・横軸・面積ごとに，太字の見出しの次の行で意味を言う（語の途中で折り返さないよう改行を明示）
    _paras(tf, [
        ("**縦軸：全域 HV**", FS_BODY, 0),
        ("Pareto フロント全体の品質", FS_BODY, 4),
        ("**横軸：計算時間（対数）**", FS_BODY, 22),
        ("立ち上がりの速さを重視する", FS_BODY, 4),
        ("**曲線の下の面積：AUC（anytime 性能）**", FS_BODY, 22),
        ("どの時点で計算を打ち切っても，\n品質の高いフロントが得られるか", FS_BODY, 4),
    ])


def s_anytime():
    s = slide_frame("全域 HV と AUC：代表 2 シナリオの推移",
                    "立ち上がりは ILS が速く，到達した全域 HV は演算子を組み込んだ Memetic が同等以上",
                    "3. 数値実験")
    w = 12.0
    _w, fh = pic(s, "anytime.png", ML + (CW - w) / 2, TOP, w=w)
    _bul(s, ML, TOP + fh + 0.25, CW, [
        "(a) ta21S：演算子を組み込んだ Memetic も ILS と同じ全域 HV に達するが，立ち上がりが遅い",
        "(b) la36L：Memetic 系は途中で ILS 系を追い越す．到達値は演算子を組み込んだ Memetic が最大",
    ], size=FS_BODY, gap=8)


def s_union_auc():
    s = slide_frame("全域 HV と AUC：全 8 シナリオ",
                    "最良の手法は評価指標によって異なる：全域 HV は Memetic＋演算子，AUC は ILS",
                    "3. 数値実験")
    tw = 5.75
    _text(s, ML, TOP, tw, 0.42, "最良に並ぶシナリオ数（/8）", FS_HEAD, color=BLUE, bold=True,
          anchor=MSO_ANCHOR.MIDDLE)
    trh = 0.57
    render_table(s, [
        ["手法", "全域HV", "高安定HV", "AUC"],
        ["ILS", "4", "6", "**6**"],
        ["ILS+PR-$k$", "4", "**8**", "**6**"],
        ["ILS+PR", "4", "**8**", "**6**"],
        ["GA", "0", "0", "0"],
        ["Memetic", "1", "0", "2"],
        ["Memetic+PR-$k$", "**7**", "**7**", "1"],
        ["Memetic+PR", "**7**", "**7**", "0"],
    ], ML, TOP + 0.5, tw, col_w=[2.3, 1.15, 1.35, 0.95], size=FS_BODY, row_h=trh)
    _text(s, ML, TOP + 0.5 + 8 * trh + 0.08, tw, 0.36, "最良に並ぶ：中央値が最良手法の中央値から 2% 以内",
          FS_NOTE, color=MUTE)
    # 右：指標ごとの読み取り（シナリオ数は表に任せ，語の途中で折り返さないよう行を明示）
    rx = ML + tw + 0.5
    rw = SW - ML - rx
    _text(s, rx, TOP, rw, 0.42, "全域 HV", FS_HEAD, color=BLUE, bold=True, anchor=MSO_ANCHOR.MIDDLE)
    _text(s, rx, TOP + 0.5, rw, 1.9,
          "Memetic＋演算子は 7 シナリオで最良に並ぶ\n"
          "ILS 系は 4 シナリオで最良に並ぶ\n"
          "→ **演算子は Memetic の全域 HV を**\v**押し上げるが，ILS ではほぼ変わらない**",
          FS_BODY, color=INK, ls=1.12, emph=BLUE)
    _divider(s, rx, TOP + 2.45, rw)   # 全域 HV と AUC の読み取りの間
    _text(s, rx, TOP + 2.65, rw, 0.42, "AUC", FS_HEAD, color=BLUE, bold=True, anchor=MSO_ANCHOR.MIDDLE)
    _text(s, rx, TOP + 3.15, rw, 1.9,
          "ILS 系は 6 シナリオで最良に並ぶ\n"
          "Memetic では演算子が各個体に発動して\n"
          "計算時間が増え，PR は AUC を\n"
          "5 シナリオで 7〜26% 下げる",
          FS_BODY, color=INK, ls=1.12)


# ---------------------------------------------------------------- 15. 結論

def s_conclusion():
    s = slide_frame("まとめ",
                    "高安定 HV は単点探索が上回り，多点探索も安定性誘導演算子を組み込めば同程度に達する",
                    "4. 結論")
    # 各項目は「結果」と「→ 現場にとっての意味」。語の途中で折り返さないよう，区切りのよい所で改行する
    items = [("仮説 1 を支持", "高安定 HV は**単点探索（ILS）**が多点探索（Memetic）を上回る\n"
                              "→ スケジュール変更コストが無視できない現場では，\v"
                              "多点探索より**単点探索**のほうが適する"),
             ("仮説 2 を支持", "安定性誘導演算子は，**多点探索の高安定 HV を単点探索と同程度まで向上**させる\n"
                              "→ 多点探索でも，安定性誘導演算子を組み込めば，\v"
                              "単点探索と同程度の質で安定性の高い解を得られる"),
             ("手法の選び方", "Pareto フロント全体の品質を重視する場合は **多点探索＋安定性誘導演算子**\n"
                              "短い計算時間で解を得る場合は **単点探索**")]
    # 見出しの列を詰めて本文の幅を取る。枠の高さは行数から決める
    bx = ML + 2.65
    bw = SW - ML - 0.2 - bx
    y, gap = TOP + 0.05, 0.17
    hhs = [0.36 * (b.count("\n") + b.count("\v") + 1) + 0.3 for _, b in items]
    pad = (BOT - y - sum(hhs) - gap * (len(items) - 1)) / len(items)   # 下端を BOT に揃える
    for i, (h, b) in enumerate(items):
        hh = hhs[i] + pad
        _rrect(s, ML, y, CW, hh, WHITE, line=CARDLINE, radius=0.06)
        _boxtext(s, ML + 0.2, y + (hh - 0.56) / 2, 0.56, 0.56, [(str(i + 1), FS_HEAD, WHITE, True)],
                 fill=BLUE, radius=0.5)
        _text(s, ML + 0.9, y + 0.08, 1.7, hh - 0.16, h, FS_HEAD, color=BLUE, bold=True,
              anchor=MSO_ANCHOR.MIDDLE)
        _text(s, bx, y + 0.06, bw, hh - 0.12, b, FS_BODY, color=INK, ls=1.12,
              anchor=MSO_ANCHOR.MIDDLE)
        y += hh + gap


def s_limits():
    s = slide_frame("限界と今後の課題",
                    "結論は本実験の条件の下で得たもので，条件を変えた再検証が今後の課題",
                    "4. 結論")
    x = ML + 0.2
    w = CW - 0.2
    _text(s, ML, TOP, CW, 0.42, "限界", FS_HEAD, color=BLUE, bold=True, anchor=MSO_ANCHOR.MIDDLE)
    _bul(s, x, TOP + 0.47, w, [
        "(i) $p$ が有意水準に近い比較は，$n$ = 10 では決着していない",
        "(ii) 安定性を開始時刻偏差で測る場合の妥当性は検証していない",
        "(iii) 外乱は単一作業の遅延に限る",
        "(iv) 初期スケジュール $S_p$ の最適化の程度が結果に与える影響は検証していない",
        "(v) 各探索構造の代表は 1 手法ずつである",
    ], size=FS_BODY, gap=4, mark="")
    _text(s, ML, 4.0, CW, 0.42, "今後の課題", FS_HEAD, color=BLUE, bold=True,
          anchor=MSO_ANCHOR.MIDDLE)
    _bul(s, x, 4.47, w, [
        "上記の条件を変えた再検証",
        "再スケジュールの範囲を限定するアプローチと安定性誘導演算子の統合",
        "安定性を交叉自体に持たせる方式の検証",
        "Pareto 支配に基づく手法（NSGA-II 等）での再検証",
    ], size=FS_BODY, gap=4)
    # 1 行に収まらないので，URL の前で改行する
    _text(s, ML, BOT - 0.62, CW, 0.62,
          "データ：各シナリオの定義と，試行ごとの値・検定は公開リポジトリに収録\v"
          "https://github.com/kitotakumi/stability_scheduling", FS_NOTE, color=MUTE,
          anchor=MSO_ANCHOR.BOTTOM)


# ---------------------------------------------------------------- 予備

def backup_divider():
    slide = prs.slides.add_slide(BLANK)
    _rect(slide, 0, 0, SW, SH, COVER)
    _text(slide, 1.15, 2.9, 10, 1.0, "予備スライド", 30, color=WHITE, bold=True)
    _page["n"] += 1


def s_backup_robust():
    s = slide_frame("［予備］頑健性と計算時間", None, "予備")
    _bul(s, ML, 1.5, CW, [
        "**高安定領域の境界**：$θ$ を 50 から 33〜67 に変えても，仮説 1・仮説 2 の結論は変わらない（全シナリオで完全分離）",
        "**計算時間**：壁時計は Memetic+PR が ILS の 4.8〜9.4 倍．ただし中央値の試行は，全手法とも予算の 7 割以内で最終の全域 HV の 99% に達する → 全域 HV の差は計算予算の不足によらない",
        "**PR か PR-$k$ か**：ILS ではどちらを組み込んでも 3 指標とも差は 2% 以内．Memetic では全域 HV・高安定 HV は PR がわずかに上回り（有意差は各 2 シナリオ），AUC は 1 回の計算が軽い PR-$k$ が 8 シナリオ中 6 つで有意に上回る",
        "**ILS で演算子により有意に向上した 2 シナリオ**：la36L・ta21L で $p$ < 0.05．Holm 補正後に残るのは ta21L のみ（la36L は $p_{adj}$ = 0.091）．中央値の向上は 6.3〜6.5%",
    ], size=15, gap=14)


def s_backup_scalar():
    s = slide_frame("［予備］重み付き和によるスカラー化", None, "予備")
    render_equation(s, r"F(S_q)=\lambda\,\hat{D}(S_p,S_q)+(1-\lambda)\,\widehat{MS}(S_q)",
                    ML, 1.45, CW, number="(2)", name="talk_F", h=0.6)
    _text(s, ML, 2.2, CW, 0.4,
          "$\\hat{D}$，$\\widehat{MS}$ は min-max 正規化値．$λ$ = 0, 0.1, …, 0.9 の 10 点で探索を独立に行う"
          "（$λ$ = 1.0 は $D$ = 0 の自明解に縮退するため除く）", 12.5, color=MUTE,
          align=PP_ALIGN.CENTER)
    _bul(s, ML, 2.85, CW, [
        "**運用との対応**：生産効率と安定性の優先度は，意思決定者が事前に重みとして与える．"
        "1 点ぶんの探索は，重みが定まった状態で再スケジューリングを 1 回行う実運用に対応する",
        "**先行研究との対応**：Rangsaritratsamee ら (2004)・Zhang ら (2013)・Fattahi & Fallahi (2010) も，"
        "生産効率と安定性を重みで 1 つのスカラーにまとめる",
        "**統制比較の共通基盤**：単点探索と多点探索に同じ目的 $F$ を共有させ，同じ安定性誘導演算子を"
        "変更なしで組み込める → 構造間で条件を揃えて比較できる",
        "**評価との対応**：各重みの探索を時間で追えば立ち上がりの速さ（AUC）が，"
        "全重みの解を合わせれば Pareto フロント全体の HV が，同じ実験から測れる",
        "**限界**：重み付き和は非凸な Pareto フロントの凹部に届かない．ただし全手法が同じスカラー化を共有し，"
        "探索中に訪問した凹部の解も記録するため，構造差の比較への影響は限定的．"
        "Pareto 支配に基づく手法（NSGA-II 等）での再検証は今後の課題",
    ], size=14.5, gap=14)


def s_backup_metrics():
    """本編では定義だけを渡す 3 指標の算出の細部。全域 HV・高安定 HV は全ての重みの解を合わせ，
    AUC は重みごとに求めて平均するという単位の違いはここで述べる（本編の全域 HV の推移の図の面積とは一致しない）。"""
    s = slide_frame("［予備］評価指標の詳細", None, "予備")
    rh = 0.52
    render_table(s, [
        ["項目", "全域 HV", "高安定 HV", "AUC"],
        ["対象の解", "全ての重み $λ$ の解を合わせる", "同左", "重み $λ$ ごと"],
        ["$D$ の範囲", "全域", "$D$ < $P_{θ}$（$θ$ = 50）", "全域"],
        ["値", "計算終了時の HV", "計算終了時の HV", "HV の推移を対数時間で平均し，\n全ての重みで平均"],
    ], ML, 1.5, CW, col_w=[2.0, 3.4, 3.0, 3.7], size=15, row_h=rh)
    _bul(s, ML, 1.5 + 4 * rh + 0.4, CW, [
        "**共通**：各シナリオで [0, 1]² に正規化し，参照点 (1.1, 1.1) で算出する．$D$ = 0 の解（$S_p$）は除く",
        "**$P_{θ}$**：全手法・全試行の非劣解の $D$ の $θ$ パーセンタイル",
        "**AUC**：時間窓は各重みで全手法共通",
        (1, "本編の全域 HV の推移の図は全ての重みの解を合わせた曲線なので，AUC はその面積とは一致しない"),
    ], size=15, gap=14)


def s_backup_why():
    s = slide_frame("［予備］なぜ ILS と Memetic か", None, "予備")
    gap = 0.4
    cw = (CW - gap) / 2
    cols = [
        ("単点探索の代表：ILS（反復局所探索）", ORANGE, [
            "摂動 → 局所探索 → 受理判定を反復し，1 つの解を改善する枠組み",
            "**局所探索と摂動が独立した段に分かれている**ため，摂動の強度（現在解から一度に"
            "どれだけ離すか）を明示的に設定でき，安定性誘導演算子も局所探索・受理判定を"
            "変えずに摂動の段へ組み込める",
            "他の単点探索は局所最適からの脱出を別の形で行う：SA は悪化解の確率的な受理，"
            "タブーサーチは再訪の禁止",
            "設定：初期解 $S_p$．insert 摂動は 2〜5 手を停滞度に応じて巡回．"
            "best の $F$ を厳密に改善したときのみ現在解を更新",
        ]),
        ("多点探索の代表：Memetic（GA＋局所探索）", GREEN, [
            "先行研究（Rangsaritratsamee ら 2004，Fattahi & Fallahi 2010）で用いられる，"
            "**GA に局所探索を組み込んだ標準的な構成**",
            "GA：operation-based 表現，GT 法でデコード，エリート保存付きトーナメント選択，"
            "PPX 交叉，逆位突然変異",
            "各個体に N5 局所探索を適用し，改善解を染色体に書き戻す Lamarckian 型．"
            "初期集団に $S_p$ を 1 個体含める",
            "局所探索をもたない素の GA は参考として比較する",
        ]),
    ]
    for k, (head, col, items) in enumerate(cols):
        x = ML + k * (cw + gap)
        _rrect(s, x, 1.45, cw, 5.45, WHITE, line=CARDLINE, radius=0.04)
        _rect(s, x, 1.45, cw, 0.55, col)
        _text(s, x + 0.2, 1.45, cw - 0.4, 0.55, head, 14.5, color=WHITE, bold=True,
              anchor=MSO_ANCHOR.MIDDLE)
        _bul(s, x + 0.2, 2.15, cw - 0.4, items, size=13, gap=10)


def s_backup_problem():
    s = slide_frame("［予備］問題設定の根拠", None, "予備")
    _bul(s, ML, 1.5, CW, [
        "**外乱を単一の作業遅延に限る理由**：作業遅延は作業集合を変えず，処理順序の再調整だけで対処できる．"
        "修正スケジュールを高安定領域に保つという前提が，最も素直に成り立つ外乱である",
        "**再スケジューリング時刻 $t_r$**：遅延の解消時刻．遅延の発生から $t_r$ までは，"
        "機械上の順序を $S_p$ から変えずに遅延を吸収して実行を続ける（right-shift）．"
        "この間に開始した作業はもう変更できないため凍結し，$t_r$ 以降の作業の機械ごとの処理順序だけを決定変数とする",
        "**安定性を順序偏差で測る理由**：変更量の測り方は，開始時刻偏差と順序偏差の二系統に大別される"
        "（Sun ら (2026) は前者の開始時刻の絶対偏差和を採る）．本研究は，探索で用いる操作"
        "（N5 近傍と安定性誘導演算子の swap 移動）がいずれも順列操作であること，"
        "および $MS$ との独立性が高いことから，順序偏差を採る",
    ], size=15, gap=16)


def s_backup_operator():
    s = slide_frame("［予備］安定性誘導演算子の設計", None, "予備")
    _bul(s, ML, 1.5, CW, [
        "**誘導解を $S_p$ に固定する理由**：(a) $S_p$ は順序偏差 $D$ を最小にする解なので，"
        "PR を「$D$ を最小にする解へ向かう移動」と一意に解釈できる．"
        "(b) $S_p$ へ向かう経路上には $D$ が小さく $MS$ の劣化も小さい解が現れると期待でき，"
        "それが高安定領域の非劣解になる",
        "**PR-$k$ で $k$ を変える理由**：$k$ が $d$ に等しければ $S_p$ まで戻り，小さければ現在解の近くに留まる．"
        "$k$ を変えて繰り返すことで，現在解と $S_p$ の間の広い範囲に解を得る",
        "**組み込み方**：両構造に「$S_p$ に近づける swap 移動」という核を共有させ，発動の契機と返し方は"
        "各構造の標準的な組み込み方に合わせる．ILS は停滞時に摂動として発動し，$k$ を 1 から $d$ まで "
        "1 ずつ増やして，上限に達するか best が改善したら 1 に戻す．Memetic は局所探索の後に個体ごとに"
        "確率的に適用し（$k$ は 1〜$d$ から一様に選ぶ），N5 で再最適化して無条件に集団へ戻す"
        "（採否はトーナメント選択に委ねる）",
        (1, "組み込み方が同じ ILS 系でも，最良値に届いていなかった la36L・ta21L だけで効果が現れる → 非対称は組み込み方の違いでは説明できない"),
        "**入れ替える作業の選び方**：実行可能な候補からランダムに選び，評価回数を $O(d)$ に抑える．"
        "全候補を評価する best 選択は $O(d^2)$ で，予備実験では全域 HV が約 +1% 増えるのに対し，"
        "最大で約 8 倍の時間を要したため採らない",
    ], size=14.5, gap=14)


def s_backup_control():
    s = slide_frame("［予備］対照実験とパラメータ感度", None, "予備")
    _bul(s, ML, 1.5, CW, [
        "**ランダム方向の対照実験**：Memetic に組み込んだ演算子による向上が，摂動一般の効果か，"
        "$S_p$ に近づける方向に固有の効果かを確かめるため，強度を揃えて方向だけをランダム化した対照を "
        "2 シナリオ（mt10・la36S）で実行した",
        (1, "ランダム方向は向上幅の 7 割前後しか再現せず，高安定 HV では演算子に完全分離で及ばない"),
        (1, "同じ高安定 HV に達するには，1.7〜2.4 倍の時間を要する"),
        "**パラメータ感度**：手法パラメータは標準値で固定した．挙動を定める 11 軸（ベースの探索構造側を含む）を "
        "2 シナリオで一要因ずつ上下に振ったが，全域 HV の変化は +0.5〜2% で，"
        "手法間の差（+7〜54%）より一桁小さい",
    ], size=15, gap=16)


def s_backup_params():
    s = slide_frame("［予備］手法のパラメータ", None, "予備")
    render_table(s, [
        ["項目", "設定"],
        ["GA・Memetic 系", "交叉率 0.85，突然変異率 0.1，集団 50（$S_p$ 以外の 49 個体は無作為生成）．operation-based 表現，GT 法でデコード"],
        ["ILS 系", "insert 摂動 2〜5 手を巡回．局所探索は N5 近傍（クリティカルブロック端の隣接 swap）"],
        ["演算子（Memetic 系）", "各個体に確率 0.3 で発動．PR は経路上の上位 3 中間解を局所探索して最良を返す"],
        ["演算子（ILS 系）", "無改善 400 反復で初回，以降 10 反復ごとに発動．PR は経路上の最良 1 解を返す"],
        ["PR-$k$ の深さ $k$", "1 から $S_p$ と処理順位が異なる作業数 $d$ までの範囲で発動ごとに変える"],
        ["記録", "選択の前に生成した全解から非劣解を保存（両構造で記録点を揃える）"],
        ["環境", "AMD Ryzen 5 7530U，Python 3.12（NumPy・DEAP・SciPy）"],
    ], ML, 1.5, CW, col_w=[2.6, 9.5], size=13.5, align=['l', 'l'])


HANG_MARKS = ("●", "→", "★", "※", "▪")


def _hang_markers(prs):
    """行頭記号（● → ★ ※ ▪）で始まる段落は，2 行目以降を記号の後ろにそろえる（ぶら下げ）。
    折り返しでも段落内の改行でも同じ位置になる。すでにぶら下げを設定した段落（_bul の見出し付きなど）は除く。"""
    def frames(slide):
        for sh in slide.shapes:
            if sh.has_text_frame:
                yield sh.text_frame
            if sh.has_table:
                for row in sh.table.rows:
                    for cell in row.cells:
                        yield cell.text_frame

    for slide in prs.slides:
        for tf in frames(slide):
            for p in tf.paragraphs:
                runs = p.runs
                if not runs or not runs[0].text or runs[0].text[0] not in HANG_MARKS:
                    continue
                pPr = p._p.get_or_add_pPr()
                if pPr.get('marL') is not None:
                    continue
                r0 = runs[0]
                mark = r0.text if r0.text.strip() == r0.text[0] else r0.text[0] + " "
                size = r0.font.size.pt if r0.font.size is not None else FS_BODY
                hang = int(Pt(_text_pt(mark, size, bold=bool(r0.font.bold))))
                pPr.set('marL', str(hang))
                pPr.set('indent', str(-hang))


# Meiryo は行の高さの中で字面が上に寄る（字面の中心が行の中心より文字の大きさの約 0.13 倍高い）ため，
# 上下中央揃えの文字は箱の中で上に寄って見える。上の余白を足して，その半分だけ文字を下げる
OPTICAL_SHIFT = 0.13


def _optical_center(prs):
    """上下中央揃えの文字（図形・テキストボックス・表のセル）を，字面が箱の中央に来るよう下げる。題名は除く。"""
    def max_size(tf):
        sizes = [r.font.size.pt for p in tf.paragraphs for r in p.runs if r.font.size is not None]
        return max(sizes) if sizes else None

    def shift(obj, tf):
        sz = max_size(tf)
        if sz:
            obj.margin_top = obj.margin_top + Pt(2 * OPTICAL_SHIFT * sz)

    for slide in prs.slides:
        for sh in slide.shapes:
            if sh.name == "title":
                continue
            if sh.has_text_frame and sh.text_frame.vertical_anchor == MSO_ANCHOR.MIDDLE:
                shift(sh.text_frame, sh.text_frame)
            if sh.has_table:
                for row in sh.table.rows:
                    for cell in row.cells:
                        if cell.vertical_anchor == MSO_ANCHOR.MIDDLE:
                            shift(cell, cell.text_frame)


# ================================================================ 本編
def build():
    title_slide_talk()
    chapter("1")
    s_intro()
    s_background()
    s_focus()     # 着目点と評価指標：高安定領域の解が求められる → 高安定 HV で評価する
    s_special()   # 研究の出発点：S_p が探索前にある → この特性が探索構造の優劣を左右するのではないか
    s_related()   # 先行研究：どちらのアプローチも解法は多点探索に集中
    s_gaps()
    s_hypotheses()
    chapter("2")
    s_problem()
    s_structures()
    s_operators()
    s_design()
    chapter("3")
    s_setup()
    s_front()
    s_highstab()
    s_band()
    s_extra()        # 追加の評価：全域 HV の推移の図 1 つで，縦軸（全域 HV）と面積（AUC）として示す
    s_anytime()
    s_union_auc()
    chapter("4")
    s_conclusion()
    s_limits()
    backup_divider()
    s_backup_setup()
    s_backup_problem()
    s_backup_scalar()
    s_backup_metrics()
    s_backup_why()
    s_backup_operator()
    s_backup_params()
    s_backup_robust()
    s_backup_control()
    _hang_markers(prs)
    _optical_center(prs)
    prs.save(OUT)
    print("saved:", OUT, " slides:", len(prs.slides._sldIdLst))


def export_pdf(pptx_path, pdf_path=None):
    """PowerPoint（COM）で pptx を PDF に書き出す。PowerShell 経由で呼ぶので pywin32 は不要。

    PowerPoint で描画するため，フォントや数式の見た目は pptx を開いたときと一致する。
    """
    import subprocess
    pdf_path = pdf_path or os.path.splitext(pptx_path)[0] + ".pdf"
    ps = (
        "$ErrorActionPreference='Stop';"
        "$app = New-Object -ComObject PowerPoint.Application;"
        f"$p = $app.Presentations.Open('{pptx_path}', $true, $false, $false);"
        f"$p.SaveAs('{pdf_path}', 32);"  # 32 = ppSaveAsPDF
        "$p.Close();"
        "if ($app.Presentations.Count -eq 0) { $app.Quit() }"
    )
    r = subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command", ps],
                       capture_output=True, text=True)
    if r.returncode != 0:
        print("PDF の書き出しに失敗（同名の PDF を開いていないか確認）:", r.stderr.strip())
    else:
        print("pdf:", pdf_path)


if __name__ == "__main__":
    import sys
    build()
    if "--no-pdf" not in sys.argv:
        export_pdf(OUT)
