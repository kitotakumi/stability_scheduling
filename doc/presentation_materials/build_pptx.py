# -*- coding: utf-8 -*-
"""学会発表（JIMA2026 秋・15 分）用 pptx を生成する（深い青テーマ）。

構成は仮説の提示までを JIMA 予稿、その後を APIEMS v2 原稿（実験計画 → 指標ごとの結果 → 結論）に合わせる。

デザイン方針:
  - テーマカラーは「深い青」の単色系。装飾を足さず余白で見せる。
  - 表紙・章扉は深い青の背景に白文字。本文は白背景。
  - 本文スライドは「タイトル → 単色の細い区切り線 → 1行キーメッセージ → コンテンツ」。
    キーメッセージは各スライドの“主張そのもの”を1文で置く。
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
KEY_TOP = 1.14
CONTENT_TOP_KEY = 1.80
CONTENT_TOP_NOKEY = 1.30
CONTENT_BOTTOM = 7.02

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

def _title_zone(slide, title, key):
    tb, tf = add_textbox(slide, ML, 0.34, CW, 0.66, anchor=MSO_ANCHOR.MIDDLE)
    _add_runs(tf.paragraphs[0], title, 24.5, color=BLUE, bold=True)
    _hline(slide, ML, LINE_Y, CW, color=BLUE, weight=1.25)
    if key:
        _rect(slide, ML, KEY_TOP + 0.02, 0.065, 0.34, BLUE)   # 短い縦アクセント
        tb, tf = add_textbox(slide, ML + 0.22, KEY_TOP, CW - 0.22, 0.42,
                             anchor=MSO_ANCHOR.MIDDLE)
        p = tf.paragraphs[0]; p.line_spacing = 1.0
        _add_runs(p, key, 15, color=BLUE, bold=True)


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
        _run(p, number, 13, color=INK)
    return hh + 0.12

# ================================================================ フロー

def content_slide(title, key, section, blocks):
    slide = prs.slides.add_slide(BLANK)
    _title_zone(slide, title, key)
    _footer(slide, section)
    y0 = CONTENT_TOP_KEY if key else CONTENT_TOP_NOKEY
    _render_blocks(slide, blocks, ML, y0, CW)
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


CHAPTERS = [("1", "研究背景と仮説", "外乱と再スケジューリング・先行研究と課題・仮説"),
            ("2", "問題設定と実験計画", "問題設定・評価指標・探索構造・安定性誘導演算子・仮説の判定"),
            ("3", "数値実験", "実験設定・高安定 HV・全域 HV と AUC"),
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
RED    = RGBColor(0xB0, 0x3A, 0x2E)
SAND   = RGBColor(0xF5, 0xEB, 0xD0)   # 高安定領域の帯
SANDLN = RGBColor(0x7A, 0x5C, 0x12)


def _text(slide, x, y, w, h, text, size, color=INK, bold=False, align=PP_ALIGN.LEFT,
          anchor=MSO_ANCHOR.TOP, ls=1.05, emph=None):
    tb, tf = add_textbox(slide, x, y, w, h, anchor=anchor)
    for i, line in enumerate(text.split("\n")):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        p.line_spacing = ls
        _add_runs(p, line, size, color=color, bold=bold, emph=emph or color)
    return tb


def _bul(slide, x, y, w, items, size=15, gap=8, color=INK, emph=BLUE, mark="▪ "):
    """items: 文字列 or (level, 文字列)。level 1 は字下げした小さめの行。"""
    tb, tf = add_textbox(slide, x, y, w, 0.5)
    for i, it in enumerate(items):
        lv, t = it if isinstance(it, tuple) else (0, it)
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.line_spacing = 1.08
        p.space_before = Pt(0 if i == 0 else (gap if lv == 0 else 3))
        if lv == 0:
            _run(p, mark, size * 0.8, color=BLUE, bold=True)
            _add_runs(p, t, size, color=color, emph=emph)
        else:
            _run(p, "　－ ", size * 0.85, color=MUTE)
            _add_runs(p, t, size * 0.9, color=color, emph=emph)
    return tb


def slide_frame(title, key, section):
    slide = prs.slides.add_slide(BLANK)
    _title_zone(slide, title, key)
    _footer(slide, section)
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
    tb, tf = add_textbox(slide, 0.95, 2.22, SW - 1.9, 1.9)
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
        ln.line.dash_style = MSO_LINE.DASH
    ln.shadow.inherit = False
    return ln


def _gantt(slide, x, y, w, ops, T=GANTT_T, rh=0.36, rg=0.12, fs=10.5, delay=None, moved=(),
           label_w=0.42, tr=None):
    """機械ごとの作業バー。delay=(機械, ジョブ, 元の長さ) なら遅延分を赤で塗る。
    moved は順序が変わった (機械, ジョブ) で赤枠で囲む。tr を与えると，その時刻に破線を引き，
    それより前（開始済み＝凍結）を白く薄める。"""
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
    return bottom + 0.05 - y


def s_intro():
    s = slide_frame("外乱と再スケジューリング",
                    "外乱が起きたら，事前に作成した初期スケジュール $S_p$ を修正する（予測リアクティブ再スケジューリング）",
                    "1. 研究背景と仮説")
    # 外乱
    by_, bh_ = 1.92, 0.72
    _rrect(s, ML, by_, CW, bh_, FILL, line=CARDLINE, radius=0.12)
    _text(s, ML + 0.3, by_, CW - 0.6, bh_,
          "製造現場では，機械故障や作業遅延などの**外乱**により，スケジュール通りに作業を実行できないことがある",
          16, color=INK, align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE, emph=BLUE)
    # 静的／動的。動的を広く取り，その下に完全リアクティブ／予測リアクティブの 2 分岐を置く
    cy, ch = 3.0, 1.2
    sw, gap = 3.3, 0.4
    dx, dw = ML + sw + gap, CW - sw - gap
    cards = [(ML, sw, "静的アプローチ", "事前に余裕を持たせ，外乱に備える", False),
             (dx, dw, "動的アプローチ", "外乱の発生後に，スケジュールを修正する", True)]
    for x, w, head, body, ours in cards:
        _connector(s, ML + CW / 2, by_ + bh_ + 0.02, x + w / 2, cy - 0.04,
                   color=BLUE if ours else MUTE, wt=2.2 if ours else 1.6, kind=1)
        _rrect(s, x, cy, w, ch, FILL if ours else WHITE, line=BLUE if ours else CARDLINE,
               radius=0.06, lw=1.8 if ours else 1.1)
        _text(s, x + 0.3, cy + 0.14, w - 0.6, 0.42, head, 18, color=BLUE if ours else MUTE,
              bold=True)
        _text(s, x + 0.3, cy + 0.62, w - 0.6, 0.52, body, 14.5, color=INK if ours else MUTE,
              ls=1.05)
    # 動的の 2 分岐。本研究に効く区別は「事前のスケジュール S_p があるか」だけなので，
    # 周期的／イベント駆動（ローリングホライズン）の区別は出さない
    ry, rh = 4.85, 1.95
    cw1 = 3.55
    px, pw = dx + cw1 + gap, dw - cw1 - gap
    _connector(s, dx + dw / 2, cy + ch + 0.02, dx + cw1 / 2, ry - 0.04, color=MUTE, wt=1.6,
               kind=1)
    _rrect(s, dx, ry, cw1, rh, WHITE, line=CARDLINE, radius=0.06, lw=1.1)
    _text(s, dx + 0.25, ry + 0.16, cw1 - 0.5, 0.42, "完全リアクティブ", 15.5, color=MUTE,
          bold=True)
    _text(s, dx + 0.25, ry + 0.66, cw1 - 0.5, rh - 0.8,
          "事前のスケジュールを持たず，ディスパッチングルールなどで作業をその場で決める",
          13, color=MUTE, ls=1.1)
    # 予測リアクティブ（本研究）。S_p の存在が以後のすべての前提
    _connector(s, dx + dw / 2, cy + ch + 0.02, px + pw / 2, ry - 0.04, color=BLUE, wt=2.6,
               kind=1)
    _rrect(s, px, ry, pw, rh, COVER, radius=0.06)
    _boxtext(s, px + pw - 1.3, ry - 0.19, 1.1, 0.38, [("本研究", 12.5, WHITE, True)],
             fill=BLUESOFT, radius=0.4)
    tb, tf = add_textbox(s, px + 0.28, ry + 0.3, pw - 0.56, rh - 0.4)
    p = tf.paragraphs[0]
    _run(p, "予測リアクティブ再スケジューリング", 15.5, color=WHITE, bold=True)
    q = tf.add_paragraph()
    q.space_before = Pt(6)
    q.line_spacing = 1.1
    _add_runs(q, "外乱が起きたら，外乱前のスケジュール（**初期スケジュール $S_p$**）を修正する",
              13.5, color=WHITE, emph=WHITE)
    q2 = tf.add_paragraph()
    q2.space_before = Pt(8)
    q2.line_spacing = 1.1
    _add_runs(q2, "→ $S_p$ が存在するので，**$S_p$ からの変更量**を定義できる",
              13.5, color=WHITE, emph=RGBColor(0xFF, 0xD9, 0x8A))


def s_background():
    s = slide_frame("再スケジューリングと安定性",
                    "スケジュールの変更は現場にコストを生むため，生産効率と安定性の二目的最適化として扱う",
                    "1. 研究背景と仮説")
    # 左：例（2 通りの修正を上下に）
    T = GANTT_T
    lx, lw_ = ML, 5.9
    tb, tf = add_textbox(s, lx, 1.72, lw_, 0.36, anchor=MSO_ANCHOR.MIDDLE)
    p = tf.paragraphs[0]
    _run(p, "例：", 12.5, color=INK, bold=True)
    _add_runs(p, "**M2 の J1 が 3 遅延（赤）** → 時刻 $t_r$ 以降の処理順序を変更",
              12.5, color=INK, emph=RED)
    cases = [("$S_p$ の処理順序を保つ", GANTT_A, (), "総所要時間 19", "変更なし", 1),
             ("M3 の順序を入れ替える", GANTT_B, {("M3", "J1"), ("M3", "J2")},
              "総所要時間 16", "順序を変更", 0)]
    ch, cg = 2.3, 0.14
    for k, (head, ops, moved, ms, chg, good) in enumerate(cases):
        cy = 2.15 + k * (ch + cg)
        _rrect(s, lx, cy, lw_, ch, WHITE, line=CARDLINE, radius=0.04)
        _text(s, lx + 0.2, cy + 0.1, lw_ - 3.3, 0.38, head, 13.5, color=INK, bold=True,
              anchor=MSO_ANCHOR.MIDDLE)
        # 良い側は青．順序を変えた案の「順序を変更」はコストの原因なので赤
        for i, txt in enumerate([ms, chg]):
            fill = BLUE if i == good else (RED if (i == 1 and moved) else FILL)
            hot = fill != FILL
            _boxtext(s, lx + lw_ - 2.95 + i * 1.40, cy + 0.1, 1.34, 0.38,
                     [(txt, 11, WHITE if hot else INK, hot)], fill=fill, radius=0.3)
        g_x, g_w, g_y = lx + 0.2, lw_ - 0.4, cy + 0.82
        _gantt(s, g_x, g_y, g_w, ops, rh=0.34, rg=0.1, fs=10.5, delay=("M2", "J1", 3),
               moved=moved, label_w=0.42, tr=GANTT_TR)
        xt = g_x + 0.42 + (g_w - 0.42) * GANTT_TR / T
        _text(s, xt - 0.25, g_y - 0.34, 0.5, 0.26, "$t_r$", 11.5, color=INK, bold=True,
              align=PP_ALIGN.CENTER)
        if k == 0:
            _text(s, g_x + 0.42, g_y - 0.33, xt - g_x - 0.75, 0.26, "開始済み（凍結）", 9.5,
                  color=MUTE, align=PP_ALIGN.RIGHT)
            _text(s, xt + 0.22, g_y - 0.33, 2.5, 0.26, "再スケジューリングの対象", 9.5, color=MUTE)

    # 右：変更のコスト → 二目的最適化
    rx = lx + lw_ + 0.45
    rw = SW - ML - rx
    # 左の「順序を変更」から右のコストへつなぐ
    _connector(s, lx + lw_ + 0.05, 2.15 + ch + cg + 0.29, rx - 0.05, 3.0, color=RED, wt=2.0,
               kind=1)
    _rrect(s, rx, 1.8, rw, 2.75, RGBColor(0xFB, 0xEE, 0xEC), line=RED, radius=0.05, lw=1.4)
    _text(s, rx + 0.22, 1.9, rw - 0.44, 0.5, "処理順序の変更は，現場にコストを生む", 15.5,
          color=RED, bold=True, anchor=MSO_ANCHOR.MIDDLE)
    costs = ["段取り替え", "資材・治具の再手配", "作業者の再配置", "下流工程・外注先\nへの予定変更の波及"]
    cw2 = (rw - 0.44 - 0.15) / 2
    for i, c in enumerate(costs):
        cx = rx + 0.22 + (i % 2) * (cw2 + 0.15)
        cy = 2.55 + (i // 2) * 0.95
        _boxtext(s, cx, cy, cw2, 0.8, [(c, 13.5, INK, True)], fill=WHITE, line=CARDLINE,
                 radius=0.12)
    _connector(s, rx + rw / 2, 4.6, rx + rw / 2, 5.0, color=BLUE, wt=3.0, kind=1)
    by, bh = 5.06, 6.92 - 5.06
    _rrect(s, rx, by, rw, bh, WHITE, line=BLUE, radius=0.05, lw=1.6)
    _text(s, rx, by + 0.06, rw, 0.44, "二目的最適化", 16, color=BLUE, bold=True,
          align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
    objs = [("生産効率", "総所要時間を小さく"), ("安定性", "初期スケジュールからの\n変更を小さく")]
    ow = (rw - 0.44 - 0.5) / 2
    for i, (head, body) in enumerate(objs):
        ox = rx + 0.22 + i * (ow + 0.5)
        oy = by + 0.55
        oh = bh - 0.55 - 0.18
        _rrect(s, ox, oy, ow, oh, FILL, line=CARDLINE, radius=0.08)
        _boxtext(s, ox, oy, ow, 0.38, [(head, 13.5, WHITE, True)], fill=BLUE, radius=0.12)
        _text(s, ox + 0.1, oy + 0.38, ow - 0.2, oh - 0.38, body, 12.5, color=INK,
              align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE, ls=1.05)
        if i == 0:
            _text(s, ox + ow, oy, 0.5, oh, "⇄", 20, color=BLUE, bold=True,
                  align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)


# ---------------------------------------------------------------- 3. 特殊性

def s_special():
    s = slide_frame("再スケジューリングの特殊性",
                    "安定性が最良の解は探索せずに得られる．問われるのは，そこから少しの変更で $MS$ をどこまで下げられるか",
                    "1. 研究背景と仮説")
    w, h = pic(s, "concept_0.png", ML - 0.05, 1.85, w=6.1)
    tx = ML + 6.25
    tw = SW - ML - tx
    # 役割：(D, MS) 平面の読み方を教え，★（探索前に得られる解）と帯（選ばれる解）を置く。
    # 主張は「高安定領域にどれだけ良い解を出せるかが問われる」まで。既存の評価形式で測れて
    # いないことは課題 2，高安定 HV の定義は評価指標のスライドで述べる。箇条書きの ★● は図の記号
    _bul(s, tx, 1.95, tw, [
        "★ $S_p$ の処理順序を保って遅延を吸収した解（right-shift）は，**探索せずに得られる**．"
        "変更量 0 で安定性は最良，$MS$ も外乱前の値＋遅延量を超えない",
        "● 処理順序を変えるほど $MS$ は下げられるが，変更量は増える（**Pareto フロント**）．"
        "変更コストが無視できない現場で選ばれるのは，$S_p$ の近傍 ＝ **高安定領域**の解",
    ], size=15, gap=12, mark="")
    _text(s, tx, 4.25, tw, 0.75,
          "→ 手法に問われるのは，フロント全体の質に加えて，\n**高安定領域にどれだけ良い解を出せるか**",
          16, color=INK, ls=1.08, emph=BLUE)
    _text(s, tx, 5.2, tw, 0.7,
          "本研究では，フロントの質を hypervolume（HV）で測り，高安定領域に限った HV を"
          "**高安定 HV** と呼ぶ", 13.5, color=INK, ls=1.08, emph=BLUE)


# ---------------------------------------------------------------- 4. 先行研究

def s_related():
    s = slide_frame("先行研究：安定性を確保する 2 つのアプローチ",
                    "安定性は「目的関数に組み込む」か「再スケジュールの範囲を限定する」かで確保されてきた",
                    "1. 研究背景と仮説")
    cards = [
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
    gap = 0.4
    cw = (CW - gap) / 2
    y, h = 1.8, 4.05
    for k, (head, refs, common) in enumerate(cards):
        x = ML + k * (cw + gap)
        _rrect(s, x, y, cw, h, WHITE, line=CARDLINE, radius=0.04)
        _rect(s, x, y, cw, 0.55, BLUE)
        _text(s, x + 0.2, y, cw - 0.4, 0.55, head, 15.5, color=WHITE, bold=True,
              anchor=MSO_ANCHOR.MIDDLE)
        tb, tf = add_textbox(s, x + 0.25, y + 0.68, cw - 0.5, 2.5)
        for i, (who, what) in enumerate(refs):
            p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
            p.space_before = Pt(0 if i == 0 else 4)
            p.line_spacing = 1.0
            _run(p, who, 13.5, color=BLUE, bold=True)
            p2 = tf.add_paragraph()
            p2.line_spacing = 1.0
            _run(p2, "　" + what, 12.5, color=INK)
        _boxtext(s, x + 0.25, y + h - 0.6, cw - 0.5, 0.46, [(common, 13.5, BLUE, True)],
                 fill=FILL, radius=0.2)
    _text(s, ML, y + h + 0.12, CW, 0.5,
          "どちらのアプローチでも，メタヒューリスティクスによる解法は GA を基軸とする**多点探索に集中**",
          19, color=INK, align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE, emph=BLUE)
    # 探索構造の 2 系統の説明（文脈を切らないよう控えめに）
    tb, tf = add_textbox(s, ML, y + h + 0.62, CW, 0.38, anchor=MSO_ANCHOR.MIDDLE)
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.CENTER
    _run(p, "単点探索", 14, color=ORANGE, bold=True)
    _run(p, "：1 つの解を少しずつ変更しながら進む　／　", 14, color=INK)
    _run(p, "多点探索", 14, color=GREEN, bold=True)
    _run(p, "：複数の解を保持し，互いに組み替えながら進む", 14, color=INK)
    _run(p, "（Blum & Roli, 2003）", 11, color=MUTE)


# ---------------------------------------------------------------- 5. 3 つの課題

def s_gaps():
    s = slide_frame("残された 3 つの課題",
                    "探索構造の差は検証されておらず，その差を測る形式も，差を補う操作もない",
                    "1. 研究背景と仮説")
    # 課題 3 は文献の事実（そういう操作がない）だけを述べ，課題 1 や仮説 1 には寄りかからない。
    # 「探索構造が届かないなら演算子で補う」は仮説 2 の側で言う
    rows = [
        ("探索構造の差が\n検証されていない",
         "解法は多点探索に集中している．$S_p$ が探索前に得られているという特性が，"
         "単点探索と多点探索の間にどんな差を生むかは分析されていない", 1.3),
        ("その差を読み取れる\n評価の形式がない",
         "比較は少数の重みでのスカラー値や，Pareto フロント全体の HV で行われ，"
         "高安定領域で手法がどう異なるかは分析されていない", 1.3),
        ("Pareto フロント全体を保って\n高安定領域の解を生む\n操作がない",
         "**目的関数に組み込む**：高安定領域の解を「選ぶ」ことはできるが，"
         "「作る」のは交叉や局所探索などの探索演算子のまま\n"
         "**範囲を限定する**：解を高安定領域に閉じ込めるので，"
         "範囲外（変更量が大きく $MS$ が小さい）の Pareto フロントは得られない\n"
         "→ Pareto フロント全体を保ったまま，高安定領域の解を能動的に作る操作は，どちらにもない", 1.95),
    ]
    y, rg = 1.9, 0.2
    hw = 4.3
    for i, (head, body, rh) in enumerate(rows):
        if i:
            y += rows[i - 1][2] + rg
        _rrect(s, ML, y, CW, rh, WHITE, line=CARDLINE, radius=0.05)
        _boxtext(s, ML + 0.22, y + (rh - 0.56) / 2, 0.56, 0.56, [(str(i + 1), 18, WHITE, True)],
                 fill=BLUE, radius=0.5)
        _text(s, ML + 1.0, y + 0.08, hw - 0.9, rh - 0.16, head, 16, color=BLUE, bold=True,
              anchor=MSO_ANCHOR.MIDDLE, ls=1.1)
        _rect(s, ML + hw + 0.2, y + 0.22, 0.015, rh - 0.44, HAIR)
        _text(s, ML + hw + 0.45, y + 0.08, CW - hw - 0.7, rh - 0.16, body, 14, color=INK,
              anchor=MSO_ANCHOR.MIDDLE, ls=1.18, emph=BLUE)


# ---------------------------------------------------------------- 6. 仮説

def s_hypotheses():
    s = slide_frame("本研究の仮説",
                    "課題 1・3 に対して 2 つの仮説を立て，どちらも高安定 HV（課題 2 への対応）で判定する",
                    "1. 研究背景と仮説")
    gap = 0.3
    cw = (CW - gap) / 2
    # 機序と主張を別の文にする（「多点探索は…伴うため，単点探索のほうが」だと主題が宙に浮く）
    hyps = [("仮説 1", "課題 1",
             "交叉で 2 親の染色体を組み替える**多点探索**は，\n"
             "$S_p$ からの小さな変更では表せない飛躍を伴う．\n"
             "そのため，$S_p$ を初期解として解を少しずつ変更する\n"
             "**単点探索**のほうが，**高安定 HV が大きい**", "concept_h1.png"),
            ("仮説 2", "課題 3",
             "多点探索に，解を $S_p$ に近づける移動\n"
             "（**安定性誘導演算子**）を組み込めば，\n"
             "その経路上に変更量の小さい解が現れるため，\n"
             "**多点探索の高安定 HV は向上する**", "concept_h2.png")]
    for k, (tag, src, text, img) in enumerate(hyps):
        x = ML + k * (cw + gap)
        _text(s, x + 0.05, 1.72, cw, 0.4, tag, 15, color=BLUE, bold=True,
              anchor=MSO_ANCHOR.MIDDLE)
        by, bh = 2.14, 1.6
        _rrect(s, x, by, cw, bh, COVER, radius=0.06)
        _text(s, x + 0.16, by + 0.05, cw - 0.32, bh - 0.1, text, 15.5, color=WHITE, ls=1.12,
              anchor=MSO_ANCHOR.MIDDLE, emph=RGBColor(0xFF, 0xD9, 0x8A))
        ih = 7.0 - (by + bh + 0.1)
        iw, _h = _img_size(asset(img))
        w_img = min(cw - 0.04, ih * iw / _h)
        pic(s, img, x + (cw - w_img) / 2, by + bh + 0.1, w=w_img)


# ---------------------------------------------------------------- 7. 問題設定

def s_problem():
    s = slide_frame("問題設定",
                    "決定変数は，遅延後に最適化対象となる作業の，機械ごとの処理順序",
                    "2. 問題設定と実験計画")
    _bul(s, ML, 1.95, 6.6, [
        "$n$ ジョブ・$m$ 機械のジョブショップ問題（JSP）に，単一作業の遅延（遅延量 $Δ$）が発生",
        "再スケジューリング時刻 $t_r$ より前に開始した作業は凍結し，以降の作業集合 $O_{opt}$ の**機械ごとの処理順序**を決定変数とする",
        "安定性は**処理順序の変更量**で測る（順序偏差 $D$）",
    ], size=15, gap=10)
    render_equation(s, r"D(S_p,S_q)=\sum_{(i,j)\in O_{opt}}\left|r^p_{i,j}-r^q_{i,j}\right|",
                    ML, 3.85, 6.6, number="(1)", name="talk_D", h=0.62)
    _text(s, ML + 0.3, 4.55, 6.3, 0.35, "$r_{i,j}$：機械 $i$ におけるジョブ $j$ の処理順位", 12.5, color=MUTE)
    _bul(s, ML, 5.1, 6.6, [
        "二目的（$MS$, $D$）は重み付き和 $F$ にまとめ，重み $λ$ を生産効率重視から安定性重視まで掃引し，得られた全非劣解で評価する",
    ], size=15)
    # 右：順序偏差の例
    rx, rw = 7.75, 4.95
    _rrect(s, rx, 1.95, rw, 4.85, RGBColor(0xF7, 0xFA, 0xFC), line=CARDLINE, radius=0.04)
    _text(s, rx + 0.25, 2.1, rw - 0.5, 0.4, "例：ある機械の処理順序", 14, color=BLUE, bold=True)
    jc = {"J1": RGBColor(0x3E, 0x61, 0x8C), "J2": RGBColor(0xD9, 0x8E, 0x45),
          "J3": RGBColor(0x6F, 0x9E, 0x7C), "J4": RGBColor(0x8A, 0x86, 0xA8)}
    seqs = [("$S_p$", ["J1", "J3", "J2", "J4"], 3.05), ("$S_q$", ["J1", "J2", "J3", "J4"], 4.55)]
    bw, bh, bx0 = 0.82, 0.55, rx + 1.0
    pos = {}
    for lab, seq, yy in seqs:
        _text(s, rx + 0.25, yy, 0.7, bh, lab, 15, color=INK, bold=True, anchor=MSO_ANCHOR.MIDDLE)
        for r, j in enumerate(seq):
            bx = bx0 + r * (bw + 0.12)
            _boxtext(s, bx, yy, bw, bh, [(j, 13.5, WHITE, True)], fill=jc[j], radius=0.1)
            _text(s, bx, yy - 0.33 if lab == "$S_p$" else yy + bh + 0.02, bw, 0.3, f"{r + 1}番目",
                  10.5, color=MUTE, align=PP_ALIGN.CENTER)
            pos[(lab, j)] = (bx + bw / 2, yy, yy + bh)
    for j in ["J1", "J2", "J3", "J4"]:
        x1, _a, y1 = pos[("$S_p$", j)]
        x2, y2, _b = pos[("$S_q$", j)]
        moved = x1 != x2
        _connector(s, x1, y1 + 0.02, x2, y2 - 0.02, color=RED if moved else HAIR,
                   wt=2.0 if moved else 1.2, arrow=moved, kind=1)
    _text(s, rx + 0.25, 5.6, rw - 0.5, 0.45, "$D$ = |3−2| ＋ |2−3| = **2**", 16, color=INK,
          align=PP_ALIGN.CENTER, emph=RED)
    _text(s, rx + 0.25, 6.1, rw - 0.5, 0.6, "$S_p$ の処理順序を保った解は $D$ = 0", 13, color=MUTE,
          align=PP_ALIGN.CENTER)


# ---------------------------------------------------------------- 8. 比較する探索構造

def _flow(slide, x, y, w, steps, bh=0.52, gap=0.3, loop_label=None, loop_from=None, loop_to=None):
    """縦に並ぶ処理ボックスを矢印でつなぐ。steps: (文字列, 強調色 or None)。
    loop_from/loop_to の段のあいだに右側の戻り矢印を描く。"""
    ys = []
    for i, (t, accent) in enumerate(steps):
        yy = y + i * (bh + gap)
        ys.append(yy)
        if accent == "shared":
            _boxtext(slide, x, yy, w, bh, [(t, 12.5, INK, False)], fill=FILL, line=CARDLINE,
                     radius=0.12)
        elif accent is not None:
            _boxtext(slide, x, yy, w, bh, [(t, 12.5, WHITE, True)], fill=accent, radius=0.12)
        else:
            _boxtext(slide, x, yy, w, bh, [(t, 12.5, INK, False)], fill=WHITE, line=CARDLINE,
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
            _text(slide, rx + 0.08, (y0 + y1) / 2 - 0.2, 1.3, 0.4, loop_label, 11, color=MUTE,
                  anchor=MSO_ANCHOR.MIDDLE)
    return ys[-1] + bh - y


def s_structures():
    s = slide_frame("比較する探索構造：単点探索と多点探索",
                    "局所探索などの条件を揃え，解の作り方（少しずつ変更するか，交叉で組み替えるか）を比較する",
                    "2. 問題設定と実験計画")
    gap = 0.4
    cw = (CW - gap) / 2
    top, ch = 1.85, 4.3
    cards = [
        ("単点探索（ILS：反復局所探索）", "現在解は常に 1 つ", ORANGE,
         [("初期解：$S_p$", None), ("摂動：数手の insert 移動", ORANGE),
          ("局所探索：N5 近傍", "shared"), ("受理判定：$F$ が改善すれば更新", None)],
         1, 3, "反復",
         "局所探索と摂動が別の段で，演算子を摂動に組み込める"),
        ("多点探索（Memetic：GA＋局所探索）", "複数の解を保持", GREEN,
         [("初期集団：$S_p$ ＋ 無作為に生成した個体", None), ("選択・交叉：2 親の染色体を組み替え", GREEN),
          ("突然変異", None), ("局所探索：N5 近傍（各個体）", "shared")],
         1, 3, "世代交代",
         "先行研究で標準的な，GA に局所探索を組み込んだ構成"),
    ]
    for k, (head, sub, col, steps, lf, lt, loop, why) in enumerate(cards):
        x = ML + k * (cw + gap)
        _rrect(s, x, top, cw, ch, WHITE, line=CARDLINE, radius=0.04)
        _rect(s, x, top, cw, 0.55, col)
        tb, tf = add_textbox(s, x + 0.2, top, cw - 0.4, 0.55, anchor=MSO_ANCHOR.MIDDLE)
        p = tf.paragraphs[0]
        _run(p, head, 15, color=WHITE, bold=True)
        _run(p, "　" + sub, 12, color=WHITE)
        tb, tf = add_textbox(s, x + 0.25, top + 0.64, cw - 0.5, 0.4, anchor=MSO_ANCHOR.MIDDLE)
        p = tf.paragraphs[0]
        _run(p, "選定理由：", 12, color=col, bold=True)
        _add_runs(p, why, 12, color=INK)
        fw = 3.9
        _flow(s, x + 0.35, top + 1.18, fw, steps, loop_from=lt, loop_to=lf, loop_label=loop)
    _boxtext(s, ML, top + ch + 0.15, CW, 0.62,
             [("揃えた条件：局所探索（N5 近傍）・目的関数 $F$・初期解に $S_p$ を含めること　→　局所探索の有無による差を除いて比較",
               13.5, WHITE, True)],
             fill=COVER, radius=0.12)


# ---------------------------------------------------------------- 9. 安定性誘導演算子

def s_operators():
    s = slide_frame("安定性誘導演算子：PR と PR-$k$",
                    "path relinking（PR）の誘導解を $S_p$ に固定し，現在解から $S_p$ へ swap 移動でたどる",
                    "2. 問題設定と実験計画")
    # 左右の列の上端・下端を揃える（左：定義と組み込み方，右：例）
    top, bot = 1.85, 6.75
    lw_ = 5.95
    lx = SW - ML - lw_
    # ---- 右：ある機械の処理順序を S_p に近づける経路
    _rrect(s, lx, top, lw_, bot - top, RGBColor(0xF7, 0xFA, 0xFC), line=CARDLINE, radius=0.04)
    _text(s, lx + 0.25, top + 0.1, lw_ - 0.5, 0.4, "例：ある機械の処理順序を $S_p$ に近づける", 14,
          color=BLUE, bold=True)
    jc = {"J1": RGBColor(0x3E, 0x61, 0x8C), "J2": RGBColor(0xD9, 0x8E, 0x45),
          "J3": RGBColor(0x6F, 0x9E, 0x7C), "J4": RGBColor(0x8A, 0x86, 0xA8)}
    SP = ["J1", "J2", "J3", "J4"]
    bw, bh, bg = 0.72, 0.5, 0.09
    bx0 = lx + 1.55

    def seq_row(yy, lab, seq, dlab, lab_color=INK):
        _text(s, lx + 0.25, yy, 1.3, bh, lab, 13.5, color=lab_color, bold=True,
              anchor=MSO_ANCHOR.MIDDLE)
        for c, j in enumerate(seq):
            bx = bx0 + c * (bw + bg)
            diff = j != SP[c]
            sp = _boxtext(s, bx, yy, bw, bh, [(j, 13.5, WHITE, True)], fill=jc[j], radius=0.1,
                          line=RED if diff else None)
            if diff:
                sp.line.width = Pt(2.6)
        _text(s, bx0 + 4 * (bw + bg) + 0.08, yy, 0.9, bh, dlab, 13.5, color=INK,
              anchor=MSO_ANCHOR.MIDDLE)

    # 目標（誘導解 S_p）を最初に示し，現在解と並べて差異を見せる
    _rrect(s, lx + 0.15, top + 0.6, lw_ - 0.3, 0.7, RGBColor(0xCF, 0xDC, 0xEA), radius=0.1)
    seq_row(top + 0.7, "$S_p$（誘導解）", SP, "$D$ = 0", lab_color=BLUE)
    path = [("現在解", ["J1", "J4", "J2", "J3"], "$D$ = 4", None),
            ("1 回目", ["J1", "J3", "J2", "J4"], "$D$ = 2", "swap：J4 と J3 を入れ替え"),
            ("2 回目", ["J1", "J2", "J3", "J4"], "$D$ = 0", "swap：J3 と J2 を入れ替え（$S_p$ に到達）")]
    y0, gy = top + 1.57, 0.46
    for r, (lab, seq, dlab, note) in enumerate(path):
        yy = y0 + r * (bh + gy)
        if note:
            _text(s, bx0 - 0.05, yy - gy, lw_ - 1.65, gy, "↓ " + note, 11.5, color=RED,
                  anchor=MSO_ANCHOR.MIDDLE)
        seq_row(yy, lab, seq, dlab)
    _text(s, lx + 0.25, bot - 0.78, lw_ - 0.5, 0.7,
          "赤枠：$S_p$ と順位が異なる作業\n"
          "1 回の swap で 1 作業を $S_p$ の順位に合わせ，$D$ が小さくなる", 12, color=MUTE, ls=1.1,
          anchor=MSO_ANCHOR.MIDDLE)
    # ---- 左：PR と PR-k（通常の PR との違いを示してから操作を述べる），組み込み方
    rx = ML
    rw = lx - 0.3 - ML
    gap = 0.18
    cards = [("PR", "通常の PR はエリート解どうしを結ぶ．\n"
                    "本研究では**誘導解を $S_p$ に固定**し，\n"
                    "現在解と $S_p$ を結ぶ経路をたどり，\n"
                    "その中間解から $F$ の小さい解を選ぶ", 1.62),
             ("PR-$k$", "$k$ 回の swap で打ち切り，\n"
                        "到達解を返す．$k$ は 1〜$d$ で変える\n"
                        "（$d$：$S_p$ と順位が異なる作業数）", 1.3)]
    cy = top
    for head, body, h in cards:
        _rrect(s, rx, cy, rw, h, WHITE, line=CARDLINE, radius=0.05)
        _boxtext(s, rx + 0.2, cy + 0.2, 1.05, 0.42, [(head, 14, WHITE, True)], fill=BROWN,
                 radius=0.3)
        _text(s, rx + 1.45, cy + 0.08, rw - 1.6, h - 0.16, body, 15, color=INK, ls=1.12,
              anchor=MSO_ANCHOR.MIDDLE, emph=BLUE)
        cy += h + gap
    # 組み込み方：残りの高さに中身を上下中央で置く
    bh2 = bot - cy
    _rrect(s, rx, cy, rw, bh2, FILL, line=CARDLINE, radius=0.05)
    tb, tf = add_textbox(s, rx + 0.25, cy + 0.06, rw - 0.5, bh2 - 0.12, anchor=MSO_ANCHOR.MIDDLE)
    p = tf.paragraphs[0]
    _run(p, "組み込み方", 15, color=BLUE, bold=True)
    for who, col, how in [("ILS", ORANGE, "停滞したときに，摂動として発動"),
                          ("Memetic", GREEN, "各個体に確率的に，改善操作として発動")]:
        q = tf.add_paragraph()
        q.space_before = Pt(8)
        _run(q, who + "：", 15, color=col, bold=True)
        _add_runs(q, how, 15, color=INK)


# ---------------------------------------------------------------- 10. 実験計画

def s_design():
    s = slide_frame("比較する手法と仮説の判定",
                    "仮説 1 は ILS と Memetic，仮説 2 は Memetic と演算子を組み込んだ Memetic の高安定 HV を比較して判定する",
                    "2. 問題設定と実験計画")
    # 研究の位置づけ
    _text(s, ML, 1.66, CW, 0.36,
          "本研究の主眼は最先端手法との性能競争ではなく，仮説を立てて条件を統制し，"
          "**探索構造と演算子の関係を切り分けること**", 14, color=INK,
          anchor=MSO_ANCHOR.MIDDLE, emph=BLUE)
    # 比較する手法：探索構造 × 安定性誘導演算子
    ty, trh, tw = 2.16, 0.4, 8.7
    render_table(s, [["探索構造", "安定性誘導演算子なし", "PR-$k$ を組み込む", "PR を組み込む"],
                     ["単点探索（ILS）", "ILS", "ILS＋PR-$k$", "ILS＋PR"],
                     ["多点探索（Memetic）", "Memetic", "Memetic＋PR-$k$", "Memetic＋PR"]],
                 ML, ty, tw, col_w=[2.5, 2.3, 2.0, 1.9], size=14.5, align=['l', 'c', 'c', 'c'],
                 row_h=trh)
    tbl = s.shapes[-1].table
    for r, col in [(1, ORANGE), (2, GREEN)]:
        for c in range(4):
            for p in tbl.cell(r, c).text_frame.paragraphs:
                for run in p.runs:
                    run.font.color.rgb = col
                    run.font.bold = True
    _text(s, ML + tw + 0.35, ty + trh, CW - tw - 0.35, 2 * trh,
          "参考：GA（局所探索なし）を\n加えて計 7 手法", 13, color=MUTE, anchor=MSO_ANCHOR.MIDDLE)
    # 仮説ごとに，比較する手法と支持とみなす条件を書き下す
    hyps = [("仮説 1", "単点探索のほうが，多点探索より高安定 HV が大きい",
             "ILS が Memetic を全シナリオで有意に上回れば，支持とみなす"),
            ("仮説 2", "安定性誘導演算子を組み込めば，多点探索の高安定 HV は向上する",
             "Memetic＋PR-$k$ と Memetic＋PR がそれぞれ Memetic を全シナリオで有意に上回れば，支持とみなす")]
    tx = ML + 1.4
    txw = SW - ML - tx
    y = 3.58
    for tag, claim, rule in hyps:
        _boxtext(s, ML, y + 0.03, 1.2, 0.4, [(tag, 14, WHITE, True)], fill=COVER, radius=0.3)
        _text(s, tx, y, txw, 0.46, "**" + claim + "**", 17, color=INK,
              anchor=MSO_ANCHOR.MIDDLE, emph=INK)
        tb, tf = add_textbox(s, tx, y + 0.48, txw, 0.38, anchor=MSO_ANCHOR.MIDDLE)
        p = tf.paragraphs[0]
        _run(p, "判定：", 15, color=BLUE, bold=True)
        _add_runs(p, rule, 15, color=INK)
        y += 1.04
        _rect(s, ML, y - 0.1, CW, 0.012, HAIR)
    # 仮説を置かない比較
    _boxtext(s, ML, y + 0.03, 1.2, 0.4, [("仮説なし", 13, WHITE, True)], fill=MUTE, radius=0.3)
    _text(s, tx, y, txw, 0.46, "ILS＋PR-$k$ と ILS＋PR も，仮説 2 と同じ形で ILS と比較する", 15,
          color=INK, anchor=MSO_ANCHOR.MIDDLE)
    # 検定
    jy = y + 0.72
    tb, tf = add_textbox(s, ML, jy, CW, 0.56)
    for k, (head, body) in enumerate([
            ("検定：", "両側 Mann–Whitney の $U$ 検定．全シナリオにわたる同じ比較は Holm 法で補正"),
            ("効果量：", "Cliff's $δ$．$|δ|$ = 1.0 は，一方の全試行が他方の全試行を上回る完全分離")]):
        q = tf.paragraphs[0] if k == 0 else tf.add_paragraph()
        q.space_before = Pt(0 if k == 0 else 2)
        _run(q, head, 12.5, color=MUTE, bold=True)
        _add_runs(q, body, 12.5, color=MUTE)


# ---------------------------------------------------------------- 9. 評価指標

def s_metrics():
    s = slide_frame("評価指標",
                    "高安定 HV は，順序偏差 $D$ が $P_{θ}$ 未満の非劣解だけで算出した HV",
                    "2. 問題設定と実験計画")
    pic(s, "metrics.png", ML + 0.05, 1.78, w=CW - 0.1)
    _bul(s, ML, 6.12, CW, [
        "$P_{θ}$：全手法・全試行の非劣解の $D$ の $θ$ パーセンタイル．"
        "$D$ = 0 の解（$S_p$ の処理順序を保った解）は全手法が探索せずに得るため HV から除く",
        "全域 HV・高安定 HV は全ての重み $λ$ の解を合わせて算出し，AUC は重みごとに算出して平均する",
    ], size=13, gap=4)


# ---------------------------------------------------------------- 10. 実験設定

def s_setup():
    s = slide_frame("実験設定",
                    "5 つのベンチマーク問題に単一作業の遅延を与え，8 シナリオを構成",
                    "3. 数値実験")
    rows = [
        ["項目", "設定"],
        ["問題・外乱", "mt10・la21・la36・la40・ta21 に単一作業の完了遅延（$Δ$ = 60〜148，対象作業の処理時間の約 0.9〜1.5 倍）"],
        ["シナリオ", "8 つ：la36S・la36M・la36L，ta21S・ta21L，mt10，la21，la40\n"
                    "la36 と ta21 は同一問題・同一 $S_p$ のまま外乱の規模だけを段階的に変え，小さい順に S・M・L"],
        ["初期スケジュール $S_p$", "500 世代の GA で生成"],
        ["重み $λ$", "0, 0.1, …, 0.9 の 10 点（$λ$ = 1.0 は $D$ = 0 の自明解に縮退するため除外）"],
        ["計算予算", "ILS 系 3000 反復，GA・Memetic 系 500 世代"],
        ["探索", "GA・Memetic 系：交叉率 0.85，突然変異率 0.1，集団 50（$S_p$ 以外の 49 個体は無作為生成）\n"
                "ILS 系：insert 摂動 2〜5 手を鋸歯状に巡回"],
        ["安定性誘導演算子", "Memetic 系：各個体に確率 0.3 で発動．PR は経路上の上位 3 中間解を局所探索して最良を返す\n"
                          "ILS 系：無改善 400 反復で初回，以降 10 反復ごとに発動．PR は経路上の最良 1 解を返す"],
        ["高安定領域の境界", "$θ$ = 50（$P_{50}$：全手法・全試行の非劣解の $D$ の中央値）"],
        ["試行", "各条件 10 試行（全乱数シード固定）"],
        ["環境", "AMD Ryzen 5 7530U，Python 3.12（NumPy・DEAP・SciPy）"],
    ]
    render_table(s, rows, ML, 1.78, CW, col_w=[2.3, 9.79], size=12.5, align=['l', 'l'], row_h=0.3)
    # 妥当性の確認（1 行ずつ）
    vy = 5.62
    _text(s, ML, vy, CW, 0.3, "妥当性の確認", 13.5, color=BLUE, bold=True, anchor=MSO_ANCHOR.MIDDLE)
    _bul(s, ML, vy + 0.33, CW, [
        "**高安定領域の境界**：$θ$ を 33〜67 に変えても，仮説 1・仮説 2 の結論は変わらない（全シナリオで完全分離）",
        "**計算予算**：中央値の試行は全手法とも，予算の 7 割以内で最終の全域 HV の 99% に到達",
        "**パラメータ感度**：挙動を定める 11 軸を 2 シナリオで一要因ずつ振っても，全域 HV の変化は +0.5〜2%（手法間の差は +7〜54%）",
    ], size=13, gap=3)


# ---------------------------------------------------------------- 11〜14. 結果

def s_front():
    s = slide_frame("結果 1　高安定領域の非劣解（代表 2 シナリオ）",
                    "ta21S では，演算子なしの Memetic は高安定領域に非劣解を持たない",
                    "3. 数値実験")
    w = 12.0
    pic(s, "front_ta21.png", ML + (CW - w) / 2, 1.72, w=w)
    _bul(s, ML, 5.3, CW, [
        "(a) ta21S：Memetic は高安定領域に非劣解を持たない",
        "(b) ta21L：Memetic は高安定領域に入るが，$MS$ で ILS に劣る",
        "どちらのシナリオでも，演算子を組み込むと両構造の Pareto フロント（点線）は高安定領域で重なる",
    ], size=18, gap=7)


def s_highstab():
    s = slide_frame("結果 2　高安定 HV（全 8 シナリオ）",
                    "仮説 1・仮説 2 とも，全 8 シナリオで支持された",
                    "3. 数値実験")
    w = 12.0
    pic(s, "highstab_all8.png", ML + (CW - w) / 2, 1.68, w=w)
    _bul(s, ML, 5.62, CW, [
        "**仮説 1**：演算子なしで ILS > Memetic（全シナリオで $p$ < 0.001，$|δ|$ = 1.0）",
        "**仮説 2**：演算子を組み込んだ Memetic > Memetic（全シナリオで $p$ < 0.001，$|δ|$ = 1.0）",
        (1, "Memetic は演算子を組み込むと ILS と同程度まで向上する一方，ILS ではほとんど向上しない"),
    ], size=18, gap=7)


def s_band():
    s = slide_frame("安定性誘導演算子の効果が現れる条件",
                    "効果の非対称は，組み込む前に「到達上限の帯」に入っているかで説明できる",
                    "3. 数値実験")
    w = 12.0
    pic(s, "highstab_band.png", ML + (CW - w) / 2, 1.68, w=w)
    _bul(s, ML, 5.6, CW, [
        "どのシナリオでも 3〜5 手法が最良値の 2% 以内に集まる → この帯を到達上限とみなす",
        "**Memetic**：演算子なしでは全シナリオで帯の外 → 組み込むと 7 つで帯の中",
        "**ILS**：演算子なしで 6 つが帯の中 → 有意に向上したのは帯の外の 2 つだけ",
    ], size=17, gap=3)
    _text(s, ML, 6.6, CW, 0.36,
          "→ 演算子の効果は，探索構造の違いそのものではなく，**組み込む前に上限までの差が残っているかで決まる**",
          17, color=INK, emph=BLUE, anchor=MSO_ANCHOR.MIDDLE)


def s_anytime():
    s = slide_frame("結果 3　全域 HV と AUC（代表 2 シナリオ）",
                    "立ち上がりは ILS が速く，到達した全域 HV は演算子を組み込んだ Memetic が同等以上",
                    "3. 数値実験")
    w = 12.0
    pic(s, "anytime.png", ML + (CW - w) / 2, 1.72, w=w)
    _bul(s, ML, 5.74, CW, [
        "(a) ta21S：演算子を組み込んだ Memetic も ILS と同じ全域 HV に達するが，立ち上がりが遅い",
        "(b) la36L：Memetic 系は途中で ILS 系を追い越し，より大きい全域 HV に達する",
    ], size=18, gap=7)


def s_union_auc():
    s = slide_frame("結果 4　全域 HV と AUC（全 8 シナリオ）",
                    "最良の手法は評価指標によって異なる：全域 HV は Memetic＋演算子，AUC は ILS",
                    "3. 数値実験")
    tw = 5.75
    _text(s, ML, 1.74, tw, 0.42, "最良に並ぶシナリオ数（/8）", 17, color=BLUE, bold=True,
          anchor=MSO_ANCHOR.MIDDLE)
    trh = 0.47
    render_table(s, [
        ["手法", "全域HV", "高安定HV", "AUC"],
        ["ILS", "4", "6", "**6**"],
        ["ILS+PR-$k$", "4", "**8**", "**6**"],
        ["ILS+PR", "4", "**8**", "**6**"],
        ["GA", "0", "0", "0"],
        ["Memetic", "1", "0", "2"],
        ["Memetic+PR-$k$", "**7**", "**7**", "1"],
        ["Memetic+PR", "**7**", "**7**", "0"],
    ], ML, 2.24, tw, col_w=[2.3, 1.15, 1.35, 0.95], size=16.5, row_h=trh)
    _text(s, ML, 2.24 + 8 * trh + 0.08, tw, 0.4, "最良に並ぶ：中央値が最良手法の中央値から 2% 以内",
          14, color=MUTE)
    # 右：指標ごとの読み取り（シナリオ数は表に任せ，語の途中で折り返さないよう行を明示）
    rx = ML + tw + 0.5
    rw = SW - ML - rx
    _text(s, rx, 1.74, rw, 0.42, "全域 HV", 18, color=BLUE, bold=True, anchor=MSO_ANCHOR.MIDDLE)
    _text(s, rx, 2.24, rw, 1.9,
          "Memetic＋演算子が最良に並ぶ\n"
          "ILS 系は 4 シナリオで 5〜38% 下回る\n"
          "→ **高安定 HV での ILS の優位は，**\n**全体として優れていることの反映ではない**",
          17, color=INK, ls=1.12, emph=BLUE)
    _text(s, rx, 4.4, rw, 0.42, "AUC", 18, color=BLUE, bold=True, anchor=MSO_ANCHOR.MIDDLE)
    _text(s, rx, 4.9, rw, 1.9,
          "ILS 系が最良に並ぶ\n"
          "Memetic では演算子が各個体に発動して\n"
          "計算時間が増え，PR は AUC を\n"
          "5 シナリオで 7〜26% 下げる",
          17, color=INK, ls=1.12)


# ---------------------------------------------------------------- 15. 結論

def s_conclusion():
    s = slide_frame("まとめ",
                    "$S_p$ が探索前に得られる再スケジューリングでは，高安定領域で探索構造と演算子の関係が分かれる",
                    "4. 結論")
    items = [("仮説 1 を支持", "高安定 HV は**単点探索（ILS）**が多点探索（Memetic）を上回る．全 8 シナリオの全試行で分離"),
             ("仮説 2 を支持", "安定性誘導演算子は**多点探索の高安定 HV を単点探索と同程度まで向上**させ，\n"
                              "単点探索ではほとんど向上させない．効果は，組み込む前に到達上限の帯に入っているかで決まる"),
             ("手法の選び方", "全域 HV を重視するなら **Memetic＋演算子**，短い計算時間で解を得るなら **ILS**．\n"
                              "高安定 HV では両者が並ぶ"),
             ("貢献", "手法を高安定 HV でも評価し，局所探索を揃えた統制比較で，探索構造と演算子の関係を切り分けた\n"
                      "→ **ある演算子が有効だという報告は，どの探索構造に組み込んだかを抜きにしては一般化できない**")]
    # 見出しの列を詰めて本文の幅を取り，各行を 1 行に収める
    bx = ML + 2.75
    bw = SW - ML - 0.2 - bx
    y, hh, gap = 1.85, 1.08, 0.15
    for i, (h, b) in enumerate(items):
        _rrect(s, ML, y, CW, hh, WHITE, line=CARDLINE, radius=0.06)
        if h != "貢献":
            _boxtext(s, ML + 0.2, y + (hh - 0.52) / 2, 0.52, 0.52, [(str(i + 1), 17, WHITE, True)],
                     fill=BLUE, radius=0.5)
        _text(s, ML + 0.95, y + 0.08, 1.7, hh - 0.16, h, 15.5, color=BLUE, bold=True,
              anchor=MSO_ANCHOR.MIDDLE)
        _text(s, bx, y + 0.06, bw, hh - 0.12, b, 14, color=INK, ls=1.12,
              anchor=MSO_ANCHOR.MIDDLE, emph=BLUE if h == "貢献" else None)
        y += hh + gap


def s_limits():
    s = slide_frame("限界と今後の課題",
                    "結論は本実験の条件の下で得たもので，条件を変えた再検証が今後の課題",
                    "4. 結論")
    x = ML + 0.2
    w = CW - 0.2
    _text(s, ML, 1.74, CW, 0.42, "限界", 18, color=BLUE, bold=True, anchor=MSO_ANCHOR.MIDDLE)
    _bul(s, x, 2.18, w, [
        "(i) $p$ が有意水準に近い比較は，$n$ = 10 では決着していない",
        "(ii) 安定性を開始時刻偏差で測る場合の妥当性は検証していない",
        "(iii) 外乱は単一作業の遅延に限る",
        "(iv) 初期スケジュール $S_p$ の最適化の程度が結果に与える影響は検証していない",
        "(v) 各探索構造の代表は 1 手法ずつである",
    ], size=16.5, gap=4, mark="")
    _text(s, ML, 4.25, CW, 0.42, "今後の課題", 18, color=BLUE, bold=True,
          anchor=MSO_ANCHOR.MIDDLE)
    _bul(s, x, 4.69, w, [
        "上記の条件を変えた再検証",
        "再スケジュールの範囲を限定するアプローチと安定性誘導演算子の統合",
        "安定性を交叉自体に持たせる方式の検証",
        "Pareto 支配に基づく手法（NSGA-II 等）での再検証",
    ], size=16.5, gap=4)
    _text(s, ML, 6.48, CW, 0.36,
          "データ：各シナリオの定義と，試行ごとの値・検定は公開リポジトリに収録"
          "（https://github.com/kitotakumi/stability_scheduling）", 13, color=MUTE,
          anchor=MSO_ANCHOR.MIDDLE)


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
        (1, "組み込み方が同じ ILS 系の中でも帯の内外で効果が分かれる → 効果の非対称は，組み込み方の違いでは説明できない"),
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


# ================================================================ 本編
def build():
    title_slide_talk()
    chapter("1")
    s_intro()
    s_background()
    s_special()
    s_related()
    s_gaps()
    s_hypotheses()
    chapter("2")
    s_problem()
    s_metrics()
    s_structures()
    s_operators()
    s_design()
    chapter("3")
    s_setup()
    s_front()
    s_highstab()
    s_band()
    s_anytime()
    s_union_auc()
    chapter("4")
    s_conclusion()
    s_limits()
    backup_divider()
    s_backup_problem()
    s_backup_scalar()
    s_backup_why()
    s_backup_operator()
    s_backup_params()
    s_backup_robust()
    s_backup_control()
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
