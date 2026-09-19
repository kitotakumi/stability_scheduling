#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""apiems2026_manuscript_v2.md から日本語版 docx を生成する。

v1 は content_ja.py に本文を手で転記する方式だったが、v2 は markdown を直接
builder のブロック列へ変換する。原稿を直したら再実行するだけで docx が出る。

usage: python make_docx_v2.py [--pages]
       --pages を付けると Word で開いてページ数を数える（Windows + Word 必須）

対応する markdown:
  ## N. 見出し            -> h1
  ### N.N 見出し          -> h2
  **Title**: / **Abstract...** / **Keywords**:  -> title / abstract / keywords
  ![キャプション](path)   -> fig (全幅)
  **表 N.** キャプション + 続く | 表 |  -> table
  $$ ... (N) $$           -> eq (builder の omml_eq1 / omml_eq2)
  - 箇条書き              -> p（builder にリスト種別がないため本文段落）
  [n] 文献                -> ref
  > 引用, --- , 作業メモ見出し  -> 除外
"""
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
V1 = os.path.join(HERE, '..', 'paper_apiems2026')
sys.path.insert(0, V1)

import builder  # noqa: E402

MD = os.path.join(HERE, 'apiems2026_manuscript_v2.md')
TEMPLATE = os.path.join(V1, 'APIEMS FullPaperTemplate.docx')
FIG_DIR = os.path.join(HERE, 'figures')
OUT = os.path.join(HERE, 'APIEMS2026_draft_v2_ja.docx')

# 本文の範囲（作業メモ・ページ予算・v1→v2 対応表を除く）
BODY_START = '## Title / Authors / Abstract / Keywords'
BODY_END = '## v1 → v2 の対応表'

AUTHORS = ['Takumi Kito']
AFFIL = ['早稲田大学大学院 創造理工学研究科 経営システム工学専攻, 東京, 日本',
         'Tel: (+81) 80-4756-3741, Email: kito@toki.waseda.jp']


def _body_lines():
    lines = open(MD, encoding='utf-8').read().split('\n')
    lo = next(i for i, l in enumerate(lines) if l.startswith(BODY_START))
    hi = next((i for i, l in enumerate(lines) if l.startswith(BODY_END)),
              len(lines))
    return lines[lo + 1:hi]


def parse():
    lines = _body_lines()
    blocks, i, eq_n = [], 0, 0
    title = abstract = keywords = None
    while i < len(lines):
        ln = lines[i].rstrip()
        s = ln.strip()

        # --- 除外 ---
        if not s or s == '---' or s.startswith('>') or s.startswith('（英:'):
            i += 1
            continue

        # --- タイトル / Abstract / Keywords ---
        if s.startswith('**Title**:'):
            title = s.split(':', 1)[1].strip()
            i += 1
            continue
        if s.startswith('**Abstract'):
            i += 1
            while i < len(lines) and not lines[i].strip():
                i += 1
            abstract = lines[i].strip()
            i += 1
            continue
        if s.startswith('**Keywords**:'):
            keywords = s.split(':', 1)[1].strip()
            blocks += [('title', title), ('authors', AUTHORS),
                       ('affil', AFFIL), ('abstract', abstract),
                       ('keywords', 'Keywords:', keywords)]
            i += 1
            continue

        # --- 見出し ---
        if s.startswith('### '):
            blocks.append(('h2', s[4:].strip()))
            i += 1
            continue
        if s.startswith('## '):
            blocks.append(('h1', s[3:].strip()))
            i += 1
            continue

        # --- 図 ---
        m = re.match(r'!\[(.*)\]\((.*)\)\s*$', s, re.S)
        if m:
            cap, path = m.group(1), m.group(2)
            blocks.append(('fig', os.path.basename(path), cap, 'full', 1.0))
            i += 1
            continue

        # --- 数式 ---
        if s == '$$':
            buf = []
            i += 1
            while i < len(lines) and lines[i].strip() != '$$':
                buf.append(lines[i].strip())
                i += 1
            i += 1
            eq_n += 1
            lab = re.search(r'\((\d)\)', ' '.join(buf))
            blocks.append(('eq', f'EQ{eq_n}', f'({lab.group(1) if lab else eq_n})'))
            continue

        # --- 表（**表 N.** キャプション の次に | 行が続く）---
        m = re.match(r'\*\*表\s*(\d+)\.\*\*\s*(.*)$', s)
        if m:
            cap = f'表 {m.group(1)}. {m.group(2)}'
            j = i + 1
            while j < len(lines) and not lines[j].strip().startswith('|'):
                j += 1
            rows = []
            while j < len(lines) and lines[j].strip().startswith('|'):
                cells = [c.strip() for c in lines[j].strip().strip('|').split('|')]
                if not all(re.fullmatch(r':?-{2,}:?', c) for c in cells):
                    rows.append(cells)
                j += 1
            n = len(rows[0])  # 先頭列 0.28、残りを均等割り（表1 は 2 列、表2 は 4 列）
            blocks.append(('table', rows[0], rows[1:],
                           [0.28] + [0.72 / (n - 1)] * (n - 1), cap))
            i = j
            continue

        # --- 参考文献 ---
        if re.match(r'^\[\d+\]\s', s):
            blocks.append(('ref', s))
            i += 1
            continue

        # --- 箇条書き / 本文 ---
        if s.startswith('- '):
            blocks.append(('p', '・' + s[2:]))
        else:
            blocks.append(('p', s))
        i += 1
    return blocks


def resolve_eqs(blocks):
    out = []
    for b in blocks:
        if b[0] == 'eq':
            omml = builder.omml_eq1() if b[1] == 'EQ1' else builder.omml_eq2()
            out.append(('eq', omml, b[2]))
        else:
            out.append(b)
    return out


def page_count(path):
    """Word で開いてページ数を返す（Windows + Word が要る）。"""
    import win32com.client
    word = win32com.client.Dispatch('Word.Application')
    word.Visible = False
    try:
        doc = word.Documents.Open(os.path.abspath(path), ReadOnly=True)
        doc.Repaginate()
        n = doc.ComputeStatistics(2)  # wdStatisticPages
        doc.Close(False)
        return n
    finally:
        word.Quit()


if __name__ == '__main__':
    blocks = parse()
    builder.build(TEMPLATE, OUT, resolve_eqs(blocks),
                  east_asia='ＭＳ 明朝', fig_dir=FIG_DIR, demote_bold=False)
    kinds = {}
    for b in blocks:
        kinds[b[0]] = kinds.get(b[0], 0) + 1
    print(f'built: {OUT}')
    print('blocks:', ', '.join(f'{k}={v}' for k, v in sorted(kinds.items())))
    if '--pages' in sys.argv:
        print(f'pages: {page_count(OUT)}')
