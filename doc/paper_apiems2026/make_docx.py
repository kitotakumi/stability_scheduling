#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""日本語版・英語版の APIEMS 原稿 docx を生成する。

usage: python make_docx.py [ja|en|both]
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import builder  # noqa: E402

TEMPLATE = os.path.join(HERE, '..', 'APIEMS FullPaperTemplate.docx')
FIG_DIR = os.path.join(HERE, 'figures')


def _resolve_eqs(blocks):
    out = []
    for b in blocks:
        if b[0] == 'eq':
            omml = builder.omml_eq1() if b[1] == 'EQ1' else builder.omml_eq2()
            out.append(('eq', omml, b[2]))
        else:
            out.append(b)
    return out


def make(lang):
    if lang == 'ja':
        import content_ja as C
        ea = 'ＭＳ 明朝'
        out = os.path.join(HERE, 'APIEMS2026_draft_ja.docx')
        demote_bold = False  # 日本語レビュー版は太字強調のまま（イタリック邦文は非慣用）
    else:
        import content_en as C
        ea = None
        out = os.path.join(HERE, 'APIEMS2026_manuscript_en.docx')
        demote_bold = True   # 投稿版はテンプレ準拠: 文中強調をイタリック化
    builder.build(TEMPLATE, out, _resolve_eqs(C.BLOCKS),
                  east_asia=ea, fig_dir=FIG_DIR, demote_bold=demote_bold)


if __name__ == '__main__':
    arg = sys.argv[1] if len(sys.argv) > 1 else 'both'
    if arg in ('ja', 'both'):
        make('ja')
    if arg in ('en', 'both'):
        make('en')
