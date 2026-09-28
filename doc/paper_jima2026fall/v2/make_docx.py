#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""JIMA2026 秋季大会 予稿の docx を生成する。

usage: python make_docx.py [--pdf]
       --pdf を付けると Word 経由で PDF も書き出し、頁数を表示する
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import content_ja as C  # noqa: E402  （APIEMS 版と同名なので先に読む）
import jima_builder  # noqa: E402

TEMPLATE = os.path.join(HERE, '..', 'jima_abst_template.docx')
OUT = os.path.join(HERE, '鬼頭拓海_JIMA2026fall_yokou.docx')
PDF = os.path.join(HERE, '鬼頭拓海_JIMA2026fall_yokou.pdf')
FIG_DIR = os.path.join(HERE, 'figures')


def export_pdf(docx_path, pdf_path):
    """Word COM で PDF 化し、頁数を返す。"""
    import win32com.client as win32
    word = win32.gencache.EnsureDispatch('Word.Application')
    word.Visible = False
    word.DisplayAlerts = 0
    doc = None
    try:
        doc = word.Documents.Open(os.path.abspath(docx_path), ReadOnly=False,
                                  AddToRecentFiles=False)
        doc.ExportAsFixedFormat(OutputFileName=os.path.abspath(pdf_path),
                                ExportFormat=17)  # wdExportFormatPDF
        pages = doc.ComputeStatistics(2)  # wdStatisticPages
    finally:
        if doc is not None:
            doc.Close(False)
        word.Quit()
    return pages


if __name__ == '__main__':
    jima_builder.build(TEMPLATE, OUT, C.BLOCKS, fig_dir=FIG_DIR)
    if '--pdf' in sys.argv[1:]:
        pages = export_pdf(OUT, PDF)
        print(' ->', PDF, f'({pages} pages)')
