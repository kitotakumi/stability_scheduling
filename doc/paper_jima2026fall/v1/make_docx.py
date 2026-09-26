#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""JIMA2026 秋季大会 予稿の docx を生成する。

usage: python make_docx.py
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import content_ja as C  # noqa: E402  （APIEMS 版と同名なので先に読む）
import jima_builder  # noqa: E402

TEMPLATE = os.path.join(HERE, '..', 'jima_abst_template.docx')
OUT = os.path.join(HERE, 'JIMA2026fall_yokou.docx')
FIG_DIR = os.path.join(HERE, 'figures')

if __name__ == '__main__':
    jima_builder.build(TEMPLATE, OUT, C.BLOCKS, fig_dir=FIG_DIR)
