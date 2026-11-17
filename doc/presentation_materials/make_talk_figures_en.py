#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""APIEMS 2026 発表（英語版）の図を生成する（JIMA の図の生成スクリプトはそのまま残す）。

英語版のスライド（build_pptx_en.py）に貼る図を，v1（make_talk_figures.py）・v2（make_talk_figures_v2.py）と
同じ関数で描き，図の中の文字だけを talk_strings_en.py の対訳で英語にする。文字の置き換えは matplotlib の
Text に文字列が入るところ（Text.set_text）で行うので，軸の名前・凡例・注記のどれも同じ表で訳せる。
対訳のない日本語が残れば，最後に一覧を出して止める。文字は Segoe UI（スライドと同じ）。

出力は v1・v2 の図と同じ名前に _en を付けたもの（例：concept_hv_v2_en.png）。英語にすると文字の置き場所が
合わなくなる図は，仮説の 2 枚をこのファイルで描き直し，ta21L のフロントは保存の直前に帯の名前の大きさだけ直す（FIXES）。
usage: python make_talk_figures_en.py [--no-data]   # --no-data は模式図だけ作る
"""
import sys

import matplotlib.pyplot as plt
import matplotlib.text as mtext

import make_talk_figures as F
import make_talk_figures_v2 as F2
import talk_strings_en as S

FONT = 'Segoe UI'
_where = {'fn': None}   # いま描いている図の関数名（FIGURES_BY_FN の訳し分けに使う）
MISSING = []

_set_text_ja = mtext.Text.set_text


def _set_text(self, s):
    if isinstance(s, str) and S.CJK.search(s):
        t = S.FIGURES_BY_FN.get(_where['fn'], {}).get(s, S.FIGURES.get(s))
        if t is None:
            t = S.normalize(s)
        if t is None:
            MISSING.append((_where['fn'], s))
        else:
            s = t
    _set_text_ja(self, s)


mtext.Text.set_text = _set_text

_save_ja = F.save


def _fix_front(fig):
    """実データのフロント（fig_front_v2）：帯の名前は帯の幅（D < P50）に収まるよう 2 行・14 pt にし，見えている帯
    （横軸の左端〜P50）の中央に置く（v2 の位置 P50/2 のままだと，英語の 1 行目の右端が境界の点線に触れる）。"""
    for ax in fig.axes:
        for t in ax.texts:
            if t.get_text() == S.FIGURES['高安定領域']:
                t.set_fontsize(14)
                t.set_linespacing(1.05)
                t.set_x(t.get_position()[0] + ax.get_xlim()[0] / 2)   # 横軸の左端は負


FIXES = {'fig_front_v2': _fix_front}   # 保存の直前に当てる，図の関数ごとの手直し


def _save(fig, name):
    fix = FIXES.get(_where['fn'])
    if fix:
        fix(fig)
    _save_ja(fig, name[:-4] + '_en.png')


F.save = F2.save = _save
_style_ja = F.style


def _style():
    _style_ja()
    # 数式（$S_p$ など）も本文と同じ字形にする（スライドの数式は Segoe UI の斜体で組む）
    plt.rcParams.update({'font.family': [FONT, 'DejaVu Sans'], 'mathtext.fontset': 'custom',
                         'mathtext.rm': FONT, 'mathtext.it': FONT + ':italic',
                         'mathtext.bf': FONT + ':bold', 'mathtext.sf': FONT})


F.style = _style


def fig_concept_h(stage):
    """仮説のスライドの図（v1 の fig_concept_h）の英語版。探索構造の名前が日本語より長いので，置き方を変える：
    単点探索の名前は v1 と同じ帯の中（軌道の下）に 2 行で，帯の幅に収まる大きさで置き，多点探索の名前は
    演算子の名前（仮説 2 の図）と同じく帯の外から書き出す。"""
    fs = F.FS_SMALL_PLANE
    fig, ax = plt.subplots(figsize=(6.0, 3.95))
    F._plane(ax, fs=fs, band_label='高安定領域', front_label=False)
    ax.xaxis.label.set_size(fs)
    ax.yaxis.label.set_size(fs)
    if stage == 1:
        F._mem(ax, 0.9)
        F._ils(ax, 1.0)
        ax.text(0.02, 0.43, 'Trajectory-based\n(ILS)', fontsize=fs + 0.5, color=F.C_ILS, ha='left',
                va='top', fontweight='bold', linespacing=1.1)
        ax.text(F.BAND_HI + 0.02, 0.995, 'Population-based (Memetic)', fontsize=fs + 1.5, color=F.C_MEM,
                ha='left', va='bottom', fontweight='bold')
        out = 'concept_h1.png'
    else:
        F._mem(ax, 0.26)
        F._ils(ax, 0.30)
        F._pr(ax)
        ax.text(F.BAND_HI + 0.02, 0.995, '安定性誘導演算子：$S_p$ へ近づける', fontsize=fs + 0.5,
                color=F.C_MPR, ha='left', va='bottom', fontweight='bold')
        out = 'concept_h2.png'
    fig.tight_layout(pad=0.35)
    F.mark_schematic(fig)
    F.save(fig, out)


def run(fn, *a, **k):
    _where['fn'] = fn.__name__
    return fn(*a, **k)


if __name__ == '__main__':
    F.style()
    # 模式図は実データの図の文字の設定（DATA_RC）を当てる前に描く
    run(F2.fig_concept_pareto_v2)
    run(F2.fig_concept_hv_v2)
    run(F2.fig_concept_sp_v2)
    run(fig_concept_h, 1)
    run(fig_concept_h, 2)
    run(F.fig_concept_op)
    run(F.fig_struct, 'ils')
    run(F.fig_struct, 'mem')
    run(F.fig_metric_auc)
    if '--no-data' not in sys.argv:   # 実データの図（データの読み込みに数分かかる）
        M, S_ = F.load()
        run(F2.fig_front_v2, M, S_)
        run(F2.fig_highstab_v2, S_)
        run(F2.fig_highstab4_v2, S_)
        run(F2.fig_anytime_v2, M)
        run(F.fig_anytime, M, S_)   # 予備の代表 2 シナリオの推移
    if MISSING:
        print('対訳のない日本語（talk_strings_en.py の FIGURES に足す）:')
        for where, s in dict.fromkeys(MISSING):
            print(f'  [{where}] {s!r}')
        sys.exit(1)
    print('done.')
