# -*- coding: utf-8 -*-
"""学会発表（JIMA2026 秋・15 分）の pptx の v2：結果の章（3. 数値実験）を組み直した版を生成する。

v1（build_pptx.py）はそのまま残し，その build() で組む。結果の章のスライド関数と図だけを差し替えるので，
v1 の他のスライドを直せば v2 にもそのまま反映される。

v1 からの変更:
  - 目次：1 章の項目に「Pareto フロントと HV」を足す
  - 1 章：二目的最適化と高安定 HV の間に「Pareto フロントと HV」のスライドを足す（両目的とも最小化であるため
    フロントは左下にあるほど良く，その品質を HV で測る）。Pareto フロントの曲線とラベルはこのスライドの図で付ける
  - 着目点と評価指標：HV の定義は前のスライドに任せて本文から外し，図から Pareto フロントの曲線とラベルを外す
  - 研究の出発点：図の★の注記（探索せずに得られる）を外し，★のラベルを「初期スケジュール S_p」にする
  - 実験設定：予備の実験設定の詳細の表を本編に移し（比較手法の行は除く），問題の規模（n × m）を足す。
    予備の実験設定の詳細は外す。妥当性は高安定領域の境界とパラメータの 2 つにする
    （計算予算は変えておらず，確かめたのは収束なので予備の結論の妥当性に任せる）
  - 安定性誘導演算子：swap の説明を「1 作業以上が S_p の順位に戻り，d が減る」にする（D は同じままの swap もある）
  - 高安定 HV（全 8 シナリオ）：仮説の判定の p を Holm 補正後の p_adj と書く
  - Pareto フロント：代表を ta21L の 1 シナリオにし，(a) 演算子なし と (b) PR を組み込む の 2 パネルに分ける。
    (b) には (a) の 2 手法を薄い帯で重ねる
  - 高安定 HV（全 8 シナリオ・演算子の効果）：図から有意水準の * を外す（判定は本文の p と δ で示す）。
    演算子の効果では，ILS の la36L・ta21L の向上の有意性を Holm 補正後の判定で書く
  - AUC と全域 HV：ta21L の推移の図と全 8 シナリオの読み取りを 1 枚にまとめる（推移を読む順に AUC → 全域 HV）。
    v1 の代表 2 シナリオの推移と集計表のスライドは予備に移す
  - まとめ：仮説 1 の「→ 現場にとっての意味」を外し，手法の選び方を「〜を重視する現場では」の形にする

前提:  python make_talk_figures_v2.py を先に実行（assets/ に v2 の図を生成）。
再生成:  python build_pptx_v2.py          （pptx と同名の PDF も書き出す．PDF 不要なら --no-pdf）
"""
import copy
import os
import sys

import build_pptx as B
from build_pptx import (ML, CW, SW, TOP, BOT, FS_BODY, FS_HEAD, FS_NOTE, BLUE, MUTE, INK, WHITE, CARDLINE,
                        MSO_ANCHOR, slide_frame, pic, _bul, _text, _divider, _paras, _cite_note, add_textbox,
                        _rrect, _boxtext, render_table)

B.OUT = os.path.join(B.HERE, "JIMA2026fall_発表_v2.pptx")

# 高安定 HV の 2 枚と研究の出発点は v1 のスライド関数のまま，貼る図だけを v2 の図に替える
V2_ASSETS = {"highstab_all8.png": "highstab_all8_v2.png", "highstab_4.png": "highstab_4_v2.png",
             "concept_sp.png": "concept_sp_v2.png"}
_asset_v1 = B.asset
B.asset = lambda name: _asset_v1(V2_ASSETS.get(name, name))

# 予備に移す v1 のスライド関数（差し替える前に取っておく）
_s_anytime_v1, _s_union_auc_v1, _s_backup_results_v1 = B.s_anytime, B.s_union_auc, B.s_backup_results


def s_pareto():
    """1 章で，二目的最適化（背景のスライド）と高安定 HV（着目点と評価指標のスライド）の間に置く。
    両目的とも最小化であるため Pareto フロントは左下にあるほど良く，その良さを HV で測ることを，
    模式図で示す。図は次の着目点と評価指標のスライドと同じ非劣解と HV で，図と本文の配置も同じ。"""
    s = slide_frame("Pareto フロントと hypervolume（HV）",
                    "両目的とも最小化であるため Pareto フロントは左下にあるほど良く，"
                    "本研究はその品質を HV で測る",
                    "1. 研究背景と仮説")
    _w, h = pic(s, "concept_pareto_v2.png", ML - 0.05, TOP, w=5.8)
    tx = ML + 5.9
    tw = SW - ML - tx
    # 「● 事実 → 帰結」の 2 組を，組の中は詰め，組の間に区切り線を引く。
    # 語の途中で折り返さないよう，区切りのよい所で改行する
    tb, tf = add_textbox(s, tx, TOP + 0.1, tw, 2.2)
    _paras(tf, [
        ("● 二目的最適化では，解が 1 つに定まらない．\n一方の目的を悪化させずには他方を\n"
         "改善できない解を**非劣解**，その集合を\n**Pareto フロント**と呼ぶ", FS_BODY, 0),
        ("→ 両目的とも最小化であるため，\nPareto フロントは**左下にあるほど良い**", FS_BODY, 14),
    ])
    _divider(s, tx, TOP + 2.45, tw)
    tb, tf = add_textbox(s, tx, TOP + 2.8, tw, h - 2.8)
    _paras(tf, [
        ("● **hypervolume（HV）**[[5]]：非劣解が支配する\n領域（参照点まで）の面積．\n"
         "値が大きいほど Pareto フロントの品質が高い", FS_BODY, 0),
        ("→ 本研究は，Pareto フロントの品質を HV で測る", FS_BODY, 14),
    ])
    _cite_note(s, [5])


def s_focus():
    """v1 の着目点と評価指標から，HV の定義を外したもの（定義は前の Pareto フロントと HV のスライドにある）。
    本文は「● 求められる解（変更量が少ない解） → 高安定領域の HV（高安定 HV）に着目して評価」の 1 組にする。
    図は Pareto フロントの曲線とラベルを外した v2 の図（曲線とラベルは前のスライドで示す）。"""
    s = slide_frame("本研究の着目点と評価指標：高安定 HV",
                    "スケジュール変更コストが無視できない場面では高安定領域の解が求められるため，"
                    "本研究は高安定領域に限った HV（高安定 HV）で手法を評価する",
                    "1. 研究背景と仮説")
    _w, h = pic(s, "concept_hv_v2.png", ML - 0.05, TOP, w=5.8)
    tx = ML + 5.9
    tw = SW - ML - tx
    # 語の途中で折り返さないよう，区切りのよい所で改行する
    tb, tf = add_textbox(s, tx, TOP + 0.1, tw, h - 0.1)
    _paras(tf, [
        ("● スケジュール変更コストが無視できない\n場面で求められるのは，**安定性の高い解**，\n"
         "すなわち Pareto フロント上の変更量が少ない解", FS_BODY, 0),
        ("→ 本研究は，**高安定領域**（全手法の非劣解のうち，\n変更量が中央値より小さい範囲）の HV に\n"
         "着目する．この HV を**高安定 HV** と呼び，\nこれで手法を評価する", FS_BODY, 14),
    ])


def _s_pareto_and_focus():
    s_pareto()
    s_focus()


def s_setup():
    """実験設定。v1 で予備にあった設定の表を本編に置き（読み上げず，聞き手が参照する表），問題の規模
    （ジョブ数 n × 機械数 m）を足す。比較手法は比較する手法と仮説の判定のスライドにあるので行を置かない。
    再スケジュール率（最適化の対象になる作業の割合）は，結果との関係に結論が出ていないので数値を出さず，
    S・M・L の意味は「対象作業が少ない順」とだけ書く。重み λ = 1.0 を除く理由は予備（スカラー化）にある。"""
    s = slide_frame("実験設定",
                    "5 つのベンチマーク問題に単一作業の遅延を与え，8 シナリオを構成",
                    "3. 数値実験")
    # 語の途中で折り返さないよう，区切りのよい所で改行する
    rows = [
        ["項目", "設定"],
        ["問題（$n$ × $m$）", "mt10（10×10），la21（15×10），la36・la40（15×15），ta21（20×20）"],
        ["外乱", "単一作業の完了遅延（$Δ$ = 60〜148，対象作業の処理時間の約 0.9〜1.5 倍）"],
        ["シナリオ", "8 つ：la36S・la36M・la36L，ta21S・ta21L，mt10，la21，la40\n"
                    "la36 と ta21 は同一問題・同一 $S_p$ のまま遅延の発生時刻を変え，\n"
                    "対象作業が少ない順に S・M・L"],
        ["初期スケジュール $S_p$", "500 世代の GA で生成"],
        ["重み $λ$", "0, 0.1, …, 0.9 の 10 点"],
        ["計算予算", "ILS 系 3000 反復，GA・Memetic 系 500 世代"],
        ["試行", "各手法・各重みで 10 試行（全乱数シード固定）"],
    ]
    render_table(s, rows, ML, TOP + 0.05, CW, col_w=[2.9, CW - 2.9], size=FS_BODY, align=['l', 'l'],
                 row_h=0.46)
    ty = TOP + 0.05 + 0.46 * 7 + 0.34 * 3 + 0.3   # 表の下端（シナリオの行は 3 行）から間をあける
    _bul(s, ML, ty, CW, [
        "**妥当性**：高安定領域の境界・パラメータを変えても，結論は変わらない",
    ], size=FS_BODY, gap=12)
    _text(s, ML, ty + 0.5, CW, 0.36, "手法のパラメータと妥当性の確認は予備スライドに掲載", FS_NOTE, color=MUTE)


def s_front():
    """代表シナリオ ta21L の Pareto フロントを，演算子なし（仮説 1 の比較）と PR を組み込んだ場合（仮説 2 の比較）の
    2 パネルで見せる。(b) に薄く重ねた演算子なしの帯で，Memetic の移動と，演算子なしの ILS との重なりを読む。"""
    s = slide_frame("高安定 HV の比較：代表シナリオ ta21L の Pareto フロント",
                    "ta21L では，演算子なしの Memetic は高安定領域にほとんど入らず，"
                    "PR を組み込むと演算子なしの ILS と重なる",
                    "3. 数値実験")
    w = 12.0
    _w, fh = pic(s, "front_ta21L_v2.png", ML + (CW - w) / 2, TOP, w=w)
    _bul(s, ML, TOP + fh + 0.25, CW, [
        "(a) 演算子なし：Memetic は高安定領域にほとんど入らず，入った解も $MS$ で ILS に劣る",
        "(b) PR を組み込む：Memetic も高安定領域に広がり，演算子なしの ILS（薄い橙）と重なる",
    ], size=FS_BODY, gap=8)


def s_anytime():
    """AUC と全域 HV の結果を 1 枚にまとめる。左は代表シナリオ ta21L（フロントと同じ）の全域 HV の推移で，
    立ち上がり（AUC）と最終値（全域 HV）の両方を読む。右は推移を読む順に AUC → 全域 HV と並べ，指標ごとに
    「ta21L（左の図）の読み取り → 全 8 シナリオ」の順に書く。全 8 シナリオでは ILS 系・Memetic（演算子なし）・
    Memetic＋演算子の 3 つの，最良に並ぶシナリオ数を並べる。
    集計表と代表 2 シナリオ（ta21S・la36L）の推移は予備に置く。"""
    s = slide_frame("AUC と全域 HV の比較",
                    "最良の手法は評価指標によって異なる：AUC は ILS，全域 HV は Memetic＋演算子",
                    "3. 数値実験")
    # 右の列は 1 章の図と本文のスライドと同じ位置・幅にする（図を少し縮めて本文の行を確保する）
    fw = 5.5
    _w, fh = pic(s, "anytime_ta21L_v2.png", ML, TOP, w=fw)
    rx = ML + 5.9
    rw = SW - ML - rx
    _text(s, rx, TOP, rw, 0.42, "AUC", FS_HEAD, color=BLUE, bold=True, anchor=MSO_ANCHOR.MIDDLE)
    # 語の途中で折り返さないよう，区切りのよい所で改行する
    _bul(s, rx, TOP + 0.5, rw, [
        "**ta21L** では，ILS が最も早く立ち上がる．\nMemetic は遅れ，演算子でさらに遅れる",
        "**全 8 シナリオ**では，ILS 系は 6，Memetic は 2，\nMemetic＋演算子は 1 シナリオで最良に並ぶ",
    ], size=FS_BODY, gap=6)
    _divider(s, rx, TOP + 2.05, rw)   # AUC と全域 HV の読み取りの間
    _text(s, rx, TOP + 2.22, rw, 0.42, "全域 HV", FS_HEAD, color=BLUE, bold=True, anchor=MSO_ANCHOR.MIDDLE)
    _bul(s, rx, TOP + 2.72, rw, [
        "**ta21L** では，演算子で Memetic の最終値が\n上がり，最も高くなる．\nILS は演算子でほぼ変わらない",
        "**全 8 シナリオ**では，Memetic＋演算子は 7，\nILS 系は 4，Memetic は 1 シナリオで最良に並ぶ",
    ], size=FS_BODY, gap=6)
    # 「最良に並ぶ」の定義は図の下に置く（右の列は本文で下端まで使う）
    _text(s, ML, BOT - 0.36, fw, 0.36, "最良に並ぶ：中央値が最良手法の中央値から 2% 以内", FS_NOTE, color=MUTE)


def s_band():
    """v1 の演算子の効果から，ILS の la36L・ta21L の向上の有意性を，15 枚目で示した検定の手順（全シナリオに
    わたる同じ比較を Holm 法で補正）に合わせて書いたもの。la36L・ta21L はどちらも演算子で向上するが，
    Holm 補正後も有意なのは ta21L だけ（la36L は p_adj = 0.091）。数は公開 CSV と同じ統計
    （最良値を 1 とした中央値，PR-k と PR の両方を含む範囲）。"""
    s = slide_frame("高安定 HV の比較：演算子の効果",
                    "演算子を組み込むと両構造とも最良値の付近に達する．ILS は組み込む前から最良値の付近にある",
                    "3. 数値実験")
    w = 12.0
    _w, fh = pic(s, "highstab_4.png", ML + (CW - w) / 2, TOP, w=w)
    cy = TOP + fh
    _bul(s, ML, cy + 0.5, CW, [
        "**Memetic**：演算子を組み込むと，最良値の 0〜48% から 90〜100% まで大幅に向上",
        "**ILS**：演算子なしで，8 シナリオ中 6 つがすでに最良値の 98.5% 以上．\n"
        "最良値に届いていなかった la36L・ta21L だけ，演算子で向上（94% → 100%）．\n"
        "Holm 補正後も有意なのは ta21L",
    ], size=FS_BODY, gap=8)
    # 「→」は箇条書きの直後に，他のスライドの「→」と同じ間隔で続ける（ILS の項目が 1 行増えたぶん下げる）
    _text(s, ML, cy + 2.10, CW, 0.42,
          "→ 演算子の効果は，**探索構造だけでは最良値に届かない場合に，その差を埋める形で現れる**",
          FS_BODY, color=INK, emph=BLUE, anchor=MSO_ANCHOR.MIDDLE)


def s_conclusion():
    """v1 のまとめから，仮説 1 の「→ 現場にとっての意味」を外し（手法の選び方と食い違わないように），
    現場とのつながりは手法の選び方で「〜を重視する現場では」の形で言う。"""
    s = slide_frame("まとめ",
                    "高安定 HV は単点探索が上回り，多点探索も安定性誘導演算子を組み込めば同程度に達する",
                    "4. 結論")
    # 語の途中で折り返さないよう，区切りのよい所で改行する
    items = [("仮説 1 を支持", "高安定 HV は**単点探索（ILS）**が多点探索（Memetic）を大きく上回る"),
             ("仮説 2 を支持", "安定性誘導演算子は，**多点探索の高安定 HV を単点探索と同程度まで向上**させる"),
             ("手法の選び方", "Pareto フロント全体の品質を重視する現場では **多点探索＋安定性誘導演算子**\n"
                              "高安定領域の品質を重視する現場では **単点探索** か **多点探索＋安定性誘導演算子**\n"
                              "短い計算時間で解を得る必要がある現場では **単点探索**")]
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
              anchor=MSO_ANCHOR.MIDDLE, emph=BLUE)
        y += hh + gap


def _replacing(fn, pairs):
    """v1 のスライド関数 fn を，本文（_bul の項目と _text の文字列）の一部だけ置き換えて組む関数にする。
    置き換え元が見つからなければ止める（v1 の文言が変わったときに黙って素通りしないように）。"""
    def wrapped():
        bul, text = B._bul, B._text
        hit = set()

        def rep(t):
            for old, new in pairs:
                if old in t:
                    hit.add(old)
                    t = t.replace(old, new)
            return t

        B._bul = lambda s, x, y, w, items, *a, **k: bul(
            s, x, y, w, [(i[0], rep(i[1])) if isinstance(i, tuple) else rep(i) for i in items], *a, **k)
        B._text = lambda s, x, y, w, h, t, *a, **k: text(s, x, y, w, h, rep(t), *a, **k)
        try:
            fn()
        finally:
            B._bul, B._text = bul, text
        missing = [old for old, _ in pairs if old not in hit]
        if missing:
            raise RuntimeError(f"{fn.__name__}: 置き換え元が見つからない: {missing}")
    return wrapped


# 安定性誘導演算子：1 回の swap で S_p の順位に戻る作業は 1 つ以上（例の 2 回目は 2 つ戻る）で，
# 確実に減るのは順位が異なる作業の数 d（D は増えないが，同じままの swap もある）
_s_operators_v2 = _replacing(B.s_operators, [
    ("1 回の swap で 1 作業が $S_p$ の順位に戻り，$D$ が減る",
     "1 回の swap で 1 作業以上が $S_p$ の順位に戻り，$d$ が減る"),
])
# 仮説の判定の p は Holm 補正後の値（15 枚目の検定の手順と論文に合わせる）
_s_highstab_v2 = _replacing(B.s_highstab, [("全シナリオで $p$ < 0.001", "全シナリオで $p_{adj}$ < 0.001")])

# 1 章の目次に「Pareto フロントと HV」のスライドを足す。1 行に収まらないので，語の途中で折り返さないよう
# 項目の区切り（_SUB1_BREAK の後ろ）で改行する（v1 の chapter は項目の行を 1 つの run で組むので，組んだ後に分ける）
_SUB1 = "外乱と再スケジューリング・Pareto フロントと HV・高安定領域・高安定 HV・特殊性・先行研究・課題・仮説"
_SUB1_BREAK = "Pareto フロントと HV・"
B.CHAPTERS = [(n, t, _SUB1 if n == "1" else sub) for n, t, sub in B.CHAPTERS]
_chapter_v1 = B.chapter


def chapter(no):
    slide = _chapter_v1(no)
    if no == "1":
        for shp in slide.shapes:
            if not shp.has_text_frame:
                continue
            for p in shp.text_frame.paragraphs:
                if len(p.runs) == 1 and p.runs[0].text == _SUB1:
                    r = p.runs[0]
                    head, tail = _SUB1.split(_SUB1_BREAK)
                    r.text = head + _SUB1_BREAK
                    p.add_line_break()
                    r2 = p.add_run()
                    r2.text = tail
                    r2._r.insert(0, copy.deepcopy(r._r.rPr))   # 文字の大きさ・色は 1 行目と同じ
                    return slide
        raise RuntimeError("chapter: 1 章の項目の行が見つからない")
    return slide


def _as_backup(fn):
    """v1 の本編のスライド関数で，題名に［予備］を付け，フッターを「予備」にしたスライドを作る。"""
    frame = B.slide_frame
    B.slide_frame = lambda title, key, section: frame("［予備］" + title, None, "予備")
    try:
        fn()
    finally:
        B.slide_frame = frame


def s_backup_results():
    _s_backup_results_v1()
    _as_backup(_s_anytime_v1)     # 代表 2 シナリオ（ta21S・la36L）の推移
    _as_backup(_s_union_auc_v1)   # 最良に並ぶシナリオ数の集計表


B.chapter = chapter
B.s_focus = _s_pareto_and_focus
B.s_setup = s_setup
B.s_front = s_front
B.s_operators = _s_operators_v2
B.s_highstab = _s_highstab_v2
B.s_band = s_band
B.s_anytime = s_anytime
B.s_conclusion = s_conclusion
B.s_union_auc = lambda: None      # 本編の集計表は予備に移す（s_backup_results で作る）
B.s_backup_setup = lambda: None   # 実験設定の詳細は本編の実験設定に移した
B.s_backup_results = s_backup_results


if __name__ == "__main__":
    B.build()
    if "--no-pdf" not in sys.argv:
        B.export_pdf(B.OUT)
