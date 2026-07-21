# -*- coding: utf-8 -*-
"""元の概念図SVGを Chrome headless で忠実にPNG化する（2倍解像度・白背景）。
seminar/ の SVG をそのまま使う（作り直さない）。出力: assets/*.png
"""
import os, re, subprocess, tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
SEM = os.path.join(os.path.dirname(HERE), "seminar")
ASSETS = os.path.join(HERE, "assets")
os.makedirs(ASSETS, exist_ok=True)

CHROME = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
if not os.path.exists(CHROME):
    CHROME = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"

SCALE = 2  # device pixel ratio


def viewbox_of(svg):
    m = re.search(r'viewBox="([\d.\s]+)"', svg)
    if m:
        parts = [float(x) for x in m.group(1).split()]
        if len(parts) == 4:
            return parts[2], parts[3]
    return 1000.0, 600.0


def strip_dark(svg):
    """@media (prefers-color-scheme: dark) { ... } ブロックを除去してライト固定にする。"""
    key = "@media (prefers-color-scheme: dark)"
    i = svg.find(key)
    while i != -1:
        b = svg.find("{", i)
        depth, j = 0, b
        while j < len(svg):
            if svg[j] == "{":
                depth += 1
            elif svg[j] == "}":
                depth -= 1
                if depth == 0:
                    break
            j += 1
        svg = svg[:i] + svg[j + 1:]
        i = svg.find(key)
    return svg


def convert(svg_name, out_name=None):
    svg_path = os.path.join(SEM, svg_name)
    out_name = out_name or (os.path.splitext(svg_name)[0] + ".png")
    out_path = os.path.join(ASSETS, out_name)
    with open(svg_path, "r", encoding="utf-8") as f:
        svg = strip_dark(f.read())          # ダークモード配色を除去
    w, h = viewbox_of(svg)
    # ライト固定した SVG を一時ファイルに書き出して参照
    sfd, svgpath = tempfile.mkstemp(suffix=".svg", dir=ASSETS)
    with os.fdopen(sfd, "w", encoding="utf-8") as f:
        f.write(svg)
    svg_uri = "file:///" + svgpath.replace("\\", "/")
    html = (
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<meta name='color-scheme' content='only light'>"
        "<style>*{margin:0;padding:0}html,body{background:#ffffff}"
        f"img{{width:{int(w)}px;height:{int(h)}px;display:block}}</style></head>"
        f"<body><img src='{svg_uri}'></body></html>"
    )
    fd, htmlpath = tempfile.mkstemp(suffix=".html", dir=ASSETS)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(html)
    try:
        cmd = [
            CHROME, "--headless=new", "--disable-gpu", "--hide-scrollbars",
            f"--force-device-scale-factor={SCALE}",
            f"--screenshot={out_path}",
            f"--window-size={int(w)},{int(h)}",
            "file:///" + htmlpath.replace("\\", "/"),
        ]
        subprocess.run(cmd, check=True, capture_output=True, timeout=60)
        print("wrote", out_name, f"({int(w*SCALE)}x{int(h*SCALE)})",
              os.path.getsize(out_path), "bytes")
    finally:
        os.remove(htmlpath)
        os.remove(svgpath)


if __name__ == "__main__":
    convert("ga_vs_ils_pr_rescheduling.svg", "concept_3struct.png")
    convert("direct_swap_focused.svg", "concept_direct_swap.png")
    convert("ga_vs_ils_rescheduling.svg", "concept_2struct.png")
    convert("N5neighborhood.svg", "concept_n5.png")
