# -*- coding: utf-8 -*-
"""PowerPoint COM で pptx を PNG 書き出し（レイアウト検証用）。"""
import os, sys, glob
import win32com.client

HERE = os.path.dirname(os.path.abspath(__file__))
PPTX = os.path.join(HERE, "APIEMS2026_ゼミ発表.pptx")
OUTDIR = os.path.join(HERE, "_preview")
os.makedirs(OUTDIR, exist_ok=True)
for f in glob.glob(os.path.join(OUTDIR, "*.png")):
    os.remove(f)

app = win32com.client.Dispatch("PowerPoint.Application")
pres = app.Presentations.Open(PPTX, WithWindow=False)
pres.Export(OUTDIR, "PNG", 1600, 900)
pres.Close()
app.Quit()
print("exported to", OUTDIR)
print(sorted(os.listdir(OUTDIR)))
