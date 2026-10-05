# -*- coding: utf-8 -*-
"""汇总 _v18_cell.py 的四个格子并出分析表。"""
import sys
try:
    sys.stdout.reconfigure(errors="replace")   # Windows 控制台是 GBK，- ≤ 这类字符会直接抛错
except Exception:
    pass

from pathlib import Path

import numpy as np
import pandas as pd

try:
    from scipy import stats
except ImportError:
    stats = None

D = Path(__file__).parent
files = sorted((D / "results" / "v18").glob("a*_t*.csv"))
if not files:
    raise SystemExit("没有找到 _v18_a*_t*.csv")
df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
print("读入 %d 行，来自 %s" % (len(df), [f.name for f in files]))

RULES = ["A", "B", "C22", "Cprime", "C_star"]

print("\n" + "=" * 104)
print("(1)(2) 各规则标定后的 FAR：验证集（选点依据）与测试集（实际工作点），30 组均值")
print("=" * 104)
print("%-6s %-7s %-9s %9s %9s %9s %9s %9s" %
      ("amp", "目标", "口径", *RULES))
for amp in sorted(df.amp.unique(), reverse=True):
    for tf in sorted(df.target.unique()):
        s = df[(df.amp == amp) & (df.target == tf)]
        for kind, pre in (("验证集", "valfar_"), ("测试集", "far_"), ("测试偏差", "far_")):
            vals = [s[pre + t].mean() for t in RULES]
            if kind == "测试偏差":
                vals = [v - tf for v in vals]
                label = "偏差"
            else:
                label = kind
            print("%-6.2f %-7.2f %-9s %9.4f %9.4f %9.4f %9.4f %9.4f"
                  % (amp, tf, label, *vals))

print("\n(3) 配对检验：D = 测试 FAR(C22) - 测试 FAR(B)，正 = C22 工作点更宽松")
print("    另列验证集上的同一差值：若验证集已为正，说明是标定过程本身的问题，")
print("    而不是验证集→测试集的泛化波动。")
for amp in sorted(df.amp.unique(), reverse=True):
    for tf in sorted(df.target.unique()):
        s = df[(df.amp == amp) & (df.target == tf)]
        for tag, a, b in (("测试集", "far_C22", "far_B"), ("验证集", "valfar_C22", "valfar_B")):
            diff = (s[a] - s[b]).values
            if stats is not None and np.any(diff != 0):
                p = stats.wilcoxon(diff).pvalue
                tt = stats.ttest_rel(s[a], s[b]).pvalue
                print("  amp=%.2f 目标=%.2f %s: 均值 %+.4f 中位 %+.4f 正差 %d/%d  "
                      "Wilcoxon p=%.4f  t p=%.4f"
                      % (amp, tf, tag, diff.mean(), np.median(diff), int((diff > 0).sum()),
                         len(diff), p, tt))
            else:
                print("  amp=%.2f 目标=%.2f %s: 均值 %+.4f 正差 %d/%d"
                      % (amp, tf, tag, diff.mean(), int((diff > 0).sum()), len(diff)))

print("\n(4) FAR 残差与 FNR 优势的关系")
print("    D_FAR = FAR_C22 - FAR_B（正 = C22 更宽松）；D_FNR = FNR_B - FNR_C22（正 = C22 更好）")
for amp in sorted(df.amp.unique(), reverse=True):
    for tf in sorted(df.target.unique()):
        s = df[(df.amp == amp) & (df.target == tf)]
        d_far = (s.far_C22 - s.far_B).values
        print("  amp=%.2f 目标=%.2f  D_FAR 均值 %+.4f" % (amp, tf, d_far.mean()))
        for name, col in (("合并", "fnr_"), ("局部", "loc_"), ("耦合", "coup_")):
            d_fnr = (s[col + "B"] - s[col + "C22"]).values
            if stats is not None and np.std(d_far) > 0:
                pr = stats.pearsonr(d_far, d_fnr)
                sr = stats.spearmanr(d_far, d_fnr)
                slope = np.polyfit(d_far, d_fnr, 1)[0]
                print("    %-4s D_FNR %+.4f  Pearson %+.3f (p=%.3f)  Spearman %+.3f  "
                      "斜率 %+.3f  外推 D_FAR=0 后剩 %+.4f"
                      % (name, d_fnr.mean(), pr.statistic, pr.pvalue, sr.statistic,
                         slope, d_fnr.mean() - slope * d_far.mean()))
            else:
                print("    %-4s D_FNR %+.4f" % (name, d_fnr.mean()))
        hi = s[(s.far_C22 - s.far_B) > 0]
        lo = s[(s.far_C22 - s.far_B) <= 0]
        if len(hi) and len(lo):
            print("    C22 占 FAR 便宜的组 %d/%d：耦合 D_FNR %+.4f；其余组 %+.4f"
                  % (len(hi), len(s), (hi.coup_B - hi.coup_C22).mean(),
                     (lo.coup_B - lo.coup_C22).mean()))
