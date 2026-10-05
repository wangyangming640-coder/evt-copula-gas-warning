# -*- coding: utf-8 -*-
"""汇总 _v20_cell.py 的 ratio × lag 八格（30 组）。"""
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
files = sorted((D / "results" / "v20").glob("r*_l*.csv"))
if not files:
    raise SystemExit("没有找到 _v20_r*_l*.csv")
df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
if "error" in df:
    bad = df[df.error.notna()]
    if len(bad):
        print("注意：%d 组失败" % len(bad))
    df = df[df.error.isna()]
print("读入 %d 行，来自 %d 个格子" % (len(df), len(files)))

RATIOS = sorted(df.ratio.unique())
LAGS = sorted(df.lag.unique())

print("\n" + "=" * 96)
print("每组危险段数（局部 / 耦合）与 C22 的危险事件数")
print("=" * 96)
print("%-8s %-6s %8s %9s %12s" % ("ratio", "lag", "n_local", "n_coupled", "n_danger_events"))
for r in RATIOS:
    for l in LAGS:
        s = df[(df.ratio == r) & (df.lag == l)]
        print("%-8.2f %-6d %8.1f %9.1f %12.1f"
              % (r, l, s.n_local.mean(), s.n_coupled.mean(), s.C22_all_n.mean()))

print("\n" + "=" * 96)
print("耦合段 ΔFNR(B - C22)，正 = C22 更好。n_coupled = 0 的格子无耦合段，Δ 无定义")
print("=" * 96)
print("%-8s %-6s %10s %10s %12s %10s" % ("ratio", "lag", "Δ耦合", "B耦合", "C22耦合", "p (Wilcoxon)"))
for r in RATIOS:
    for l in LAGS:
        s = df[(df.ratio == r) & (df.lag == l)]
        if s.n_coupled.mean() < 0.5:
            print("%-8.2f %-6d %10s %10s %10s %12s" % (r, l, "—", "—", "—", "无耦合段"))
            continue
        diff = (s.B_coup_fnr - s.C22_coup_fnr).values
        p = stats.wilcoxon(diff).pvalue if (stats is not None and np.any(diff != 0)) else float("nan")
        print("%-8.2f %-6d %+10.4f %10.4f %10.4f %12s"
              % (r, l, diff.mean(), s.B_coup_fnr.mean(), s.C22_coup_fnr.mean(),
                 ("%.4f" % p) if np.isfinite(p) else "全为 0"))

print("\n" + "=" * 96)
print("合并口径事件级 FNR（30 组均值）")
print("=" * 96)
print("%-8s %-6s %9s %9s %9s" % ("ratio", "lag", "B", "C22", "C_star"))
for r in RATIOS:
    for l in LAGS:
        s = df[(df.ratio == r) & (df.lag == l)]
        print("%-8.2f %-6d %9.4f %9.4f %9.4f"
              % (r, l, s.B_all_fnr.mean(), s.C22_all_fnr.mean(), s.C_star_all_fnr.mean()))

print("\n" + "=" * 96)
print("Early-FNR(20 min)，合并口径。看 C22 的及时性是否随 lag 劣化")
print("=" * 96)
print("%-8s %-6s %9s %9s %9s %12s" % ("ratio", "lag", "A", "B", "C22", "C22 - B"))
for r in RATIOS:
    for l in LAGS:
        s = df[(df.ratio == r) & (df.lag == l)]
        print("%-8.2f %-6d %9.4f %9.4f %9.4f %+12.4f"
              % (r, l, s.A_e20.mean(), s.B_e20.mean(), s.C22_e20.mean(),
                 (s.C22_e20 - s.B_e20).mean()))
