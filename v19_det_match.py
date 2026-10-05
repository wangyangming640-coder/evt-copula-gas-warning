# -*- coding: utf-8 -*-
"""严格共同 FAR 的敏感性分析：在测试集 DET 曲线上按同一 FAR 插值比较。

v18 若确认 C22 的实测 FAR 系统性偏高，则「C22 耦合优势」可能部分来自工作点更宽松。
本脚本把 FAR 残差彻底消掉：对每条规则扫完整候选网格，得到测试集上的 (FAR, FNR)
曲线，再做单调包络 + 线性插值，在**同一个 FAR** 处取 FNR。

口径说明：插值点由测试集曲线确定，属 oracle 对齐，**不是**可部署的标定流程；
它的用途只是回答「若 FAR 严格对齐，排序是否还在」，与正文的验证集标定互不替代。

对照两个 FAR 水平：0.05（正文基准）与 0.01。
幅度 0.78 / 0.40，30 组种子。
"""
import sys
try:
    sys.stdout.reconfigure(errors="replace")   # Windows 控制台是 GBK，- ≤ 这类字符会直接抛错
except Exception:
    pass

import sys
from pathlib import Path

import numpy as np
import pandas as pd

import strict_pipeline as sp

D = Path(__file__).parent
R = int(sys.argv[1]) if len(sys.argv) > 1 else 30
SEED0 = 20260924
FAR_LEVELS = [0.05, 0.01]
RULES = ["B", "C22"]


def kind_masks(segs, val_end, n_test):
    loc = np.zeros(n_test, dtype=bool)
    cou = np.zeros(n_test, dtype=bool)
    for s, e, k in segs:
        if s < val_end:
            continue
        tgt = loc if k == "local" else cou
        a, b = max(0, s - val_end), min(n_test, e - val_end)
        tgt[a:b] = True
    return loc, cou


def det_curve(X, seg, gpd, mk, cfg, y_masks):
    """扫候选网格 → (FAR, FNR) 点集，已做单调包络。返回 (far, fnr) 升序数组。"""
    q0 = max(float(m["q"]) for m in gpd)
    far, fnr = [], []
    for p in cfg.b_q_grid:
        if p <= q0:
            continue
        q = np.array([sp.gpd_quantile(m, float(p)) for m in gpd])
        if not np.all(np.isfinite(q)):
            continue
        al = mk(X, q)
        for j, ym in enumerate(y_masks):
            f = sp.evaluate_rule(al, ym, cfg.interval_sec, cfg.early_tolerance_steps)
            if j == 0:
                far.append(f["far_per_hour"])
            fnr.append(f["event_fnr"])
    n = len(y_masks)
    far = np.asarray(far)
    fnr = np.asarray(fnr).reshape(-1, n)
    return far, fnr


def interp_at(far, fnr, level):
    """单调包络后线性插值；曲线未达到该 FAR 时返回 nan。"""
    order = np.argsort(far)
    f, v = far[order], fnr[order]
    keep = np.concatenate([[True], np.diff(f) > 0])
    f, v = f[keep], v[keep]
    v = np.minimum.accumulate(v)          # FNR 随 FAR 单调不增
    if level < f[0] or level > f[-1]:
        return np.nan
    return float(np.interp(level, f, v))


(D / "results" / "v19").mkdir(parents=True, exist_ok=True)
OUT = D / "results" / "v19" / "det_match.csv"


def dump(rows):
    pd.DataFrame(rows).to_csv(OUT, index=False, encoding="utf-8-sig")


rows, done = [], set()
if OUT.exists():
    prev = pd.read_csv(OUT)
    rows = prev.to_dict("records")
    done = set((float(a), int(b)) for a, b in zip(prev.amp, prev.rep))
    print("续跑：已有 %d 组" % len(done), flush=True)

for amp in (0.78, 0.40):
    for r in range(R):
        if (amp, r + 1) in done:
            continue
        d = dict(sp.SimConfig().__dict__)
        d.update({"seed": SEED0 + r, "local_amp": amp, "coupled_amp": amp,
                  "bootstrap_gpd": 0, "bootstrap_metrics": 0})
        cfg = sp.SimConfig(**d)
        X, y, segs = sp.generate_dataset(cfg)
        data, split = sp.split_time_series(X, y)
        Xte, yte = data["test"]
        gpd, marg, cop, alt, _ = sp.fit_training_models(data["train"][0], cfg)
        loc_m, cou_m = kind_masks(segs, split.val_end, len(yte))
        masks = [yte, loc_m, cou_m]

        rec = {"amp": amp, "rep": r + 1}
        for tag, mk in (("B", sp.rule_B), ("C22", sp.rule_C_count)):
            far, fnr = det_curve(Xte, None, gpd, mk, cfg, masks)
            for j, name in enumerate(("all", "loc", "coup")):
                for lv in FAR_LEVELS:
                    rec["%s_%s_%g" % (tag, name, lv)] = interp_at(far, fnr[:, j], lv)
        rows.append(rec)
        dump(rows)
        print("  amp=%.2f rep %d/%d 完成" % (amp, r + 1, R), flush=True)

df = pd.DataFrame(rows)

try:
    from scipy import stats
except ImportError:
    stats = None

print("\n" + "=" * 100)
print("严格共同 FAR：在同一 FAR 处插值取 FNR（30 组均值），Δ = B - C22（正 = C22 更好）")
print("=" * 100)
print("%-6s %-6s %-9s %10s %10s %10s" % ("amp", "FAR*", "口径", "B", "C22", "Δ(B-C22)"))
for amp in (0.78, 0.40):
    s = df[df.amp == amp]
    for lv in FAR_LEVELS:
        for name, cn in (("all", "合并"), ("loc", "局部"), ("coup", "耦合")):
            b = s["B_%s_%g" % (name, lv)]
            c = s["C22_%s_%g" % (name, lv)]
            ok = b.notna() & c.notna()
            diff = (b - c)[ok]
            note = ""
            if stats is not None and len(diff) > 5:
                p = stats.wilcoxon(diff).pvalue if np.any(diff != 0) else 1.0
                note = "  p=%.4f (n=%d)" % (p, len(diff))
            print("%-6.2f %-6.2f %-9s %10.4f %10.4f %10.4f%s"
                  % (amp, lv, cn, b[ok].mean(), c[ok].mean(), diff.mean(), note))

print("\n可插值组数（曲线未覆盖该 FAR 的组会被丢弃，如实列出）")
for amp in (0.78, 0.40):
    s = df[df.amp == amp]
    for lv in FAR_LEVELS:
        b = s["B_all_%g" % lv]
        c = s["C22_all_%g" % lv]
        print("  amp=%.2f FAR*=%.2f: B 可插值 %d/%d，C22 可插值 %d/%d"
              % (amp, lv, int(b.notna().sum()), len(s),
                 int(c.notna().sum()), len(s)))

print("\n已写出 results/v19/det_match.csv")
