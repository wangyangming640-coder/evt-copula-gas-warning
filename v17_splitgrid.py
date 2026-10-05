# -*- coding: utf-8 -*-
"""⑥ 拆分 B/C22 的候选网格：网格共享到底有没有左右基准结果。

代码事实：B 与 C22 各自独立挑 argmin（calibrate_by_far 只看自己那条规则的
FAR 曲线），C22 的候选点不进入 B 的排序。**唯一共用的是网格的范围** ——
b_q_grid 下界 0.901 是为了让式(22) 能凑到目标 FAR 才从 0.98 降下来的。

所以本脚本给 B 与 C22 各发一套独立网格，看：
  (1) p_B 在窄网格 [0.98, 0.9995] 与宽网格 [0.90, 0.9999] 之间会不会动；
  (2) 有没有哪条规则被钉在网格下界（= 目标 FAR 在该范围内不可达）；
  (3) ΔFNR(B - C22) 在窄/宽/生产网格下是否稳定。

若 (1) 不动、(2) 不钉、(3) 稳定 —— 网格共享不构成 ⑥ 意义上的污染。
幅度 0.78 与 0.40，8 组种子，FAR*=0.05。
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
R = int(sys.argv[1]) if len(sys.argv) > 1 else 8
TF = 0.05
SEED0 = 20260924
GRIDS = {
    "窄[0.980,0.9995]": np.linspace(0.980, 0.9995, 250),
    "宽[0.900,0.9999]": np.linspace(0.900, 0.9999, 400),
}


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


def cands(X, grid, gpd, mk):
    q0 = max(float(m["q"]) for m in gpd)
    out = []
    for p in grid:
        if p <= q0:
            continue
        q = np.array([sp.gpd_quantile(m, float(p)) for m in gpd])
        if not np.all(np.isfinite(q)):
            continue
        out.append((float(p), q, mk(X, q)))
    return out


rows = []
for amp in (0.78, 0.40):
    for r in range(R):
        d = dict(sp.SimConfig().__dict__)
        d.update({"seed": SEED0 + r, "local_amp": amp, "coupled_amp": amp,
                  "bootstrap_gpd": 0, "bootstrap_metrics": 0})
        cfg = sp.SimConfig(**d)
        X, y, segs = sp.generate_dataset(cfg)
        data, split = sp.split_time_series(X, y)
        Xv, yv = data["val"]
        Xte, yte = data["test"]
        gpd, marg, cop, alt, _ = sp.fit_training_models(data["train"][0], cfg)
        assert cfg.b_q_grid is not None
        loc_m, cou_m = kind_masks(segs, split.val_end, len(yte))
        base = {"amp": amp, "rep": r + 1}

        # 生产网格（现实现）作为参照
        cand_prod, _ = sp.b_threshold_candidates(Xv, cfg, gpd)
        prod = {}
        for tag, mk in (("B", sp.rule_B), ("C22", sp.rule_C_count)):
            rr = sp.calibrate_by_far(Xv, yv, [(p, mk(Xv, q)) for p, q, a in cand_prod], TF, cfg)
            p_sel = float(rr[0]["param"])
            q_sel = next(q for p, q, a in cand_prod if p == p_sel)
            al = mk(Xte, q_sel)
            prod[tag] = (
                p_sel, rr[0]["metrics"]["far_per_hour"],
                sp.evaluate_rule(al, yte, cfg.interval_sec, cfg.early_tolerance_steps),
                sp.evaluate_rule(al, loc_m, cfg.interval_sec, cfg.early_tolerance_steps),
                sp.evaluate_rule(al, cou_m, cfg.interval_sec, cfg.early_tolerance_steps))

        for gname, grid in GRIDS.items():
            rec = dict(base, grid=gname)
            for tag, mk in (("B", sp.rule_B), ("C22", sp.rule_C_count)):
                cs = cands(Xv, grid, gpd, mk)
                rr = sp.calibrate_by_far(Xv, yv, [(p, a) for p, q, a in cs], TF, cfg)
                p_sel = float(rr[0]["param"])
                q_sel = next(q for p, q, a in cs if p == p_sel)
                qmin = min(p for p, q, a in cs)
                al = mk(Xte, q_sel)
                m = sp.evaluate_rule(al, yte, cfg.interval_sec, cfg.early_tolerance_steps)
                ml = sp.evaluate_rule(al, loc_m, cfg.interval_sec, cfg.early_tolerance_steps)
                mc = sp.evaluate_rule(al, cou_m, cfg.interval_sec, cfg.early_tolerance_steps)
                rec["p_%s" % tag] = p_sel
                rec["pmin_%s" % tag] = qmin
                rec["atfloor_%s" % tag] = int(abs(p_sel - qmin) < 1e-9)
                rec["valfar_%s" % tag] = rr[0]["metrics"]["far_per_hour"]
                rec["far_%s" % tag] = m["far_per_hour"]
                rec["fnr_%s" % tag] = m["event_fnr"]
                rec["loc_%s" % tag] = ml["event_fnr"]
                rec["coup_%s" % tag] = mc["event_fnr"]
            rec["p_B_prod"] = prod["B"][0]
            rec["p_C22_prod"] = prod["C22"][0]
            rec["far_B_prod"] = prod["B"][2]["far_per_hour"]
            rec["far_C22_prod"] = prod["C22"][2]["far_per_hour"]
            rec["fnr_B_prod"] = prod["B"][2]["event_fnr"]
            rec["fnr_C22_prod"] = prod["C22"][2]["event_fnr"]
            rec["loc_B_prod"] = prod["B"][3]["event_fnr"]
            rec["loc_C22_prod"] = prod["C22"][3]["event_fnr"]
            rec["coup_B_prod"] = prod["B"][4]["event_fnr"]
            rec["coup_C22_prod"] = prod["C22"][4]["event_fnr"]
            rec["n_prod"] = len(cand_prod)
            rows.append(rec)
        print("  amp=%.2f rep %d/%d 完成" % (amp, r + 1, R), flush=True)

df = pd.DataFrame(rows)
(D / "results").mkdir(exist_ok=True)
df.to_csv(D / "results" / "v17_splitgrid.csv", index=False, encoding="utf-8-sig")

print("\n" + "=" * 100)
print("(1) p_B 是否会因网格范围而移动（窄网格 / 宽网格 / 生产网格三者对照）")
print("=" * 100)
print("%-6s %-18s %10s %10s %10s %8s" % ("amp", "grid", "p_B", "p_C22", "p_B_prod", "B钉下界"))
for amp in (0.78, 0.40):
    for gname in GRIDS:
        s = df[(df.amp == amp) & (df.grid == gname)]
        print("%-6.2f %-18s %10.5f %10.5f %10.5f %8d"
              % (amp, gname, s.p_B.mean(), s.p_C22.mean(), s.p_B_prod.mean(),
                 int(s.atfloor_B.sum())))
    s = df[(df.amp == amp) & (df.grid == list(GRIDS)[0])]
    print("%-6.2f %-18s %10.5f %10.5f %10s %8s"
          % (amp, "生产 b_q_grid", s.p_B_prod.mean(), s.p_C22_prod.mean(), "—", "—"))

print("\n(2) 工作点是否被打到网格下界（= 目标 FAR 在该范围内不可达，等 FAR 对齐失效）")
for amp in (0.78, 0.40):
    for gname in GRIDS:
        s = df[(df.amp == amp) & (df.grid == gname)]
        for tag in ("B", "C22"):
            f = s[s["atfloor_%s" % tag] == 1]
            note = ("；钉住时 p=%.5f，验证集 FAR=%.4f，测试集 FAR=%.4f（目标 %.2f）"
                    % (f["p_%s" % tag].mean(), f["valfar_%s" % tag].mean(),
                       f["far_%s" % tag].mean(), TF)) if len(f) else ""
            print("  amp=%.2f %-18s %-4s 钉下界 %d/%d 次%s"
                  % (amp, gname, tag, len(f), len(s), note))

print("\n(3) ΔFNR(B - C22)：合并 / 局部 / 耦合，三种网格下是否稳定")
print("%-6s %-18s %12s %12s %12s %12s" % ("amp", "grid", "Δ合并", "Δ局部", "Δ耦合", "B.FAR/C22.FAR"))
for amp in (0.78, 0.40):
    for gname in GRIDS:
        s = df[(df.amp == amp) & (df.grid == gname)]
        print("%-6.2f %-18s %12.4f %12.4f %12.4f %6.4f/%.4f"
              % (amp, gname, (s.fnr_B - s.fnr_C22).mean(), (s.loc_B - s.loc_C22).mean(),
                 (s.coup_B - s.coup_C22).mean(), s.far_B.mean(), s.far_C22.mean()))
    s = df[df.amp == amp]
    print("%-6.2f %-18s %12.4f %12.4f %12.4f %6.4f/%.4f"
          % (amp, "生产（宽+裁q0）", (s.fnr_B_prod - s.fnr_C22_prod).mean(),
             (s.loc_B_prod - s.loc_C22_prod).mean(),
             (s.coup_B_prod - s.coup_C22_prod).mean(),
             s.far_B_prod.mean(), s.far_C22_prod.mean()))

print("\n已写出 results/v17_splitgrid.csv")
