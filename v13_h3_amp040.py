# -*- coding: utf-8 -*-
"""H3 在非饱和基准（amp=0.40）下的重跑。

H3 说 C 相对 B 的改善量随 λ_U 增大而增大。原结论（据 amp=0.78 基准）是方向相反，
且约 86% 的反向单调由「等 FAR 标定把 p_C 推高」解释。但 amp=0.78 落在饱和区
（FAR*=0.05 下幅度 ≥0.55 时 B 的事件级 FNR 恒为 0），H3 在那里同样无从检验。
这里把基准换到 amp=0.40 重做同一张表。

统计量：局部危险段（单点事件）上的 ΔFNR(B - C22)。
之所以限定局部段：耦合段上 B 会触底，混进来会把饱和效应掺进 H3。
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
from scipy import stats

import strict_pipeline as sp

D = Path(__file__).parent
R = int(sys.argv[1]) if len(sys.argv) > 1 else 8
LEVELS = [0.0, 0.50, 0.90]        # cross_strength，与论文 4.6 节同档
FIXED_PC = [0.98, 0.985, 0.99]    # 共同 p_C 水平。原表用的 0.955 在新基准下不可用：
                                  # 部分测点选中的 POT 阈值分位达到 0.975，低于它的分位点
                                  # 落在 GPD 外推区间之外，gpd_quantile 会直接报错。
AMP = 0.40
TF = 0.05
SEED0 = 20260924


def local_mask(segs, val_end, n_test):
    m = np.zeros(n_test, dtype=bool)
    for s, e, k in segs:
        if k == "local" and s >= val_end:
            m[max(0, s - val_end):min(n_test, e - val_end)] = True
    return m


rows = []
for cs in LEVELS:
    for r in range(R):
        base = sp.SimConfig()
        d = dict(base.__dict__)
        d.update({"seed": SEED0 + r, "cross_strength": cs,
                  "local_amp": AMP, "coupled_amp": AMP,
                  "bootstrap_gpd": 0, "bootstrap_metrics": 0})
        cfg = sp.SimConfig(**d)
        X, y, segs = sp.generate_dataset(cfg)
        data, split = sp.split_time_series(X, y)
        gpd, marg, cop, alt, _ = sp.fit_training_models(data["train"][0], cfg)
        Xv, yv = data["val"]
        Xte, yte = data["test"]
        cal = sp.validation_calibration(Xv, yv, cfg, gpd, marg, cop["theta"],
                                        cop["family"], target=TF)
        lm = local_mask(segs, split.val_end, len(yte))
        b = sp.rule_B(Xte, cal["B"]["thresholds"])
        fnr_b = sp.evaluate_rule(b, lm, cfg.interval_sec, cfg.early_tolerance_steps)["event_fnr"]

        rec = {"cross_strength": cs, "rep": r + 1,
               "pC_calibrated": cal["C22"]["p"],
               "tau_mean": cop["tau_mean"], "lambda_u": cop["lambda_u"],
               "fnr_B_local": fnr_b}
        # (1) 各自按等 FAR 标定
        c = sp.rule_C_count(Xte, cal["C22"]["thresholds"])
        rec["fnr_C_local_each"] = sp.evaluate_rule(c, lm, cfg.interval_sec,
                                                   cfg.early_tolerance_steps)["event_fnr"]
        # (2) 把 p_C 固定在共同水平（门限由训练集 GPD 映射，不经验证集分位估计）
        for pf in FIXED_PC:
            try:
                q = np.array([sp.gpd_quantile(m, pf) for m in gpd])
            except ValueError:
                # 个别测点的 POT 阈值高于该档，落到外推区间之外；记 NaN 而不是让整轮崩掉
                rec["fnr_C_local_p%g" % pf] = float("nan")
                continue
            cf = sp.rule_C_count(Xte, q)
            rec["fnr_C_local_p%g" % pf] = sp.evaluate_rule(
                cf, lm, cfg.interval_sec, cfg.early_tolerance_steps)["event_fnr"]
        rows.append(rec)
    print("  cross_strength=%.2f 完成 %d 组" % (cs, R), flush=True)

df = pd.DataFrame(rows)
df["d_each_"] = df.fnr_B_local - df.fnr_C_local_each
for pf in FIXED_PC:
    df["d_p%g" % pf] = df.fnr_B_local - df["fnr_C_local_p%g" % pf]
(D / "results").mkdir(exist_ok=True)
df.to_csv(D / "results" / "v13_h3_amp040.csv", index=False, encoding="utf-8-sig")

print("\n实测相依强度（未强制标定 τ，按达成值报告）")
print(df.groupby("cross_strength")[["tau_mean", "lambda_u", "pC_calibrated"]].mean().to_string())

COLS = [("各自标定", "d_each_")] + [("共同 p_C=%g" % pf, "d_p%g" % pf) for pf in FIXED_PC]
print("\n局部段 ΔFNR(B - C22)，按 λ_U 档（负 = B 更好）")
print("%-16s %10s %10s %10s %14s %8s" % ("口径", "λ_U≈0.03", "λ_U≈0.42", "λ_U≈0.76",
                                          "强-独立", "p"))
lv = sorted(df.cross_strength.unique())
for label, col in COLS:
    g = df.groupby("cross_strength")[col].mean()
    piv = df.pivot_table(index="rep", columns="cross_strength", values=col)
    diff = piv[lv[-1]] - piv[lv[0]]
    try:
        p = stats.ttest_rel(piv[lv[-1]], piv[lv[0]]).pvalue
    except Exception:
        p = float("nan")
    print("%-16s %10.3f %10.3f %10.3f %14.3f %8.4g"
          % (label, g[lv[0]], g[lv[1]], g[lv[2]], diff.mean(), p))

print("\n已写出 results/v13_h3_amp040.csv")
