# -*- coding: utf-8 -*-
"""设计变量的扫描：局部段占比 × 传播滞后。

针对三条外部质疑：

(1) 「50% 局部段是人为压低 C」——把 danger_ratio_local 拉到 0（全耦合）与 1（全局部），
    看 B 与 C22/C* 的排序怎么变。若全耦合下 C22 反超，则合并口径的结论确实是设定产物。
(2) 「10 分钟传播滞后与『两点同时超限』冲突」——扫 propagation_lag ∈ {0, 10, 20, 40}。
    lag=0 时三个测点同起同落，计数判据的「同时」条件最松；lag 越大越难同时。
(3) 「危险段长度 / 滞后」这个比值——lag=40 时最长滞后 80 步，已接近段长下界 160 的一半。

幅度固定 0.40（非饱和），FAR*=0.05，8 组种子（与 _v13 同种子集，可跨脚本对照）。
统计量按危险段类型拆开；局部段占比为 0 或 1 时对应格为空，取 nan。
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
RATIOS = [0.0, 0.5, 1.0]
LAGS = [0, 10, 20, 40]
AMP = 0.40
TF = 0.05
SEED0 = 20260924
RULES = ["A", "B", "C22", "C_star"]


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


rows = []
for ratio in RATIOS:
    for lag in LAGS:
        for r in range(R):
            d = dict(sp.SimConfig().__dict__)
            d.update({"seed": SEED0 + r, "local_amp": AMP, "coupled_amp": AMP,
                      "danger_ratio_local": ratio, "propagation_lag": lag,
                      "bootstrap_gpd": 0, "bootstrap_metrics": 0})
            cfg = sp.SimConfig(**d)
            X, y, segs = sp.generate_dataset(cfg)
            data, split = sp.split_time_series(X, y)
            Xv, yv = data["val"]
            Xte, yte = data["test"]
            try:
                gpd, marg, cop, alt, _ = sp.fit_training_models(data["train"][0], cfg)
                cal = sp.validation_calibration(Xv, yv, cfg, gpd, marg, cop["theta"],
                                                cop["family"], target=TF)
            except Exception as exc:                      # 标定/拟合失败就整格记 NaN
                rows.append({"ratio": ratio, "lag": lag, "rep": r + 1,
                             "error": type(exc).__name__ + ":" + str(exc)[:60]})
                continue

            lam = sp.all_three_joint_prob(sp.pit_observations(Xte, marg), cop["theta"])
            alarms = {
                "A": sp.rule_A(Xte, cal["A"]["k"], cfg.fixed_thresholds),
                "B": sp.rule_B(Xte, cal["B"]["thresholds"]),
                "C22": sp.rule_C_count(Xte, cal["C22"]["thresholds"]),
                "C_star": sp.rule_C_count(Xte, cal["C_star"]["thresholds"]),
            }
            loc_m, cou_m = kind_masks(segs, split.val_end, len(yte))

            # 各规则的报警密度：C22 的「两点同时」在 lag 大时到底还剩多少重叠
            rec = {"ratio": ratio, "lag": lag, "rep": r + 1,
                   "tau": cop["tau_mean"], "lambda_u": cop["lambda_u"],
                   "p_B": cal["B"]["p"], "p_C22": cal["C22"]["p"],
                   "n_local": int(len([1 for s, e, k in segs if k == "local" and s >= split.val_end])),
                   "n_coupled": int(len([1 for s, e, k in segs if k == "coupled" and s >= split.val_end])),
                   "n_danger_steps": int(yte.sum())}
            for rule, al in alarms.items():
                m = sp.evaluate_rule(al, yte, cfg.interval_sec, cfg.early_tolerance_steps)
                ml = sp.evaluate_rule(al, loc_m, cfg.interval_sec, cfg.early_tolerance_steps)
                mc = sp.evaluate_rule(al, cou_m, cfg.interval_sec, cfg.early_tolerance_steps)
                rec["%s_event_fnr" % rule] = m["event_fnr"]
                rec["%s_far" % rule] = m["far_per_hour"]
                rec["%s_delay" % rule] = m["mean_delay_min"]
                rec["%s_e20" % rule] = m["early_fnr_20min"]
                rec["%s_local_fnr" % rule] = ml["event_fnr"]
                rec["%s_coupled_fnr" % rule] = mc["event_fnr"]
            rows.append(rec)
        print("  ratio=%.1f lag=%2d 完成 %d 组" % (ratio, lag, R), flush=True)

df = pd.DataFrame(rows)
(D / "results").mkdir(exist_ok=True)
df.to_csv(D / "results" / "v15_design.csv", index=False, encoding="utf-8-sig")

errs = df[df.get("error").notna()] if "error" in df else df.iloc[0:0]
print("\n失败格 %d / %d" % (len(errs), len(df)))

print("\n" + "=" * 96)
print("事件级 FNR（8 组均值）。local=该规则的局部段 FNR，coup=耦合段 FNR，all=合并")
print("=" * 96)
print("%-6s %-5s | %-19s | %-19s | %-19s" % ("ratio", "lag", "B", "C22", "C_star"))
print("%-6s %-5s | %7s %6s %5s | %7s %6s %5s | %7s %6s %5s"
      % ("", "", "local", "coup", "all", "local", "coup", "all", "local", "coup", "all"))
for ratio in RATIOS:
    for lag in LAGS:
        s = df[(df.ratio == ratio) & (df.lag == lag)]
        if not len(s):
            continue
        cells = []
        for rule in ["B", "C22", "C_star"]:
            cells.append("%7.3f %6.3f %5.3f" % (s["%s_local_fnr" % rule].mean(),
                                                s["%s_coupled_fnr" % rule].mean(),
                                                s["%s_event_fnr" % rule].mean()))
        print("%-6.1f %-5d | %s | %s | %s" % (ratio, lag, cells[0], cells[1], cells[2]))

print("\n耦合段 ΔFNR(B - C22) 与 ΔFNR(B - C*)（负 = B 更好）")
print("%-6s %-5s %12s %12s %12s" % ("ratio", "lag", "Δ(B-C22)", "Δ(B-C*)", "C22 FAR"))
for ratio in RATIOS:
    for lag in LAGS:
        s = df[(df.ratio == ratio) & (df.lag == lag)]
        if not len(s):
            continue
        print("%-6.1f %-5d %12.4f %12.4f %12.4f"
              % (ratio, lag, (s.B_coupled_fnr - s.C22_coupled_fnr).mean(),
                 (s.B_coupled_fnr - s.C_star_coupled_fnr).mean(), s.C22_far.mean()))

print("\nEarly-FNR(20 min)，合并口径")
print("%-6s %-5s %8s %8s %8s" % ("ratio", "lag", "A", "B", "C22"))
for ratio in RATIOS:
    for lag in LAGS:
        s = df[(df.ratio == ratio) & (df.lag == lag)]
        if not len(s):
            continue
        print("%-6.1f %-5d %8.4f %8.4f %8.4f"
              % (ratio, lag, s.A_e20.mean(), s.B_e20.mean(), s.C22_e20.mean()))

print("\n已写出 results/v15_design.csv")
