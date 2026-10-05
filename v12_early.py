# -*- coding: utf-8 -*-
"""新基准幅度 + Early-FNR 的等 FAR 比较。

两件事一起做：

(1) **Early-FNR**。此前只有 event_fnr（有没有报过）与 time_fnr（覆盖了多少危险步）、
    mean_delay（均值延迟）。三者都不回答「报得够不够早」。evaluate_rule 现在多返回
    early_fnr_{5,10,20,30}min：危险段开始后 T0 分钟内一个报警步都没有，该段即算早期漏报。
    窗口与报警集合无关，所以该指标和 event_fnr 一样对报警集合单调。

(2) **基准幅度**。此前基准 amp=0.78 落在饱和区（FAR*=0.05 下 amp≥0.55 时 B 的
    事件级 FNR 恒为 0），「C 不比 B 好」在事件级上无法与「测不出来」区分。这里把基准
    降到 0.40（用户指定区间 0.30—0.40 的上端，离饱和边界最近、与旧基准可比性最好），
    并保留 0.78 一档作对照。

**不改 SimConfig 默认值**：默认仍是 0.78/0.72。原因是改了会让此前所有脚本的产物在同一次
重跑下悄悄换口径、与论文里已有的表对不上。本脚本显式传 amp，并在输出里逐档打印实际幅度。

口径：A/B/C22/C*/C′ 五条规则，全部只在 validation 上按目标 FAR 标定，测试集只评价一次。
30 组重复用 seed+r，配对比较。
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

import strict_pipeline as sp

D = Path(__file__).parent
R = int(sys.argv[1]) if len(sys.argv) > 1 else 30
AMPS = [0.40, 0.78]
TARGETS = [0.01, 0.05]
SEED0 = 20260924
RULES = ["A", "B", "C22", "C_star", "Cprime"]


def kind_masks(segs, val_end, n_test):
    """把测试集内的危险段按 local / coupled 拆成两个掩码。"""
    loc = np.zeros(n_test, dtype=bool)
    cou = np.zeros(n_test, dtype=bool)
    for s, e, k in segs:
        if s < val_end:
            continue
        tgt = loc if k == "local" else cou
        a, b = max(0, s - val_end), min(n_test, e - val_end)
        tgt[a:b] = True
    return loc, cou


def alarms_of(cal, Xte, marg, theta):
    """按标定结果在测试集上还原五条规则的报警序列。"""
    lam = sp.all_three_joint_prob(sp.pit_observations(Xte, marg), theta)
    return {
        "A": sp.rule_A(Xte, cal["A"]["k"], cfg_fixed),
        "B": sp.rule_B(Xte, cal["B"]["thresholds"]),
        "C22": sp.rule_C_count(Xte, cal["C22"]["thresholds"]),
        "C_star": sp.rule_C_count(Xte, cal["C_star"]["thresholds"]),
        "Cprime": sp.rule_Cprime_from_score(
            lam, sp.rule_B(Xte, cal["Cprime"]["thresholds"]), cal["Cprime"]["lam_c"]),
    }


rows = []
for amp in AMPS:
    for r in range(R):
        base = sp.SimConfig()
        d = dict(base.__dict__)
        d.update({"seed": SEED0 + r, "local_amp": amp, "coupled_amp": amp,
                  "bootstrap_gpd": 0, "bootstrap_metrics": 0})
        cfg = sp.SimConfig(**d)
        cfg_fixed = cfg.fixed_thresholds
        X, y, segs = sp.generate_dataset(cfg)
        data, split = sp.split_time_series(X, y)
        Xv, yv = data["val"]
        Xte, yte = data["test"]
        gpd, marg, cop, alt, _ = sp.fit_training_models(data["train"][0], cfg)
        loc_m, cou_m = kind_masks(segs, split.val_end, len(yte))

        for tf in TARGETS:
            cal = sp.validation_calibration(Xv, yv, cfg, gpd, marg, cop["theta"],
                                            cop["family"], target=tf)
            al = alarms_of(cal, Xte, marg, cop["theta"])
            for rule, alarm in al.items():
                m = sp.evaluate_rule(alarm, yte, cfg.interval_sec, cfg.early_tolerance_steps)
                ml = sp.evaluate_rule(alarm, loc_m, cfg.interval_sec, cfg.early_tolerance_steps)
                mc = sp.evaluate_rule(alarm, cou_m, cfg.interval_sec, cfg.early_tolerance_steps)
                rows.append({
                    "amp": amp, "rep": r + 1, "target_far": tf, "rule": rule,
                    **{k: v for k, v in m.items()},
                    "local_event_fnr": ml["event_fnr"], "local_time_fnr": ml["time_fnr"],
                    "coupled_event_fnr": mc["event_fnr"], "coupled_time_fnr": mc["time_fnr"],
                    "n_local": ml["n_danger_events"], "n_coupled": mc["n_danger_events"],
                    "tau": cop["tau_mean"], "lambda_u": cop["lambda_u"],
                })
        print("  amp=%.2f rep %2d/%d 完成" % (amp, r + 1, R), flush=True)

df = pd.DataFrame(rows)
(D / "results").mkdir(exist_ok=True)
df.to_csv(D / "results" / "v12_early_replicates.csv", index=False, encoding="utf-8-sig")

METRICS = ["event_fnr", "time_fnr", "local_event_fnr", "coupled_event_fnr",
           "early_fnr_5min", "early_fnr_10min", "early_fnr_20min", "early_fnr_30min",
           "far_per_hour", "mean_delay_min"]

print("\n" + "=" * 100)
print("各规则测试指标（%d 组均值 ± 标准差），全部按目标 FAR 在 validation 上标定" % R)
print("=" * 100)
for amp in AMPS:
    for tf in TARGETS:
        sub = df[(df.amp == amp) & (df.target_far == tf)]
        g = sub.groupby("rule")[METRICS].agg(["mean", "std"])
        print("\n--- 幅度 %.2f，目标 FAR*=%.2f  (λ_U≈%.3f，测试集 local %d / coupled %d) ---"
              % (amp, tf, sub.lambda_u.mean(), round(sub.n_local.mean()),
                 round(sub.n_coupled.mean())))
        for m in METRICS:
            line = "  %-18s" % m
            for rule in RULES:
                line += "  %s %6.4f" % (rule.ljust(7), g.loc[rule, (m, "mean")])
            print(line)

print("\n" + "=" * 100)
print("配对 Wilcoxon（同一组重复、同一目标 FAR 下逐组比较；n=%d）" % R)
print("=" * 100)
for amp in AMPS:
    for tf in TARGETS:
        sub = df[(df.amp == amp) & (df.target_far == tf)]
        piv = {m: sub.pivot_table(index="rep", columns="rule", values=m) for m in METRICS}
        print("\n--- 幅度 %.2f，FAR*=%.2f ---" % (amp, tf))
        for m in ["event_fnr", "local_event_fnr", "coupled_event_fnr", "early_fnr_10min"]:
            p = piv[m]
            for a, b in [("B", "C22"), ("B", "C_star"), ("B", "Cprime"), ("B", "A")]:
                d = p[a] - p[b]
                if np.allclose(d, 0):
                    verdict = "两组逐组完全相同（p 无定义）"
                else:
                    try:
                        w = stats.wilcoxon(p[a], p[b])
                        verdict = "p=%.4g  均值差 %+.4f" % (w.pvalue, d.mean())
                    except ValueError as e:
                        verdict = "无法计算：%s" % e
                print("  %-18s %-8s vs %-8s : %s" % (m, a, b, verdict))

print("\n已写出 results/v12_early_replicates.csv")
