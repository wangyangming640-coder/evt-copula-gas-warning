# -*- coding: utf-8 -*-
"""等 FAR 标定的对齐质量：单个 (幅度, 目标 FAR) 格的 30 组重复。

usage: D:/python.exe _v18_cell.py <amp> <target_far> <R>

对每条规则（A/B/C22/Cprime/C_star）记录验证集 FAR（选点依据）、测试集 FAR
（实际工作点）、误报事件数，以及合并/局部/耦合三种口径的事件级 FNR。
拆分并行是为了不让 4 个格子串行排队；分析见 _v18_report.py。
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import strict_pipeline as sp

D = Path(__file__).parent
AMP = float(sys.argv[1])
TF = float(sys.argv[2])
R = int(sys.argv[3]) if len(sys.argv) > 3 else 30
SEED0 = 20260924
RULES = ["A", "B", "C22", "Cprime", "C_star"]


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


def alarms(X, cal, cfg, marg, theta, cdf_fn):
    return {
        "A": sp.rule_A(X, cal["A"]["k"], cfg.fixed_thresholds),
        "B": sp.rule_B(X, cal["B"]["thresholds"]),
        "C22": sp.rule_C_count(X, cal["C22"]["thresholds"]),
        "Cprime": sp.rule_Cprime(X, cal["Cprime"]["thresholds"], cal["Cprime"]["lam_c"],
                                 marg, theta, cdf_fn)[0],
        "C_star": sp.rule_C_count(X, cal["C_star"]["thresholds"]),
    }


(D / "results" / "v18").mkdir(parents=True, exist_ok=True)
OUT = D / "results" / "v18" / ("a%03d_t%03d.csv" % (round(AMP * 100), round(TF * 100)))


def dump(rows):
    """每组落盘一次：进程被打断也只丢当前这一组。"""
    pd.DataFrame(rows).to_csv(OUT, index=False, encoding="utf-8-sig")


rows, done = [], set()
if OUT.exists():
    prev = pd.read_csv(OUT)
    if "error" in prev:
        prev = prev[prev.error.isna()]
    rows = prev.to_dict("records")
    done = set(int(x) for x in prev.rep)
    print("续跑：已有 %d 组" % len(done), flush=True)

for r in range(R):
    if r + 1 in done:
        continue
    d = dict(sp.SimConfig().__dict__)
    d.update({"seed": SEED0 + r, "local_amp": AMP, "coupled_amp": AMP,
              "bootstrap_gpd": 0, "bootstrap_metrics": 0})
    cfg = sp.SimConfig(**d)
    X, y, segs = sp.generate_dataset(cfg)
    data, split = sp.split_time_series(X, y)
    Xv, yv = data["val"]
    Xte, yte = data["test"]
    try:
        gpd, marg, cop, alt, _ = sp.fit_training_models(data["train"][0], cfg)
        cdf_fn = sp.gumbel_C if cop["family"] == "gumbel" else sp.frank_C
        cal = sp.validation_calibration(Xv, yv, cfg, gpd, marg, cop["theta"],
                                        cop["family"], target=TF)
    except Exception as exc:
        rows.append({"amp": AMP, "target": TF, "rep": r + 1,
                     "error": type(exc).__name__ + ":" + str(exc)[:60]})
        dump(rows)
        continue
    av = alarms(Xv, cal, cfg, marg, cop["theta"], cdf_fn)
    ae = alarms(Xte, cal, cfg, marg, cop["theta"], cdf_fn)
    loc_m, cou_m = kind_masks(segs, split.val_end, len(yte))

    rec = {"amp": AMP, "target": TF, "rep": r + 1}
    for tag in RULES:
        mv = sp.evaluate_rule(av[tag], yv, cfg.interval_sec, cfg.early_tolerance_steps)
        rec["valfar_" + tag] = mv["far_per_hour"]
        rec["far_" + tag] = sp.evaluate_rule(
            ae[tag], yte, cfg.interval_sec, cfg.early_tolerance_steps)["far_per_hour"]
        rec["nfa_" + tag] = sp.evaluate_rule(
            ae[tag], yte, cfg.interval_sec, cfg.early_tolerance_steps)["n_false_alarm_events"]
        for name, ym in (("fnr", yte), ("loc", loc_m), ("coup", cou_m)):
            rec["%s_%s" % (name, tag)] = sp.evaluate_rule(
                ae[tag], ym, cfg.interval_sec, cfg.early_tolerance_steps)["event_fnr"]
    rows.append(rec)
    dump(rows)
    print("  amp=%.2f target=%.2f rep %d/%d" % (AMP, TF, r + 1, R), flush=True)

print("amp=%.2f target=%.2f 完成 %d 组" % (AMP, TF, len(rows)), flush=True)
