# -*- coding: utf-8 -*-
"""单个 (ratio, lag) 格的 30 组重复，供并行拼接。

usage: D:/python.exe _v20_cell.py <ratio> <lag> <R>

_v15 用 8 组跑出方向，ratio=0 的耦合段 ΔFNR 只在 lag=0 过 p<0.05，
需要 30 组把 p 值钉下来。每格单独写一个 csv，避免并行写冲突。
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import strict_pipeline as sp

D = Path(__file__).parent
RATIO = float(sys.argv[1])
LAG = int(sys.argv[2])
R = int(sys.argv[3]) if len(sys.argv) > 3 else 30
AMP = 0.40
TF = 0.05
SEED0 = 20260924


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


(D / "results" / "v20").mkdir(parents=True, exist_ok=True)
OUT = D / "results" / "v20" / ("r%02d_l%02d.csv" % (round(RATIO * 100), LAG))


def dump(rows):
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
              "danger_ratio_local": RATIO, "propagation_lag": LAG,
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
    except Exception as exc:
        rows.append({"ratio": RATIO, "lag": LAG, "rep": r + 1,
                     "error": type(exc).__name__ + ":" + str(exc)[:60]})
        dump(rows)
        continue

    alarms = {
        "A": sp.rule_A(Xte, cal["A"]["k"], cfg.fixed_thresholds),
        "B": sp.rule_B(Xte, cal["B"]["thresholds"]),
        "C22": sp.rule_C_count(Xte, cal["C22"]["thresholds"]),
        "C_star": sp.rule_C_count(Xte, cal["C_star"]["thresholds"]),
    }
    loc_m, cou_m = kind_masks(segs, split.val_end, len(yte))
    rec = {"ratio": RATIO, "lag": LAG, "rep": r + 1,
           "n_local": int(len([1 for s, e, k in segs if k == "local" and s >= split.val_end])),
           "n_coupled": int(len([1 for s, e, k in segs if k == "coupled" and s >= split.val_end]))}
    for rule, al in alarms.items():
        for name, ym in (("all", yte), ("loc", loc_m), ("coup", cou_m)):
            m = sp.evaluate_rule(al, ym, cfg.interval_sec, cfg.early_tolerance_steps)
            rec["%s_%s_fnr" % (rule, name)] = m["event_fnr"]
            rec["%s_%s_n" % (rule, name)] = m["n_danger_events"]
            if name == "all":
                rec["%s_far" % rule] = m["far_per_hour"]
                rec["%s_delay" % rule] = m["mean_delay_min"]
                for w in (10, 20, 30):
                    rec["%s_e%d" % (rule, w)] = m["early_fnr_%dmin" % w]
    rows.append(rec)
    dump(rows)
    print("  ratio=%.2f lag=%d rep %d/%d" % (RATIO, LAG, r + 1, R), flush=True)

print("ratio=%.2f lag=%d 完成 %d 组" % (RATIO, LAG, len(rows)), flush=True)
