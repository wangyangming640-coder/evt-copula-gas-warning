# -*- coding: utf-8 -*-
"""核对 4.5 节那句机制解释：等 FAR 标定后，B 的逐测点门限是否真的高于 A 的。

A 的门限 = k_A * cfg.fixed_thresholds（三测点同乘一个 k）
B 的门限 = 训练集 GPD 在 p_B 分位处的逐测点分位点
两者都在原始浓度空间，可逐测点直接比。输出比值 b_i/a_i 的均值与各测点值。
"""
import numpy as np
import pandas as pd

import strict_pipeline as sp

for amp in (0.78, 0.40):
    rows = []
    for r in range(6):
        d = dict(sp.SimConfig().__dict__)
        d.update({"seed": 20260924 + r, "local_amp": amp, "coupled_amp": amp,
                  "bootstrap_gpd": 0, "bootstrap_metrics": 0})
        cfg = sp.SimConfig(**d)
        X, y, _ = sp.generate_dataset(cfg)
        data, split = sp.split_time_series(X, y)
        gpd, marg, cop, alt, _ = sp.fit_training_models(data["train"][0], cfg)
        cal = sp.validation_calibration(data["val"][0], data["val"][1], cfg, gpd, marg,
                                        cop["theta"], cop["family"], target=0.05)
        a = cal["A"]["k"] * np.asarray(cfg.fixed_thresholds, dtype=float)
        b = np.asarray(cal["B"]["thresholds"], dtype=float)
        rows.append({"rep": r + 1, "k_A": cal["A"]["k"], "p_B": cal["B"]["p"],
                     "a1": a[0], "a2": a[1], "a3": a[2],
                     "b1": b[0], "b2": b[1], "b3": b[2],
                     "ratio_mean": float(np.mean(b / a))})
    df = pd.DataFrame(rows)
    print("=== amp=%.2f ===" % amp)
    print(df.to_string(index=False))
    print("逐测点 b/a 均值: %s" % np.round((df[["b1", "b2", "b3"]].values /
                                          df[["a1", "a2", "a3"]].values).mean(axis=0), 3))
    print("整体 b/a 均值 %.3f，b>a 的组数 %d/6" % (df.ratio_mean.mean(),
                                                  int((df.ratio_mean > 1).sum())))
    print()
