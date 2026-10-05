# -*- coding: utf-8 -*-
"""Copula 的作用位置：原始 p* 工作点 vs 等 FAR 标定后的工作点。

回答「Copula 到底有没有参与最终判别」：

(A) **不经 FAR 标定**，直接按报告（七）的流程取定 p*（多点事件目标概率），由式(17)
    反解 α*、再由边缘反解门限，套式(22) 的计数判据。看：同一个 p* 下，Gumbel
    （λ_U = 0.42）与 Frank（λ_U = 0）给出的门限/报警集合差多少；以及 p* 本身
    对可达 FAR 的控制有多弱。
(B) **经等 FAR 标定**（本文正文用的方式）后，两者还剩多少差别。

若 (A) 里两族差别明显、而 (B) 里差别消失，则「Copula 被 FAR 标定吸收」得到直接证据，
且连带说明：把 Copula 的作用说成「决定工作点」是错的，它决定的是「在给定 FAR 下
选哪个 p*」。

幅度 0.40，10 组种子。
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import strict_pipeline as sp

D = Path(__file__).parent
R = int(sys.argv[1]) if len(sys.argv) > 1 else 10
AMP = 0.40
SEED0 = 20260924
P_STARS = [0.001, 0.005, 0.01, 0.05, 0.10]
TF = 0.05


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
for r in range(R):
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
    except Exception as exc:
        rows.append({"rep": r + 1, "error": type(exc).__name__})
        continue
    loc_m, cou_m = kind_masks(segs, split.val_end, len(yte))

    rec = {"rep": r + 1, "tau": cop["tau_mean"],
           "theta_G": cop["theta"], "theta_F": alt["theta"],
           "lambda_u_G": cop["lambda_u"], "lambda_u_F": alt["lambda_u"]}
    try:
        cal = sp.validation_calibration(Xv, yv, cfg, gpd, marg, cop["theta"],
                                        cop["family"], target=TF)
        rec["B_far"] = sp.evaluate_rule(sp.rule_B(Xte, cal["B"]["thresholds"]), yte,
                                        cfg.interval_sec, cfg.early_tolerance_steps)["far_per_hour"]
    except Exception as exc:
        rec["cal_error"] = type(exc).__name__
        rows.append(rec)
        continue

    fams = {"G": (cop["theta"], sp.gumbel_C), "F": (alt["theta"], sp.frank_C)}
    alarms = {}
    for tag, (th, cdf) in fams.items():
        for ps in P_STARS:
            try:
                alpha, q = sp.joint_target_thresholds(marg, th, ps, cdf)
            except ValueError:
                rec["%s_q_%g" % (tag, ps)] = np.nan
                continue
            # 训练集上的门限一律用训练集分位反解，这里 marg 就是训练集拟合的边际
            al = sp.rule_C_count(Xte, q)
            alarms["%s_%g" % (tag, ps)] = al
            m = sp.evaluate_rule(al, yte, cfg.interval_sec, cfg.early_tolerance_steps)
            rec["%s_alpha_%g" % (tag, ps)] = float(alpha)
            rec["%s_q_%g" % (tag, ps)] = float(np.mean(q))
            rec["%s_far_%g" % (tag, ps)] = m["far_per_hour"]
            rec["%s_fnr_%g" % (tag, ps)] = m["event_fnr"]
            rec["%s_local_%g" % (tag, ps)] = sp.evaluate_rule(
                al, loc_m, cfg.interval_sec, cfg.early_tolerance_steps)["event_fnr"]
            rec["%s_coup_%g" % (tag, ps)] = sp.evaluate_rule(
                al, cou_m, cfg.interval_sec, cfg.early_tolerance_steps)["event_fnr"]

    # 同一 p* 下两族的报警集合分歧（步级），以及门限差
    for ps in P_STARS:
        a, b = alarms.get("G_%g" % ps), alarms.get("F_%g" % ps)
        if a is None or b is None:
            continue
        union = np.logical_or(a, b).sum()
        rec["disagree_%g" % ps] = float(np.logical_xor(a, b).sum() / union) if union else 0.0
        rec["dthr_%g" % ps] = abs(rec.get("G_q_%g" % ps, np.nan) - rec.get("F_q_%g" % ps, np.nan))
    rows.append(rec)
    print("  rep %2d/%d 完成（τ=%.3f）" % (r + 1, R, cop["tau_mean"]), flush=True)

df = pd.DataFrame(rows)
(D / "results").mkdir(exist_ok=True)
df.to_csv(D / "results" / "v16_copula_raw.csv", index=False, encoding="utf-8-sig")
ok = df[df.get("cal_error").isna() if "cal_error" in df else slice(None)]

print("\n" + "=" * 92)
print("(A) 不经 FAR 标定，直接按目标多点概率 p* 定门限：两族的可达工作点")
print("=" * 92)
print("%-8s %10s %10s %10s %10s %10s %10s" %
      ("p*", "α*_G", "α*_F", "FAR_G", "FAR_F", "FNR_G", "FNR_F"))
for ps in P_STARS:
    c = lambda k: ok["%s_%s" % (k, ps)].mean()
    print("%-8g %10.5f %10.5f %10.4f %10.4f %10.4f %10.4f"
          % (ps, c("G_alpha"), c("F_alpha"), c("G_far"), c("F_far"),
             c("G_fnr"), c("F_fnr")))

print("\n同一 p* 下两族的门限差（三测点均值）与报警集合分歧率")
for ps in P_STARS:
    print("  p*=%-6g 门限差 %.5f   报警集合分歧 %.2f%%"
          % (ps, ok["dthr_%g" % ps].mean(), 100 * ok["disagree_%g" % ps].mean()))

print("\n参考：B 经等 FAR 标定后的 FAR = %.4f（目标 %.2f）" % (ok["B_far"].mean(), TF))
print("已写出 results/v16_copula_raw.csv")
