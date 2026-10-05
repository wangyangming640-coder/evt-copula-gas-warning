"""θ̂ 修正两种口径的对照：式(8) 的边际分位口径 vs 报告五（四）字面要求的口径。

报告 L128 要求「把式(6)—(8)中的超越率 ζ_u 替换为 θ̂ζ_u」。本脚本把这一条按字面
实现一遍，与不修正的口径并排跑同一套数据，用实测数字说明两者的差别与代价。
"""
import sys
try:
    sys.stdout.reconfigure(errors="replace")   # Windows 控制台是 GBK，- ≤ 这类字符会直接抛错
except Exception:
    pass

from pathlib import Path
import numpy as np
import pandas as pd

import strict_pipeline as sp

OUT = Path(__file__).parent / "results" / "decluster_compare"
OUT.mkdir(parents=True, exist_ok=True)

base = sp.SimConfig()

# ---- 1. 门限随分位点的变化 ----
X, y, _ = sp.generate_dataset(base)
data, _ = sp.split_time_series(X, y)
Xtr = data["train"][0]
gpd, _, _, _, _ = sp.fit_training_models(Xtr, base)

rows = []
for p in [0.98, 0.99, 0.995, 0.998, 0.9995]:
    for j, m in enumerate(gpd):
        q_raw = sp.gpd_quantile(m, p, use_decluster=False)
        q_cor = sp.gpd_quantile(m, p, use_decluster=True)
        rows.append({
            "p": p, "sensor": j, "u": m["u"], "q_POT": m["q"],
            "theta_hat": m["theta_hat"], "raw_tail_prob": m["raw_tail_prob"],
            "cluster_tail_prob": m["cluster_tail_prob"],
            "threshold_marginal": q_raw, "threshold_decluster": q_cor,
            # 修正后返回值在原始（未修正）GPD 下的实际超越概率
            "implied_exceed_prob": np.nan if not np.isfinite(q_cor)
            else (1 - p) / m["theta_hat"],
        })
df = pd.DataFrame(rows)
df.to_csv(OUT / "threshold_compare.csv", index=False, encoding="utf-8-sig")
print("极值指数与门限对照（前 10 行）：")
print(df.drop(columns=["raw_tail_prob", "cluster_tail_prob"]).head(10).to_string(index=False))

# ---- 2. 两套口径各自跑一次完整流程 ----
summ = []
for flag in (False, True):
    d = dict(base.__dict__)
    d["use_decluster_rate"] = flag
    d["bootstrap_gpd"] = 0
    d["bootstrap_metrics"] = 0
    cfg = sp.SimConfig(**d)
    res = sp.run_one(cfg, save_dir=None)
    tag = "theta_hat_corrected" if flag else "marginal_quantile"
    cal = res["calibration"]
    for rule, m in res["test_metrics"].items():
        # 每行填**该规则自己的**门限；原先无论哪条规则都填 B 的门限、列名却叫 thresholds，
        # 会把「C22 的门限」误读成 [0.97, 0.91, 0.87]（实际是 [0.83, 0.77, 0.73]）。
        own = cal.get(rule, {}).get("thresholds")
        summ.append({
            "convention": tag, "use_decluster_rate": flag, "rule": rule,
            "B_p": cal["B"]["p"], "B_grid_dropped": cal["B"]["n_grid_dropped"],
            "thresholds": None if own is None else np.round(np.asarray(own), 4).tolist(),
            "event_fnr": m["event_fnr"], "time_fnr": m["time_fnr"],
            "far_per_hour": m["far_per_hour"], "mean_delay_min": m["mean_delay_min"],
            "n_alarm_events": m["n_alarm_events"],
        })
    print("\n[%s] B 标定 p=%.6f（网格丢弃 %d 点）门限=%s"
          % (tag, cal["B"]["p"], cal["B"]["n_grid_dropped"],
             np.round(cal["B"]["thresholds"], 4).tolist()))
    tf = [float(np.quantile(data["train"][0][:, j], p))
          for j, p in enumerate([cal["B"]["p"]] * 3)]
    print("   同一 p 在训练集上的经验分位（不受口径影响）：%s" % np.round(tf, 4).tolist())

sm = pd.DataFrame(summ)
sm.to_csv(OUT / "convention_compare.csv", index=False, encoding="utf-8-sig")
print("\n两套口径的测试指标：")
print(sm[["convention", "rule", "event_fnr", "time_fnr", "far_per_hour",
          "n_alarm_events"]].to_string(index=False))
