# V8 交叉核对: 用 strict_pipeline 自己的函数（只作外部一致性对照，不作为主证据）
# 1) sp.at_least_two_prob 在我自己反解出的 alpha* 处是否 = 0.01
# 2) joint_target_thresholds 在真实拟合边际上: 正相依 theta 是否给出更小 alpha*、更高门限 q
import os
import numpy as np
import strict_pipeline as sp

cfg = sp.SimConfig(n_days=90, n_danger=50, bootstrap_gpd=0, bootstrap_metrics=0)
X, y, segs = sp.generate_dataset(cfg)
data, split = sp.split_time_series(X, y)
gpd, marginals, main, alt, diag = sp.fit_training_models(data["train"][0], cfg)
th_fit = float(main["theta"])
print(f"[数据] n_days={cfg.n_days} n_danger={cfg.n_danger}; 拟合 Gumbel theta={th_fit:.4f} tau={main.get('tau_mean', float('nan')):.4f}")

p_star = 0.01
print("\n[1] 用 sp.at_least_two_prob 回代 (应与 0.01 相符):")
mine = {"indep/gumbel1": (0.058903, 1.0), "gumbel1.5094": (0.016112, 1.5094),
        "gumbel4": (0.010680, 4.0), "gumbel50": (0.010025, 50.0)}
for name, (al, th) in mine.items():
    v = sp.at_least_two_prob(al, th)
    print(f"  {name:<16} alpha*={al:.6f} theta={th:<7} -> sp.P>=2={v:.12f}  (mine 1-3C2+2C3)")

print("\n[2] 门限方向 (同一目标 p*=0.01, 同一拟合边际):")
for th in [1.0, th_fit, 4.0]:
    al, q = sp.joint_target_thresholds(marginals, th, p_star)
    print(f"  theta={th:<8.4f} alpha*={al:.6f}  q=[{', '.join(f'{v:.6f}' for v in q)}]")
