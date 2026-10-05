# 「独立边际」对照是否真的独立：四种段内排布的依赖强度实测对比
# 判据是 tau_全体测试集 ≈ 0（整段序列上无依赖）。tau_危险掩码 是在并集掩码上
# 算的，掩码本身是过程的函数，条件在该并集上会制造依赖（选择效应），故它不为 0
# 并不说明对照失败；把它一并列出只是为了让这个陷阱显形。
import numpy as np
import pandas as pd
from scipy import stats
import strict_pipeline as sp

pairs = [(0, 1), (0, 2), (1, 2)]


def tau_on(X, mask):
    if mask.sum() < 10:
        return np.nan
    return float(np.mean([stats.kendalltau(X[mask, i], X[mask, j])[0] for i, j in pairs]))


base = sp.SimConfig()
rows = []
for tag, kw in [("lagged(默认)", {}),
                ("jitter(旧修法)", {"coupled_jitter": True}),
                ("disjoint(互斥子段)", {"coupled_layout": "disjoint"}),
                ("independent(现方案)", {"coupled_layout": "independent"})]:
    d = dict(base.__dict__); d.update(kw)
    if not tag.startswith("lagged"):
        d["cross_strength"] = 0.0      # 独立边际对照要求去掉基底耦合
    cfg = sp.SimConfig(**d)
    X, y, segs = sp.generate_dataset(cfg)
    data, _ = sp.split_time_series(X, y)
    Xte, yte = data["test"]
    seg = np.zeros(len(yte), dtype=bool)
    for s, e in sp.danger_segments_from_mask(yte):
        seg[s:e] = True
    # 预先登记的窗口（外生，不受已实现过程影响）
    te_segs = [x for x in segs if x[0] >= len(y) * 0.8]
    win = np.zeros(len(yte), dtype=bool)
    for s, e, _ in te_segs:
        win[s:e] = True
    rows.append({
        "layout": tag,
        "tau_全体测试集": round(tau_on(Xte, np.ones(len(yte), bool)), 4),
        "tau_正常段": round(tau_on(Xte, ~yte), 4),
        "tau_登记窗口": round(tau_on(Xte, win), 4),
        "tau_危险掩码(选择污染)": round(tau_on(Xte, seg), 4),
        "危险段数": len(sp.danger_segments_from_mask(yte)),
    })

print(pd.DataFrame(rows).to_string(index=False))
print("\n判据：independent 的「tau_全体测试集」与「tau_正常段」应为 0（或与 0 无异）。")
