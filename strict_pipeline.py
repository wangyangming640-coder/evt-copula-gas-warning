"""
严格版：基于极值理论与 Gumbel Copula 的多测点瓦斯联合预警

严格对应《基于极值理论与Copula的多测点瓦斯联合预警_开题报告_修改稿》中的核心要求：
1. 90 d、30 s、3 测点；危险段约占 3%~5%；危险段真值来自生成器；
2. 时间顺序 60%/20%/20%，测试集不参与任何参数选择；
3. POT + GPD：候选分位数、MRL、参数稳定性、KS 拟合优度、去聚集、GPD CI；
4. PIT + Kendall tau + Gumbel Copula；上尾依赖；联合超越概率；Copula GOF；
5. A/B/C 三规则；每种规则只有一个自由参数；验证集只负责预先标定参数；
6. 测试集只用于生成 DET/FAR-FNR 曲线和最终评价，不反向选参数；
7. 危险段 bootstrap 2000 次、30 个独立数据集、Wilcoxon 配对检验；
8. 噪声、Kendall tau、危险段长度、run length 敏感性；零危险和独立边际阴性对照。

说明：
- 这是本科论文级“严格实验框架”，不是现场工程报警系统。
- GPD 的 KS p 值属于参数已估计后的近似 GOF 指标；论文中应如实表述。
- 为避免测试集泄漏，DET 曲线的横轴参数网格在实验前固定，不在测试集上挑选单个参数。
"""
from __future__ import annotations

import sys
try:
    sys.stdout.reconfigure(errors="replace")   # Windows 控制台是 GBK，- ≤ 这类字符会直接抛错
except Exception:
    pass


from dataclasses import dataclass, asdict
from pathlib import Path
import json
import math
import warnings
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import brentq

warnings.filterwarnings("ignore", category=RuntimeWarning)


# ============================================================
# 1. 配置
# ============================================================

@dataclass
class SimConfig:
    # 270 天 × 150 段 × 平均约 210 个 30s 时间步 ≈ 4.05% 危险占比（与原 90 天/50 段同）
    # 提量只为功效：测试集危险段数由 10 提到 30，事件级 FNR 的分辨率由 0.10 降到 0.033。
    # 原 90 天/50 段下 H2、H3 的判定都卡在 10 个事件上，差异小于 0.10 一律不可信。
    n_days: int = 270
    interval_sec: int = 30
    n_sensors: int = 3
    seed: int = 20260924

    mu_log: Tuple[float, ...] = (math.log(0.45), math.log(0.42), math.log(0.40))
    rho: Tuple[float, ...] = (0.97, 0.97, 0.97)
    sigma_log: Tuple[float, ...] = (0.055, 0.055, 0.055)
    cross_strength: float = 0.50
    sigma_noise: float = 0.015

    n_danger: int = 150
    danger_ratio_local: float = 0.50
    min_len: int = 160
    max_len: int = 260
    local_amp: float = 0.78
    coupled_amp: float = 0.72
    propagation_lag: int = 20

    fixed_thresholds: Tuple[float, ...] = (1.0, 1.0, 1.0)

    # POT 候选；报告要求的四档
    pot_q_grid: Tuple[float, ...] = (0.90, 0.925, 0.95, 0.975)
    run_length: int = 20
    bootstrap_gpd: int = 100
    bootstrap_metrics: int = 2000

    # 极值指数 θ̂ 的用法。报告五（三）(四)要求把 ζ_u 换成 θ̂ζ_u 再代入式(7)(8)。
    # 修正对重现水平式(7)是成立的（P(X > x_T) = 1/(θ̂T)），但式(8) 把 q_p 定义为
    # 边际 p 分位（P(X ≤ q_p) = p），乘以 θ̂<1 之后返回值就不再满足这个定义，
    # 式(10)(11) 的 PIT 也随之失去一致口径。故默认关闭（=按式(8)字面口径），
    # 打开时走 B_q_grid_decluster 网格，用于「按报告字面实现」的对照诊断。
    use_decluster_rate: bool = False
    B_q_grid_decluster: Tuple[float, ...] = tuple(np.linspace(0.995, 0.99999, 100))

    # Copula 族：gumbel（λ_U > 0）与 frank（λ_U = 0，用作上尾相依的对照）
    copula_family: str = "gumbel"

    # 危险段内各测点抬升的排布方式：
    #   "lagged"      各测点固定错开 j*propagation_lag（默认，对应报告表2 的传播滞后）
    #   "independent" 各测点抬升窗口位置独立抽取，用于「独立边际」阴性对照
    # 独立对照只能走 "independent"。实测三种看似可行的做法里：随机错位（jitter）
    # 把段内 τ 从 0.211 升到 0.254，互斥子区间把 τ 压成 -0.167，二者都不是独立——
    # 只要三测点共享同一个确定性抬升剖面，观测序列就正相关，与错位无关。
    coupled_layout: str = "lagged"
    coupled_jitter: bool = False

    # 敏感性/对照实验的重复次数：各档位共用同一组种子，避免「档位×随机数」混淆。
    # 3 次重复下 H3 的每个 λ_U 档位只有 3 个观测，无法判定方向；提到 10。
    sens_replicates: int = 10
    # 依赖性敏感性直接指定目标 Kendall τ（报告表2 的 0.1 / 0.3 / 0.5），
    # 由 calibrate_cross_strength 反解 cross_strength
    target_taus: Tuple[float, ...] = (0.1, 0.3, 0.5)

    # DET 的参数网格是实验前固定的，不由测试集选择。
    # 分位网格名义上从 0.901 起，再由 b_threshold_candidates 按**实际拟合到的**
    # POT 阈值分位 q₀ 逐数据集裁掉 p ≤ q₀ 的点（GPD 外推在该区间无定义），
    # 裁掉的点进 dropped 计数、不静默截断。q₀ 由 choose_pot_threshold 数据驱动，
    # 取值在 pot_q_grid = (0.90, 0.925, 0.95, 0.975) 中，所以名义下界必须低于 0.90。
    #
    # 原实现把下界写死 0.98，后果是式(22) 在低相依数据上凑不出目标 FAR：
    # 实测独立边际数据下 p_C 取到网格最低点 0.98 时验证集 FAR 仅 0.0046，
    # 离目标 0.05 差一个量级且无法再往下，等误报率对齐会静默失效
    # （C22 被摆在远比 B 保守的工作点上，比较不再是等 FAR 的）。
    A_multiplier_grid: Tuple[float, ...] = tuple(np.linspace(0.70, 1.50, 81))
    B_q_grid: Tuple[float, ...] = tuple(np.linspace(0.901, 0.9995, 250))
    C_pc_grid: Tuple[float, ...] = tuple(np.geomspace(1e-5, 0.20, 100))

    # 统一的名义 FAR 点；报告示例给出 1%、5%
    target_fars: Tuple[float, ...] = (0.01, 0.05)

    # 报告 C′（式19 的 Λ_3）的唯一自由参数 λ_c：Λ_3 越小越「同时极端」。
    # 预注册网格，与其余网格一样实验前固定、不由测试集选点。
    Cprime_lam_grid: Tuple[float, ...] = tuple(np.geomspace(1e-8, 0.5, 100))
    # 报告式(22) 用 Copula 换算门限时的唯一自由参数：目标多点事件概率 p*
    C_star_grid: Tuple[float, ...] = tuple(np.geomspace(1e-5, 0.2, 100))
    early_tolerance_min: float = 5.0

    @property
    def n_steps(self) -> int:
        return int(self.n_days * 24 * 3600 / self.interval_sec)

    @property
    def b_q_grid(self) -> Tuple[float, ...]:
        """B 的分位点网格随 θ̂ 口径切换，两套都是实验前固定的。"""
        return self.B_q_grid_decluster if self.use_decluster_rate else self.B_q_grid

    @property
    def early_tolerance_steps(self) -> int:
        return int(round(self.early_tolerance_min * 60 / self.interval_sec))


@dataclass
class Split:
    train_end: int
    val_end: int


# ============================================================
# 2. 数据生成：危险段不跨 train/val/test 边界
# ============================================================

def ar1(n, mu, rho, sigma, rng):
    x = np.empty(n)
    x[0] = mu + rng.normal(0, sigma / math.sqrt(max(1e-10, 1 - rho**2)))
    eps = rng.normal(0, sigma, n)
    for t in range(1, n):
        x[t] = mu + rho * (x[t - 1] - mu) + eps[t]
    return x


def make_safe_danger_segments(cfg: SimConfig, rng: np.random.Generator):
    """在三个时间分区内部生成危险段，避免真值事件跨分割边界。"""
    n = cfg.n_steps
    tr = int(n * 0.60)
    va = int(n * 0.80)
    blocks = [(0, tr), (tr, va), (va, n)]

    # 让三个区都有事件，测试集不会因为边界截断而产生事件定义偏差
    counts = [round(cfg.n_danger * 0.60), round(cfg.n_danger * 0.20)]
    counts.append(cfg.n_danger - sum(counts))

    segments = []
    occupied = np.zeros(n, dtype=bool)
    for (lo, hi), count in zip(blocks, counts):
        attempts = 0
        while sum(1 for s, e, _ in segments if lo <= s < hi) < count:
            attempts += 1
            if attempts > count * 5000:
                raise RuntimeError("无法在指定分区内放置足够危险段，请降低危险段长度或数量。")
            L = int(rng.integers(cfg.min_len, cfg.max_len + 1))
            margin = 30
            if hi - lo <= L + 2 * margin:
                raise RuntimeError("分区长度不足以生成危险段。")
            start = int(rng.integers(lo + margin, hi - L - margin + 1))
            end = start + L
            if occupied[max(lo, start - 10):min(hi, end + 10)].any():
                continue
            occupied[start:end] = True
            kind = "local" if rng.random() < cfg.danger_ratio_local else "coupled"
            segments.append((start, end, kind))
    segments.sort()
    return segments


def smooth_pulse(L: int):
    ramp = max(5, L // 5)
    s = np.ones(L)
    s[:ramp] = np.linspace(0, 1, ramp)
    s[-ramp:] = np.linspace(1, 0, ramp)
    return s


def generate_dataset(cfg: SimConfig):
    rng = np.random.default_rng(cfg.seed)
    T = cfg.n_steps

    common = ar1(T, 0.0, 0.98, 0.035, rng)
    log_x = np.zeros((T, cfg.n_sensors))
    for j in range(cfg.n_sensors):
        own = ar1(T, cfg.mu_log[j], cfg.rho[j], cfg.sigma_log[j], rng)
        own_center = own - cfg.mu_log[j]
        s = float(np.clip(cfg.cross_strength, 0.0, 0.95))
        scale = np.std(own_center) / max(np.std(common), 1e-12)
        log_x[:, j] = (
            cfg.mu_log[j]
            + math.sqrt(1 - s) * own_center
            + math.sqrt(s) * common * scale
            + rng.normal(0, cfg.sigma_noise, T)
        )
    X = np.exp(log_x)

    segments = make_safe_danger_segments(cfg, rng)
    # 真值标签取各测点实际抬升区间的并集：coupled 段的第 j 个测点抬升到
    # end + j*lag 才结束，旧版只标 [start, end) 会漏标后半段抬升。
    elev = np.zeros(T, dtype=bool)
    for start, end, kind in segments:
        if kind == "local":
            j = int(rng.integers(0, cfg.n_sensors))
            X[start:end, j] += cfg.local_amp * smooth_pulse(end - start)
            elev[start:end] = True
        elif cfg.coupled_layout == "independent":
            # 各测点抬升窗口的位置各自独立抽取，不以同一区间为共同载体。
            # 这是「独立边际」对照的必要条件：只要三测点共享同一个确定性
            # 抬升剖面，观测序列就正相关（实测段内 τ≈0.21），而那与随机错位
            # 无关——错位独立并不能消除共同剖面。窗口按段中心独立偏移，
            # 因此各传感器过程相互独立，偶然重叠也算独立抽样。
            L = end - start
            half = max(1, L // 2)
            for j in range(cfg.n_sensors):
                off = int(rng.integers(-half, half + 1))
                s = max(0, start + off)
                e = min(T, end + off)
                if e - s >= 2:
                    X[s:e, j] += cfg.coupled_amp * smooth_pulse(e - s)
                    elev[s:e] = True
        else:
            max_lag = cfg.propagation_lag * (cfg.n_sensors - 1)
            for j in range(cfg.n_sensors):
                if cfg.coupled_jitter:
                    # 各测点独立随机错位：切断「固定 j*lag」制造的真实同期依赖
                    off = int(rng.integers(0, max_lag + 1))
                else:
                    off = j * cfg.propagation_lag
                s = start + off
                e = min(end + off, T)
                if s < T:
                    X[s:e, j] += cfg.coupled_amp * smooth_pulse(e - s)
                    elev[s:e] = True
    return X, elev, segments


def split_time_series(X, y):
    n = len(X)
    tr = int(n * 0.60)
    va = int(n * 0.80)
    return {
        "train": (X[:tr], y[:tr]),
        "val": (X[tr:va], y[tr:va]),
        "test": (X[va:], y[va:]),
    }, Split(tr, va)


# ============================================================
# 3. 游程去聚集 + GPD
# ============================================================

def decluster_exceedances(x, u, run_length):
    idx = np.flatnonzero(np.asarray(x) > u)
    if len(idx) == 0:
        return np.array([]), np.array([], dtype=int)
    clusters = []
    cur = [idx[0]]
    for k in idx[1:]:
        if k - cur[-1] <= run_length:
            cur.append(k)
        else:
            clusters.append(cur)
            cur = [k]
    clusters.append(cur)
    excess = np.array([np.max(np.asarray(x)[c]) - u for c in clusters])
    peak_idx = np.array([c[int(np.argmax(np.asarray(x)[c]))] for c in clusters], dtype=int)
    return excess, peak_idx


def mrl_diagnostic(x, q_grid):
    x = np.asarray(x)
    rows = []
    for q in q_grid:
        u = float(np.quantile(x, q))
        e = x[x > u] - u
        rows.append({"q": q, "u": u, "n": len(e), "mrl": np.mean(e) if len(e) else np.nan,
                     "se": np.std(e, ddof=1) / math.sqrt(len(e)) if len(e) > 1 else np.nan})
    return pd.DataFrame(rows)


def fit_gpd_at_threshold(x, u, run_length):
    excess, peak_idx = decluster_exceedances(x, u, run_length)
    if len(excess) < 20:
        return None
    xi, loc, beta = stats.genpareto.fit(excess, floc=0)
    if beta <= 0 or not np.isfinite(xi) or not np.isfinite(beta):
        return None
    ks = stats.kstest(excess, "genpareto", args=(xi, 0, beta))
    # 极值指数 θ̂ = n_c / n_u：去聚集簇数比原始超越次数
    n_raw = int(np.sum(np.asarray(x) > u))
    theta_hat = len(excess) / n_raw if n_raw else np.nan
    return {
        "u": float(u), "shape": float(xi), "scale": float(beta),
        "n_clusters": int(len(excess)), "n_raw": n_raw, "theta_hat": float(theta_hat),
        "excess": excess,
        "peak_idx": peak_idx, "ks_stat": float(ks.statistic), "ks_pvalue": float(ks.pvalue),
    }


def parameter_stability(x, q_grid, run_length):
    rows = []
    for q in q_grid:
        u = float(np.quantile(x, q))
        fit = fit_gpd_at_threshold(x, u, run_length)
        rows.append({
            "q": q, "u": u,
            "shape": np.nan if fit is None else fit["shape"],
            "scale": np.nan if fit is None else fit["scale"],
            "n_clusters": 0 if fit is None else fit["n_clusters"],
            "n_raw": 0 if fit is None else fit["n_raw"],
            "theta_hat": np.nan if fit is None else fit["theta_hat"],
            "ks_pvalue": np.nan if fit is None else fit["ks_pvalue"],
        })
    return pd.DataFrame(rows)


def choose_pot_threshold(x, cfg: SimConfig):
    mrl_q = np.linspace(0.90, 0.99, 37)
    mrl = mrl_diagnostic(x, mrl_q)
    stab = parameter_stability(x, cfg.pot_q_grid, cfg.run_length)

    # 选择规则：优先在“样本仍足够 + KS 不差 + shape 相对稳定”的候选中取最低阈值。
    valid = stab[(stab.n_clusters >= 30) & np.isfinite(stab["shape"]) & np.isfinite(stab["ks_pvalue"])]
    selected_q = None
    if len(valid):
        for q in cfg.pot_q_grid:
            row = valid[valid.q == q]
            if row.empty:
                continue
            xi = float(row["shape"].iloc[0])
            higher = valid[valid.q >= q]
            med = float(np.median(higher.shape))
            stable = abs(xi - med) <= 0.25
            if float(row["ks_pvalue"].iloc[0]) >= 0.05 and stable:
                selected_q = q
                break
        if selected_q is None:
            selected_q = float(valid.sort_values(["ks_pvalue", "n_clusters"], ascending=[False, False]).iloc[0].q)
    else:
        raise ValueError("所有 POT 候选阈值的去聚集样本都不足。")

    u = float(np.quantile(x, selected_q))
    fit = fit_gpd_at_threshold(x, u, cfg.run_length)
    if fit is None:
        raise ValueError("选定 POT 阈值后 GPD 拟合失败。")
    fit["q"] = selected_q
    fit["mrl"] = mrl
    fit["stability"] = stab
    fit["raw_tail_prob"] = 1.0 - selected_q                       # ζ_u：式(8) 的边际口径
    fit["cluster_tail_prob"] = fit["theta_hat"] * (1.0 - selected_q)  # θ̂ζ_u：式(7) 的重现水平口径
    return fit


def bootstrap_gpd_ci(excess, n_boot=200, seed=1234):
    rng = np.random.default_rng(seed)
    xi = []
    beta = []
    e = np.asarray(excess)
    for _ in range(n_boot):
        sample = rng.choice(e, size=len(e), replace=True)
        try:
            a, _, b = stats.genpareto.fit(sample, floc=0)
            if b > 0 and np.isfinite(a) and np.isfinite(b):
                xi.append(a); beta.append(b)
        except Exception:
            pass
    if len(xi) < max(30, n_boot // 5):
        return {"xi_ci": [np.nan, np.nan], "scale_ci": [np.nan, np.nan], "n_valid": len(xi)}
    return {
        "xi_ci": np.quantile(xi, [0.025, 0.975]).tolist(),
        "scale_ci": np.quantile(beta, [0.025, 0.975]).tolist(),
        "n_valid": len(xi),
    }


def gpd_quantile(model, q, use_decluster=None):
    """式(8)：P(X ≤ q_p) = p 的边际分位点（use_decluster=False）。

    use_decluster=True 时把 ζ_u 换成 θ̂ζ_u，得到的是「每 T 个观测才出现一次
    独立超越事件」意义下的水平（式(7) 的重现水平口径）：P(X > 返回值) = (1-p)/θ̂，
    当 (1-p) ≥ θ̂ζ_u 时该水平落到 POT 阈值以下、超出 GPD 外推区间，返回 nan。
    """
    q0 = model["q"]
    if q <= q0:
        raise ValueError("GPD 外推分位点必须高于 POT 阈值分位点。")
    if use_decluster is None:
        use_decluster = bool(model.get("use_decluster_rate", False))
    pu = model["cluster_tail_prob"] if use_decluster else model["raw_tail_prob"]
    tail_cond = (1.0 - q) / pu
    if tail_cond >= 1.0:
        return float("nan")
    xi, beta, u = model["shape"], model["scale"], model["u"]
    if abs(xi) < 1e-8:
        y = -beta * np.log(tail_cond)
    else:
        y = beta / xi * (tail_cond ** (-xi) - 1.0)
    return float(u + y)


def gpd_return_level(model, T_steps, interval_sec=30, use_decluster=True):
    """式(7) 重现水平：平均每 T_steps 个观测（按独立超越事件计）出现一次的水平。

    θ̂ 修正在此处成立：P(X > x_T) = 1/(θ̂ T)，故 x_T = u + (β/ξ)[(T θ̂ζ_u)^ξ - 1]。
    """
    if use_decluster:
        pu = model["cluster_tail_prob"]
    else:
        pu = model["raw_tail_prob"]
    xi, beta, u = model["shape"], model["scale"], model["u"]
    lam = T_steps * pu
    if abs(xi) < 1e-8:
        y = beta * np.log(lam)
    else:
        y = beta / xi * (lam ** xi - 1.0)
    return float(u + y)


# ============================================================
# 4. 半参数边际 PIT：主体经验分布 + GPD 尾部
# ============================================================

class TailMarginal:
    """式(10)(11) 的半参数边缘分布：阈值以下经验 CDF，阈值以上 GPD 外推。

    接缝处必须连续，否则它不是一个合法的分布函数、PIT 也不是均匀的。
    旧实现下支用 ranks/(n+1)、上支用标称 q0，在 x=u 处跳约 q0/(n+1)；
    这里把上支的基准改为实测的 F_emp(u) = n_u/(n+1)，使 F(u) 两侧相等。
    """

    def __init__(self, x_train, gpd_model):
        self.x = np.sort(np.asarray(x_train))
        self.n = len(self.x)
        self.model = gpd_model
        self.u = gpd_model["u"]
        self.q = gpd_model["q"]
        # 实测（而非标称）的阈值以下概率，n_u = #{x ≤ u}
        n_u = int(np.searchsorted(self.x, self.u, side="right"))
        self.p_below = n_u / (self.n + 1.0)

    def cdf(self, x):
        x = np.asarray(x, dtype=float)
        out = np.empty_like(x)
        below = x <= self.u
        ranks = np.searchsorted(self.x, x[below], side="right")
        out[below] = ranks / (self.n + 1.0)
        z = np.maximum(x[~below] - self.u, 0.0)
        xi, beta = self.model["shape"], self.model["scale"]
        if abs(xi) < 1e-8:
            surv = np.exp(-z / beta)
        else:
            surv = np.maximum(1 + xi * z / beta, 1e-12) ** (-1 / xi)
        # 1 - F(x) = (1 - F_emp(u))·P(X > x | X > u)，在 x=u 处与上支连续
        out[~below] = 1.0 - (1.0 - self.p_below) * surv
        return np.clip(out, 1e-8, 1 - 1e-8)

    def ppf(self, p):
        """分位函数，式(8) F(q_p)=p 的反解，用于报告的 Copula 门限换算。

        必须与 cdf 用同一套映射：报告第五（七）要求「由式(17) 解出 α*，再由各测点
        边缘分布反解门限 q_{p,i}＝F_i^{-1}(1-α*)」。若反解用的映射与 pit_observations
        的映射不是同一个，解出的门限代入式(17) 就不再命中目标概率 p*。
        """
        p = float(np.clip(p, 1e-12, 1.0 - 1e-12))
        if p <= self.p_below:
            # 经验支：F = ranks/(n+1)，故 rank r 对应 x[r-1]
            r = min(max(int(math.ceil(p * (self.n + 1.0))), 1), self.n)
            return float(self.x[r - 1])
        # 上支：1 - F = (1 - p_below)·surv = 1 - p
        surv = (1.0 - p) / (1.0 - self.p_below)
        xi, beta = self.model["shape"], self.model["scale"]
        z = -beta * math.log(surv) if abs(xi) < 1e-8 else beta / xi * (surv ** (-xi) - 1.0)
        return float(self.u + z)


# ============================================================
# 5. Copula：Gumbel（λ_U>0）与 Frank（λ_U=0） + GOF
#
# 符号提醒：报告里 θ 既被用作 Gumbel 参数、又被用作极值指数。
# 本文件中极值指数一律写 theta_hat，Copula 参数写 theta。
# ============================================================

def pseudo_observations(X):
    """秩型伪观测；仅作为诊断对照保留，Copula 拟合改用 TailMarginal 的 PIT。"""
    X = np.asarray(X)
    n, d = X.shape
    U = np.empty_like(X, dtype=float)
    for j in range(d):
        U[:, j] = stats.rankdata(X[:, j], method="average") / (n + 1.0)
    return np.clip(U, 1e-8, 1 - 1e-8)


def pit_observations(X, marginals):
    """式(10)(11) 的 PIT：半参数边际（经验 CDF + GPD 尾部）给出的 u_i = F_i(x_i)。

    与秩伪观测的差别在尾部：秩只给到 i/(n+1)，而 U_i 必须是 F_i(x_i) 本身，
    式(16) 的 1 - u_i - u_j + C(u_i,u_j) 才等于 P(X_i > q_i, X_j > q_j)。
    """
    return np.column_stack([m.cdf(X[:, j]) for j, m in enumerate(marginals)])


def kendall_tau_matrix(X):
    d = X.shape[1]
    tau = np.eye(d)
    p = np.ones((d, d))
    for i in range(d):
        for j in range(i + 1, d):
            t, pv = stats.kendalltau(X[:, i], X[:, j])
            tau[i, j] = tau[j, i] = t
            p[i, j] = p[j, i] = pv
    return tau, p


def gumbel_theta_from_tau(tau):
    tau = float(tau)
    if tau <= 0:
        # 报告式(13) 在 τ≤0 时无解（Gumbel 只覆盖正相依）；此处显式退回独立 Copula
        return 1.0
    return 1.0 / (1.0 - min(tau, 0.999999))


def gumbel_C(U, theta):
    U = np.clip(np.asarray(U, dtype=float), 1e-12, 1 - 1e-12)
    z = np.sum((-np.log(U)) ** theta, axis=-1)
    return np.exp(-(z ** (1.0 / theta)))


def frank_C(U, theta):
    """Frank Copula 的 d 维 CDF：C(u) = -1/θ ln(1 + Π(e^{-θu_i}-1)/(e^{-θ}-1)^{d-1})。"""
    U = np.clip(np.asarray(U, dtype=float), 1e-12, 1 - 1e-12)
    theta = float(theta)
    if abs(theta) < 1e-8:
        return np.prod(U, axis=-1)
    a = math.expm1(-theta)
    prod = np.prod(np.expm1(-theta * U), axis=-1)
    return -np.log1p(prod / a ** (U.shape[-1] - 1)) / theta


def _debye1(t):
    """一阶 Debye 函数 D1(t) = (1/t)∫₀ᵗ x/(eˣ-1) dx。"""
    if t < 1e-6:
        return 1.0 - t / 4.0
    if t > 40.0:
        return (math.pi ** 2 / 6.0 - math.exp(-t) * (t + 1.0)) / t
    x = np.linspace(0.0, t, 4001)
    with np.errstate(invalid="ignore", divide="ignore"):
        f = np.where(x < 1e-12, 1.0, x / np.expm1(x))
    return float(np.trapezoid(f, x) / t)


def frank_tau_of_theta(theta):
    """Frank Copula 的 Kendall τ = ±(1 - 4/|θ| + 4D1(|θ|)/|θ|)，τ(-θ) = -τ(θ)。"""
    if abs(theta) < 1e-9:
        return 0.0
    t = abs(theta)
    val = 1.0 - 4.0 / t + 4.0 * _debye1(t) / t
    return val if theta > 0 else -val


def frank_theta_from_tau(tau):
    """由 Kendall τ 反解 Frank 参数（τ 在 θ 上单调增，τ(-θ) = -τ(θ)）。"""
    tau = float(tau)
    if tau == 0.0:
        return 0.0
    sign = 1.0 if tau > 0 else -1.0
    target = abs(tau)
    if target >= 1.0:
        raise ValueError("Frank Copula 的 |τ| 必须小于 1。")
    hi = 1.0
    while frank_tau_of_theta(hi) < target:
        hi *= 2.0
        if hi > 1e5:
            raise ValueError("Frank 参数反解超出范围。")
    th = brentq(lambda t: frank_tau_of_theta(t) - target, 1e-9, hi, xtol=1e-12, rtol=1e-12)
    return sign * float(th)


def sample_gumbel(n, d, theta, rng):
    """Marshall-Olkin 正稳定表示生成 Gumbel Copula 样本。"""
    alpha = 1.0 / theta
    if abs(alpha - 1.0) < 1e-10:
        V = np.ones(n)
    else:
        U0 = rng.uniform(-np.pi / 2, np.pi / 2, size=n)
        W = rng.exponential(size=n)
        V = (
            np.sin(alpha * (U0 + np.pi / 2))
            / np.cos(U0) ** (1 / alpha)
            * (np.cos(U0 - alpha * (U0 + np.pi / 2)) / W) ** ((1 - alpha) / alpha)
        )
    E = rng.exponential(size=(n, d))
    return np.clip(np.exp(-(E / V[:, None]) ** alpha), 1e-8, 1 - 1e-8)


def sample_frank(n, d, theta, rng):
    """Archimedean 条件分布逐分量逆变换生成 Frank Copula 样本（d ≤ 3）。

    取 ψ(s) = -1/θ ln(1 + A e^{-s})，A = e^{-θ}-1，记 β(u) = e^{-φ(u)} = (e^{-θu}-1)/A、
    B_i = A·Π_{k≤i}β(u_k)（B<0）。则
      C(u_2|u_1)     = ψ'(s_2)/ψ'(s_1)     = [B_2/(1+B_2)] / [B_1/(1+B_1)]
      C(u_3|u_1,u_2) = ψ''(s_3)/ψ''(s_2)   = [B_3/(1+B_3)²] / [B_2/(1+B_2)²]
    二者都对 β 单调增且在 β=1 处等于 1，故用 brentq 反解。

    适用范围 θ ≥ 0：θ<0 时 Frank 的 CDF 仍然合法，但 ψ 不再是 Laplace 变换，
    上面这条条件式在 β∈(0,1) 上会超出 [0,1] 且非单调（|θ| 越大越明显），
    故负相依分支不在此实现，直接报错而不是返回错样本。
    """
    if d > 3:
        raise NotImplementedError("Frank 条件抽样目前只实现 d ≤ 3。")
    if theta < -1e-8:
        raise NotImplementedError(
            "sample_frank 仅实现 θ ≥ 0 分支（本研究的 τ>0 恒对应 θ>0）。")
    if abs(theta) < 1e-8:
        return np.clip(rng.uniform(size=(n, d)), 1e-8, 1 - 1e-8)
    A = math.expm1(-theta)
    W = rng.uniform(size=(n, d))
    out = np.empty((n, d))
    for i in range(n):
        u1 = W[i, 0]
        b1 = math.expm1(-theta * u1) / A
        B1 = A * b1
        if d == 1:
            out[i, 0] = u1
            continue
        B2 = W[i, 1] * B1 / (1.0 + B1 * (1.0 - W[i, 1]))
        b2 = B2 / B1
        u2 = -math.log1p(A * b2) / theta
        out[i, 0], out[i, 1] = u1, min(max(u2, 1e-12), 1 - 1e-12)
        if d == 2:
            continue
        C2 = 1.0 + B2
        g = lambda b: (B2 * b) / (1.0 + B2 * b) ** 2 * (C2 ** 2 / B2) - W[i, 2]
        b3 = brentq(g, 1e-12, 1.0, xtol=1e-12, rtol=1e-12)
        u3 = -math.log1p(A * b3) / theta
        out[i, 2] = min(max(u3, 1e-12), 1 - 1e-12)
    return np.clip(out, 1e-8, 1 - 1e-8)


def copula_lambda_u(family, theta):
    """上尾相依系数 λ_U：Gumbel = 2 - 2^{1/θ}；Frank = 0。"""
    if family == "gumbel":
        return float(2.0 - 2.0 ** (1.0 / theta))
    if family == "frank":
        return 0.0
    raise ValueError(family)


def copula_cvm(U, theta, grid=None, cdf_fn=gumbel_C):
    """在固定的少量观测网格上计算经验 Copula 与理论 Copula 的 CvM 距离。"""
    U = np.asarray(U)
    if grid is None:
        rng = np.random.default_rng(20260924)
        idx = rng.choice(len(U), size=min(200, len(U)), replace=False)
        grid = U[idx]
    emp = np.empty(len(grid))
    for i, g in enumerate(grid):
        emp[i] = np.mean(np.all(U <= g, axis=1))
    theo = cdf_fn(grid, theta)
    return float(np.mean((emp - theo) ** 2))


def copula_gof(U, theta, B=50, seed=12345, family="gumbel"):
    """参数估计后的近似 parametric-bootstrap GOF。为控制计算量使用固定 200 点网格。"""
    sampler = sample_gumbel if family == "gumbel" else sample_frank
    cdf_fn = gumbel_C if family == "gumbel" else frank_C
    if family == "frank" and theta < 0:
        # τ<0 时 Frank 无法条件抽样（见 sample_frank），GOF 不给出，如实标注
        return {"cvm": np.nan, "pvalue": np.nan, "n_boot": 0, "n_sample": 0, "grid_size": 0,
                "family": family, "unsupported_reason": "frank 负相依分支不支持参数 bootstrap"}
    rng = np.random.default_rng(seed)
    n_use = min(len(U), 3000)
    idx = rng.choice(len(U), size=n_use, replace=False)
    U0 = U[idx]
    grid_idx = rng.choice(n_use, size=min(200, n_use), replace=False)
    grid = U0[grid_idx]
    observed = copula_cvm(U0, theta, grid=grid, cdf_fn=cdf_fn)
    sims = []
    for _ in range(B):
        Us = sampler(n_use, U.shape[1], theta, rng)
        sims.append(copula_cvm(Us, theta, grid=grid, cdf_fn=cdf_fn))
    sims = np.asarray(sims)
    p = (1 + np.sum(sims >= observed)) / (B + 1)
    return {"cvm": observed, "pvalue": float(p), "n_boot": B, "n_sample": n_use,
            "grid_size": len(grid), "family": family}


def fit_copula(U, family="gumbel", gof_boot=50, seed=20260924):
    """Kendall τ 反解参数（τ 对严格增的边际变换不变，故 PIT 口径不影响 τ）。"""
    if family not in ("gumbel", "frank"):
        raise ValueError(family)
    tau, p = kendall_tau_matrix(U)
    pair = np.array([tau[i, j] for i in range(U.shape[1]) for j in range(i + 1, U.shape[1])])
    tau_bar = float(np.mean(pair))
    if family == "gumbel":
        theta = gumbel_theta_from_tau(tau_bar)
    else:
        theta = frank_theta_from_tau(tau_bar)
    return {
        "family": family,
        "tau_matrix": tau,
        "p_matrix": p,
        "pair_tau": pair,
        "tau_mean": tau_bar,
        "theta": theta,
        "theta_of_tau_used": float(tau_bar),
        "lambda_u": copula_lambda_u(family, theta),
        "gof": copula_gof(U, theta, gof_boot, seed=seed, family=family),
    }


def fit_gumbel(U, gof_boot=50):
    return fit_copula(U, "gumbel", gof_boot)


def pit_uniformity(U):
    """PIT 均匀性诊断：给出名义 K-S 与其按有效样本量修正后的判定。

    本数据 lag-1 自相关约 0.97，K-S 的独立同分布前提不成立，名义 p 值会被极度高估
    （实测 p≈1e-29，而按 AR(1) 有效样本量 n_eff = n(1-ρ)/(1+ρ) ≈ 2400 计，
    5% 临界值 1.36/√n_eff ≈ 0.028，实测统计量 0.015 并不显著）。
    """
    U = np.asarray(U, dtype=float).ravel()
    n = len(U)
    ks = stats.kstest(U, "uniform")
    rho1 = float(np.corrcoef(U[:-1], U[1:])[0, 1]) if n > 2 else 0.0
    n_eff = n * (1 - rho1) / (1 + rho1) if rho1 < 1 else 1.0
    crit = 1.36 / math.sqrt(max(n_eff, 1.0))
    return {
        "n": int(n), "ks_stat": float(ks.statistic), "ks_pvalue_naive": float(ks.pvalue),
        "lag1_autocorr": rho1, "n_eff": float(n_eff), "ks_crit_n_eff_5pct": float(crit),
        "significant_at_n_eff": bool(ks.statistic > crit),
    }


def all_three_joint_prob(U, theta, cdf_fn=gumbel_C):
    """式(19) 的 Λ_3(t)：三测点**同时**超越各自当前水平的联合超越概率。

    报告 式(19)：Λ_3(t) = 1 - Σ a_i(t) + Σ_{i<j} C_ij(a_i(t),a_j(t)) - C(a_1,a_2,a_3)，
    其中 a_i(t) = F_i(x_i(t))，即 pit_observations 返回的 u_i。故
        Λ_3(t) = P(X_1 > x_1(t), X_2 > x_2(t), X_3 > x_3(t))。
    三个测点同时处于高浓度时取小值、常态时接近 1，是「同时极端程度」的连续度量。
    注意它与式(17) 的 P_≥2（至少两个）不是同一个泛函：P_3 ≤ P_≥2，恒有 Λ_3 ≤ P_≥2。
    """
    U = np.asarray(U, dtype=float)
    if U.ndim == 1:
        U = U[None, :]
    return np.clip(1.0 - np.sum(U, axis=1) + _pair_copula_sum(U, theta, cdf_fn)
                   - cdf_fn(U, theta), 0.0, 1.0)


def _pair_copula_sum(U, theta, cdf_fn):
    """Σ_{i<j} C_ij(a_i, a_j)，三元模型里各二元 Copula 共用同一 θ。"""
    d = U.shape[1]
    return sum(cdf_fn(U[:, [i, j]], theta) for i in range(d) for j in range(i + 1, d))


def at_least_k_tail_prob(U, theta, k=2, cdf_fn=gumbel_C):
    """对 d=3 计算当前观测水平以上“至少 k 个”联合超越概率（式(16)(17)）。"""
    U = np.asarray(U, dtype=float)
    if U.ndim == 1:
        U = U[None, :]
    n, d = U.shape
    if d != 3 or k != 2:
        raise NotImplementedError("当前严格版实现 d=3 且 k=2。")
    u1, u2, u3 = U[:, 0], U[:, 1], U[:, 2]
    pairs = []
    for a, b in [(u1, u2), (u1, u3), (u2, u3)]:
        C2 = cdf_fn(np.column_stack([a, b]), theta)
        pairs.append(1 - a - b + C2)
    # 三维容斥项 P(X1>x1, X2>x2, X3>x3) 就是式(19) 的 Λ_3(t)
    return np.clip(np.sum(pairs, axis=0) - 2 * all_three_joint_prob(U, theta, cdf_fn), 0.0, 1.0)


def at_least_two_prob(alpha, theta, cdf_fn=gumbel_C):
    """式(17) 在三测点同取边缘超越概率 α 时的解析值（不依赖数据）。

    a = 1 - α 代入容斥可得闭式：P_{≥2}(α) = 1 - 3·C_2(a,a) + 2·C_3(a,a,a)。
    关于 α 严格单调递增（α=0 时为 0、α=1 时为 1），故可二分反解——
    这正是报告第五（七）里 Copula 的职责。
    """
    a = 1.0 - float(alpha)
    C2 = float(np.asarray(cdf_fn(np.array([[a, a]]), theta)).ravel()[0])
    C3 = float(np.asarray(cdf_fn(np.array([[a, a, a]]), theta)).ravel()[0])
    return float(np.clip(1.0 - 3.0 * C2 + 2.0 * C3, 0.0, 1.0))


def joint_target_thresholds(marginals, theta, p_star, cdf_fn=gumbel_C):
    """报告第五（七）：由式(17) 解出共同边缘超越概率 α*，再由边缘反解各测点门限。

    「给定目标概率 p*，由式(17) 解出各测点共同的边缘超越概率 α*，使 P_{≥2}(α*)=p*，
    再由各测点边缘分布反解出对应门限 q_{p,i}」——这是报告给 Copula 规定的唯一职责
    （它不进入式(22) 的逐时刻判别式）。因此门限由 Copula 参数 θ 决定：正相依越强，
    同一边际 α 下的 P_{≥2} 越大，要达到同一 p* 就得取更小的 α*、更高的门限。
    """
    if not 0.0 < p_star < 1.0:
        raise ValueError("目标联合概率 p* 必须在 (0,1) 内。")
    lo, hi = 1e-12, 1.0 - 1e-12
    if at_least_two_prob(hi, theta, cdf_fn) < p_star:
        raise ValueError("P_{≥2} 在 α→1 处仍小于 p*，无法反解。")
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if abs(at_least_two_prob(mid, theta, cdf_fn) - p_star) < 1e-14:
            break
        if at_least_two_prob(mid, theta, cdf_fn) < p_star:
            lo = mid
        else:
            hi = mid
    alpha_star = 0.5 * (lo + hi)
    q = np.array([m.ppf(1.0 - alpha_star) for m in marginals], dtype=float)
    return alpha_star, q


# ============================================================
# 6. 三规则
# ============================================================

def rule_A(X, multiplier, fixed_thresholds):
    return np.any(X >= multiplier * np.asarray(fixed_thresholds), axis=1)


def rule_B(X, q):
    return np.any(X >= q, axis=1)


def rule_C(X, q, p_c, marginals, theta, cdf_fn=gumbel_C):
    """【旧实现，保留作对照】B 的门限 ∧ 式(17) 的 P_≥2 ≤ p_c。

    与报告不符之处有两处，故不再作为报告规则的实现：
      (1) 边缘门限用的是 B 的 q_{p,i}，而报告要求 C 自己的更低门限 q_{pC,i}；
      (2) 联合量用的是式(17) 的 P_≥2（至少两个），而报告的 C′ 用式(19) 的 Λ_3（三个同时）。
    另外它与 rule_B 共用同一组门限，故其报警步集恒为 B 的子集——在此实现下
    「C 的漏报率低于 B」在结构上就不可能成立。
    """
    U = pit_observations(X, marginals)
    b = rule_B(X, q)
    p_joint = at_least_k_tail_prob(U, theta, k=2, cdf_fn=cdf_fn)
    return b & (p_joint <= p_c), p_joint


def rule_Cprime(X, q_pC, lam_c, marginals, theta, cdf_fn=gumbel_C):
    """报告式(19) 的 C′：C'_t=1 ⟺ Λ_3(t) ≤ λ_c 且 ∃i: x_i(t) ≥ q_{pC,i}。

    与旧实现的两点差别正是报告强调的关键：边缘门限取 C 自己的 q_{pC,i}（等误报率
    对齐下比 B 低），联合量取式(19) 的 Λ_3（三个同时）而非式(17) 的 P_≥2（至少两个）。
    报告称该写法的唯一自由参数是 λ_c，故此处 pC 沿用式(22) 标定出的 C 自己的分位水平，
    只对 λ_c 做标定。
    """
    b = np.any(np.asarray(X) >= np.asarray(q_pC), axis=1)
    lam = all_three_joint_prob(pit_observations(X, marginals), theta, cdf_fn)
    return rule_Cprime_from_score(lam, b, lam_c), lam


def rule_Cprime_from_score(lam, b, lam_c):
    """Λ_3 与 λ_c 无关，故先算一次 Λ_3 再扫 λ_c，避免重复计算 Copula。"""
    return np.asarray(b) & (np.asarray(lam) <= lam_c)


def rule_C_count(X, q):
    """报告式(22) 的字面判据：C(t)=1 ⟺ #{i: x_i(t) ≥ q_{pC,i}} ≥ 2。

    注意它完全不使用 Copula，只是「至少两个测点各自超过门限」的计数规则。
    """
    return np.sum(np.asarray(X) >= np.asarray(q), axis=1) >= 2


def joint_score(X, marginals, theta, cdf_fn=gumbel_C):
    """式(16)(17) 的联合超越概率 p_joint；只依赖 Copula 与边缘，与门限 q 无关。"""
    U = pit_observations(X, marginals)
    return at_least_k_tail_prob(U, theta, k=2, cdf_fn=cdf_fn)


def rule_C_from_score(score, b, p_c):
    """b & (p_joint ≤ p_c)：把 q 与 p_c 两个自由度分开，避免重复计算 p_joint。"""
    return np.asarray(b) & (np.asarray(score) <= p_c)


# ============================================================
# 7. 事件级评价
# ============================================================

def alarm_events(alarm):
    alarm = np.asarray(alarm, dtype=bool)
    if len(alarm) == 0:
        return []
    starts = np.flatnonzero(alarm & ~np.r_[False, alarm[:-1]])
    ends = np.flatnonzero(alarm & ~np.r_[alarm[1:], False]) + 1
    return list(zip(starts, ends))


def danger_segments_from_mask(mask):
    mask = np.asarray(mask, dtype=bool)
    if len(mask) == 0:
        return []
    starts = np.flatnonzero(mask & ~np.r_[False, mask[:-1]])
    ends = np.flatnonzero(mask & ~np.r_[mask[1:], False]) + 1
    return list(zip(starts, ends))


def match_events(alarm, danger_segments, early_tolerance_steps):
    """报警段与危险段的重叠匹配。

    旧实现按「报警段的**起点**落在 [ds-tol, de) 内」匹配，且每个报警段只消耗一次。
    那个口径在报警集合包含关系下不单调：把一段长报警切碎，可以凭空造出一个落在
    窗口内的新起点，于是**子集反而可能多检出事件**（已用本函数的最小反例验证：
    FNR(B)=1.0 而 FNR(C)=0.0，尽管 C ⊂ B）。因此 event_fnr / FAR / mean_delay
    都不能再被当作「集合越全越好」的单调指标。

    改为按重叠匹配后：报警步集变大 ⇔ 报警区间之并集变大，故「危险段被检出」单调不减，
    事件级漏报率对报警集合单调（time_fnr 本来就单调）。容差只加在危险段起点一侧，
    表示允许提前报警。不再做贪心消耗，一个报警段可同时覆盖多个危险段。
    """
    events = alarm_events(alarm)
    matched_event = np.zeros(len(events), dtype=bool)
    detected = np.zeros(len(danger_segments), dtype=bool)
    delays = []
    for ei, (es, ee) in enumerate(events):
        for di, (ds, de) in enumerate(danger_segments):
            if es < de and ee > ds - early_tolerance_steps:
                matched_event[ei] = True
                detected[di] = True
    for ds, de in danger_segments:
        starts = [es for es, ee in events if es < de and ee > ds - early_tolerance_steps]
        if starts:
            delays.append(min(starts) - ds)
    return matched_event, detected, [d for d in delays if d >= 0]


# Early-FNR 的窗口（分钟）。含义：危险段开始后 T0 分钟内（含起点前的提前容忍）
# 一个报警步都没有，该段即算「早期漏报」。窗口与报警集合无关，所以该指标和
# event_fnr 一样对报警集合单调递减——报警集变大，早期漏报只可能变少。
EARLY_WINDOWS_MIN: Tuple[int, ...] = (5, 10, 20, 30)


def early_fnr(alarm, danger, tol_steps, window_steps):
    """危险段中「起点 window_steps 步内（含起点前 tol_steps 步的提前容忍）没有任何报警」
    的比例。这是**漏报率**口径，1.0 最差，与 event_fnr / time_fnr 同向。

    窗口右端必须裁到危险段终点 de。不裁的话，段长小于窗口时（本研究的危险段长 160—260 步，
    最长窗口 60 步，暂时碰不到；但短段配置会踩），危险已经结束之后的报警会被记成该段的
    「早期命中」——实测：危险段 [100,110)、报警在第 150 步，30 分钟窗口给出 early_fnr=0.0
    而 event_fnr=1.0。裁剪后窗口 ⊆ [ds-tol, de)，于是「整段没被检出 ⇒ 早期也没报」成立，
    early_fnr ≥ event_fnr 恒成立。

    空危险段返回 0.0，与 event_fnr 的哨兵值一致；不要返回 nan，它会静默污染下游 mean。
    """
    a = np.asarray(alarm, dtype=bool)
    if len(danger) == 0:
        return 0.0
    miss = 0
    for ds, de in danger:
        lo = max(0, int(ds) - tol_steps)
        hi = min(len(a), int(ds) + window_steps + 1, int(de))
        if hi <= lo or not a[lo:hi].any():
            miss += 1
    return miss / len(danger)


def evaluate_rule(alarm, danger_mask, interval_sec, early_tolerance_steps):
    danger = danger_segments_from_mask(danger_mask)
    events = alarm_events(alarm)
    matched_event, detected, delays = match_events(alarm, danger, early_tolerance_steps)

    n_danger = len(danger)
    event_fnr = (n_danger - int(detected.sum())) / max(n_danger, 1)

    total_danger_steps = sum(de - ds for ds, de in danger)
    covered = sum(int(np.sum(np.asarray(alarm)[ds:de])) for ds, de in danger)
    time_fnr = 1 - covered / max(total_danger_steps, 1)

    mean_delay = np.mean(delays) * interval_sec / 60 if delays else np.nan

    # FAR 按报告式(24)：FP / 测试集小时数。FP 是「未与任何危险段重叠的报警段」数。
    tp_alarm = int(matched_event.sum())
    fp = len(events) - tp_alarm
    hours = len(alarm) * interval_sec / 3600
    far = fp / max(hours, 1e-12)

    return {
        "event_fnr": float(event_fnr),
        "time_fnr": float(time_fnr),
        "far_per_hour": float(far),
        "mean_delay_min": float(mean_delay) if np.isfinite(mean_delay) else np.nan,
        **{"early_fnr_%dmin" % w: early_fnr(
            alarm, danger, early_tolerance_steps,
            int(round(w * 60 / interval_sec))) for w in EARLY_WINDOWS_MIN},
        "n_danger_events": int(n_danger),
        "n_alarm_events": int(len(events)),
        "n_matched": int(tp_alarm),
        "n_detected_danger": int(detected.sum()),
        "n_false_alarm_events": int(fp),
    }


# ============================================================
# 8. 参数标定：只用 validation；测试只画 DET
# ============================================================

def calibrate_by_far(X, y, candidates, target_far, cfg):
    """在预注册网格上挑离目标 FAR 最近的点；candidates = [(参数, 报警序列), ...]。"""
    et = cfg.early_tolerance_steps
    rows = []
    for param, alarm in candidates:
        m = evaluate_rule(alarm, y, cfg.interval_sec, et)
        rows.append({"param": param, "far_gap": abs(m["far_per_hour"] - target_far), "metrics": m})
    rows.sort(key=lambda r: (r["far_gap"], r["metrics"]["event_fnr"]))
    return rows


def b_threshold_candidates(X, cfg, gpd_models):
    """B 的 (p, 门限, 报警) 候选网格。

    两类点没有定义，一律丢弃并计数、不静默截断：
      (1) p ≤ 该数据实际拟合到的 POT 阈值分位 q₀ —— GPD 外推区间之外；
      (2) θ̂ 口径下 tail_cond ≥ 1 的点（会退回 POT 阈值以下）。
    """
    out, dropped = [], []
    q0_max = max(float(m["q"]) for m in gpd_models)
    for p in cfg.b_q_grid:
        if float(p) <= q0_max:
            dropped.append(float(p))
            continue
        q = np.array([gpd_quantile(m, float(p)) for m in gpd_models])
        if not np.all(np.isfinite(q)):
            dropped.append(float(p))
            continue
        out.append((float(p), q, rule_B(X, q)))
    return out, dropped


def validation_calibration(Xv, yv, cfg, gpd_models, marginals, theta, family="gumbel",
                           target: float = 0.05):
    """返回 A/B/C 及式(22) 计数规则的单一固定参数。参数只从 validation 确定。

    target 是各规则共同对齐的误报水平。默认 0.05 是报告明确给出的 FAR 示例之一，
    预注册为基准标定点；显式传入其它值只用于「同一流程在多个 f₀ 上的稳健性」检查，
    标定仍只发生在验证集上。
    """
    cdf_fn = gumbel_C if family == "gumbel" else frank_C

    # A：唯一自由参数 k
    A_rows = calibrate_by_far(
        Xv, yv, [(float(k), rule_A(Xv, k, cfg.fixed_thresholds)) for k in cfg.A_multiplier_grid],
        target, cfg)

    # B：唯一自由参数 p；门限由训练集 GPD 映射，不使用验证集分位数估计
    B_cands, B_dropped = b_threshold_candidates(Xv, cfg, gpd_models)
    if not B_cands:
        raise RuntimeError("B 的分位点网格全部落到 POT 阈值以下，无法标定。")
    B_rows = calibrate_by_far(Xv, yv, [(p, a) for p, q, a in B_cands], target, cfg)
    B_p = float(B_rows[0]["param"])
    B_q = next(q for p, q, a in B_cands if p == B_p)

    # C：唯一自由参数 p_c；边缘 EVT 门限固定为 B 的门限。
    # p_joint 与门限无关，只算一次再扫 p_c（不改变结果，只为省重复计算）。
    score_v = joint_score(Xv, marginals, theta, cdf_fn)
    b_v = rule_B(Xv, B_q)
    C_rows = calibrate_by_far(
        Xv, yv, [(float(pc), rule_C_from_score(score_v, b_v, float(pc))) for pc in cfg.C_pc_grid],
        target, cfg)

    # 式(22) 计数规则的唯一自由参数是自己的分位水平 p（与 B 共用 p 网格）。
    # 这就是报告指定的**主规则**：它的门限是 C 自己的、比 B 低的分位水平，
    # 因此它并不与 B 共用门限，也不恒为 B 的子集。
    C22_rows = calibrate_by_far(Xv, yv, [(p, rule_C_count(Xv, q)) for p, q, a in B_cands], target, cfg)
    C22_p = float(C22_rows[0]["param"])
    C22_q = next(q for p, q, a in B_cands if p == C22_p)

    # 报告 C′（式19）：边缘门限沿用 C 自己的 q_{pC,i}，只标定 λ_c；
    # 联合量用 Λ_3（三个同时），与旧实现的 P_≥2（至少两个）不同。
    Cprime_rows = calibrate_by_far(
        Xv, yv, [(float(lc), rule_Cprime(Xv, C22_q, float(lc), marginals, theta, cdf_fn)[0])
                 for lc in cfg.Cprime_lam_grid],
        target, cfg)

    # 报告第五（七）的 Copula 用法：由式(17) 反解 α* 再反解门限，判别式仍是式(22) 的计数。
    # 唯一自由参数是目标多点事件概率 p*；门限随 Copula 参数 θ 变化（这是 Copula 真正起作用的地方）。
    C_star_cands, C_star_q_of, C_star_alpha = [], {}, {}
    for p_star in cfg.C_star_grid:
        try:
            alpha_star, q_star = joint_target_thresholds(marginals, theta, float(p_star), cdf_fn)
        except ValueError:
            continue
        C_star_cands.append((float(p_star), rule_C_count(Xv, q_star)))
        C_star_q_of[float(p_star)] = q_star
        C_star_alpha[float(p_star)] = float(alpha_star)
    if not C_star_cands:
        raise RuntimeError("Copula 门限换算网格全部无法反解，无法标定 C_star。")
    C_star_rows = calibrate_by_far(Xv, yv, C_star_cands, target, cfg)
    C_star_p = float(C_star_rows[0]["param"])
    C_star_q = C_star_q_of[C_star_p]

    return {
        "target_far": target,
        "copula_family": family,
        "use_decluster_rate": bool(cfg.use_decluster_rate),
        "A": {"k": float(A_rows[0]["param"]), "validation_metrics": A_rows[0]["metrics"]},
        "B": {"p": B_p, "thresholds": B_q, "validation_metrics": B_rows[0]["metrics"],
              "n_grid_dropped": len(B_dropped), "dropped_grid_head": B_dropped[:5]},
        "C": {"p_c": float(C_rows[0]["param"]), "B_p": B_p, "thresholds": B_q,
              "validation_metrics": C_rows[0]["metrics"]},
        "C22": {"p": C22_p, "thresholds": C22_q, "validation_metrics": C22_rows[0]["metrics"]},
        "Cprime": {"lam_c": float(Cprime_rows[0]["param"]), "pC": C22_p, "thresholds": C22_q,
                   "validation_metrics": Cprime_rows[0]["metrics"]},
        "C_star": {"p_star": C_star_p, "alpha_star": C_star_alpha[C_star_p],
                   "thresholds": C_star_q, "validation_metrics": C_star_rows[0]["metrics"]},
    }


def calibrate_c_only(Xv, yv, cfg, marginals, theta, B_q, cdf_fn, target=0.05):
    """固定 B 的边缘门限，只为 C 标定 p_c；供 Copula 族对照复用。"""
    score = joint_score(Xv, marginals, theta, cdf_fn)
    b = rule_B(Xv, B_q)
    rows = calibrate_by_far(
        Xv, yv, [(float(pc), rule_C_from_score(score, b, float(pc))) for pc in cfg.C_pc_grid],
        target, cfg)
    return float(rows[0]["param"]), rows[0]["metrics"]


def fixed_det_on_test(X, y, cfg, gpd_models, marginals, theta, B_p_fixed, family="gumbel"):
    """使用预先固定参数网格生成测试 DET；不从测试曲线挑选最终参数。"""
    et = cfg.early_tolerance_steps
    cdf_fn = gumbel_C if family == "gumbel" else frank_C
    rows = []
    for k in cfg.A_multiplier_grid:
        rows.append({"rule": "A", "parameter": float(k),
                     **evaluate_rule(rule_A(X, k, cfg.fixed_thresholds), y, cfg.interval_sec, et)})
    B_cands, _ = b_threshold_candidates(X, cfg, gpd_models)
    for p, q, a in B_cands:
        rows.append({"rule": "B", "parameter": p, **evaluate_rule(a, y, cfg.interval_sec, et)})
        rows.append({"rule": "C22", "parameter": p,
                     **evaluate_rule(rule_C_count(X, q), y, cfg.interval_sec, et)})
    # C 与 B 共用边缘门限 B_p_fixed，只扫 p_c；仍是预注册网格，不是测试集调参。
    B_q = np.array([gpd_quantile(g, B_p_fixed) for g in gpd_models])
    score = joint_score(X, marginals, theta, cdf_fn)
    b = rule_B(X, B_q)
    for pc in cfg.C_pc_grid:
        rows.append({"rule": "C", "parameter": float(pc), "B_p_fixed": B_p_fixed,
                     **evaluate_rule(rule_C_from_score(score, b, float(pc)), y, cfg.interval_sec, et)})

    # C′（式19）：两个自由度都要扫——自己的边缘分位水平 pC 与 λ_c。
    # Λ_3 与它们都无关，只算一次。为控制耗时，pC 取 B 网格的等间隔子集（预注册）。
    lam = all_three_joint_prob(pit_observations(X, marginals), theta, cdf_fn)
    step = max(1, len(B_cands) // 24)
    for p, q, _ in B_cands[::step]:
        bp = rule_B(X, q)
        for lc in cfg.Cprime_lam_grid:
            rows.append({"rule": "Cprime", "parameter": float(lc), "pC": p,
                         **evaluate_rule(rule_Cprime_from_score(lam, bp, float(lc)),
                                         y, cfg.interval_sec, et)})

    # C_star：门限由 Copula 按式(17) 换算，唯一自由参数是目标多点事件概率 p*。
    for p_star in cfg.C_star_grid:
        try:
            _, q_star = joint_target_thresholds(marginals, theta, float(p_star), cdf_fn)
        except ValueError:
            continue
        rows.append({"rule": "C_star", "parameter": float(p_star),
                     **evaluate_rule(rule_C_count(X, q_star), y, cfg.interval_sec, et)})
    return pd.DataFrame(rows)


def operating_point_at_far(det_df, target_far):
    """等误报率对比：在预注册网格上取离目标 FAR 最近的点，并如实报出实际 FAR。

    原实现按 FAR 排序做线性插值，隐含「FAR 对参数单调」的假设。实测规则 A 的
    FAR 对倍率非单调（连续超越段被事件合并，k=0.90→0.0787、1.00→0.0324、
    1.05→0.0833），插值结果不可信。改为最近网格点 + 报出实际 FAR 与两侧包夹点，
    把对齐误差显式暴露出来而非藏进插值。
    """
    out = {}
    for rule, g in det_df.groupby("rule"):
        g = g.dropna(subset=["far_per_hour", "event_fnr"]).sort_values("far_per_hour")
        if g.empty:
            out[rule] = {"parameter": None, "far": np.nan, "event_fnr": np.nan, "abs_gap": np.nan}
            continue
        i = int(np.argmin(np.abs(g.far_per_hour.to_numpy() - target_far)))
        best = g.iloc[i]
        lo = g.iloc[max(i - 1, 0)]
        hi = g.iloc[min(i + 1, len(g) - 1)]
        out[rule] = {
            "parameter": float(best.parameter),
            "far": float(best.far_per_hour),
            "event_fnr": float(best.event_fnr),
            "abs_gap": float(abs(best.far_per_hour - target_far)),
            "far_lo": float(lo.far_per_hour), "far_hi": float(hi.far_per_hour),
            "fnr_lo": float(lo.event_fnr), "fnr_hi": float(hi.event_fnr),
            "n_grid": int(len(g)),
        }
    return out


# ============================================================
# 9. 危险段 bootstrap：保持段内相关结构
# ============================================================

def bootstrap_event_metrics(alarm, danger_mask, interval_sec, early_tolerance_steps, B=2000, seed=1):
    """危险段重抽 bootstrap；同时保留每段的报警覆盖与首次延迟统计。"""
    danger = danger_segments_from_mask(danger_mask)
    alarm = np.asarray(alarm, dtype=bool)
    events = alarm_events(alarm)
    rng = np.random.default_rng(seed)
    if not danger:
        return {}

    # 为每个危险段预先计算：是否命中、覆盖率、首次报警延迟、各窗口的早期命中。
    # 命中判定必须与 evaluate_rule / match_events 用同一个口径（重叠匹配）。旧版这里按
    # 「报警段的起点落在 [ds-tol, de) 内」判，与点估计的口径不同——实测会出现点估计
    # event_fnr=0.0 而自助区间 [1.0, 1.0] 这种区间根本不包含估计值的自相矛盾。
    hit = np.zeros(len(danger), dtype=float)
    cover = np.zeros(len(danger), dtype=float)
    delay = np.full(len(danger), np.nan)
    early = {w: np.zeros(len(danger), dtype=float) for w in EARLY_WINDOWS_MIN}
    for i, (ds, de) in enumerate(danger):
        hit_events = [es for es, ee in events if es < de and ee > ds - early_tolerance_steps]
        hit[i] = 1.0 if hit_events else 0.0
        cover[i] = np.mean(alarm[ds:de])
        if hit_events:
            d = min(hit_events) - ds
            if d >= 0:
                delay[i] = d * interval_sec / 60
        for w in EARLY_WINDOWS_MIN:
            lo = max(0, ds - early_tolerance_steps)
            hi = min(len(alarm), ds + int(round(w * 60 / interval_sec)) + 1, de)
            early[w][i] = 0.0 if (hi > lo and alarm[lo:hi].any()) else 1.0

    vals = {"event_fnr": [], "time_fnr": [], "mean_delay_min": []}
    vals.update({"early_fnr_%dmin" % w: [] for w in EARLY_WINDOWS_MIN})
    if B <= 0:  # 30 组主比较关闭 bootstrap 时不要触发空数组分位数
        return {k: {"estimate": float("nan"), "ci95": [float("nan")] * 2, "n_boot": 0} for k in vals}
    n = len(danger)
    for _ in range(B):
        idx = rng.integers(0, n, n)
        vals["event_fnr"].append(1 - np.mean(hit[idx]))
        vals["time_fnr"].append(1 - np.mean(cover[idx]))
        for w in EARLY_WINDOWS_MIN:
            vals["early_fnr_%dmin" % w].append(np.mean(early[w][idx]))
        dd = delay[idx]
        vals["mean_delay_min"].append(np.nanmean(dd) if np.any(np.isfinite(dd)) else np.nan)
    out = {}
    for k, v in vals.items():
        a = np.asarray(v, dtype=float)
        out[k] = {
            "estimate": float(np.nanmean(a)),
            "ci95": [float(np.nanpercentile(a, 2.5)), float(np.nanpercentile(a, 97.5))],
            "n_boot": int(B),
        }
    return out


# ============================================================
# 10. 单次数据集完整流程
# ============================================================

def fit_training_models(X_train, cfg):
    """训练集上拟合 POT-GPD、半参数边际与 Copula（同时拟合对照族）。

    Copula 拟合在 TailMarginal 的 PIT 上做（式(10)(11) 口径），不再用秩伪观测：
    秩只给到 i/(n+1)，尾部与 F_i(x_i) 不一致，会让式(16) 的
    1 - u_i - u_j + C(u_i,u_j) 不等于 P(X_i > q_i, X_j > q_j)。
    注意 τ 对严格增变换不变，故切换 PIT 口径不改变 θ 的估计值。
    """
    gpd_models = []
    diagnostics = []
    for j in range(cfg.n_sensors):
        m = choose_pot_threshold(X_train[:, j], cfg)
        m["gpd_ci"] = bootstrap_gpd_ci(m["excess"], cfg.bootstrap_gpd, seed=1000 + j)
        m["use_decluster_rate"] = bool(cfg.use_decluster_rate)
        gpd_models.append(m)
        diagnostics.append(m)
    marginals = [TailMarginal(X_train[:, j], gpd_models[j]) for j in range(cfg.n_sensors)]
    U = pit_observations(X_train, marginals)
    main = fit_copula(U, cfg.copula_family, gof_boot=50)
    main["pit_convention"] = "tail_marginal"
    main["pit_uniformity"] = [pit_uniformity(U[:, j]) for j in range(cfg.n_sensors)]
    # 秩伪观测口径的 τ，用于核对「PIT 口径切换不改变 τ」
    U_rank = pseudo_observations(X_train)
    main["tau_mean_rank_check"] = float(np.mean(
        [stats.kendalltau(U_rank[:, i], U_rank[:, j])[0]
         for i in range(cfg.n_sensors) for j in range(i + 1, cfg.n_sensors)]))
    alt_family = "frank" if cfg.copula_family == "gumbel" else "gumbel"
    alt = fit_copula(U, alt_family, gof_boot=50)
    alt["pit_convention"] = "tail_marginal"
    return gpd_models, marginals, main, alt, diagnostics


def run_one(cfg: SimConfig, save_dir: Path | None = None):
    X, y, segments = generate_dataset(cfg)
    split_data, split = split_time_series(X, y)
    Xtr, ytr = split_data["train"]
    Xv, yv = split_data["val"]
    Xte, yte = split_data["test"]

    gpd_models, marginals, copula, copula_alt, diagnostics = fit_training_models(Xtr, cfg)
    family = copula["family"]
    theta = copula["theta"]
    cdf_fn = gumbel_C if family == "gumbel" else frank_C
    cal = validation_calibration(Xv, yv, cfg, gpd_models, marginals, theta, family)

    # 最终固定参数在测试集上运行
    B_alarm = rule_B(Xte, cal["B"]["thresholds"])
    score_te = joint_score(Xte, marginals, theta, cdf_fn)
    lam_te = all_three_joint_prob(pit_observations(Xte, marginals), theta, cdf_fn)
    alarms = {
        "A": rule_A(Xte, cal["A"]["k"], cfg.fixed_thresholds),
        "B": B_alarm,
        # C22 是报告指定的主规则（式22）：自己更低的分位门限 + 至少两点超限
        "C22": rule_C_count(Xte, cal["C22"]["thresholds"]),
        # C_star 是报告第五（七）里 Copula 的用法：门限由式(17) 反解、判别式仍是式(22)
        "C_star": rule_C_count(Xte, cal["C_star"]["thresholds"]),
        # Cprime 是报告式(19) 的连续型对照 C′：Λ_3 ≤ λ_c 且 ∃i: x_i ≥ q_{pC,i}
        "Cprime": rule_Cprime_from_score(lam_te, rule_B(Xte, cal["Cprime"]["thresholds"]),
                                        cal["Cprime"]["lam_c"]),
        # 旧实现保留作对照：B 的门限 ∧ 式(17) 的 P_≥2
        "C": rule_C_from_score(score_te, B_alarm, cal["C"]["p_c"]),
    }
    # Copula 族对照：另一族单独标定 p_c 后再评测试集（λ_U=0 的 Frank 是 H3 的阴性对照）
    alt_family = copula_alt["family"]
    cdf_alt = gumbel_C if alt_family == "gumbel" else frank_C
    pc_alt, _ = calibrate_c_only(Xv, yv, cfg, marginals, copula_alt["theta"],
                                cal["B"]["thresholds"], cdf_alt)
    score_alt_te = joint_score(Xte, marginals, copula_alt["theta"], cdf_alt)
    alarms["C_" + alt_family] = rule_C_from_score(score_alt_te, B_alarm, pc_alt)

    test_metrics = {r: evaluate_rule(a, yte, cfg.interval_sec, cfg.early_tolerance_steps)
                    for r, a in alarms.items()}

    # 测试 DET：预注册参数网格，不选点
    det = fixed_det_on_test(Xte, yte, cfg, gpd_models, marginals, theta, cal["C"]["B_p"], family)
    equal_far = {str(f): operating_point_at_far(det, f) for f in cfg.target_fars}

    # 测试集 bootstrap：用于最终不确定性，不参与参数选择
    boot = {}
    for r, a in alarms.items():
        boot[r] = bootstrap_event_metrics(a, yte, cfg.interval_sec, cfg.early_tolerance_steps,
                                          B=cfg.bootstrap_metrics, seed=9000 + cfg.seed)

    result = {
        "config": asdict(cfg),
        "split": asdict(split),
        "danger_occupancy": float(np.mean(y)),
        "segments": segments,
        "gpd_models": gpd_models,
        "copula": copula,
        "copula_alt": copula_alt,
        "copula_alt_p_c": pc_alt,
        "calibration": cal,
        "test_metrics": test_metrics,
        "equal_far_test": equal_far,
        "det": det,
        "bootstrap": boot,
        "test_alarm_score_C": score_te,
        "extremal_index": {
            "theta_hat": [float(m["theta_hat"]) for m in gpd_models],
            "n_raw": [int(m["n_raw"]) for m in gpd_models],
            "n_clusters": [int(m["n_clusters"]) for m in gpd_models],
            "mean_cluster_len": [float(m["n_raw"] / max(m["n_clusters"], 1)) for m in gpd_models],
            "use_decluster_rate": bool(cfg.use_decluster_rate),
        },
    }

    if save_dir is not None:
        save_single_result(result, save_dir)
    return result


# ============================================================
# 11. 30 组实验、阴性对照、敏感性、Wilcoxon
# ============================================================

def jsonable(obj):
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [jsonable(v) for v in obj]
    return obj


def save_single_result(result, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # 大数组单独 CSV
    result["det"].to_csv(out_dir / "test_det.csv", index=False, encoding="utf-8-sig")
    rows = [{"rule": r, **m} for r, m in result["test_metrics"].items()]
    pd.DataFrame(rows).to_csv(out_dir / "test_metrics.csv", index=False, encoding="utf-8-sig")

    gpd_rows = []
    for j, m in enumerate(result["gpd_models"]):
        gpd_rows.append({
            "sensor": j, "q": m["q"], "u": m["u"], "shape": m["shape"], "scale": m["scale"],
            "n_clusters": m["n_clusters"], "n_raw": m["n_raw"], "theta_hat": m["theta_hat"],
            "mean_cluster_len": m["n_raw"] / max(m["n_clusters"], 1),
            "raw_tail_prob": m["raw_tail_prob"], "cluster_tail_prob": m["cluster_tail_prob"],
            "ks_stat": m["ks_stat"], "ks_pvalue": m["ks_pvalue"],
            "xi_ci_low": m["gpd_ci"]["xi_ci"][0], "xi_ci_high": m["gpd_ci"]["xi_ci"][1],
            "scale_ci_low": m["gpd_ci"]["scale_ci"][0], "scale_ci_high": m["gpd_ci"]["scale_ci"][1],
        })
    pd.DataFrame(gpd_rows).to_csv(out_dir / "gpd_parameters.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(result["copula"]["tau_matrix"]).to_csv(out_dir / "kendall_tau.csv", index=False)
    # Copula 族对照：Gumbel（λ_U>0）vs Frank（λ_U=0）
    fam_rows = []
    for tag in ("copula", "copula_alt"):
        c = result[tag]
        fam_rows.append({
            "tag": tag, "family": c["family"], "tau_mean": c["tau_mean"],
            "tau_mean_rank_check": c.get("tau_mean_rank_check", np.nan),
            "theta": c["theta"], "lambda_u": c["lambda_u"],
            "gof_cvm": c["gof"]["cvm"], "gof_pvalue": c["gof"]["pvalue"],
            "gof_boot": c["gof"]["n_boot"],
        })
    pd.DataFrame(fam_rows).to_csv(out_dir / "copula_families.csv", index=False, encoding="utf-8-sig")
    # 等 FAR 对齐表：报出实际 FAR 与包夹点，暴露对齐误差
    eq_rows = []
    for f, per_rule in result["equal_far_test"].items():
        for rule, d in per_rule.items():
            eq_rows.append({"target_far": float(f), "rule": rule, **d})
    pd.DataFrame(eq_rows).to_csv(out_dir / "equal_far_test.csv", index=False, encoding="utf-8-sig")
    clean_gpd = []
    for m in result["gpd_models"]:
        clean_gpd.append({
            "q": m["q"], "u": m["u"], "shape": m["shape"], "scale": m["scale"],
            "n_clusters": m["n_clusters"], "n_raw": m["n_raw"], "theta_hat": m["theta_hat"],
            "raw_tail_prob": m["raw_tail_prob"], "cluster_tail_prob": m["cluster_tail_prob"],
            "ks_stat": m["ks_stat"], "ks_pvalue": m["ks_pvalue"],
            "gpd_ci": m["gpd_ci"],
        })
    clean = {
        "config": result["config"], "split": result["split"],
        "danger_occupancy": result["danger_occupancy"],
        "gpd_models": clean_gpd, "copula": result["copula"], "copula_alt": result["copula_alt"],
        "copula_alt_p_c": result["copula_alt_p_c"],
        "calibration": result["calibration"], "test_metrics": result["test_metrics"],
        "equal_far_test": result["equal_far_test"], "bootstrap": result["bootstrap"],
        "extremal_index": result["extremal_index"],
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(jsonable(clean), f, ensure_ascii=False, indent=2)

    # 阈值诊断原始表
    for j, m in enumerate(result["gpd_models"]):
        m["mrl"].to_csv(out_dir / f"mrl_T{j}.csv", index=False)
        m["stability"].to_csv(out_dir / f"stability_T{j}.csv", index=False)


def wilcoxon_paired(df, a, b, metric):
    x = df[df.rule == a].sort_values("rep")[metric].to_numpy()
    y = df[df.rule == b].sort_values("rep")[metric].to_numpy()
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 5:
        return {"n": int(mask.sum()), "stat": np.nan, "pvalue": np.nan}
    try:
        res = stats.wilcoxon(x[mask], y[mask], alternative="two-sided", zero_method="wilcox")
        return {"n": int(mask.sum()), "stat": float(res.statistic), "pvalue": float(res.pvalue)}
    except Exception:
        return {"n": int(mask.sum()), "stat": np.nan, "pvalue": np.nan}


def run_30_replicates(base_cfg: SimConfig, out_dir: Path, R=30):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    equal_rows = []
    for r in range(R):
        # 30 组主比较先固定模型流程并汇总点估计；
        # 危险段 bootstrap=2000 属于不确定性分析，可在代表性数据集/最终汇总阶段单独运行，
        # 避免 30×2000 的重复计算拖垮本科阶段实验。
        d = asdict(base_cfg)
        d.update({"seed": base_cfg.seed + r, "bootstrap_gpd": 0, "bootstrap_metrics": 0})
        cfg = SimConfig(**d)
        res = run_one(cfg, save_dir=None)
        for rule, m in res["test_metrics"].items():
            rows.append({"rep": r + 1, "rule": rule, **m, "danger_occupancy": res["danger_occupancy"]})
        for f, per_rule in res["equal_far_test"].items():
            for rule, d in per_rule.items():
                equal_rows.append({"rep": r + 1, "target_far": float(f), "rule": rule, **d})
        pd.DataFrame(rows).to_csv(out_dir / "replicates_progress.csv", index=False, encoding="utf-8-sig")

    df = pd.DataFrame(rows)
    eq = pd.DataFrame(equal_rows)
    df.to_csv(out_dir / "all_replicates.csv", index=False, encoding="utf-8-sig")
    eq.to_csv(out_dir / "equal_far_all.csv", index=False, encoding="utf-8-sig")

    summary = df.groupby("rule").agg({"event_fnr": ["mean", "std"], "time_fnr": ["mean", "std"],
                                        "far_per_hour": ["mean", "std"], "mean_delay_min": ["mean", "std"]})
    summary.to_csv(out_dir / "replicate_summary.csv", encoding="utf-8-sig")

    tests = {}
    alt_rule = "C_frank" if base_cfg.copula_family == "gumbel" else "C_gumbel"
    # H2/H3 的判定对象是报告指定的主规则：式(22) 的 C22，以及它的 Copula 门限版 C_star
    # 与连续型对照 Cprime。旧实现 C 只作对照。
    for metric in ["event_fnr", "time_fnr", "far_per_hour", "mean_delay_min"]:
        tests[f"A_vs_B_{metric}"] = wilcoxon_paired(df, "A", "B", metric)
        for cand in ["C", "C22", "C_star", "Cprime"]:
            if cand in set(df.rule):
                tests[f"B_vs_{cand}_{metric}"] = wilcoxon_paired(df, "B", cand, metric)
        tests[f"B_vs_{alt_rule}_{metric}"] = wilcoxon_paired(df, "B", alt_rule, metric)
    with open(out_dir / "wilcoxon.json", "w", encoding="utf-8") as f:
        json.dump(tests, f, ensure_ascii=False, indent=2)
    return df, eq, tests


def negative_control(cfg: SimConfig, mode: str):
    c = SimConfig(**asdict(cfg))
    if mode == "zero_danger":
        # 用生成器的正常序列，但不注入危险段
        rng = np.random.default_rng(c.seed)
        T = c.n_steps
        common = ar1(T, 0, 0.98, 0.035, rng)
        log_x = np.zeros((T, c.n_sensors))
        for j in range(c.n_sensors):
            own = ar1(T, c.mu_log[j], c.rho[j], c.sigma_log[j], rng)
            oc = own - c.mu_log[j]
            s = c.cross_strength
            log_x[:, j] = c.mu_log[j] + math.sqrt(1-s)*oc + math.sqrt(s)*common*(np.std(oc)/np.std(common)) + rng.normal(0,c.sigma_noise,T)
        X = np.exp(log_x); y = np.zeros(T, dtype=bool)
    elif mode == "independent":
        c.cross_strength = 0.0
        # 各测点抬升窗口独立抽取，测点序列真正相互独立。
        # 注意：随机错位（jitter）曾是这里的方案，但它不消除共同抬升剖面，
        # 实测段内 τ 反而从 0.211 升到 0.254；互斥子区间（disjoint）也不对，
        # 它把 τ 压成 -0.167。两种做法都不是「独立」。
        c.coupled_layout = "independent"
        c.coupled_jitter = False
        X, y, _ = generate_dataset(c)
        # 保留危险标签以检验 H3 的结构性问题，但测点之间真值独立
    else:
        raise ValueError(mode)
    return X, y


def calibrate_cross_strength(base_cfg: SimConfig, target_tau, tol=0.01, max_iter=40):
    """二分反解 cross_strength，使正常态 Kendall τ 命中目标（报告表2 的口径）。

    原实现把 cross_strength 的档位直接当作 τ 的档位（0.10/0.30/0.50），
    两者并非同一量纲；这里改为按达成值报告。
    """
    def tau_at(s):
        d = asdict(base_cfg)
        d["cross_strength"] = float(s)
        d["coupled_jitter"] = False
        X, y, _ = generate_dataset(SimConfig(**d))
        n = X[~y]
        return float(np.mean([stats.kendalltau(n[:, i], n[:, j])[0]
                              for i, j in [(0, 1), (0, 2), (1, 2)]]))

    lo, hi = 0.0, 0.98
    t_lo, t_hi = tau_at(lo), tau_at(hi)
    if not (t_lo - 1e-9 <= target_tau <= t_hi + 1e-9):
        raise ValueError("目标 τ=%.3f 不在 cross_strength∈[0, 0.98] 的可达区间 [%.4f, %.4f] 内"
                         % (target_tau, t_lo, t_hi))
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        t = tau_at(mid)
        if abs(t - target_tau) <= tol:
            return float(mid), float(t)
        if t < target_tau:
            lo = mid
        else:
            hi = mid
    mid = 0.5 * (lo + hi)
    return float(mid), float(tau_at(mid))


def run_sensitivity(base_cfg: SimConfig, out_dir: Path):
    """按开题报告表2做敏感性实验：噪声、依赖强度、危险段长度、run length。

    各档位共用同一组种子并重复 R 次：原实现给第 i 档配 seed+i，档位与随机数
    完全绑定，档间差异无法与随机波动区分。
    """
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    R = max(1, base_cfg.sens_replicates)

    # 依赖性档位先在正常态 τ 上反解 cross_strength
    tau_cal = []
    for t in base_cfg.target_taus:
        s, achieved = calibrate_cross_strength(base_cfg, float(t))
        tau_cal.append({"target_tau": float(t), "cross_strength": s, "achieved_tau": achieved})
    pd.DataFrame(tau_cal).to_csv(out_dir / "tau_calibration.csv", index=False, encoding="utf-8-sig")

    factors = {
        "sigma_noise": [0.010, 0.015, 0.025],
        "kendall_tau": [c["cross_strength"] for c in tau_cal],
        "danger_length": [120, 210, 300],
        "run_length": [10, 20, 40],
        # 报告表2 没有这一项，但它直接决定 coupled 危险段内是否会出现「多点同时超限」，
        # 也就是规则 C / 式(22) 能否被触发的决定性参数，故补进来。
        "propagation_lag": [5, 20, 40],
        # 危险段抬升幅度。这一项是决定性的：幅度大时 B 的门限被轻松越过、事件级 FNR
        # 触底为 0，H2/H3 在事件级无从检验（实测幅度 ≥0.55 时 B 的事件级 FNR 恒为 0，
        # 基准 0.78 正落在饱和区）；幅度小到 0.2 以下则所有规则一起失效。
        # 只在 0.3—0.4 的过渡区，各规则的事件级与逐时级差异才真正可分辨。
        # 报告表2 未含此项，但缺了它就无法判断 H2 的结论是不是饱和造成的，
        # 故补进来并把实测的饱和阈值一并报告。
        "danger_amp": [0.30, 0.45, 0.60, 0.78],
    }
    tau_of_level = {c["cross_strength"]: c["target_tau"] for c in tau_cal}
    rows = []
    for name, vals in factors.items():
        for val in vals:
            for r_i in range(R):
                d = asdict(base_cfg)
                d.update({"seed": base_cfg.seed + r_i, "bootstrap_gpd": 0, "bootstrap_metrics": 0})
                if name == "sigma_noise":
                    d["sigma_noise"] = val
                elif name == "kendall_tau":
                    d["cross_strength"] = float(val)
                elif name == "danger_length":
                    d["min_len"] = int(val * 0.75); d["max_len"] = int(val * 1.25)
                elif name == "run_length":
                    d["run_length"] = int(val)
                elif name == "propagation_lag":
                    d["propagation_lag"] = int(val)
                elif name == "danger_amp":
                    # local/coupled 段同步缩放，避免把「段类型构成」与「幅度」混在一起
                    d["local_amp"] = float(val)
                    d["coupled_amp"] = float(val)
                cfg = SimConfig(**d)
                r = run_one(cfg, save_dir=None)
                for rule, m in r["test_metrics"].items():
                    rows.append({"factor": name, "level": val,
                                 "level_label": tau_of_level.get(val, val),
                                 "rep": r_i + 1, "rule": rule, **m,
                                 "danger_occupancy": r["danger_occupancy"],
                                 "tau_mean": r["copula"]["tau_mean"],
                                 "lambda_u": r["copula"]["lambda_u"],
                                 "theta_hat_mean": float(np.mean(r["extremal_index"]["theta_hat"]))})
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "sensitivity_results.csv", index=False, encoding="utf-8-sig")
    df.groupby(["factor", "level", "rule"])[
        ["event_fnr", "time_fnr", "far_per_hour", "mean_delay_min"]
    ].agg(["mean", "std"]).to_csv(out_dir / "sensitivity_summary.csv", encoding="utf-8-sig")
    return df


def run_negative_controls(base_cfg: SimConfig, out_dir: Path):
    """零危险与独立边际阴性对照；参数仍按训练/验证/测试封存流程估计。"""
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for mode in ["zero_danger", "independent"]:
        for r_i in range(max(1, base_cfg.sens_replicates)):
            d = asdict(base_cfg)
            d.update({"seed": base_cfg.seed + r_i, "bootstrap_gpd": 0, "bootstrap_metrics": 0})
            cfg = SimConfig(**d)
            X, y = negative_control(cfg, mode)
            data, _ = split_time_series(X, y)
            Xtr, ytr = data["train"]; Xv, yv = data["val"]; Xte, yte = data["test"]
            gpd, marg, cop, cop_alt, _ = fit_training_models(Xtr, cfg)
            cal = validation_calibration(Xv, yv, cfg, gpd, marg, cop["theta"], cop["family"])
            # 依赖强度诊断。判据是 tau_all_test ≈ 0：整段序列上无依赖，才说明
            # 测点之间真的独立。tau_in_danger 是在「并集危险掩码」上算的，而该掩码
            # 是过程的函数，条件在并集上本身就会制造依赖（选择效应），所以它不为 0
            # 并不能说明对照失败——只有 tau_all_test / tau_train_normal 才是判据。
            # tau_train_all 是整个训练集的 τ（含危险段），不能当「正常段 τ」用。
            pairs = [(0, 1), (0, 2), (1, 2)]
            seg_mask = np.zeros(len(yte), dtype=bool)
            for s, e in danger_segments_from_mask(yte):
                seg_mask[s:e] = True
            tau_in = float(np.mean([stats.kendalltau(Xte[seg_mask, i], Xte[seg_mask, j])[0]
                                    for i, j in pairs])) if seg_mask.sum() > 10 else np.nan
            tau_all = float(np.mean([stats.kendalltau(Xte[:, i], Xte[:, j])[0] for i, j in pairs]))
            norm_mask = ~ytr
            tau_normal = float(np.mean([stats.kendalltau(Xtr[norm_mask, i], Xtr[norm_mask, j])[0]
                                        for i, j in pairs])) if norm_mask.sum() > 10 else np.nan
            B_alarm = rule_B(Xte, cal["B"]["thresholds"])
            cdf_fn = gumbel_C if cop["family"] == "gumbel" else frank_C
            score = joint_score(Xte, marg, cop["theta"], cdf_fn)
            lam = all_three_joint_prob(pit_observations(Xte, marg), cop["theta"], cdf_fn)
            alarms = {
                "A": rule_A(Xte, cal["A"]["k"], cfg.fixed_thresholds),
                "B": B_alarm,
                "C22": rule_C_count(Xte, cal["C22"]["thresholds"]),
                "C_star": rule_C_count(Xte, cal["C_star"]["thresholds"]),
                "Cprime": rule_Cprime_from_score(lam, rule_B(Xte, cal["Cprime"]["thresholds"]),
                                                cal["Cprime"]["lam_c"]),
                "C": rule_C_from_score(score, B_alarm, cal["C"]["p_c"]),
            }
            for rule, alarm in alarms.items():
                m = evaluate_rule(alarm, yte, cfg.interval_sec, cfg.early_tolerance_steps)
                rows.append({"control": mode, "rep": r_i + 1, "rule": rule, **m,
                             "tau_all_test": tau_all, "tau_train_normal": tau_normal,
                             "tau_train_all": cop["tau_mean"], "tau_in_danger": tau_in,
                             "lambda_u": cop["lambda_u"],
                             "n_danger_test": len(danger_segments_from_mask(yte))})
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "negative_controls.csv", index=False, encoding="utf-8-sig")
    return df


# ============================================================
# 12. 图表
# ============================================================

def make_plots(result, out_dir):
    import matplotlib.pyplot as plt
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    # MRL
    for j, m in enumerate(result["gpd_models"]):
        plt.figure(figsize=(7, 4))
        d = m["mrl"]
        plt.errorbar(d["u"], d["mrl"], yerr=1.96*d["se"], fmt="o", markersize=3)
        plt.axvline(m["u"], linestyle="--")
        plt.xlabel("Threshold u"); plt.ylabel("Mean excess"); plt.title(f"T{j} MRL")
        plt.tight_layout(); plt.savefig(out_dir / f"mrl_T{j}.png", dpi=180); plt.close()

        s = m["stability"]
        plt.figure(figsize=(7, 4))
        plt.plot(s["u"], s["shape"], marker="o")
        plt.axvline(m["u"], linestyle="--")
        plt.xlabel("Threshold u"); plt.ylabel("GPD shape xi"); plt.title(f"T{j} parameter stability")
        plt.tight_layout(); plt.savefig(out_dir / f"stability_T{j}.png", dpi=180); plt.close()

    det = result["det"]
    plt.figure(figsize=(7, 5))
    for r in ["A", "B", "C22", "C_star", "Cprime", "C"]:
        g = det[det.rule == r].sort_values("far_per_hour")
        if len(g):
            plt.plot(g.far_per_hour, g.event_fnr, label=r)
    plt.xlabel("FAR (events/hour)"); plt.ylabel("Event FNR"); plt.title("Test DET curves")
    plt.legend(); plt.tight_layout(); plt.savefig(out_dir / "DET_test.png", dpi=180); plt.close()


if __name__ == "__main__":
    cfg = SimConfig()
    out = Path(__file__).parent / "results" / "strict_results"
    res = run_one(cfg, out)
    make_plots(res, out / "figures")
    print("完成严格版单数据集实验")
    print("危险占比:", res["danger_occupancy"])
    ei = res["extremal_index"]
    print("极值指数 theta_hat =", [round(v, 4) for v in ei["theta_hat"]],
          " 平均簇长 =", [round(v, 1) for v in ei["mean_cluster_len"]])
    print("Copula 主族 %s：tau=%s θ=%.4f λ_U=%.4f gof_p=%.3f"
          % (res["copula"]["family"], np.round(res["copula"]["pair_tau"], 4),
             res["copula"]["theta"], res["copula"]["lambda_u"], res["copula"]["gof"]["pvalue"]))
    print("Copula 对照族 %s：θ=%.4f λ_U=%.4f gof_p=%.3f  标定 p_c=%.3g"
          % (res["copula_alt"]["family"], res["copula_alt"]["theta"],
             res["copula_alt"]["lambda_u"], res["copula_alt"]["gof"]["pvalue"],
             res["copula_alt_p_c"]))
    print("标定：A k=%.3f | B p=%.5f（网格丢弃 %d 点）| 式(22) p=%.5f | "
          "C* p*=%.4g (α*=%.4g) | C′ λ_c=%.4g | 旧 C p_c=%.4g"
          % (res["calibration"]["A"]["k"], res["calibration"]["B"]["p"],
             res["calibration"]["B"]["n_grid_dropped"], res["calibration"]["C22"]["p"],
             res["calibration"]["C_star"]["p_star"], res["calibration"]["C_star"]["alpha_star"],
             res["calibration"]["Cprime"]["lam_c"], res["calibration"]["C"]["p_c"]))
    print("两套边缘门限（B 的 p=%.5f vs 式(22) 自己的 p_C=%.5f）："
          % (res["calibration"]["B"]["p"], res["calibration"]["C22"]["p"]))
    print("   B  :", np.round(res["calibration"]["B"]["thresholds"], 4).tolist())
    print("   C22:", np.round(res["calibration"]["C22"]["thresholds"], 4).tolist())
    print("   C* :", np.round(res["calibration"]["C_star"]["thresholds"], 4).tolist(),
          " ← 由 Copula 按式(17) 反解，与 C22 不同即说明 Copula 真的在起作用")
    print(pd.DataFrame(res["test_metrics"]).T)
    print("等 FAR 对齐（实际 FAR / 实际 FNR / 与目标 FAR 的差）：")
    for f, per_rule in res["equal_far_test"].items():
        for rule, d in per_rule.items():
            print("   target=%-5s %-8s far=%.4f fnr=%.4f gap=%.4f"
                  % (f, rule, d["far"], d["event_fnr"], d["abs_gap"]))
