# 独立实现式(17) P_{>=2}(alpha) 与二分反解。不调用 strict_pipeline。
# 口径：d=3 测点同取边缘超越概率 alpha（即门限处 F_i(q)=1-alpha=a），
# 报告式(16): P_{>=2} = sum_{i<j}[a_i+a_j-1+C_ij(1-a_i,1-a_j)] - 2 P3
# 报告式(17): P3 = 1 - sum(1-a_i) + sum C_ij - C(1-a1,1-a2,1-a3)
# 由容斥可化为 P_{>=2}(a) = 3*(1-2a+C2) - 2*(1-3a+3C2-C3) = 1 - 3*C2 + 2*C3
import numpy as np, math
from scipy.optimize import brentq

# ---- copula CDFs (自己按定义写) ----
def C_gumbel(U, theta):
    U = np.atleast_2d(np.asarray(U, float))
    z = np.sum((-np.log(U)) ** theta, axis=-1)
    return np.exp(-(z ** (1.0 / theta)))

def C_frank(U, theta):
    U = np.atleast_2d(np.asarray(U, float))
    theta = float(theta)
    if abs(theta) < 1e-14:
        return np.prod(U, axis=-1)
    A = math.expm1(-theta)                      # e^{-theta}-1
    prod = np.prod(np.expm1(-theta * U), axis=-1)
    return -np.log1p(prod / A ** (U.shape[-1] - 1)) / theta

def C_indep(U, theta=None):
    U = np.atleast_2d(np.asarray(U, float)); return np.prod(U, axis=-1)

def C_comono(U, theta=None):
    U = np.atleast_2d(np.asarray(U, float)); return np.min(U, axis=-1)

# ---- 式(17) 的两个独立写法，互相校验 ----
def Pge2_closed(alpha, C):
    """1 - 3 C2 + 2 C3"""
    a = 1.0 - float(alpha)
    C2 = float(C([[a, a]])[0]); C3 = float(C([[a, a, a]])[0])
    return 1.0 - 3.0 * C2 + 2.0 * C3

def Pge2_report(alpha, C):
    """直接照报告式(16)(17)逐项写：3 个二元项 + 三元项"""
    a = 1.0 - float(alpha)
    pairs = [(a, a), (a, a), (a, a)]
    P2 = sum(1.0 - u - v + float(C([[u, v]])[0]) for u, v in pairs)   # 式(16) 的 sum 项
    P3 = 1.0 - 3.0 * a + sum(float(C([[u, v]])[0]) for u, v in pairs) - float(C([[a, a, a]])[0])
    return P2 - 2.0 * P3                                              # 式(16)

def alpha_star(p_star, C, lo=1e-15, hi=1-1e-15):
    f = lambda al: Pge2_closed(al, C) - p_star
    return brentq(f, lo, hi, xtol=1e-15, rtol=1e-15, maxiter=200)

p_star = 0.01
cases = [
    ("独立 (indep)",              C_indep,  None),
    ("Gumbel theta=1",            C_gumbel, 1.0),
    ("Gumbel theta=1.5094",       C_gumbel, 1.5094),
    ("Gumbel theta=4",            C_gumbel, 4.0),
    ("Gumbel theta=50",           C_gumbel, 50.0),
    ("完全同单调 (comonotone)",   C_comono, None),
    ("Frank theta=0",             C_frank,  0.0),
    ("Frank theta=-10",           C_frank,  -10.0),
]
claim = {"独立 (indep)":0.058903, "Gumbel theta=1":0.058903, "Gumbel theta=1.5094":0.016112,
         "Gumbel theta=4":0.010680, "Gumbel theta=50":0.010025, "完全同单调 (comonotone)":0.010,
         "Frank theta=0":0.058903, "Frank theta=-10":0.239433}

print(f"目标 p* = {p_star}")
print(f"{'case':<28}{'alpha* 本次':>16}{'报告/结论':>14}{'差':>14}{'P>=2 两种写法差':>18}")
for name, C, th in cases:
    if th is None:
        a_star = alpha_star(p_star, C)
    else:
        a_star = alpha_star(p_star, lambda U, C=C, th=th: C(U, th))
    Cw = (lambda U, C=C, th=th: C(U, th)) if th is not None else C
    d = abs(Pge2_closed(a_star, Cw) - Pge2_report(a_star, Cw))
    cl = claim[name]
    print(f"{name:<28}{a_star:>16.6f}{cl:>14.6f}{a_star-cl:>14.2e}{d:>18.1e}")

# 独立性交叉核对：Gumbel theta=1 是否等于独立；Frank theta=0 是否等于独立
print("\n[交叉] P>=2 在 alpha=0.058903 处:",
      "gumbel(1)=", round(Pge2_closed(0.058903, lambda U: C_gumbel(U,1.0)),12),
      "indep=", round(Pge2_closed(0.058903, C_indep),12))
