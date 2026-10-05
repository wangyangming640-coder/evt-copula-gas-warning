# 蒙特卡洛独立验证：在 alpha* 处抽样，看经验 P_{>=2} 是否 = p*
# 抽样用 Archimedean 条件分布逐分量反演（Hofert 式），只依赖生成元 psi 及其 1、2 阶导。
# 不调用 strict_pipeline 的任何抽样/概率函数。
import numpy as np, math
from scipy.optimize import brentq

rng = np.random.default_rng(20260925)

# ---------- 生成元 psi 及导数 ----------
def gumbel_psi(t, th):
    return np.exp(-(np.asarray(t, float) ** (1.0 / th)))
def gumbel_psi1(t, th):
    a = 1.0 / th; t = np.asarray(t, float)
    return -a * t ** (a - 1) * np.exp(-(t ** a))
def gumbel_psi2(t, th):
    a = 1.0 / th; t = np.asarray(t, float)
    return -a * np.exp(-(t ** a)) * ((a - 1) * t ** (a - 2) - a * t ** (2 * a - 2))
def gumbel_psiinv(u, th):
    return (-np.log(u)) ** th

def frank_psi(t, th):
    A = math.expm1(-th)
    return -np.log1p(A * np.exp(-np.asarray(t, float))) / th
def frank_psi1(t, th):
    A = math.expm1(-th); t = np.asarray(t, float)
    B = 1.0 + A * np.exp(-t)
    return (A * np.exp(-t)) / (th * B)
def frank_psi2(t, th):
    A = math.expm1(-th); t = np.asarray(t, float)
    B = 1.0 + A * np.exp(-t)
    return -(A * np.exp(-t)) / (th * B * B)
def frank_psiinv(u, th):
    A = math.expm1(-th)
    return -np.log((np.exp(-th * u) - 1.0) / A)

def _bisect(ratio_fun, w, hi, iters=100):
    """ratio_fun 从 1 单调降到 0；求 ratio_fun(s)=w 的 s∈[0,hi]。"""
    lo = np.zeros_like(w); H = np.full_like(w, hi)
    for _ in range(iters):
        mid = 0.5 * (lo + H)
        ok = ratio_fun(mid) >= w
        lo = np.where(ok, mid, lo); H = np.where(ok, H, mid)
    return 0.5 * (lo + H)

def sample_arch3(n, fam, th):
    if fam == "gumbel":
        psi  = lambda t: gumbel_psi(t, th)
        psi1 = lambda t: gumbel_psi1(t, th)
        psi2 = lambda t: gumbel_psi2(t, th)
        psiinv = lambda u: gumbel_psiinv(u, th)
        hi = 30.0 ** th
    elif fam == "frank":
        psi  = lambda t: frank_psi(t, th)
        psi1 = lambda t: frank_psi1(t, th)
        psi2 = lambda t: frank_psi2(t, th)
        psiinv = lambda u: frank_psiinv(u, th)
        hi = 60.0 / max(abs(th), 1e-6) + 120.0
    u1 = rng.random(n); s1 = psiinv(u1)
    w2 = rng.random(n)
    s2 = _bisect(lambda s: psi1(s1 + s) / psi1(s1), w2, hi)
    S2 = s1 + s2
    w3 = rng.random(n)
    s3 = _bisect(lambda s: psi2(S2 + s) / psi2(S2), w3, hi)
    # s_k = psi^{-1}(u_k)，故 u_k = psi(s_k)
    return np.column_stack([psi(s1), psi(s2), psi(s3)])

# ---------- 解析 copula CDF ----------
def C_gumbel(U, theta):
    U = np.atleast_2d(np.asarray(U, float))
    z = np.sum((-np.log(U)) ** theta, axis=-1)
    return np.exp(-(z ** (1.0 / theta)))
def C_frank(U, theta):
    U = np.atleast_2d(np.asarray(U, float)); theta = float(theta)
    if abs(theta) < 1e-14: return np.prod(U, axis=-1)
    A = math.expm1(-theta)
    return -np.log1p(np.prod(np.expm1(-theta * U), axis=-1) / A ** (U.shape[-1] - 1)) / theta

def Pge2_closed(alpha, C):
    a = 1.0 - float(alpha)
    return 1.0 - 3.0 * float(C([[a, a]])[0]) + 2.0 * float(C([[a, a, a]])[0])
def empirical_Pge2(U, alpha):
    return float(np.mean(np.sum(U > (1.0 - alpha), axis=1) >= 2))

if __name__ == "__main__":
    print("=== 抽样器自检 (N=200000) ===", flush=True)
    n = 200_000
    for fam, th in [("gumbel", 1.5094), ("gumbel", 4.0), ("frank", 10.0), ("frank", -10.0)]:
        U = sample_arch3(n, fam, th)
        marg = np.abs(np.mean(U, axis=0) - 0.5).max()
        C = (lambda UU: C_gumbel(UU, th)) if fam == "gumbel" else (lambda UU: C_frank(UU, th))
        e2 = max(abs(np.mean((U[:, 0] <= p) & (U[:, 1] <= q)) - float(C([[p, q]])[0]))
                 for p, q in [(0.3, 0.4), (0.6, 0.7), (0.5, 0.5)])
        e3 = max(abs(np.mean((U[:, 0] <= p) & (U[:, 1] <= q) & (U[:, 2] <= r)) - float(C([[p, q, r]])[0]))
                 for p, q, r in [(0.3, 0.4, 0.5), (0.6, 0.6, 0.7), (0.5, 0.5, 0.5)])
        print(f"  {fam:7s} th={th:>7}: |E[U]-0.5|max={marg:.4f}  2D-CDFerr={e2:.4f}  3D-CDFerr={e3:.4f}", flush=True)
