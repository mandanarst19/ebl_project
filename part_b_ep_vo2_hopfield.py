"""
Part B — Equilibrium Propagation in a Hopfield-type network with VO2-like bistable nodes
=====================================================================================

Model (energy-based, continuous Hopfield form)
  state s in R^n (hidden + output units), inputs x clamped
  E(s; theta, x) = sum_i U_alpha(s_i) - 1/2 s^T W s - s^T (A x + b)
  U_alpha(s)     = s^4/4 - alpha s^2/2          (Landau double-well local potential)
  cost           C(s, y) = 1/2 ||s_out - y||^2

Physical reading of alpha (phenomenological, NOT a device model):
  alpha < 0 : single-well node  -> smooth, monostable response (above the transition)
  alpha > 0 : double-well node  -> two local states + hysteresis, a stand-in for the
              insulator/metal bistability of VO2 inside its hysteresis window.
The real VO2 neuristor is a thermal oscillator, not a gradient system; this model only
keeps the feature that matters for energy-based learning theory: multistability.

Sufficient condition for a unique equilibrium: E is strictly convex in s when
  alpha + lambda_max(W) < 0      (Hess_s E = diag(3 s^2 - alpha) - W)
which plays the role that convexity of Q plays in the linear resistive case.

EP (Scellier & Bengio 2017), symmetric nudging (Laborieux et al. 2021):
  free phase    s0     = relax(s_reset,  beta=0)
  nudged phases s(+-b) = relax(s0,  +-beta)            (minimise E + beta C)
  grad_theta L  ~ [dE/dtheta(s(+b)) - dE/dtheta(s(-b))] / (2 beta)

Experiments
  B1  number of distinct equilibria vs alpha (random fixed parameters).
  B2  training on XOR from ONE parameter initialisation but different reset states
      s_reset (the state the hardware relaxes from). Unique-equilibrium regime should
      give identical learning curves; bistable regime need not.
  B3  gradient fidelity: cosine(EP estimate, finite-difference gradient) vs beta,
      for monostable and bistable nodes; counts nudged phases that leave the free basin.

Run:  python part_b_ep_vo2_hopfield.py
Outputs: figures/partB_*.png, results/partB_summary.json
"""

import json
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.makedirs("figures", exist_ok=True)
os.makedirs("results", exist_ok=True)

N_IN, N_HID, N_OUT = 2, 6, 1
N = N_HID + N_OUT
OUT = slice(N_HID, N)
X = np.array([[-1, -1], [-1, 1], [1, -1], [1, 1]], float)
Y = -(X[:, :1] * X[:, 1:])          # XOR in +-1 coding, shape (4, 1)


# --------------------------------------------------------------------------
# Parameters
# --------------------------------------------------------------------------
def init_params(rng, scale=0.5):
    W = rng.normal(0, scale / np.sqrt(N), (N, N))
    W = np.triu(W, 1); W = W + W.T
    return dict(W=W, A=rng.normal(0, scale, (N, N_IN)), b=rng.normal(0, 0.1, N))


def flatten(p):
    iu = np.triu_indices(N, 1)
    return np.concatenate([p["W"][iu], p["A"].ravel(), p["b"]])


def unflatten(v):
    iu = np.triu_indices(N, 1); k = len(iu[0])
    W = np.zeros((N, N)); W[iu] = v[:k]; W = W + W.T
    A = v[k:k + N * N_IN].reshape(N, N_IN)
    return dict(W=W, A=A, b=v[k + N * N_IN:])


# --------------------------------------------------------------------------
# Dynamics
# --------------------------------------------------------------------------
def grad_s(s, p, x, alpha, beta=0.0, y=None):
    g = s**3 - alpha * s - s @ p["W"] - (x @ p["A"].T + p["b"])
    if beta != 0.0:
        g[:, OUT] += beta * (s[:, OUT] - y)
    return g


def relax(s0, p, x, alpha, beta=0.0, y=None, dt=0.1, tol=1e-8, max_steps=40000):
    """Gradient flow ds/dt = -dE/ds (explicit Euler) until the force vanishes."""
    s = s0.copy()
    for t in range(max_steps):
        g = grad_s(s, p, x, alpha, beta, y)
        s -= dt * g
        if t % 20 == 0 and np.abs(g).max() < tol:
            return s, True
    return s, bool(np.abs(grad_s(s, p, x, alpha, beta, y)).max() < 1e-5)


def dE_dtheta(s, x):
    """Mean over batch of dE/dtheta, flattened like the parameter vector (local, Hebbian terms)."""
    iu = np.triu_indices(N, 1)
    dW = -(s.T @ s) / len(s)
    dA = -(s.T @ x) / len(s)
    db = -s.mean(0)
    return np.concatenate([dW[iu], dA.ravel(), db])


def loss(s):
    return 0.5 * np.mean(np.sum((s[:, OUT] - Y) ** 2, 1))


def ep_gradient(p, s_reset, alpha, beta):
    s0, ok0 = relax(s_reset, p, X, alpha)
    sp, okp = relax(s0, p, X, alpha, +beta, Y)
    sm, okm = relax(s0, p, X, alpha, -beta, Y)
    g = (dE_dtheta(sp, X) - dE_dtheta(sm, X)) / (2 * beta)
    return g, s0, sp, sm, ok0 and okp and okm


# --------------------------------------------------------------------------
# B1: multiplicity of equilibria
# --------------------------------------------------------------------------
def count_equilibria(p, x, alpha, rng, n_init=300):
    S0 = rng.uniform(-2, 2, (n_init, N))
    Xb = np.repeat(x[None, :], n_init, 0)
    S, _ = relax(S0, p, Xb, alpha, max_steps=20000)
    return len(np.unique(np.round(S, 3), axis=0))


def experiment_B1(rng):
    p = init_params(np.random.default_rng(123))
    lam = np.linalg.eigvalsh(p["W"]).max()
    alphas = np.linspace(-1.5, 1.5, 13)
    counts = []
    for a in alphas:
        counts.append(np.mean([count_equilibria(p, x, a, rng) for x in X]))
    print("[B1] lambda_max(W) =", round(lam, 3), " -> convexity guaranteed for alpha <", round(-lam, 3))
    for a, c in zip(alphas, counts):
        print(f"      alpha={a:+.2f}  mean #stable equilibria per input = {c:.2f}")
    return alphas, np.array(counts), lam


# --------------------------------------------------------------------------
# B2: training from different reset states
# --------------------------------------------------------------------------
def keep_convex(p, alpha, margin=0.05):
    """Rescale W so that alpha + lambda_max(W) <= -margin (energy stays strictly convex in s).
    A simple retraction onto the convex-energy set (keeps W symmetric with zero diagonal)."""
    lam = np.linalg.eigvalsh(p["W"]).max()
    cap = -alpha - margin
    if lam > cap:
        p["W"] = p["W"] * (cap / lam)
    return p


def train(p0, s_reset, alpha, beta=0.1, lr=0.05, epochs=400, constrain=False):
    p = {k: v.copy() for k, v in p0.items()}
    if constrain:
        p = keep_convex(p, alpha)
    theta = flatten(p)
    L, convex = [], []
    for ep in range(epochs):
        g, s0, *_ = ep_gradient(p, s_reset, alpha, beta)
        L.append(loss(s0))
        convex.append(alpha + np.linalg.eigvalsh(p["W"]).max() < 0)
        theta = theta - lr * g
        p = unflatten(theta)
        if constrain:
            p = keep_convex(p, alpha); theta = flatten(p)
    s0, _ = relax(s_reset, p, X, alpha)
    acc = np.mean(np.sign(s0[:, OUT]) == np.sign(Y))
    n_eq = int(np.mean([count_equilibria(p, x, alpha, np.random.default_rng(5), 200) for x in X]) + 0.5)
    return np.array(L + [loss(s0)]), acc, np.array(convex), n_eq


B2_CONDITIONS = [("mono", -1.0, False), ("mono+convex", -1.0, True), ("bistable", 1.0, False)]


def experiment_B2(n_resets=8, epochs=400):
    p0 = init_params(np.random.default_rng(7))
    out = {}
    for name, alpha, constrain in B2_CONDITIONS:
        curves, accs, convex, n_eqs = [], [], [], []
        for r in range(n_resets):
            s_reset = np.random.default_rng(1000 + r).uniform(-1.5, 1.5, (1, N)).repeat(4, 0)
            L, acc, cvx, n_eq = train(p0, s_reset, alpha, epochs=epochs, constrain=constrain)
            curves.append(L); accs.append(acc); convex.append(cvx); n_eqs.append(n_eq)
        curves = np.array(curves); convex = np.array(convex)
        spread = curves.std(0)
        diverge = int(np.argmax(spread > 1e-6)) if np.any(spread > 1e-6) else None
        lost = int(np.argmax(~convex[0])) if np.any(~convex[0]) else None
        out[name] = dict(alpha=alpha, curves=curves, accs=np.array(accs), convex=convex,
                         n_eq=n_eqs, diverge_epoch=diverge, convexity_lost_epoch=lost)
        f = curves[:, -1]
        print(f"[B2] {name:12s} (alpha={alpha:+.0f}): final loss over {n_resets} resets "
              f"mean={f.mean():.4f} std={f.std():.1e} range=[{f.min():.4f}, {f.max():.4f}]")
        print(f"      XOR acc per reset={np.round(accs, 2).tolist()} | #equilibria/input after training={n_eqs}")
        print(f"      convexity (alpha+lambda_max(W)<0) first lost at epoch {lost} | "
              f"learning curves of different resets first differ (std>1e-6) at epoch {diverge}")
    return out


# --------------------------------------------------------------------------
# B3: gradient fidelity vs beta
# --------------------------------------------------------------------------
def fd_gradient(p, s_start, alpha, h=1e-5):
    """Gradient of the loss of the equilibrium continued from s_start (same basin)."""
    th = flatten(p); g = np.zeros_like(th)
    for k in range(len(th)):
        e = np.zeros_like(th); e[k] = h
        sp, _ = relax(s_start, unflatten(th + e), X, alpha, tol=1e-11)
        sm, _ = relax(s_start, unflatten(th - e), X, alpha, tol=1e-11)
        g[k] = (loss(sp) - loss(sm)) / (2 * h)
    return g


def experiment_B3(betas=(1e-3, 1e-2, 3e-2, 0.1, 0.3, 1.0, 3.0), n_params=6):
    res = {}
    for alpha in (-1.0, 1.0):
        cos = np.zeros((n_params, len(betas))); jumps = np.zeros((n_params, len(betas)))
        cvx = np.zeros(n_params, bool)
        for j in range(n_params):
            p = init_params(np.random.default_rng(200 + j), scale=0.5)
            cvx[j] = alpha + np.linalg.eigvalsh(p["W"]).max() < 0
            s_reset = np.random.default_rng(300 + j).uniform(-1.5, 1.5, (1, N)).repeat(4, 0)
            s0, _ = relax(s_reset, p, X, alpha, tol=1e-11)
            gt = fd_gradient(p, s0, alpha)
            for i, b in enumerate(betas):
                g, s0b, sp, sm, _ = ep_gradient(p, s0, alpha, b)
                cos[j, i] = g @ gt / (np.linalg.norm(g) * np.linalg.norm(gt) + 1e-300)
                # basin jump test: release the nudge and see whether the state returns to s0
                back_p, _ = relax(sp, p, X, alpha); back_m, _ = relax(sm, p, X, alpha)
                jumps[j, i] = max(np.abs(back_p - s0).max(), np.abs(back_m - s0).max()) > 1e-3
        res[alpha] = dict(cos=cos, jumps=jumps, convex=cvx)
        print(f"[B3] alpha={alpha:+.1f}  (energy provably convex in {int(cvx.sum())}/{n_params} networks)")
        for i, b in enumerate(betas):
            print(f"      beta={b:<6g} cos(EP, true) median={np.median(cos[:, i]):+.4f} "
                  f"min={cos[:, i].min():+.4f} | nudged phase left the free basin in "
                  f"{int(jumps[:, i].sum())}/{n_params} networks")
    return betas, res


# --------------------------------------------------------------------------
def main():
    rng = np.random.default_rng(0)
    alphas, counts, lam = experiment_B1(rng)
    B2 = experiment_B2()
    betas, B3 = experiment_B3()

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    ax[0].plot(alphas, counts, "o-")
    ax[0].axvline(-lam, c="k", ls="--", label=r"$\alpha=-\lambda_{\max}(W)$ (convexity edge)")
    ax[0].set_xlabel(r"$\alpha$  (single well $\to$ double well)")
    ax[0].set_ylabel("distinct equilibria per input"); ax[0].set_title("B1: multistability")
    ax[0].legend(fontsize=8)

    for (name, alpha, _), c in zip(B2_CONDITIONS, ("C0", "C2", "C3")):
        d = B2[name]
        for k, L in enumerate(d["curves"]):
            ax[1].semilogy(L, c=c, alpha=0.6, lw=1, label=f"{name}" if k == 0 else None)
        if d["convexity_lost_epoch"] is not None:
            ax[1].axvline(d["convexity_lost_epoch"], c=c, ls=":", lw=1)
    ax[1].set_xlabel("epoch"); ax[1].set_ylabel("XOR loss (free phase)")
    ax[1].set_title("B2: same $\\theta_0$, 8 reset states (dotted: convexity lost)"); ax[1].legend(fontsize=8)

    for alpha, c in ((-1.0, "C0"), (1.0, "C3")):
        cos = B3[alpha]["cos"]
        ax[2].semilogx(betas, np.median(cos, 0), "o-", c=c, label=rf"$\alpha={alpha:+g}$ median")
        ax[2].fill_between(betas, cos.min(0), cos.max(0), color=c, alpha=0.2)
    ax[2].set_xlabel(r"nudging strength $\beta$"); ax[2].set_ylabel("cosine(EP, true gradient)")
    ax[2].set_title("B3: gradient fidelity"); ax[2].legend(fontsize=8)
    fig.tight_layout(); fig.savefig("figures/partB_ep_vo2.png", dpi=150)

    summary = dict(
        B1=dict(alphas=alphas.tolist(), mean_equilibria=counts.tolist(), lambda_max_W=float(lam)),
        B2={k: dict(alpha=v["alpha"], final_loss=v["curves"][:, -1].tolist(), accuracy=v["accs"].tolist(),
                    equilibria_after_training=v["n_eq"], convexity_lost_epoch=v["convexity_lost_epoch"],
                    curves_diverge_epoch=v["diverge_epoch"]) for k, v in B2.items()},
        B3={str(a): dict(betas=list(betas), cos_median=np.median(v["cos"], 0).tolist(),
                         cos_min=v["cos"].min(0).tolist(), basin_jumps=v["jumps"].sum(0).tolist())
            for a, v in B3.items()},
    )
    with open("results/partB_summary.json", "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
