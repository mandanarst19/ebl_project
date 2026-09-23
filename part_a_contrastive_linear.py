"""
Part A — Contrastive Learning in a linear resistive network
==========================================================

Re-implementation of the setting of
  A.-M. Huijzer, T. Chaffey, B. Besselink, H. J. van Waarde,
  "Convergence of energy-based learning in linear resistive networks",
  arXiv:2503.00349 (v2).

Notation follows the paper:
  D = [D_I; D_O]            incidence matrix, rows split into input / output nodes
  g in R^B, G = diag(g)     conductances, constrained to C_eps = {g >= eps}
  p_O(g) = -(D_O G D_O^T)^{-1} D_O G D_I^T p_I            (eq. 12)
  v(g)   = D^T [p_I; p_O(g)]                               (eq. 13)
  v^D    = D^T [p_I; p_O^D]                                (eq. 15)
  Q(g)   = v_D^T G v_D - v(g)^T G v(g)                     (eq. 16)
  grad Q = v_D^2 - v(g)^2                                  (Lemma 1: local rule)
  Hess Q = 2 diag(v) W diag(v), W = D_O^T (D_O G D_O^T)^{-1} D_O    (Lemma 5)
  K      = (2/eps)(||D_I|| + sqrt(N_I N_O)||D_O||)^2 ||p_I||^2     (Theorem 4)
  g <- P_C(g - gamma grad Q), convergence guaranteed for gamma in (0, 2/K).

Experiments
  A1  correctness: local rule and Hessian formula vs finite differences; convexity check.
  A2  paper setting (only g >= eps): sweep gamma over ~8 decades.
      Observation to test: grad Q is invariant under g -> c g (v(cg) = v(g)), so
      running with step gamma from g0 is *exactly* the unit-step iteration started
      at g0/gamma with floor eps/gamma, rescaled by gamma. The step size then only
      re-scales the initial condition. A2b verifies this identity numerically.
  A3  hardware-realistic setting (eps <= g <= g_max): the rescaling argument breaks.
      Find the empirical critical step size and how convergence is lost
      (fixed point vs. period-2 cycle), and compare to 2/K and to local curvature.

Run:  python part_a_contrastive_linear.py
Outputs: figures/partA_*.png, results/partA_summary.json
"""

import json
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.makedirs("figures", exist_ok=True)
os.makedirs("results", exist_ok=True)


# --------------------------------------------------------------------------
# Network
# --------------------------------------------------------------------------
class ResistiveNetwork:
    """Linear resistive network: input nodes driven by voltage sources, output nodes free."""

    def __init__(self, D_I, D_O):
        self.D_I = np.asarray(D_I, float)  # (N_I, B)
        self.D_O = np.asarray(D_O, float)  # (N_O, B)
        self.N_I, self.B = self.D_I.shape
        self.N_O = self.D_O.shape[0]

    @classmethod
    def crossbar(cls, n_in, n_out):
        """Complete bipartite graph between input and output nodes (paper, Sec. VI-A)."""
        B = n_in * n_out
        D_I = np.zeros((n_in, B))
        D_O = np.zeros((n_out, B))
        k = 0
        for i in range(n_in):
            for o in range(n_out):
                D_I[i, k], D_O[o, k] = 1.0, -1.0
                k += 1
        return cls(D_I, D_O)

    def output_potentials(self, g, p_I):
        A = (self.D_O * g) @ self.D_O.T
        b = -(self.D_O * g) @ self.D_I.T @ p_I
        return np.linalg.solve(A, b)

    def branch_voltages(self, p_I, p_O):
        return self.D_I.T @ p_I + self.D_O.T @ p_O

    def free_voltages(self, g, p_I):
        return self.branch_voltages(p_I, self.output_potentials(g, p_I))

    def Q(self, g, p_I, p_OD):
        vD = self.branch_voltages(p_I, p_OD)
        v = self.free_voltages(g, p_I)
        return vD @ (g * vD) - v @ (g * v)

    def grad_Q(self, g, p_I, p_OD):
        """Contrastive rule: branch k only needs its clamped and free voltage."""
        vD = self.branch_voltages(p_I, p_OD)
        v = self.free_voltages(g, p_I)
        return vD**2 - v**2

    def hess_Q(self, g, p_I):
        v = self.free_voltages(g, p_I)
        A = (self.D_O * g) @ self.D_O.T
        W = self.D_O.T @ np.linalg.solve(A, self.D_O)
        return 2.0 * (v[:, None] * W * v[None, :])

    def K_bound(self, p_I, eps):
        nDI = np.linalg.norm(self.D_I, 2)
        nDO = np.linalg.norm(self.D_O, 2)
        return (2.0 / eps) * (nDI + np.sqrt(self.N_I * self.N_O) * nDO) ** 2 * (p_I @ p_I)


def T_map(net, g, p_I, p_OD, gamma, lo, hi):
    """One step of Algorithm 1 with projection onto the box [lo, hi]^B."""
    return np.clip(g - gamma * net.grad_Q(g, p_I, p_OD), lo, hi)


def run_cl(net, p_I, p_OD, g0, gamma, lo, hi=np.inf, n_steps=20000, record_every=20):
    g = g0.copy()
    errs, Qs = [], []
    for t in range(n_steps):
        g = T_map(net, g, p_I, p_OD, gamma, lo, hi)
        if t % record_every == 0 or t == n_steps - 1:
            errs.append(np.linalg.norm(net.output_potentials(g, p_I) - p_OD))
            Qs.append(net.Q(g, p_I, p_OD))
    return g, np.array(errs), np.array(Qs)


def classify(net, g, errs, Qs, p_I, p_OD, gamma, lo, hi, tol):
    """converged / slow-but-monotone / period-2 cycle / other."""
    if errs[-1] < tol:
        return "converged"
    g1 = T_map(net, g, p_I, p_OD, gamma, lo, hi)
    g2 = T_map(net, g1, p_I, p_OD, gamma, lo, hi)
    if np.linalg.norm(g2 - g) < 1e-8 * (1 + np.linalg.norm(g)) and np.linalg.norm(g1 - g) > 1e-6:
        return "period-2 cycle"
    if np.all(np.diff(Qs) <= 1e-12 * (1 + abs(Qs[0]))) and Qs[-1] < Qs[0]:
        return "monotone, not yet converged"
    return "other"


# --------------------------------------------------------------------------
# A1: checks
# --------------------------------------------------------------------------
def checks(net, p_I, p_OD, rng, h=1e-6):
    g = rng.uniform(0.5, 5.0, net.B)
    gq = net.grad_Q(g, p_I, p_OD)
    fd = np.array([(net.Q(g + h * e, p_I, p_OD) - net.Q(g - h * e, p_I, p_OD)) / (2 * h)
                   for e in np.eye(net.B)])
    H = net.hess_Q(g, p_I)
    Hfd = np.array([(net.grad_Q(g + h * e, p_I, p_OD) - net.grad_Q(g - h * e, p_I, p_OD)) / (2 * h)
                    for e in np.eye(net.B)]).T
    min_eig = min(np.linalg.eigvalsh(net.hess_Q(rng.uniform(0.1, 10, net.B), p_I)).min()
                  for _ in range(50))
    return (np.linalg.norm(gq - fd) / np.linalg.norm(fd),
            np.linalg.norm(H - Hfd) / np.linalg.norm(Hfd), float(min_eig))


def sweep(net, p_I, p_OD, g0, gammas, lo, hi, n_steps, tol):
    rows = []
    for gm in gammas:
        g, errs, Qs = run_cl(net, p_I, p_OD, g0, gm, lo, hi, n_steps)
        lam = np.linalg.eigvalsh(net.hess_Q(g, p_I)).max()
        rows.append(dict(gamma=float(gm), final_err=float(errs[-1]),
                         outcome=classify(net, g, errs, Qs, p_I, p_OD, gm, lo, hi, tol),
                         lam_max_final=float(lam), gamma_times_lam=float(gm * lam),
                         frac_at_upper=float(np.mean(g >= hi - 1e-9)) if np.isfinite(hi) else 0.0,
                         errs=errs))
    return rows


def main(seed=0):
    rng = np.random.default_rng(seed)
    n_in, n_out, eps, g_max = 10, 8, 0.1, 20.0
    n_steps, tol = 20000, 1e-6
    net = ResistiveNetwork.crossbar(n_in, n_out)
    p_I = np.arange(1, n_in + 1, dtype=float)          # as in the paper, Sec. VI-B
    g_teacher = rng.uniform(eps, 10.0, net.B)          # Assumption 1 (realisable target)
    p_OD = net.output_potentials(g_teacher, p_I)
    g0 = np.full(net.B, 2.0)

    # ---------------- A1
    ge, he, me = checks(net, p_I, p_OD, rng)
    print(f"[A1] grad rule vs FD rel.err = {ge:.1e} | Hessian (Lemma 5) vs FD rel.err = {he:.1e}"
          f" | min Hessian eig at 50 random g = {me:.1e}")

    K = net.K_bound(p_I, eps)
    gamma_K = 2.0 / K
    print(f"[A ] K = {K:.3e}  ->  2/K = {gamma_K:.3e}")

    gammas = np.logspace(np.log10(gamma_K), 4, 40)

    # ---------------- A2: paper setting (lower bound only)
    rowsA2 = sweep(net, p_I, p_OD, g0, gammas, eps, np.inf, n_steps, tol)
    outc = [r["outcome"] for r in rowsA2]
    print("[A2] outcomes (g >= eps only):", {o: outc.count(o) for o in set(outc)})
    conv = [r for r in rowsA2 if r["outcome"] == "converged"]
    print(f"[A2] largest step tested that converged: {max(r['gamma'] for r in conv):.1e}"
          f"  (= {max(r['gamma'] for r in conv)/gamma_K:.1e} x 2/K)")
    print(f"[A2] gamma*lambda_max(Hess Q(g_final)) over converged runs: "
          f"max = {max(r['gamma_times_lam'] for r in conv):.3f}  (local stability needs < 2)")

    # ---------------- A2b: rescaling identity
    gm = 37.0
    gA = g0.copy(); gB = g0 / gm
    dev = 0.0
    for _ in range(500):
        gA = T_map(net, gA, p_I, p_OD, gm, eps, np.inf)
        gB = T_map(net, gB, p_I, p_OD, 1.0, eps / gm, np.inf)
        dev = max(dev, np.linalg.norm(gA - gm * gB) / np.linalg.norm(gA))
    print(f"[A2b] max rel. deviation between (gamma={gm}, g0) and gamma*(unit step, g0/gamma): {dev:.1e}")

    # ---------------- A3: bounded conductances
    rowsA3 = sweep(net, p_I, p_OD, g0, gammas, eps, g_max, n_steps, tol)
    outc3 = [r["outcome"] for r in rowsA3]
    print(f"[A3] outcomes ({eps} <= g <= {g_max}):", {o: outc3.count(o) for o in set(outc3)})
    # refine threshold between last converged and first failure
    conv3 = [r["gamma"] for r in rowsA3 if r["outcome"] in ("converged", "monotone, not yet converged")]
    fail3 = [r["gamma"] for r in rowsA3 if r["outcome"] not in ("converged", "monotone, not yet converged")]
    lo_g, hi_g = max(conv3), min(g for g in fail3 if g > max(conv3))
    for _ in range(25):
        mid = np.sqrt(lo_g * hi_g)
        g, errs, Qs = run_cl(net, p_I, p_OD, g0, mid, eps, g_max, n_steps)
        if classify(net, g, errs, Qs, p_I, p_OD, mid, eps, g_max, tol) == "converged":
            lo_g = mid
        else:
            hi_g = mid
    gamma_c = lo_g
    g_c, _, _ = run_cl(net, p_I, p_OD, g0, gamma_c, eps, g_max, n_steps)
    lam_c = np.linalg.eigvalsh(net.hess_Q(g_c, p_I)).max()
    first_fail = next(r for r in rowsA3 if r["gamma"] > gamma_c)
    print(f"[A3] critical step size gamma_c ~= {gamma_c:.4f} (= {gamma_c/gamma_K:.1e} x 2/K)")
    print(f"[A3] at gamma_c: 2/lambda_max(Hess Q(g*)) = {2/lam_c:.4f}")
    print(f"[A3] just above gamma_c the iteration ends in: {first_fail['outcome']}"
          f" (fraction of conductances pinned at g_max: {first_fail['frac_at_upper']:.2f})")

    # every gamma <= 2/K should be monotone or converged (Theorem 4 regime)
    thm_ok = all(r["outcome"] in ("converged", "monotone, not yet converged")
                 for r in rowsA2 + rowsA3 if r["gamma"] <= gamma_K * (1 + 1e-9))

    # ---------------- figures
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    for rows, lab, c in [(rowsA2, r"$g\geq\epsilon$ (paper)", "C0"),
                         (rowsA3, rf"$\epsilon\leq g\leq {g_max:g}$", "C3")]:
        ax[0].loglog([r["gamma"] for r in rows], [max(r["final_err"], 1e-16) for r in rows],
                     "o-", ms=3, c=c, label=lab)
    ax[0].axvline(gamma_K, c="k", ls="--", label=r"$2/K$ (Theorem 4)")
    ax[0].axvline(gamma_c, c="C3", ls=":", label=r"empirical $\gamma_c$ (bounded)")
    ax[0].set_xlabel(r"step size $\gamma$"); ax[0].set_ylabel(rf"$\|p_O-p_O^D\|$ after {n_steps} steps")
    ax[0].set_title("A2/A3: convergence vs. step size"); ax[0].legend(fontsize=8)

    c2 = [r for r in rowsA2 if r["outcome"] == "converged"]
    ax[1].semilogx([r["gamma"] for r in c2], [r["gamma_times_lam"] for r in c2], "o", ms=3, c="C0",
                   label=r"$g\geq\epsilon$")
    c3 = [r for r in rowsA3 if r["outcome"] == "converged"]
    ax[1].semilogx([r["gamma"] for r in c3], [r["gamma_times_lam"] for r in c3], "s", ms=3, c="C3",
                   label="bounded")
    ax[1].axhline(2, c="k", ls="--", label="local stability limit")
    ax[1].set_xlabel(r"$\gamma$"); ax[1].set_ylabel(r"$\gamma\,\lambda_{\max}(\nabla^2 Q(g^*))$")
    ax[1].set_title("Curvature of the minimiser that is reached"); ax[1].legend(fontsize=8)

    for r in rowsA3:
        if r["gamma"] in (rowsA3[5]["gamma"], max(c3, key=lambda z: z["gamma"])["gamma"], first_fail["gamma"]):
            ax[2].semilogy(np.arange(len(r["errs"])) * 20, np.maximum(r["errs"], 1e-16),
                           label=rf"$\gamma$={r['gamma']:.2e} ({r['outcome']})")
    ax[2].set_xlabel("iteration"); ax[2].set_ylabel(r"$\|p_O-p_O^D\|$")
    ax[2].set_title("A3: traces (bounded conductances)"); ax[2].legend(fontsize=7)
    fig.tight_layout(); fig.savefig("figures/partA_stepsize.png", dpi=150)

    strip = lambda rows: [{k: v for k, v in r.items() if k != "errs"} for r in rows]
    summary = dict(n_in=n_in, n_out=n_out, branches=net.B, eps=eps, g_max=g_max,
                   n_steps=n_steps, tol=tol, grad_rel_err=ge, hess_rel_err=he, min_hess_eig=me,
                   K=K, two_over_K=gamma_K, rescaling_identity_max_dev=dev,
                   gamma_c_bounded=gamma_c, two_over_lam_at_gamma_c=2 / lam_c,
                   outcome_above_gamma_c=first_fail["outcome"],
                   theorem4_regime_all_monotone=thm_ok,
                   sweep_unbounded=strip(rowsA2), sweep_bounded=strip(rowsA3))
    with open("results/partA_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[A ] every gamma <= 2/K gave monotone decrease of Q: {thm_ok}")


if __name__ == "__main__":
    main()
