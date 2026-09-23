# Energy-based learning: from linear resistive networks to multistable (VO₂-like) nodes

A small exploratory project to test my understanding of energy-based learning before
applying to a PhD on systems and control theory for energy-based learning.

* **Part A** — Contrastive Learning (CL) on a linear resistive crossbar, re-implementing the
  setting of Huijzer, Chaffey, Besselink & van Waarde, *Convergence of energy-based learning in
  linear resistive networks*, arXiv:2503.00349.
* **Part B** — Equilibrium Propagation (EP) on a Hopfield-type network whose nodes have a Landau
  double-well potential, a *phenomenological* stand-in for the bistability of VO₂ inside its
  hysteresis window. It is not a device model: a real VO₂ neuristor is a thermal oscillator,
  not a gradient system. The model keeps only the feature relevant here, multistability.

```
pip install -r requirements.txt
python part_a_contrastive_linear.py     # ~1 min
python part_b_ep_vo2_hopfield.py        # ~2 min
```
Figures go to `figures/`, all numbers to `results/*.json`.

## Part A — Contrastive Learning in a linear resistive network

Notation as in the paper: $Q(g)=v_D^\top G v_D - v(g)^\top G v(g)$, local rule
$\nabla Q = v_D^2 - v(g)^2$, update $g\leftarrow P_{\mathcal C}(g-\gamma\nabla Q)$,
guaranteed convergence for $\gamma<2/K$ (Theorem 4).
Setup: 10×8 crossbar (80 branches), $p_I=[1,\dots,10]$, teacher conductances $U(0.1,10)$,
$g^0=2$, $\epsilon=0.1$, 20 000 iterations.

| Check / experiment | Result (seed 0) |
|---|---|
| A1 local rule vs finite differences | rel. error 3e-8 |
| A1 Hessian formula (Lemma 5) vs finite differences | rel. error 4e-9 |
| A1 convexity: min Hessian eigenvalue at 50 random $g$ | −3e-15 (numerically ≥ 0) |
| $2/K$ | 2.7e-7 |
| Every $\gamma\le 2/K$ gives monotone decrease of $Q$ | yes |
| A2 ($g\ge\epsilon$ only): largest tested $\gamma$ that converged | 1e4 (≈ 4e10 × $2/K$); no failure found |
| A2b: $(\gamma, g^0)$ run vs $\gamma\cdot$(unit-step run from $g^0/\gamma$, floor $\epsilon/\gamma$) | identical to 1e-15 |
| A3 ($0.1\le g\le 20$): critical step size $\gamma_c$ | ≈ 1.84 (≈ 7e6 × $2/K$) |
| A3: $2/\lambda_{\max}(\nabla^2Q(g^*))$ at $\gamma_c$ | ≈ 1.86 |
| A3: behaviour just above $\gamma_c$ | stable period-2 cycle, 60 % of conductances pinned at $g_{\max}$ |

**Interpretation.**
1. The bound of Theorem 4 holds, and it is very conservative in this example, as the paper
   itself reports in its Sec. VI-B.
2. The reason large steps still work: branch voltages are invariant under $g\to cg$, so
   $\nabla Q$ is homogeneous of degree 0. With only a lower bound, a run with step $\gamma$ is
   exactly a rescaled unit-step run started from $g^0/\gamma$ (A2b). The step size only moves the
   initial condition, and the iteration settles on a minimiser whose curvature satisfies
   $\gamma\lambda_{\max}<2$ (middle panel of `figures/partA_stepsize.png`).
3. A hardware upper bound on conductance breaks this invariance. A finite critical step size
   then appears; it matches the local-curvature limit $2/\lambda_{\max}$ at the flattest
   reachable minimiser, and beyond it convergence is lost through a period-2 cycle
   (a flip-type instability), not through divergence.

These are observations on one small example, not theorems.

## Part B — EP with multistable nodes

$E(s)=\sum_i\big(\tfrac14 s_i^4-\tfrac{\alpha}{2}s_i^2\big)-\tfrac12 s^\top Ws-s^\top(Ax+b)$,
cost $\tfrac12\|s_{\rm out}-y\|^2$, symmetric-nudging EP, task XOR (2 inputs, 6 hidden, 1 output).
$\alpha<0$: single well (monostable). $\alpha>0$: double well (bistable).
Sufficient condition for a unique equilibrium: $\alpha+\lambda_{\max}(W)<0$ (energy strictly convex in $s$).

| Experiment | Result |
|---|---|
| B1 distinct equilibria per input vs $\alpha$ (fixed random $W$, $\lambda_{\max}(W)=0.72$) | 1 for $\alpha\le-0.25$; 2 at 0.25; ≈10 at 1.0; ≈25 at 1.5 |
| B2 monostable ($\alpha=-1$), same $\theta_0$, 8 reset states | curves identical until epoch 93; convexity condition lost at epoch 67; after training 7/8 networks have 2 equilibria/input; final loss 0.001–0.57 |
| B2 monostable + rescaling $W$ to keep the energy convex | all 8 curves identical (std 3e-10), 100 % XOR accuracy, but higher loss (0.24) |
| B2 bistable ($\alpha=+1$) | curves differ from epoch 0; final loss 0.01–1.64; accuracy 50–100 %; 4–16 equilibria/input |
| B3 cosine(EP, finite-difference gradient), $\beta\le0.1$ | ≥ 0.99 in both regimes |
| B3 at $\beta=0.3$ | monostable: min 0.94, no basin changes · bistable: min 0.12, nudged phase left the free basin in 3/6 networks |

**Interpretation.**
1. With a unique equilibrium, learning does not depend on the state the network is reset to;
   with several equilibria it does, and the free-phase loss shows jumps when the equilibrium
   reached from the reset state switches basin.
2. Training itself can push a network out of the unique-equilibrium regime (B2, monostable).
   Keeping the energy convex restores reset-independence but costs expressivity.
3. In the bistable regime a finite nudge can move the nudged phase to another basin, so the
   EP estimate stops approximating the gradient of the free-phase loss at smaller $\beta$
   than in the monostable regime. For large $\beta$ both regimes degrade (finite-$\beta$ bias).

## Limitations
* Single small examples, few seeds; no statistics across network sizes.
* Part B's node potential is phenomenological, not a VO₂ device model.
* The convexity-keeping step in B2 is a simple rescaling of $W$, not an exact projection.
