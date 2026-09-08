# Dahl Friction Discrete Shell Bending

**DahlFrictionDiscreteShellBending** is a constitutive model for thin-shell bending with **internal-friction hysteresis** (cloth deformation memory). Real fabric dissipates energy while bending and unbending: a crumpled garment keeps its folds because internal friction moments resist any change of the current curvature, recoverably and without permanent rest-shape change. Purely elastic discrete-shell bending (as if the fabric had no internal friction) lets crumpled cloth spring open under its own stored bending energy.

The model superposes, on the standard discrete-shell dihedral hinge, a **Dahl-type friction moment** $F$ conjugate to the dihedral angle $\theta$: rate-independent, bounded by a saturation moment, following an exact exponential approach so the state stays admissible by construction. The elastic part is **identical to [DiscreteShellBending](./discrete_shell_bending.md)**; with `friction_moment_per_length = 0` the law degenerates to it exactly.

Use this constitution **instead of** `DiscreteShellBending` (it contains the elastic term; applying both double-counts bending).

Reference:

- [Discrete Shell](https://www.cs.columbia.edu/cg/pdfs/10_ds.pdf)
- Miguel et al., *Modeling and Estimation of Internal Friction in Cloth*, EGSR 2013 (motivation for internal-friction cloth models; the constant-parameter Dahl law here is the scalar special case)

## User-Defined UID 4294967297 DahlFrictionDiscreteShellBending

For a shell bending element defined by four vertices at positions $\mathbf{x}_0$, $\mathbf{x}_1$, $\mathbf{x}_2$, $\mathbf{x}_3$, where $(\mathbf{x}_1, \mathbf{x}_2)$ is the shared edge of the two adjacent triangles. The rest configuration constants $L_0$ (rest edge length), $\bar{h}$ (average height) and $\bar\theta$ (rest dihedral angle) are the same as for [DiscreteShellBending](./discrete_shell_bending.md), with the discrete-shell weight $w = L_0/\bar{h} = 3L_0^2/A$.

### Per-edge parameters

| Parameter | Unit | Meaning |
| --- | --- | --- |
| `bending_stiffness` $\kappa$ | $\mathrm{N\cdot m}$ | Elastic bending stiffness (same meaning as DiscreteShellBending). |
| `friction_moment_per_length` $\hat{m}$ | $\mathrm{N\cdot m/m}$ | Saturated internal-friction moment per unit crease length. The per-edge saturation moment is $M_e = \hat{m} L_0$, which keeps the total moment across a crease invariant under mesh refinement. `0` disables friction. |
| `friction_transition_angle` $\ell$ | $\mathrm{rad}$ | Angle over which the friction moment reaches $1 - e^{-1} \approx 63\%$ of saturation when evolving from $F=0$. The initial slope is $\sigma_e = M_e/\ell$. A mesh-independent material constant. |

### State (per edge, history dependent)

- $\theta_c$: committed dihedral angle at the last accepted frame
- $F_c$: committed friction moment, always $|F_c| \le M_e$

Both are initialized from the rest configuration as $(\bar\theta, 0)$ — fresh, never-bent cloth — and are included in engine dump/recover.

### Anchored incremental friction potential

With the increment $d = \operatorname{wrap}(\theta - \theta_c)$, $s = \operatorname{sign}(d)$, $x = |d|/\ell$:

$$
W(d;\,F_c) = F_c\,d + (M_e - s F_c)\,\ell\,\bigl[x - (1 - e^{-x})\bigr]
$$

$$
\frac{dW}{dd} = s M_e + (F_c - s M_e)\,e^{-x} \;\; (= F_{\mathrm{new}}), \qquad
\frac{d^2W}{dd^2} = \frac{M_e - s F_c}{\ell}\,e^{-x} \;\;\ge 0
$$

At $d = 0$ the law uses $s=0$, which reproduces the exact one-sided limits $W=0$, $dW/dd = F_c$ and the average tangent $d^2W/dd^2 = M_e/\ell$. The scalar law is $C^1$ in $d$; the returned Hessian is a consistent generalized tangent everywhere.

**Anchoring contract:** every trial evaluation during the Newton/line-search loop of a frame reads the same committed $(\theta_c, F_c)$; the state commits exactly once per frame from the frame's final positions (the same bookkeeping the engine uses for frictional contact anchors). With recoverable internal storage $U(F) = F^2/(2\sigma_e)$, the dissipation $W - \Delta U = \int F^2/M_e\,|d\theta| \ge 0$ for every monotonic segment: the law is passive, and holds with no motion dissipate nothing.

### Total per-edge energy and derivatives

$$
P(\theta) = \kappa\,w\,(\operatorname{wrap}(\theta - \bar\theta))^2 + W(\operatorname{wrap}(\theta-\theta_c);\,F_c)
$$

$$
\frac{dP}{d\theta} = 2\kappa w\,(\operatorname{wrap}(\theta-\bar\theta)) + F_{\mathrm{new}}, \qquad
\frac{d^2P}{d\theta^2} = 2\kappa w + \frac{M_e - s F_c}{\ell} e^{-x}
$$

Spatial gradients/Hessians w.r.t. the four vertices follow the standard chain rule through the dihedral angle derivatives, with a make-SPD projection on the $12\times12$ stencil Hessian, exactly like the other discrete-shell constitutions.

### State update (accepted frames only)

$$
F^{new} = s M_e + (F_c - s M_e)\, e^{-|d|/\ell}, \qquad \theta_c \leftarrow \theta,\; F_c \leftarrow F^{new}
$$

The update is a convex combination of $F_c$ and $s M_e$, so admissibility $|F|\le M_e$ holds by construction. The law is rate-independent (no time or velocity enters).

### Notes and limitations

- The wrapped increment is continuous against the committed state; a hinge that rotates through $\pm\pi$ between two frames produces one bounded spurious reversal ($|F|\le M_e$), after which the state re-synchronizes. The same wrap convention is used by the plastic discrete-shell constitutions.
- Friction is bend-only: there is no membrane/shear internal friction, no rate-dependent damping, and no plastic rest-shape change. Recoverable deformation memory only.
- A `positions`-only NPZ capture is not a restart of this constitution: the committed $(\theta_c, F_c)$ state must come from engine dump/recover.

### Example

```python
from uipc.constitution import DahlFrictionDiscreteShellBending

dahl = DahlFrictionDiscreteShellBending()
# kappa=2e-5 N*m (as used for DiscreteShellBending), saturated friction moment
# 1.5e-4 N*m per meter of crease, transition angle 0.05 rad:
dahl.apply_to(cloth_mesh, 2e-5, 1.5e-4, 0.05)
```
