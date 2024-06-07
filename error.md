# Computation of the error in stellacode

## Some theory

Since the total magnetic field has no normal component wrt the LCFS $S_p=\partial\Omega_p$, we search for $J$ such that
$$B_{\rm plasma}(r)\cdot n(r) + B_{\rm coils}\cdot n(r) = 0 \quad \forall r\in S_p,$$
where $B_{\rm plasma}$ is given by an equilibrium code, ${\sf n}$ is the normal vector, and
$$B_{\rm coils}(r)={\rm BS}(J)(r) := \frac{\mu_0}{4\pi}\int_S\frac{J(r')\times(r-r')}{|r-r'|^3}dr'\quad \forall r\in S_p.$$
See
* `biot_et_savart`, `biot_et_savart_op` (which do not take into account the magnetic constant $\frac{\mu_0}{4\pi}$)
* `BiotSavartOperator` (which does take into account the magnetic constant).

Since the current $J$ is tangent to $S$, we consider the vector field $K$ such that $K\cdot ${\sf n}$ = 0$ on $S$ and
$$J(r) = K(\theta(r),\zeta(r)) \quad \forall r\in S.$$
This vector field can be expressed in terms of a current potential $K = ${\sf n}$\times \nabla\phi$, which itself can be expressed as a Fourier series plus two periodic terms
$$\phi(\theta,\zeta)=I_t\theta+I_p\zeta+\sum_{m,n}\varphi_{m,n}\sin(m\theta+n\zeta),$$
where $I_t,I_p$ are the net toroidal and poloidal currents.

References :
* [An Introduction to Stellarators: From magnetic fields to symmetries and optimization](https://arxiv.org/abs/1908.05360v2), L.-M. Imbert-Gérard, E.J. Paul, A.M. Wright, 2020
* [An improved current potential method for fast computation of stellarator coil shapes](https://arxiv.org/abs/1609.04378), M. Landreman, 2017

## EMCost

To get this current, we run an optimization algorithm to solve
$$\min \chi^2 = \min \left(\chi_B^2+\lambda\chi_{\rm coils}^2\right)$$
where usually
$$\chi_B^2 := \int_{S_p}((B_{target}-B_{coils})\cdot {\sf n})^2d^2x
\quad\text{and}\quad
\chi_{\rm coils}^2 := \int_S|K|^2(x)d^2x.$$
The idea is that the first penalization term accounts for equilibrium properties and the second for the coil structure.

*Remark: Small $|K|$ means small current density means less current potential contours, large $|K|$ means more current potential contours.*

In the code, `cost_B` stands for $\chi_B^2$, `cost_J` for $\chi_{\rm coils}^2$, and `em_cost` for $\chi^2$.

## Other costs

It is also possible to use other definitions for $\chi_B$ and $\chi_{\rm coils}$, or to add other penalization terms, such as :
* a penalization to ensure a minimal distance coil to plasma (see `DistanceCost`)
* a penalization for negative poloidal currents (see `PoloidalCurrentCost`)
* ...


## W7-X equilibrium

The goal is to find a current $J$ over a CWS $S$ that reproduces the target magnetic field $B_T$ of W7-X in the plasma with a given precision :
$$|B(x)-B_T(x)|/|B_T(x)| \leq 10^{-4} \quad \forall x\in \Omega_P.$$

