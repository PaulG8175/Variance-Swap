# Variance Swap Pricing under Stochastic Volatility

> 🚀 **Personal project**, done independently outside coursework.

Pricing of **Variance Swaps** under a stochastic volatility model, with comparison between the theoretical (continuous) and real (discrete) variance swap, and a static replication result via a continuum of European options.

---

## Products

### Theoretical Variance Swap
$$VS^{\text{theo}} = e^{-rT}\mathbb{E}\left[\frac{1}{T}\int_0^T \sigma^2(s)\*ds - K^2\right]$$

### Real (Discrete) Variance Swap
$$VS^{\text{real}} = e^{-rT}\mathbb{E}\left[\frac{1}{T}\sum_{i=1}^{N}\ln\left(\frac{S(t_i)}{S(t_{i-1})}\right)^{2} - K^2\right]$$

---

## Model

Log-normal stochastic volatility model:

$$dS(t) = S(t)\left(rdt + \sigma(t)\*dW(t)\right)$$

$$d\sigma(t) = \sigma(t)\nu\*dZ(t), \quad d\langle W,Z\rangle_t = \rho\*dt$$

**Closed-form solution for σ(t):**
$$\sigma(t) = \sigma_0 \exp\left(\nu Z(t) - \tfrac{1}{2}\nu^2 t\right)$$

**Closed-form price (theoretical swap):**
$$VS^{\text{theo}} = e^{-rT}\left(\sigma_0^2\frac{e^{\nu^2 T}-1}{\nu^2} - K^2\right)$$

---

## Parameters

| Parameter | Value |
|-----------|-------|
| `S₀` | 1 |
| `σ₀` | 0.35 |
| `ν` | 0.8 |
| `ρ` | −0.5 |
| `K` | 0.35 |
| `r` | 0.01 |
| `T` | 3 months |
| `N` (steps) | 63 (dt = 1/252) |

> All simulations use **Uniform random variables only** via Box-Muller transform.

---

## Variance reduction

### Antithetic variables
For each draw $(\varepsilon_1, \varepsilon_2)$, also evaluate at $(-\varepsilon_1, -\varepsilon_2)$:

$$\hat{VS} = \frac{1}{2}\left(VS(\varepsilon) + VS(-\varepsilon)\right)$$

---

## Key results

**Effect of ν:**

- $VS^{\text{real}}$ increases with $\nu$: higher vol-of-vol → larger realised variance
- The gap $VS^{\text{real}} - VS^{\text{theo}}$ widens with $\nu$ (discrete approximation less accurate)

**Effect of ρ:**

- Both $VS^{\text{real}}$ and $VS^{\text{theo}}$ are **independent of ρ**: the integrated variance $\int_0^T \sigma^2(s)\,ds$ does not depend on the correlation between $W$ and $Z$
- The difference $VS^{\text{real}} - VS^{\text{theo}}$ remains roughly constant across ρ

**Static replication (Q12):**
$$VS^{\text{theo}} = \frac{2}{T}\left(\int_0^{S_0 e^{rT}} \frac{P(S_0,k)}{k^2}\*dk + \int_{S_0 e^{rT}}^{\infty} \frac{C(S_0,k)}{k^2}\*dk\right) - e^{-rT}K^2$$

The variance swap can be replicated **statically** by a continuum of OTM puts and calls weighted by $1/k^2$.

---

## Topics covered

| Q | Content |
|---|---------|
| Q1 | Solution of the SDE for S(t) |
| Q2 | Already-started variance swap (t₀ < 0) |
| Q3 | Why Black-Scholes is not suitable |
| Q4 | Closed-form σ(t) and VS_theo |
| Q5 | Simulation of correlated BM (W, Z) |
| Q6 | Simulation of (S(t), σ(t)) + MC pricing |
| Q7 | Antithetic variables + 90% CI comparison |
| Q8 | VS vs ν ∈ [0, 1.5] |
| Q9 | VS vs ρ ∈ ]−1, 1[ |
| Q10 | Taylor-integral decomposition (put/call) |
| Q11 | Itô derivation of integrated variance |
| Q12 | Static replication by puts and calls (1/k²) |
| Q15 | Density of S(T) vs ν |
| Q16 | Density of S(T) vs ρ (skew effect) |

---

## Run

```bash
pip install numpy matplotlib
python variance_swap_commented.py
```

## Dependencies

`numpy` · `matplotlib`
