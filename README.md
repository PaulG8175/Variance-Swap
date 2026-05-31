# 📊 PRB222 — Variance Swap & Réplication par Options Européennes

Projet numérique portant sur la valorisation d'un **Variance Swap** dans un modèle à volatilité stochastique, et sa réplication statique par un continuum d'options vanilles.

---

## 🎯 Objectif

Calculer et comparer :
- **VS_théo** : variance swap théorique (formule fermée)
- **VS_réel** : variance swap réel (variance réalisée discrète, Monte Carlo)

Via deux estimateurs MC (standard et antithétique), puis étudier la réplication par options européennes.

---

## 📂 Structure

```
├── variance_swap_commented.py   # Script principal commenté
```

---

## ⚙️ Modèle

Modèle à volatilité stochastique (type log-normal sur σ) :

```
dS(t) = S(t)(r dt + σ(t) dW(t))
dσ(t) = σ(t) ν dZ(t)
d<W,Z>_t = ρ dt
```

La formule fermée de σ(t) est :

```
σ(t) = σ₀ · exp(ν·Z(t) − 0.5·ν²·t)
```

Et le prix théorique :

```
VS_théo = e^{-rT} · (σ₀² · (exp(ν²T) − 1) / ν² − K²)
```

---

## ⚙️ Paramètres

| Paramètre | Valeur |
|-----------|--------|
| `S0` | 1 |
| `K` | 0.35 |
| `T` | 3 mois |
| `r` | 0.01 |
| `σ₀` | 0.35 |
| `ν` | 0.8 |
| `ρ` | -0.5 |
| `N` (pas) | 63 (dt = 1/252) |

> Les simulations utilisent uniquement des lois **Uniformes** via la méthode de Box-Muller.

---

## 📋 Questions traitées

| Q | Contenu |
|---|---------|
| Q1 | Solution de l'EDS de S(t) |
| Q2 | VS déjà commencé (t0 < 0) |
| Q3 | Pourquoi B&S n'est pas pertinent pour ce produit |
| Q4 | Calcul de σ(t) et formule fermée de VS_théo |
| Q5 | Simulation des BM corrélés (W, Z) |
| Q6 | Simulation de (S(t), σ(t)) + MC classique |
| Q7 | Variables antithétiques + comparaison des IC à 90% |
| Q8 | VS en fonction de ν ∈ [0, 1.5] |
| Q9 | VS en fonction de ρ ∈ ]−1, 1[ |
| Q10 | Formule de Taylor intégrale (décomposition put/call) |
| Q11 | Expression de la variance intégrée via Itô |
| Q12 | Réplication statique du VS par puts et calls (1/k²) |
| Q15 | Densité de S(T) en fonction de ν |
| Q16 | Densité de S(T) en fonction de ρ (skew) |

---

## 🚀 Lancement

```bash
pip install numpy matplotlib
python variance_swap_commented.py
```

---

## 📦 Dépendances

- `numpy`
- `matplotlib`
