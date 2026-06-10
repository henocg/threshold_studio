"""Outils d'ajustement de la GPD, partagés par toutes les méthodes.

La loi de Pareto généralisée (GPD) modélise les excès Y = X − u au-dessus d'un
seuil u (théorème de Pickands-Balkema-de Haan). Elle a deux paramètres : ξ
(indice de queue : ξ > 0 = queue épaisse, ξ ≈ 0 = exponentielle, ξ < 0 = bornée)
et σ (échelle). On fixe ici la localisation à 0 (loc=0) car les excès partent
de 0 par construction.
"""
import numpy as np
import scipy.stats as st


def fit_gpd(excesses: np.ndarray):
    """Ajuste une GPD aux excès par maximum de vraisemblance (loc = 0 fixé).

    Renvoie ξ, σ, la stat. et la p-value de Kolmogorov-Smirnov (qualité de
    l'ajustement), la log-vraisemblance et les critères AIC/BIC (comparaison de
    modèles : plus petit = meilleur). Une p-value KS élevée = bon ajustement.
    """
    if len(excesses) < 5:                          # trop peu d'excès → ajustement non fiable
        return dict(xi=np.nan, sigma=np.nan, ks_stat=np.nan,
                    ks_pval=np.nan, ll=np.nan, aic=np.nan, bic=np.nan)
    xi, _, sigma = st.genpareto.fit(excesses, floc=0)            # MLE, localisation forcée à 0
    ks_stat, ks_pval = st.kstest(excesses, "genpareto", args=(xi, 0, sigma))  # test d'adéquation
    ll = float(np.sum(st.genpareto.logpdf(excesses, c=xi, loc=0, scale=sigma)))  # log-vraisemblance
    r = len(excesses)
    aic = 4 - 2 * ll                               # AIC = 2k − 2ll, ici k = 2 paramètres
    bic = 2 * np.log(r) - 2 * ll                   # BIC = k·ln(r) − 2ll, ici k = 2
    return dict(xi=xi, sigma=sigma, ks_stat=ks_stat, ks_pval=ks_pval,
                ll=ll, aic=aic, bic=bic)


def return_levels(u: float, xi: float, sigma: float, n: int, r: int,
                  periods=(10, 25, 50, 100, 200, 500)):
    """Niveaux de retour z(T) à partir d'un ajustement GPD.

    z(T) est le montant dépassé en moyenne une fois toutes les T périodes. n est
    le nombre total d'observations ; prob_u = r/n est la probabilité de dépasser
    le seuil. Formule GPD inversée (cas ξ≈0 traité à part pour éviter la division
    par zéro).
    """
    if np.isnan(xi) or np.isnan(sigma):
        return []
    prob_u = r / n                                  # fréquence empirique de dépassement de u
    rows = []
    for T in periods:
        if abs(xi) > 1e-4:                          # cas général ξ ≠ 0
            z = u + (sigma / xi) * ((T * prob_u) ** xi - 1)
        else:                                       # cas limite ξ → 0 (exponentielle)
            z = u + sigma * np.log(T * prob_u)
        rows.append({"T": T, "z": max(u, z)})       # le niveau ne peut pas être sous le seuil
    return rows


def ad_test_genpareto(excesses: np.ndarray, xi: float, sigma: float) -> float:
    """p-value approchée du test d'Anderson-Darling pour l'ajustement GPD.

    Anderson-Darling pèse davantage la queue que Kolmogorov-Smirnov, donc plus
    sensible aux écarts sur les extrêmes. La statistique A² est mappée vers une
    p-value approximative par paliers (table d'interpolation grossière).
    """
    try:
        n = len(excesses)
        u = np.sort(st.genpareto.cdf(excesses, c=xi, loc=0, scale=sigma))  # transformée en uniformes
        u = np.clip(u, 1e-9, 1 - 1e-9)              # évite log(0) aux bornes
        i = np.arange(1, n + 1)
        A2 = -n - np.sum((2 * i - 1) * (np.log(u) + np.log(1 - u[::-1]))) / n  # statistique A²
        # Interpolation grossière A² → p-value (paliers)
        if A2 < 0.2:
            return 0.99
        elif A2 < 0.34:
            return 0.75
        elif A2 < 0.46:
            return 0.5
        elif A2 < 0.6:
            return 0.25
        elif A2 < 0.75:
            return 0.15
        elif A2 < 1.0:
            return 0.10
        elif A2 < 1.5:
            return 0.05
        else:
            return 0.01
    except Exception:
        return np.nan
