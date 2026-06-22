"""Estimateur de Hill : indice de queue d'une distribution à queue épaisse.

Théorie. Pour une distribution de type Fréchet (queue en loi de puissance), Hill
estime l'indice de queue à partir des k plus grandes valeurs :
    ξ̂_k = (1/k) Σ_{i=1}^{k} ln X_(n−i+1) − ln X_(n−k)
où X_(1) ≤ … ≤ X_(n) sont les statistiques d'ordre. On trace ξ̂_k en fonction de
k (ou du seuil u = X_(n−k)) et on cherche une *zone de stabilité* (plateau) :
elle indique l'indice de queue et le seuil au-delà duquel la GPD est valable.
ATTENTION : Hill n'est valable que pour ξ > 0 (queue épaisse, Fréchet).
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from utils.style import ORANGE, GRIS, apply_layout


def compute(charges: np.ndarray) -> pd.DataFrame:
    """Estimateur de Hill ξ̂_k pour k = 2 … n−2, vectorisé en O(n).

    Le log est exigé > 0 donc on ne garde que les charges strictement positives.
    Astuce de perf : une somme cumulée rend chaque ξ̂_k calculable en O(1).
    """
    X = np.sort(charges[np.isfinite(charges) & (charges > 0)])   # stat. d'ordre croissantes (>0)
    n = len(X)
    if n < 4:                                       # pas assez de points pour estimer
        return pd.DataFrame(columns=["k", "xi", "lo", "hi", "u"])
    log_X = np.log(X)
    # Somme cumulée des log : permet d'obtenir la somme des k plus grands en O(1).
    cum     = np.concatenate([[0.0], np.cumsum(log_X)])
    k_arr   = np.arange(2, n - 1)                   # nombre d'excès considérés
    sum_top = cum[n] - cum[n - k_arr]               # Σ des k plus grands ln X
    xi  = sum_top / k_arr - log_X[n - k_arr - 1]    # ξ̂_k = moyenne − ln X_(n−k)
    se  = np.abs(xi) / np.sqrt(k_arr)               # erreur-type asymptotique ξ/√k
    u   = X[n - k_arr - 1]                          # seuil correspondant u = X_(n−k)
    return pd.DataFrame({"k": k_arr, "xi": xi,
                         "lo": xi - 1.96 * se,       # bornes de l'IC 95 %
                         "hi": xi + 1.96 * se,
                         "u": u})


def plot(df: pd.DataFrame, u_selected: float) -> go.Figure:
    """Graphe de Hill : ξ̂ en fonction du seuil u (lecture directe en euros).

    On choisit u* dans la zone où ξ̂ se stabilise (plateau horizontal).
    """
    fig = go.Figure()

    # ξ̂ en fonction du seuil u (lecture en euros) : repérer le plateau.
    fig.add_trace(go.Scatter(
        x=df["u"] / 1e3, y=df["xi"],
        mode="lines", line=dict(color=ORANGE, width=1.8),
        name="ξ̂ vs u", customdata=df["k"],
        hovertemplate="u = %{x:.0f} k€<br>ξ̂ = %{y:.4f}<br>k = %{customdata}<extra></extra>",
    ))

    fig.add_vline(x=u_selected / 1e3, line_dash="dash",
                  line_color=ORANGE, line_width=2,
                  annotation_text=f"u* = {u_selected/1e3:.0f} k€",
                  annotation_position="top right",
                  annotation_font_color=ORANGE)
    # Ligne ξ = 0 : frontière de validité (Hill suppose ξ > 0).
    fig.add_hline(y=0, line_dash="dot", line_color=GRIS, line_width=1)

    fig.update_xaxes(title_text="Seuil u (k€)")
    fig.update_yaxes(title_text="ξ̂ (Hill)")
    apply_layout(fig)
    return fig
