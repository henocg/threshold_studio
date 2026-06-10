"""Mean Residual Life (MRL) : méthode graphique de choix de seuil.

Théorie. La *mean residual life* (ou mean excess function) est définie par
    e(u) = E[X − u | X > u]
c.-à-d. la moyenne des dépassements au-dessus du seuil u. Propriété clé : pour
une loi de Pareto généralisée (GPD) d'indice ξ < 1, e(u) est une fonction
*linéaire* de u. La stratégie de lecture consiste donc à repérer le plus petit
seuil u* à partir duquel la courbe devient approximativement linéaire : au-delà,
les excès se modélisent par une GPD, ce qui sépare les sinistres « graves » des
attritionnels. Une pente croissante signale de plus une queue épaisse (poids non
négligeable des extrêmes).
"""
import numpy as np
import pandas as pd
import scipy.stats as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from utils.style import NAVY, ORANGE, GRIS, apply_layout


def compute(charges: np.ndarray, n_points: int = 200) -> pd.DataFrame:
    """Calcule la courbe MRL e(u) et son IC 95 % sur n_points seuils (min → P99).

    Renvoie un DataFrame (u, mrl, lo, hi, r) où r = nombre d'excès au seuil u ;
    r sert de garde-fou : un seuil n'est lisible que s'il reste assez d'excès.
    """
    c = charges[np.isfinite(charges)]                 # on écarte NaN / inf
    if len(c) == 0:
        return pd.DataFrame(columns=["u", "mrl", "lo", "hi", "r"])

    # Grille de seuils du minimum au 99e percentile (au-delà, trop peu d'excès
    # pour que la moyenne soit stable).
    thresholds = np.linspace(float(np.min(c)), float(np.percentile(c, 99)), n_points)
    rows = []
    for u in thresholds:
        exc = c[c > u] - u                            # excès X − u au seuil courant
        r = len(exc)                                  # nombre de dépassements
        if r >= 3:                                    # ≥ 3 excès → moyenne + écart-type définis
            m, se = exc.mean(), exc.std(ddof=1) / np.sqrt(r)   # e(u) et son erreur-type
            rows.append((u, m, m - 1.96 * se, m + 1.96 * se, r))  # IC 95 % (approx. normale)
        else:
            rows.append((u, np.nan, np.nan, np.nan, r))    # trop peu d'excès → NaN
    return pd.DataFrame(rows, columns=["u", "mrl", "lo", "hi", "r"])


def plot(df: pd.DataFrame, u_selected: float) -> go.Figure:
    """Graphe MRL en deux panneaux : (A) e(u) + IC 95 %  |  (B) nombre d'excès r_u.

    Le panneau (B) accompagne la lecture : plus u monte, moins il reste d'excès,
    donc plus la fin de la courbe (A) est incertaine.
    """
    valid = df.dropna(subset=["mrl"])             # on ne trace que les seuils exploitables

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=["(A) Mean Residual Life Plot",
                        "(B) Nombre d'excès  r_u"],
        horizontal_spacing=0.12,
    )

    # (A) Bande de confiance 95 % : polygone fermé hi (aller) puis lo inversé (retour).
    fig.add_trace(go.Scatter(
        x=np.concatenate([valid["u"] / 1e3, valid["u"].values[::-1] / 1e3]),
        y=np.concatenate([valid["hi"] / 1e3, valid["lo"].values[::-1] / 1e3]),
        fill="toself", fillcolor="rgba(30,58,138,0.12)",
        line=dict(color="rgba(0,0,0,0)"),
        hoverinfo="skip", name="IC 95%",
    ), row=1, col=1)

    # (A) Courbe MRL empirique e(u) : chercher où elle devient linéaire.
    fig.add_trace(go.Scatter(
        x=valid["u"] / 1e3, y=valid["mrl"] / 1e3,
        mode="lines+markers",
        line=dict(color=NAVY, width=2.5), marker=dict(size=5, color=NAVY),
        name="MRL empirique",
        hovertemplate="u = %{x:.0f} k€<br>e(u) = %{y:.0f} k€<extra></extra>",
    ), row=1, col=1)

    # Repère vertical sur le seuil u* choisi par l'utilisateur.
    fig.add_vline(x=u_selected / 1e3, line_dash="dash",
                  line_color=ORANGE, line_width=2,
                  annotation_text=f"u* = {u_selected/1e3:.0f} k€",
                  annotation_position="top right",
                  annotation_font_color=ORANGE, row=1, col=1)

    # (B) Nombre de dépassements r_u en fonction du seuil (décroissant).
    fig.add_trace(go.Scatter(
        x=df["u"] / 1e3, y=df["r"],
        mode="lines+markers",
        line=dict(color=ORANGE, width=2), marker=dict(size=5, color=ORANGE),
        name="r_u", fill="tozeroy", fillcolor="rgba(227,97,32,0.08)",
        hovertemplate="u = %{x:.0f} k€<br>r = %{y}<extra></extra>",
    ), row=1, col=2)

    # Seuil pratique r = 20 : en-dessous, l'estimation GPD devient peu fiable.
    fig.add_hline(y=20, line_dash="dot", line_color=GRIS, line_width=1.5,
                  annotation_text="r = 20 (min)", annotation_position="top left",
                  row=1, col=2)
    fig.add_vline(x=u_selected / 1e3, line_dash="dash",
                  line_color=ORANGE, line_width=2,
                  annotation_text="u*", annotation_position="top right",
                  annotation_font_color=ORANGE, row=1, col=2)

    fig.update_xaxes(title_text="Seuil u (k€)")
    fig.update_yaxes(title_text="E[X − u | X > u]  (k€)", row=1, col=1)
    fig.update_yaxes(title_text="Nombre de dépassements r",  row=1, col=2)
    apply_layout(fig)
    fig.update_xaxes(zerolinecolor="#CFD3DF", row=1, col=1)
    fig.update_yaxes(zerolinecolor="#CFD3DF", row=1, col=1)
    return fig


def qq_plot(excesses: np.ndarray, xi: float, sigma: float) -> go.Figure:
    """QQ plot des excès au-dessus de u* contre la GPD ajustée (loc = 0).

    Principe : on confronte les quantiles empiriques des excès aux quantiles
    théoriques de la GPD estimée. Des points alignés sur la diagonale y = x
    valident l'ajustement à la loi de Pareto généralisée (donc le choix de u*).
    """
    exc = np.sort(excesses[np.isfinite(excesses)])    # excès triés croissants = quantiles empiriques
    r   = len(exc)

    fig = go.Figure()
    # Garde-fou : sans assez d'excès ou sans ajustement valide, pas de QQ plot.
    if r < 5 or np.isnan(xi) or np.isnan(sigma) or sigma <= 0:
        fig.add_annotation(
            text="Pas assez d'excès pour le QQ plot (r &lt; 5).",
            x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False,
            font=dict(color=GRIS),
        )
        apply_layout(fig, height=380)
        return fig

    probs = (np.arange(1, r + 1) - 0.5) / r           # positions de tracé (i−0.5)/r
    theo  = st.genpareto.ppf(probs, c=xi, loc=0, scale=sigma)  # quantiles théoriques GPD
    lim   = float(max(theo[-1], exc[-1])) / 1e3        # borne commune pour la diagonale

    fig.add_trace(go.Scatter(
        x=[0, lim], y=[0, lim],
        mode="lines", line=dict(color=GRIS, width=1.5, dash="dash"),
        name="y = x  (ajustement parfait)", hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=theo / 1e3, y=exc / 1e3,
        mode="markers", marker=dict(size=6, color=NAVY, opacity=0.8),
        name="Excès observés",
        hovertemplate="Théorique = %{x:.0f} k€<br>Empirique = %{y:.0f} k€<extra></extra>",
    ))

    fig.update_xaxes(title_text="Quantiles théoriques GPD (k€)")
    fig.update_yaxes(title_text="Quantiles empiriques des excès (k€)")
    apply_layout(fig, height=380)
    return fig
