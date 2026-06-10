"""Graphe de stabilité des paramètres de la GPD.

Théorie. Par la propriété de stabilité de la GPD, si les excès au-dessus de u
suivent une GPD(ξ, σ_u), alors au-dessus de u' > u ils suivent encore une GPD de
*même* ξ, avec une échelle modifiée σ* = σ_u − ξ·u qui, elle, reste *constante*.
Méthode : on ajuste une GPD à chaque seuil et on trace ξ̂(u) et σ*(u) ; on choisit
u* à partir d'où les deux courbes deviennent (à peu près) horizontales.
"""
import numpy as np
import pandas as pd
import scipy.stats as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from utils.style import NAVY, ORANGE, GRIS, apply_layout


def compute(charges: np.ndarray, n_points: int = 100) -> pd.DataFrame:
    """Ajuste une GPD à chaque seuil (min → P99) → séries de stabilité ξ̂(u), σ*(u)."""
    c = charges[np.isfinite(charges)]
    q_lo = float(np.min(c)) if len(c) else 0.0
    q_hi = float(np.nanpercentile(c, 99))
    rows = []
    for u in np.linspace(q_lo, q_hi, n_points):     # balayage des seuils candidats
        exc = charges[charges > u] - u              # excès au-dessus de u
        if len(exc) < 5:                            # trop peu d'excès → ajustement ignoré
            continue
        try:
            xi, _, sigma = st.genpareto.fit(exc, floc=0)             # MLE de la GPD
            _, ks_pval   = st.kstest(exc, "genpareto", args=(xi, 0, sigma))  # qualité d'ajustement
            rows.append({"u": u, "xi": xi, "sigma": sigma,
                         "sigma_star": sigma - xi * u,   # échelle modifiée (doit être constante)
                         "ks_pval": ks_pval, "r": len(exc)})
        except Exception:                            # MLE qui ne converge pas → on saute ce seuil
            continue
    return pd.DataFrame(rows)


def plot(df: pd.DataFrame, u_selected: float) -> go.Figure:
    """Graphe de stabilité GPD : (A) ξ̂(u) vs u  |  (B) σ*(u) vs u.

    On cherche le seuil à partir duquel chaque courbe devient plate ; la couleur
    des points encode la qualité d'ajustement (p-value KS) pour aider à trancher.
    """
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=[
            "(A) Stabilité de ξ : constant au-dessus de u*",
            "(B) Stabilité de σ* = σ − ξu : constant au-dessus de u*",
        ],
        horizontal_spacing=0.14,
    )

    # (A) ξ̂(u) ; couleur = p-value KS (vert/jaune = bon ajustement).
    fig.add_trace(go.Scatter(
        x=df["u"] / 1e3, y=df["xi"],
        mode="lines+markers",
        line=dict(color=NAVY, width=2.5),
        marker=dict(size=8, color=df["ks_pval"], colorscale="Viridis",
                    showscale=True, cmin=0, cmax=1,
                    colorbar=dict(title="KS p-val", x=0.44, len=0.85, thickness=12)),
        name="ξ̂(u)",
        hovertemplate="u = %{x:.0f} k€<br>ξ̂ = %{y:.4f}<br>KS p = %{marker.color:.3f}<extra></extra>",
    ), row=1, col=1)

    fig.add_hline(y=0, line_dash="dot", line_color=GRIS, line_width=1, row=1, col=1)
    fig.add_vline(x=u_selected / 1e3, line_dash="dash",
                  line_color=ORANGE, line_width=2,
                  annotation_text=f"u* = {u_selected/1e3:.0f} k€",
                  annotation_position="top right",
                  annotation_font_color=ORANGE, row=1, col=1)

    # (B) σ*(u) = σ − ξu ; couleur = nombre d'excès r (diminue quand u monte).
    # Échelle Viridis (jamais blanche) + contour des marqueurs → tous les points
    # restent visibles, même ceux à faible r (sinon ils disparaissaient en blanc).
    fig.add_trace(go.Scatter(
        x=df["u"] / 1e3, y=df["sigma_star"] / 1e3,
        mode="lines+markers",
        line=dict(color=ORANGE, width=2.5),
        marker=dict(size=8, color=df["r"], colorscale="Viridis",
                    showscale=True, cmin=df["r"].min(), cmax=df["r"].max(),
                    line=dict(width=0.8, color="#3A4252"),
                    colorbar=dict(title="r (excès)", x=1.01, len=0.85, thickness=12)),
        name="σ*(u)",
        hovertemplate="u = %{x:.0f} k€<br>σ* = %{y:.1f} k€<br>r = %{marker.color:.0f} excès<extra></extra>",
    ), row=1, col=2)

    fig.add_hline(y=0, line_dash="dot", line_color=GRIS, line_width=1, row=1, col=2)
    fig.add_vline(x=u_selected / 1e3, line_dash="dash",
                  line_color=ORANGE, line_width=2,
                  annotation_text=f"u* = {u_selected/1e3:.0f} k€",
                  annotation_position="top right",
                  annotation_font_color=ORANGE, row=1, col=2)

    fig.update_xaxes(title_text="Seuil u (k€)")
    fig.update_yaxes(title_text="ξ̂(u)  :  indice de forme GPD", row=1, col=1)
    fig.update_yaxes(title_text="σ*(u) = σ − ξu  (k€)",         row=1, col=2)
    apply_layout(fig, margin_r=80)
    return fig
