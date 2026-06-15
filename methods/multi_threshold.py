"""Détection de seuils multiples : PELT (ruptures) + Jenks Natural Breaks (jenkspy).

Contrairement aux méthodes EVT (MRL, Hill…) qui cherchent UN seuil de bascule
vers la queue, ces deux méthodes découpent toute la distribution en plusieurs
classes : utile pour calibrer des tranches (ex. tranches XL en réassurance).

• PELT (Pruned Exact Linear Time) : détection de ruptures dans le signal trié.
  Le nombre de ruptures n'est pas imposé : il est piloté par une *pénalité*
  (pénalité haute ⇒ moins de ruptures). Coût « rbf » = changement de
  distribution.
• Jenks Natural Breaks : partitionne en k classes en minimisant la variance
  intra-classe (coupures « naturelles » de la distribution).

Les deux peuvent tourner sur ln(charges) (ratios/queue) ou sur les charges
brutes (écarts en €) ; les seuils sont toujours renvoyés en €.
"""
import numpy as np
import jenkspy
import ruptures as rpt
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from utils.style import NAVY, ORANGE, PURPLE, VERT, GRIS, apply_layout


# ── Public API ────────────────────────────────────────────────────────────────

def compute(
    charges: np.ndarray,
    pen: float = 5.0,
    jenks_k: int = 4,
    use_log: bool = True,
) -> dict:
    """
    Detect multiple thresholds on the claim distribution.

    use_log=True : détection sur ln(charges) (queue/ratios, recommandé pour les
    sinistres graves). use_log=False : sur les charges brutes (écarts en €).
    PELT : rpt.Pelt(model="rbf") (défauts min_size=2, jump=5), seuil = dernier
    point de chaque segment. Jenks : jenkspy.jenks_breaks. Les bornes sont
    toujours renvoyées en € pour l'affichage.
    """
    c = np.sort(charges[np.isfinite(charges) & (charges > 0)])
    signal = np.log(c) if use_log else c

    # PELT, piloté par la pénalité (bkps[:-1] exclut la sentinelle). Le coût rbf
    # construit une matrice n×n : au-delà de PELT_CAP points on sous-échantillonne
    # uniformément (puis on remappe les ruptures sur les vraies valeurs en €).
    PELT_CAP = 2000
    pelt_u: list[float] = []
    if len(c) >= 2:
        idx = (np.round(np.linspace(0, len(signal) - 1, PELT_CAP)).astype(int)
               if len(signal) > PELT_CAP else np.arange(len(signal)))
        sig = signal[idx]
        bkps = rpt.Pelt(model="rbf").fit(sig.reshape(-1, 1)).predict(pen=pen)
        pelt_u = [float(c[idx[bp - 1]]) for bp in bkps[:-1]]

    # Jenks ; bornes intérieures reconverties en € (exp si échelle log).
    # jenkspy exige n_classes <= nombre de valeurs uniques → on borne k_eff.
    jenks_vals: list[float] = []
    k_eff = min(jenks_k, len(np.unique(signal)))
    if k_eff >= 2:
        breaks = jenkspy.jenks_breaks(signal.tolist(), n_classes=k_eff)
        inner = breaks[1:-1]
        jenks_vals = [float(np.exp(b)) for b in inner] if use_log else [float(b) for b in inner]

    return dict(
        charges=c,
        pelt_u=pelt_u,
        jenks_vals=jenks_vals,
        use_log=use_log,
    )


def plot(result: dict, u_selected: float) -> go.Figure:
    """
    Two-panel figure (both on the claims distribution, breaks detected on log) :
      (A) PELT : ruptures (pilotées par la pénalité)
      (B) Jenks : natural breaks
    """
    c          = result["charges"]
    pelt_u     = result["pelt_u"]
    jenks_vals = result["jenks_vals"]
    _scale     = "ln(charges)" if result.get("use_log", True) else "charges brutes"

    ALT_COLORS = [ORANGE, PURPLE, VERT, "#C53B5A"]

    # CDF empirique (fonction de répartition) : proportion cumulée de sinistres ≤ x.
    ecdf = np.arange(1, len(c) + 1) / len(c) * 100 if len(c) else np.array([])

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=[
            f"(A) PELT : Ruptures sur {_scale}",
            f"(B) Jenks Natural Breaks : sur {_scale}",
        ],
        horizontal_spacing=0.12,
        specs=[[{"secondary_y": True}, {"secondary_y": True}]],
    )

    for col_i, (breaks, label) in [(1, (pelt_u, "PELT")), (2, (jenks_vals, "Jenks"))]:
        fig.add_trace(go.Histogram(
            x=c / 1e3, nbinsx=50,
            marker_color=NAVY, opacity=0.65,
            name="Distribution", showlegend=(col_i == 1),
            hovertemplate="[%{x:.0f} k€]  Count : %{y}<extra></extra>",
        ), row=1, col=col_i, secondary_y=False)

        fig.add_trace(go.Scatter(
            x=c / 1e3, y=ecdf,
            mode="lines", line=dict(color="#0B1B36", width=2),
            name="Fréquence cumulée", showlegend=(col_i == 1),
            hovertemplate="[%{x:.0f} k€]  CDF : %{y:.1f} %<extra></extra>",
        ), row=1, col=col_i, secondary_y=True)

        for i, bv in enumerate(breaks):
            color = ALT_COLORS[i % len(ALT_COLORS)]
            pct   = float(np.mean(c <= bv)) * 100
            pos   = "top left" if i % 2 == 0 else "top right"
            fig.add_vline(
                x=bv / 1e3, line_dash="dash", line_color=color, line_width=2,
                annotation_text=f"{label} {i+1}<br>{bv/1e3:.0f} k€  P{pct:.0f}",
                annotation_position=pos, annotation_font_color=color,
                row=1, col=col_i,
            )

        fig.add_vline(
            x=u_selected / 1e3, line_dash="dot", line_color=GRIS, line_width=1.5,
            annotation_text=f"u* = {u_selected/1e3:.0f} k€",
            annotation_position="bottom right", annotation_font_color=GRIS,
            row=1, col=col_i,
        )

    fig.update_xaxes(title_text="Charge (k€)")
    fig.update_yaxes(title_text="Nombre de sinistres", secondary_y=False)
    fig.update_yaxes(title_text="Fréquence cumulée (%)", range=[0, 100],
                     showgrid=False, secondary_y=True)
    apply_layout(fig, height=440, margin_t=50)
    return fig
