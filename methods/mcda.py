"""MCDA : analyse multicritère avec pondération par entropie de Shannon.

Idée. Plutôt que de juger un seuil sur un seul critère, on évalue chaque seuil
candidat sur plusieurs mesures de qualité d'ajustement GPD (KS, χ², RMSE de la
CDF : toutes « plus petit = mieux »), puis on les agrège en un score global. Les
poids ne sont PAS arbitraires : ils viennent de l'entropie de Shannon de chaque
critère : un critère qui discrimine bien entre candidats (faible entropie) reçoit
un poids plus fort. Le seuil optimal est celui de score minimal.
"""
import numpy as np
import scipy.stats as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from utils.style import NAVY, PURPLE, ORANGE, apply_layout


def compute(charges: np.ndarray, n_candidates: int = 100,
            n_bins_chi2: int = 5) -> dict:
    """Exécute la MCDA ; renvoie scores, poids et matrice de décision."""
    r_min = 10                                       # minimum d'excès pour qu'un candidat soit retenu
    c = charges[np.isfinite(charges)]
    q_lo = float(np.min(c)) if len(c) else 0.0
    q_hi = float(np.nanpercentile(c, 99))
    # Seuils candidats régulièrement espacés, gardés s'ils laissent ≥ r_min excès.
    candidates = np.array([
        u for u in np.linspace(q_lo, q_hi, n_candidates)
        if np.sum(charges > u) >= r_min
    ])
    if len(candidates) == 0:
        return {}

    # ── Matrice de décision R : une ligne par candidat, 3 critères de qualité ──
    rows_R, u_valid = [], []
    for u in candidates:
        exc = charges[charges > u] - u
        r   = len(exc)
        try:
            xi, _, sigma = st.genpareto.fit(exc, floc=0)         # ajustement GPD du candidat
            if sigma <= 0 or np.isnan(xi):
                continue
            # Critère 1 : Kolmogorov-Smirnov (écart max CDF empirique/théorique).
            ks, _ = st.kstest(exc, "genpareto", args=(xi, 0, sigma))
            # Critère 2 : χ² sur des classes équiprobables sous la GPD ajustée.
            q_bounds = st.genpareto.ppf(
                np.linspace(0, 1, n_bins_chi2 + 1)[1:-1], c=xi, loc=0, scale=sigma)
            obs, _ = np.histogram(exc, bins=np.concatenate([[0], q_bounds, [np.inf]]))
            exp    = np.full(n_bins_chi2, r / n_bins_chi2)        # effectifs attendus (équiprobables)
            chi2   = np.sum((obs - exp) ** 2 / np.maximum(exp, 1e-10))
            # Critère 3 : RMSE entre CDF empirique et CDF théorique.
            exc_s  = np.sort(exc)
            cdf_e  = (np.arange(1, r + 1) - 0.5) / r              # CDF empirique (positions de tracé)
            cdf_t  = st.genpareto.cdf(exc_s, c=xi, loc=0, scale=sigma)
            rmse   = np.sqrt(np.mean((cdf_e - cdf_t) ** 2))
            rows_R.append([ks, chi2, rmse])
            u_valid.append(u)
        except Exception:                            # ajustement impossible → candidat ignoré
            continue

    if not rows_R:
        return {}

    # ── Pondération par entropie de Shannon ───────────────────────────────────
    R = np.array(rows_R)
    m = len(u_valid)
    P = R / np.maximum(R.sum(axis=0), 1e-12)         # normalise chaque colonne (critère) en proba
    with np.errstate(divide="ignore", invalid="ignore"):
        log_P = np.where(P > 0, np.log(P), 0.0)      # 0·log0 = 0 par convention
    E = -(1.0 / np.log(m)) * np.sum(P * log_P, axis=0)   # entropie de chaque critère ∈ [0,1]
    W = (1 - E) / (1 - E).sum()                      # poids : + un critère discrimine, + il pèse
    S = R @ W                                        # score global pondéré (plus petit = mieux)

    best_idx = int(np.argmin(S))                     # seuil optimal = score minimal
    return dict(
        u_candidates=np.array(u_valid),
        R=R, P=P, E=E, W=W, S=S,
        criteria=["KS stat", "χ² stat", "RMSE CDF"],
        best_idx=best_idx,
        u_star=float(u_valid[best_idx]),
        S_star=float(S[best_idx]),
    )


def plot(result: dict, u_selected: float) -> go.Figure:
    """Three-panel MCDA plot: (A) score vs rang  |  (B) score vs u  |  (C) poids."""
    if not result:
        fig = go.Figure()
        fig.add_annotation(text="Pas assez de données", x=0.5, y=0.5, showarrow=False)
        return fig

    u_cands  = result["u_candidates"]
    S        = result["S"]
    W        = result["W"]
    E        = result["E"]
    criteria = result["criteria"]
    m        = len(u_cands)
    sel_idx  = int(np.argmin(np.abs(u_cands - u_selected)))

    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=[
            "(A) Score S_j vs rang du candidat",
            "(B) Score S_j vs seuil u",
            "(C) Poids des critères",
        ],
        horizontal_spacing=0.10,
    )

    bar_colors = [ORANGE if i == sel_idx else NAVY for i in range(m)]
    fig.add_trace(go.Bar(
        x=list(range(m)), y=S,
        marker_color=bar_colors, opacity=0.80,
        name="Score S_j", customdata=u_cands / 1e3,
        hovertemplate="Rang %{x}<br>u = %{customdata:.0f} k€<br>S_j = %{y:.4f}<extra></extra>",
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=list(range(m)), y=S,
        mode="lines", line=dict(color=NAVY, width=1.5), showlegend=False,
    ), row=1, col=1)
    fig.add_vline(x=sel_idx, line_dash="dash", line_color=ORANGE, line_width=2,
                  annotation_text=f"u*={u_selected/1e3:.0f}k€",
                  annotation_position="top right",
                  annotation_font_color=ORANGE, row=1, col=1)

    fig.add_trace(go.Scatter(
        x=u_cands / 1e3, y=S,
        mode="lines+markers",
        line=dict(color=NAVY, width=2),
        marker=dict(size=6, color=S, colorscale="Blues_r", showscale=False),
        name="S_j vs u",
        hovertemplate="u = %{x:.0f} k€<br>S_j = %{y:.4f}<extra></extra>",
    ), row=1, col=2)
    fig.add_vline(x=u_selected / 1e3, line_dash="dash", line_color=ORANGE, line_width=2,
                  annotation_text=f"u*={u_selected/1e3:.0f}k€",
                  annotation_position="top right",
                  annotation_font_color=ORANGE, row=1, col=2)

    fig.add_trace(go.Bar(
        x=W, y=criteria, orientation="h",
        marker=dict(color=[NAVY, PURPLE, ORANGE][:len(W)], opacity=0.85),
        text=[f"{w:.3f}" for w in W], textposition="outside",
        customdata=E, name="Poids w_j",
        hovertemplate="%{y}<br>w = %{x:.4f}<br>E = %{customdata:.4f}<extra></extra>",
    ), row=1, col=3)

    fig.update_xaxes(title_text="Rang du candidat j", row=1, col=1)
    fig.update_xaxes(title_text="Seuil u (k€)",       row=1, col=2)
    fig.update_xaxes(title_text="Poids w_j",           row=1, col=3)
    fig.update_yaxes(title_text="Score global S_j",    row=1, col=1)
    fig.update_yaxes(title_text="Score global S_j",    row=1, col=2)
    apply_layout(fig, margin_t=50, margin_r=60)
    fig.update_layout(title=dict(
        text=f"<b>MCDA · u* = {u_selected/1e3:.0f} k€  |  Score = {S[sel_idx]:.4f}</b>",
        font=dict(size=12),
    ))
    return fig
