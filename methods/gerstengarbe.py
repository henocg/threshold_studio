"""Test de Gerstengarbe : détection de rupture par test séquentiel (façon Mann-Kendall).

Principe. On parcourt les données triées et on construit deux statistiques de
type Mann-Kendall : une série progressive normalisée H_k et sa version
rétrograde H̃_k. Leur point d'intersection (changement de signe de −H_k − H̃_k)
marque une *rupture* dans la structure des données : interprétée ici comme le
seuil séparant le régime « normal » du régime « extrême ». Avantage : valable
quel que soit le signe de ξ (contrairement à Hill).
"""
import numpy as np
import plotly.graph_objects as go

from utils.style import NAVY, ORANGE, GRIS, apply_layout


def compute(charges: np.ndarray) -> dict:
    """Construit les séries H_k et H̃_k et détecte les croisements (tous les points).

    Accumulation incrémentale en O(n²). k* = premier croisement (changement de
    signe), converti en seuil u* via la statistique d'ordre correspondante.
    """
    X = np.sort(charges[np.isfinite(charges)])      # données triées croissantes
    n_full = len(X)
    delta       = np.diff(X)                         # incréments successifs (croissants)
    delta_tilde = np.diff(X[::-1])                   # incréments de la série inversée (rétrograde)

    H  = np.zeros(n_full)
    Ht = np.zeros(n_full)
    s_H = s_Ht = 0
    for k in range(2, n_full):                       # accumulation séquentielle
        # Comptage de Mann-Kendall : combien d'incréments antérieurs sont < l'incrément courant.
        s_H  += int(np.sum(delta[:k - 1]        < delta[k - 1]))
        s_Ht += int(np.sum(delta_tilde[:k - 1] < delta_tilde[k - 1]))
        a_k = k * (k - 1) / 4.0                       # espérance sous H0 (pas de tendance)
        b_k = np.sqrt(k * (k - 1) * (2 * k + 5) / 72.0)  # écart-type sous H0
        if b_k == 0:
            continue
        H[k]  = (s_H  - a_k) / b_k                    # statistique normalisée (progressive)
        Ht[k] = (s_Ht - a_k) / b_k                    # statistique normalisée (rétrograde)

    k_arr        = np.arange(n_full)
    # Croisement = changement de signe de (−H_k − H̃_k) : rupture détectée.
    sign_changes = np.where(np.diff(np.sign(-H[2:] - Ht[2:])))[0]

    crossings = [
        {"k": int(k_arr[2:][c]),
         "u": float(X[max(0, n_full - int(k_arr[2:][c]) - 1)])}   # k ↦ seuil X_(n−k)
        for c in sign_changes
    ]

    if crossings:
        k_star = crossings[0]["k"]                    # premier croisement retenu comme rupture
        u_star = crossings[0]["u"]
    else:
        k_star = n_full // 4                          # repli si aucun croisement détecté
        u_star = float(X[max(0, n_full - k_star - 1)])

    return {
        "k_arr":     k_arr[2:],
        "H":        -H[2:],
        "Ht":        Ht[2:],
        "X":         X,
        "crossings": crossings,
        "k_star":    k_star,
        "u_star":    u_star,
        "n_full":    n_full,
    }


def plot(result: dict) -> go.Figure:
    """Graphe de Gerstengarbe : vue globale des courbes −H_k et H̃_k.

    On visualise leur intersection (= rupture). Le zoom sur les petits k se fait
    directement à la souris grâce à l'interactivité de Plotly.
    """
    k_arr  = result["k_arr"]
    H      = result["H"]
    Ht     = result["Ht"]
    X      = result["X"]
    n_full = result["n_full"]

    # Seuil associé à chaque k : u = X_(n−k) → permet de lire le seuil en survol.
    idx = np.clip(n_full - k_arr - 1, 0, n_full - 1)
    u_k = X[idx] / 1e3                                # en k€

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=k_arr, y=H, customdata=u_k,
        mode="lines", line=dict(color=NAVY, width=2),
        name="−H_k",
        hovertemplate="k = %{x}<br>seuil u = %{customdata:.0f} k€<br>−H_k = %{y:.2f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=k_arr, y=Ht, customdata=u_k,
        mode="lines", line=dict(color=ORANGE, width=2),
        name="H̃_k",
        hovertemplate="k = %{x}<br>seuil u = %{customdata:.0f} k€<br>H̃_k = %{y:.2f}<extra></extra>",
    ))
    fig.add_hline(y=0, line_dash="dot", line_color=GRIS, line_width=1)

    fig.update_xaxes(title_text="k")
    fig.update_yaxes(title_text="Statistiques de test  (−H_k , H̃_k)")
    apply_layout(fig)
    return fig
