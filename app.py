"""Threshold Studio : LSN Ré Walbaum.  Lancement : streamlit run app.py"""
from collections import defaultdict

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils.gpd_utils import fit_gpd
from methods import mrl, hill, gerstengarbe, gpd_stability, mcda, multi_threshold

# ─── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Threshold Studio : LSN Ré Walbaum",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Visual design tokens ─────────────────────────────────────────────────────
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter+Tight:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter Tight', ui-sans-serif, system-ui, -apple-system, sans-serif;
}

[data-testid="stHeader"] {
    background: #fff !important;
    border-bottom: 1px solid #E4E6EE;
}
[data-testid="stHeader"]::after {
    content: '';
    display: block;
    height: 3px;
    background: linear-gradient(95deg,#1E3A8A 0%,#6B2DA6 45%,#C53B5A 78%,#E36120 100%);
}

[data-testid="stSidebar"] {
    background: #FAFBFD !important;
    border-right: 1px solid #E4E6EE !important;
}
[data-testid="stSidebar"] [data-testid="stVerticalBlock"] {
    gap: 0.5rem !important;
}

[data-testid="stMetric"] {
    background: #fff;
    border: 1px solid #E4E6EE;
    border-radius: 10px;
    padding: 14px 16px;
    box-shadow: 0 1px 2px rgba(11,27,54,.04);
}
[data-testid="stMetricValue"] {
    font-size: 22px !important;
    font-weight: 600 !important;
    letter-spacing: -.02em !important;
    color: #0B1B36 !important;
}
[data-testid="stMetricLabel"] {
    font-size: 10.5px !important;
    font-weight: 600 !important;
    letter-spacing: .08em !important;
    text-transform: uppercase !important;
    color: #939BAE !important;
}

[data-testid="stTabs"] [role="tablist"] {
    background: #fff;
    border: 1px solid #E4E6EE;
    border-radius: 10px;
    padding: 4px;
    gap: 2px;
}
[data-testid="stTabs"] [role="tab"] {
    border-radius: 7px !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    color: #5C657A !important;
    padding: 8px 14px !important;
}
[data-testid="stTabs"] [role="tab"][aria-selected="true"] {
    background: linear-gradient(180deg,#FBFCFE,#F4F6FB) !important;
    border: 1px solid #E4E6EE !important;
    color: #0B1B36 !important;
}

.card-title {
    font-size: 14px; font-weight: 600; color: #0B1B36;
    letter-spacing: -.005em; margin-bottom: 4px;
}
.card-hint {
    font-size: 11.5px; color: #939BAE; line-height: 1.5;
}

.thresh-hero {
    background: linear-gradient(180deg, #0E2F5C 0%, #061B36 100%);
    color: #fff;
    border-radius: 10px;
    padding: 14px 14px 12px;
    margin-bottom: 10px;
    position: relative;
    overflow: hidden;
}
.thresh-hero::before {
    content:''; position:absolute; right:-30px; top:-60px;
    width:200px; height:200px;
    background:radial-gradient(circle,rgba(227,97,32,.40) 0%,transparent 60%);
}
.thresh-hero h5 {
    font-size:10.5px; font-weight:600; letter-spacing:.08em;
    text-transform:uppercase; color:rgba(255,255,255,.55); margin:0 0 8px;
}
.thresh-hero .val {
    font-size:34px; font-weight:600; letter-spacing:-.02em; line-height:1.05;
    font-variant-numeric: tabular-nums;
}
.thresh-hero .sub {
    font-size:11px; color:rgba(255,255,255,.60); margin-top:3px;
}
.thresh-hero .sub b { color:#ECA033; }

::-webkit-scrollbar { width:8px; height:8px; }
::-webkit-scrollbar-track { background:transparent; }
::-webkit-scrollbar-thumb { background:#D9DCE5; border-radius:999px; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# ─── Helpers ─────────────────────────────────────────────────────────────────
def fmt_k(v):
    return f"{v/1e3:.0f} k€"

def fmt_M(v):
    if v >= 1e6:
        return f"{v/1e6:.2f} M€"
    return f"{v/1e3:.0f} k€"

STATUS_COLORS = {"ok": "#137A45", "warn": "#B45309", "bad": "#B22A2A"}


# ─── Sidebar : Partie 1 : chargement du fichier ───────────────────────────────
with st.sidebar:
    st.markdown("### Threshold Studio")
    st.markdown(
        '<div style="font-size:10.5px;color:#939BAE;letter-spacing:.06em;'
        'text-transform:uppercase;margin-bottom:12px;">'
        'LSN Ré Walbaum · Détection seuil grave</div>',
        unsafe_allow_html=True,
    )

    st.markdown("**01 : Source de données**")
    uploaded = st.file_uploader(
        "CSV ou Excel (colonnes numériques)",
        type=["csv", "xlsx", "xls"],
        label_visibility="collapsed",
    )

    file_loaded = False
    df_raw = None
    sheet_name = None

    if uploaded is not None:
        try:
            if uploaded.name.lower().endswith(".csv"):
                df_raw = pd.read_csv(uploaded)
                file_loaded = True
            else:
                excel_file = pd.ExcelFile(uploaded)
                sheet_names = excel_file.sheet_names

                st.markdown("**Feuille à analyser :**")
                selected_sheet = st.selectbox(
                    "Feuille",
                    options=sheet_names,
                    label_visibility="collapsed",
                    key="sheet_selector",
                )
                df_raw = pd.read_excel(uploaded, sheet_name=selected_sheet)
                sheet_name = selected_sheet
                file_loaded = True

            st.success(f"✓ {uploaded.name}  ·  {len(df_raw)} lignes")
        except Exception as e:
            st.error(f"Erreur lecture : {e}")
            df_raw = None
            file_loaded = False

    st.session_state["file_loaded"] = file_loaded


# ─── État vide : aucun fichier chargé ────────────────────────────────────────
if not file_loaded:
    st.markdown(
        """
        <div style="display:flex;flex-direction:column;align-items:center;
                    justify-content:center;min-height:62vh;text-align:center;">
            <div style="font-size:52px;margin-bottom:20px;opacity:.55;">&#128194;</div>
            <div style="font-size:20px;font-weight:600;color:#0B1B36;margin-bottom:10px;">
                Aucun fichier chargé
            </div>
            <div style="font-size:13px;color:#5C657A;max-width:420px;line-height:1.7;">
                Importez un fichier <strong>CSV</strong> ou <strong>Excel</strong> via
                le panneau latéral pour lancer l'analyse POT et la détection du seuil
                de sinistre grave.
            </div>
            <div style="margin-top:24px;display:flex;gap:10px;flex-wrap:wrap;justify-content:center;">
                <span style="background:#F4F6FB;border:1px solid #E4E6EE;border-radius:8px;
                             padding:6px 14px;font-size:12px;color:#5C657A;">.csv</span>
                <span style="background:#F4F6FB;border:1px solid #E4E6EE;border-radius:8px;
                             padding:6px 14px;font-size:12px;color:#5C657A;">.xlsx</span>
                <span style="background:#F4F6FB;border:1px solid #E4E6EE;border-radius:8px;
                             padding:6px 14px;font-size:12px;color:#5C657A;">.xls</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.stop()


def _is_qualitative(series, col_selected: str) -> bool:
    """True for columns suitable as categorical filters (not the target variable)."""
    if series.name == col_selected:
        return False
    k  = series.dtype.kind
    nu = series.nunique(dropna=True)
    if str(series.dtype) == "category":      return 1 < nu <= 100
    if k in ("O", "U", "S", "b"):            return 1 < nu <= 100
    if k in ("i", "u") and nu <= 30:         return True  # low-cardinality integer codes
    return False


# ─── Sidebar : Partie 2 : contrôles d'analyse ────────────────────────────────
with st.sidebar:
    # ── 02 Variables ──────────────────────────────────────────────────────
    st.markdown("**02 : Variables**")
    num_cols = df_raw.select_dtypes(include=np.number).columns.tolist()
    if not num_cols:
        st.error("Aucune colonne numérique.")
        st.stop()

    col_selected = st.selectbox(
        "Colonne charge", num_cols, label_visibility="collapsed"
    )

    st.divider()

    # ── 03 Filtres ────────────────────────────────────────────────────────
    st.markdown("**03 : Filtres**")
    cat_cols = [c for c in df_raw.columns if _is_qualitative(df_raw[c], col_selected)]

    df_filtered = df_raw.copy()
    applied_filters = {}
    if cat_cols:
        for col_f in cat_cols:
            modalities = sorted(df_raw[col_f].dropna().unique().tolist(), key=str)
            sel = st.multiselect(
                col_f,
                options=modalities,
                default=modalities,
                key=f"filter_{col_f}",
            )
            if len(sel) < len(modalities):
                df_filtered = df_filtered[df_filtered[col_f].isin(sel)]
                applied_filters[col_f] = sel
    else:
        st.caption("Aucune colonne qualitative détectée.")

    st.markdown(f"*{len(df_filtered):,} lignes après filtres*")
    st.divider()

    # ── 03b Traitement de la variable quantitative ────────────────────────
    st.markdown(f"**Valeurs de** `{col_selected}`")
    excl_nan = st.checkbox("Exclure les valeurs manquantes (NaN)", value=False,
                           key="excl_nan")
    excl_neg = st.checkbox("Exclure les valeurs négatives (< 0)", value=False,
                           key="excl_neg")
    excl_zero = st.checkbox("Exclure les valeurs nulles (= 0)", value=False,
                            key="excl_zero")

    charges_raw = df_filtered[col_selected].values.astype(float)
    mask = np.ones(len(charges_raw), dtype=bool)
    if excl_nan:
        mask &= np.isfinite(charges_raw)
    if excl_neg:
        mask &= charges_raw >= 0
    if excl_zero:
        mask &= charges_raw != 0
    charges = charges_raw[mask]

    # ── 03c Intervalle de valeurs analysées ───────────────────────────────
    # Restreint les données utilisées par TOUTES les méthodes (différent de la
    # « Plage d'affichage » qui ne zoome que l'axe x des graphiques).
    st.markdown("**Intervalle de valeurs analysées**")
    range_note = None
    _cf = charges[np.isfinite(charges)]
    if len(_cf) and float(np.min(_cf)) < float(np.max(_cf)):
        _vmin, _vmax = float(np.min(_cf)), float(np.max(_cf))
        range_mode = st.radio(
            "Mode de bornes", ["€", "Percentiles"],
            horizontal=True, key="range_mode", label_visibility="collapsed",
        )
        if range_mode == "€":
            lo_e, hi_e = st.slider(
                "Bornes (€)", min_value=_vmin, max_value=_vmax,
                value=(_vmin, _vmax),
                step=max(1000.0, (_vmax - _vmin) / 100),
                format="%.0f", key="range_eur", label_visibility="collapsed",
            )
            range_active = (lo_e, hi_e) != (_vmin, _vmax)
            if range_active:
                range_note = f"{lo_e/1e3:.0f}–{hi_e/1e3:.0f} k€"
        else:
            lo_p, hi_p = st.slider(
                "Bornes (percentiles %)", min_value=0, max_value=100,
                value=(0, 100), step=1, format="%d %%",
                key="range_pct", label_visibility="collapsed",
            )
            lo_e = float(np.nanpercentile(_cf, lo_p))
            hi_e = float(np.nanpercentile(_cf, hi_p))
            range_active = (lo_p, hi_p) != (0, 100)
            if range_active:
                range_note = f"P{lo_p}–P{hi_p}"
        if range_active:
            charges = charges[(charges >= lo_e) & (charges <= hi_e)]
    st.caption(f"Intervalle : {range_note}" if range_note else "Intervalle : complet")

    st.divider()

    n_total = len(df_filtered)
    n = len(charges)

    if n < 10:
        st.error("Moins de 10 valeurs après filtrage : ajustez les options.")
        st.stop()

    charges = np.sort(charges)

    st.divider()

    # ── 04 Seuil u* ───────────────────────────────────────────────────────
    st.markdown("**04 : Seuil  u***")
    u_min = float(np.nanpercentile(charges, 40))
    u_max = float(np.nanpercentile(charges, 97))
    if u_min >= u_max:
        u_max = float(np.nanmax(charges))
    default_u = float(np.clip(np.nanpercentile(charges, 75), u_min, u_max))

    # État géré par Streamlit via key="u_val" (évite le décalage d'un rerun).
    # On (ré)ajuste la valeur stockée dans la plage courante : les bornes
    # changent si le fichier/les filtres changent, sinon la valeur serait hors limites.
    if "u_val" not in st.session_state:
        st.session_state["u_val"] = default_u
    else:
        st.session_state["u_val"] = float(np.clip(st.session_state["u_val"], u_min, u_max))

    u_selected = st.slider(
        "Seuil u* (€)",
        min_value=u_min, max_value=u_max,
        step=max(1000.0, (u_max - u_min) / 100),
        format="%.0f",
        label_visibility="collapsed",
        key="u_val",
    )

    exceedances = int(np.sum(charges > u_selected))
    pct_exc     = exceedances / n * 100
    pct_below   = float(np.mean(charges <= u_selected)) * 100   # percentile de u* (P-level)

    st.markdown(f"""
    <div class="thresh-hero">
      <h5>Seuil retenu  u*</h5>
      <div class="val">{u_selected/1e3:.0f}<span style="font-size:16px;font-weight:400;
           color:rgba(255,255,255,.6);margin-left:3px">k€</span><span style="font-size:14px;
           font-weight:500;color:#ECA033;margin-left:8px">· P{pct_below:.0f}</span></div>
      <div class="sub">{exceedances} dépassements  ·  <b>{pct_exc:.1f} %</b> de la base</div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # ── 05 Méthodes ───────────────────────────────────────────────────────
    st.markdown("**05 : Méthodes**")
    ALL_METHODS = ["MRL", "Hill", "Gerstengarbe", "GPD Stabilité", "MCDA", "PELT & Jenks"]
    # État géré par Streamlit via key="methods_sel" (évite le décalage d'un rerun
    # qu'introduisait le couple default=... + écriture manuelle de session_state).
    selected_methods = st.multiselect(
        "Méthodes à analyser",
        options=ALL_METHODS,
        label_visibility="collapsed",
        key="methods_sel",
    )

    st.divider()

    # ── 06 Plage d'affichage (quantiles) ──────────────────────────────────
    st.markdown("**06 : Plage d'affichage**")
    q_range = st.slider(
        "Quantiles affichés (%)",
        min_value=0, max_value=100,
        value=(0, 100), step=5,
        format="%d %%",
        label_visibility="collapsed",
        key="q_range",
        help="Limite la plage de seuils/charges affichée (axe x) des graphiques. "
             "Ex : 50–100 % = médiane → maximum.",
    )
    q_lo_pct, q_hi_pct = q_range
    if (q_lo_pct, q_hi_pct) != (0, 100):
        _QZOOM = (float(np.nanpercentile(charges, q_lo_pct)) / 1e3,
                  float(np.nanpercentile(charges, q_hi_pct)) / 1e3)
    else:
        _QZOOM = (None, None)
    st.caption(f"Affichage : quantiles {q_lo_pct} → {q_hi_pct} %")

    st.divider()
    st.markdown(
        '<div style="font-size:10px;color:#939BAE;text-align:center;margin-top:8px;">'
        'v1.0 · LSN Ré Walbaum</div>',
        unsafe_allow_html=True,
    )


# ─── Ajustement GPD au seuil courant (pour le QQ plot du MRL) ────────────────
excesses  = charges[charges > u_selected] - u_selected
fit       = fit_gpd(excesses)
xi_hat    = fit["xi"]    if not np.isnan(fit["xi"])    else 0.0
sigma_hat = fit["sigma"] if not np.isnan(fit["sigma"]) else 1.0


# ─── En-tête de page ─────────────────────────────────────────────────────────
st.markdown("""
<div style="font-size:22px;font-weight:600;letter-spacing:-.02em;color:#0B1B36;margin-bottom:4px;">
    Détection du seuil de sinistre grave
</div>
<div style="font-size:12.5px;color:#5C657A;line-height:1.5;max-width:720px;margin-bottom:12px;">
    Étude POT (Peaks-Over-Threshold) · Comparaison de 5 méthodes graphiques pour
    sélectionner u* et ajustement GPD pour évaluer la qualité du seuil retenu.
</div>
""", unsafe_allow_html=True)

# ─── Statistiques descriptives ───────────────────────────────────────────────
_n_nan = int(np.sum(np.isnan(charges)))
_filter_note = f"{n_total:,} lignes après filtres · {n:,} sélectionnées"
if range_note:
    _filter_note += f" · intervalle {range_note}"
if _n_nan:
    _filter_note += f" (dont {_n_nan:,} NaN)"
st.markdown(
    '<div class="card-title">Statistiques descriptives : Variable analysée</div>'
    f'<div class="card-hint">Colonne : <b>{col_selected}</b> · {_filter_note}</div>',
    unsafe_allow_html=True,
)
st.markdown("<div style='margin-top:8px'></div>", unsafe_allow_html=True)

_s1 = st.columns(4)
_s1[0].metric("Observations",  str(n))
_s1[1].metric("Minimum",       fmt_k(float(np.nanmin(charges))))
_s1[2].metric("Q1  (P25)",     fmt_k(float(np.nanpercentile(charges, 25))))
_s1[3].metric("Médiane",       fmt_k(float(np.nanmedian(charges))))

_s2 = st.columns(4)
_s2[0].metric("Moyenne",       fmt_k(float(np.nanmean(charges))))
_s2[1].metric("Q3  (P75)",     fmt_k(float(np.nanpercentile(charges, 75))))
_s2[2].metric("P95",           fmt_M(float(np.nanpercentile(charges, 95))))
_s2[3].metric("Maximum",       fmt_M(float(np.nanmax(charges))))

st.markdown("<div style='margin-top:8px'></div>", unsafe_allow_html=True)

# ─── Calcul des méthodes sélectionnées (mis en cache, une entrée par méthode) ──
# _cv : incrémenter si l'une des fonctions compute() des modules change
_RESULT_KEY = {"MRL": "mrl_df", "Hill": "hill_df", "Gerstengarbe": "gers_res",
               "GPD Stabilité": "gpd_df", "MCDA": "mcda_res"}


@st.cache_data(show_spinner=False)
def compute_method(method: str, c_bytes: bytes, _cv: str = "5"):
    ch = np.sort(np.frombuffer(c_bytes, dtype=np.float64))
    if method == "MRL":           return mrl.compute(ch)
    if method == "Hill":          return hill.compute(ch)
    if method == "Gerstengarbe":  return gerstengarbe.compute(ch)
    if method == "GPD Stabilité": return gpd_stability.compute(ch)
    if method == "MCDA":          return mcda.compute(ch)


# Rien n'est calculé tant qu'aucune méthode n'est cochée : dès que l'utilisateur
# en sélectionne une dans le panneau latéral, Streamlit relance le script et la
# méthode choisie s'affiche aussitôt (calcul mis en cache par méthode).
if not selected_methods:
    st.info("Sélectionnez une ou plusieurs méthodes dans le panneau latéral "
            "pour afficher les analyses.")
    st.stop()

# Gerstengarbe est en O(n²) : on prévient avant un calcul potentiellement long
# (le résultat est ensuite mis en cache, donc une seule fois par jeu de données).
GERS_WARN_N = 20000
if "Gerstengarbe" in selected_methods and n > GERS_WARN_N:
    st.warning(
        f"Gerstengarbe : calcul long sur {n:,} lignes (complexité en n²). "
        f"Patientez quelques secondes, ou décochez cette méthode dans le panneau "
        f"latéral pour accélérer.",
        icon="⏳",
    )

computed = {}
with st.spinner("Calcul des méthodes en cours…"):
    for _m in selected_methods:
        if _m in _RESULT_KEY:
            computed[_RESULT_KEY[_m]] = compute_method(_m, charges.tobytes())


@st.cache_data(show_spinner=False)
def compute_multi_threshold(c_bytes: bytes, pen: float, jenks_k: int,
                            use_log: bool, _cv: str = "5") -> dict:
    ch = np.sort(np.frombuffer(c_bytes, dtype=np.float64))
    return multi_threshold.compute(ch, pen=pen, jenks_k=jenks_k, use_log=use_log)


def _zoom_charge_axes(fig):
    """Recadre les graphiques sur la plage de quantiles choisie.

    Limite l'axe x des panneaux basés sur charge/seuil (k€), PUIS recalcule
    l'axe y correspondant sur les seuls points visibles, afin que la courbe
    remplisse tout l'espace au lieu de se tasser dans une bande. Les axes non
    monétaires (k, rang, poids) ne sont pas touchés."""
    lo, hi = _QZOOM
    if lo is None:
        return

    # x-axis ids (ex. 'x', 'x2') des panneaux charge/seuil → axe + y-axis ancré
    charge_x_to_y = {}
    x_axis_obj = {}
    for ax in fig.select_xaxes():
        title = (ax.title.text or "")
        if title.startswith("Seuil u") or title.startswith("Charge"):
            yid = ax.anchor if ax.anchor and ax.anchor != "free" else "y"
            xid = "x" + yid[1:]
            charge_x_to_y[xid] = yid
            x_axis_obj[xid] = ax

    # Étendue réelle des points dans la fenêtre [lo, hi] (histogrammes exclus).
    # On cale x ET y sur les données présentes pour éviter tout espace vide.
    x_bounds = defaultdict(lambda: [np.inf, -np.inf])
    y_bounds = defaultdict(lambda: [np.inf, -np.inf])
    for tr in fig.data:
        xid = tr.xaxis or "x"
        if xid not in charge_x_to_y or tr.type == "histogram":
            continue
        if tr.x is None or tr.y is None:
            continue
        x = np.asarray(tr.x, dtype=float)
        y = np.asarray(tr.y, dtype=float)
        m = (x >= lo) & (x <= hi) & np.isfinite(x) & np.isfinite(y)
        if m.any():
            yid = charge_x_to_y[xid]
            x_bounds[xid][0] = min(x_bounds[xid][0], float(x[m].min()))
            x_bounds[xid][1] = max(x_bounds[xid][1], float(x[m].max()))
            y_bounds[yid][0] = min(y_bounds[yid][0], float(y[m].min()))
            y_bounds[yid][1] = max(y_bounds[yid][1], float(y[m].max()))

    for xid, ax in x_axis_obj.items():
        xmin, xmax = x_bounds[xid]
        if np.isfinite(xmin) and np.isfinite(xmax):
            pad = (xmax - xmin) * 0.02 or 1.0
            ax.range = [xmin - pad, xmax + pad]
        else:
            ax.range = [lo, hi]

    for yid, (ymin, ymax) in y_bounds.items():
        if not np.isfinite(ymin) or not np.isfinite(ymax):
            continue
        pad = (ymax - ymin) * 0.06 or abs(ymax) * 0.06 or 1.0
        fig.layout["yaxis" + yid[1:]].range = [ymin - pad, ymax + pad]


def _chart_card(title, hint, fig):
    _zoom_charge_axes(fig)
    st.markdown(
        f'<div class="card-title">{title}</div>'
        f'<div class="card-hint">{hint}</div>',
        unsafe_allow_html=True,
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": True})


# Guides de lecture en langage simple (public non familier de l'EVT).
HOW_TO_READ = {
    "MRL":
        "**Axe horizontal :** le seuil testé (en k€). **Axe vertical :** le montant moyen "
        "qui dépasse ce seuil.\n\n"
        "**À chercher :** l'endroit à partir duquel la courbe devient une ligne à peu près "
        "**droite** : c'est un bon candidat de seuil. Le graphe de droite montre combien de "
        "sinistres dépassent chaque seuil : méfiez-vous des zones où il en reste très peu.",
    "Hill":
        "Chaque point estime le « poids » de la queue de distribution selon le nombre de plus "
        "grosses valeurs prises en compte.\n\n"
        "**À chercher :** une zone où la courbe reste à peu près **plate** (un palier) ; le seuil "
        "correspondant est un bon candidat. À n'utiliser que pour des distributions à queue lourde.",
    "Gerstengarbe":
        "Deux courbes sont tracées. Le **point où elles se croisent** marque une rupture dans les "
        "données : la frontière entre sinistres « normaux » et « extrêmes ».\n\n"
        "**À faire :** survolez le croisement pour lire directement le seuil correspondant.",
    "GPD Stabilité":
        "**À chercher :** sur chaque courbe, l'endroit à partir duquel elle devient à peu près "
        "**horizontale (plate)** ; au-delà, le modèle est fiable.\n\n"
        "La couleur des points indique la qualité de l'ajustement (jaune = bon, violet = moins bon).",
    "MCDA":
        "Cette méthode combine plusieurs mesures de qualité pour **noter chaque seuil possible**.\n\n"
        "**À retenir :** le meilleur seuil est celui qui obtient le **score le plus bas**.",
    "PELT & Jenks":
        "Ces deux méthodes **découpent la distribution en plusieurs tranches**. Les traits "
        "verticaux marquent les seuils détectés.\n\n"
        "**Réglages :** augmentez la pénalité (PELT) pour obtenir **moins** de seuils ; ajustez le "
        "nombre de classes (Jenks). L'échelle « log » convient mieux aux montants très étalés.",
}


def _how_to_read(method: str):
    """Affiche un encart repliable expliquant comment lire le graphique de la méthode."""
    txt = HOW_TO_READ.get(method)
    if txt:
        with st.expander("Comment lire ce graphique ?"):
            st.markdown(txt)


def _seuil_retenu_input(method: str, default: float) -> dict:
    """Saisie du seuil retenu pour une méthode : valeur unique ou intervalle (€)."""
    st.markdown(
        '<div class="card-title" style="margin-top:6px;">Seuil retenu pour cette méthode</div>',
        unsafe_allow_html=True,
    )
    mode = st.radio(
        "Type de seuil retenu", ["Valeur unique", "Intervalle"],
        horizontal=True, key=f"retmode_{method}", label_visibility="collapsed",
    )
    if mode == "Valeur unique":
        val = st.number_input(
            "Seuil retenu (€)", min_value=0.0, value=float(default),
            step=1000.0, format="%.0f", key=f"retval_{method}",
        )
        return {"mode": "Valeur unique", "lo": float(val), "hi": None}

    c1, c2 = st.columns(2)
    lo = c1.number_input("Borne basse (€)", min_value=0.0, value=float(default),
                         step=1000.0, format="%.0f", key=f"retlo_{method}")
    hi = c2.number_input("Borne haute (€)", min_value=0.0, value=float(default),
                         step=1000.0, format="%.0f", key=f"rethi_{method}")
    return {"mode": "Intervalle", "lo": float(lo), "hi": float(hi)}


def _retenu_str(r: dict) -> str:
    """Représentation lisible (k€) d'un seuil retenu."""
    if r["mode"] == "Seuils détectés":
        vals = sorted(r.get("values", []))
        if not vals:
            return "Aucun seuil détecté"
        return ", ".join(
            f"{v/1e3:.0f} k€ (P{float(np.mean(charges <= v)) * 100:.0f})" for v in vals)
    if r["mode"] == "Intervalle":
        lo, hi = sorted([r["lo"], r["hi"]])
        return f"{lo/1e3:.0f} – {hi/1e3:.0f} k€"
    return f"{r['lo']/1e3:.0f} k€"


def _build_synthese_pdf(meta: dict, rows: list[tuple]) -> bytes:
    """PDF (texte + tableau) de la synthèse des seuils retenus, via fpdf2."""
    from fpdf import FPDF  # noqa: PLC0415

    def _t(s) -> str:
        # Les polices cœur de fpdf2 sont en latin-1 : remplacer les caractères absents.
        s = (str(s).replace("€", " EUR").replace("–", "-").replace("’", "'")
             .replace("…", "...").replace("·", "-"))
        return s.encode("latin-1", "replace").decode("latin-1")

    pdf = FPDF(orientation="P", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    pdf.set_font("Helvetica", "B", 15)
    pdf.cell(0, 10, _t("Synthèse des seuils retenus"), new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(120, 120, 120)
    pdf.cell(0, 6, _t("Threshold Studio - LSN Ré Walbaum"), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)

    # ── Bloc métadonnées ──────────────────────────────────────────────────────
    pdf.set_text_color(11, 27, 54)
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 7, _t("Paramètres de l'analyse"), new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    for label, value in meta.items():
        pdf.set_font("Helvetica", "B", 10)
        pdf.cell(45, 6, _t(f"{label} :"))
        pdf.set_font("Helvetica", "", 10)
        pdf.multi_cell(0, 6, _t(value), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)

    # ── Tableau des seuils retenus ────────────────────────────────────────────
    from fpdf.fonts import FontFace  # noqa: PLC0415

    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 7, _t("Seuils retenus par méthode"), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(1)

    pdf.set_font("Helvetica", "", 10)
    headings = FontFace(emphasis="BOLD", color=(255, 255, 255), fill_color=(14, 47, 92))
    with pdf.table(col_widths=(32, 26, 42), text_align="LEFT",
                   headings_style=headings) as table:
        table.row([_t("Méthode"), _t("Type"), _t("Seuil(s) retenu(s)")])
        for row in rows:
            table.row([_t(c) for c in row])

    out = pdf.output()
    return bytes(out)


# ─── Onglets des méthodes ─────────────────────────────────────────────────────
TAB_DESC = {
    "MRL":          ("Mean Residual Life",
                     "e(u) = E[X−u | X>u]. La courbe doit devenir linéaire et croissante au-dessus de u*."),
    "Hill":         ("Hill estimator",
                     "Estime le « poids » de la queue de distribution selon le nombre k de plus grandes "
                     "valeurs retenues. On choisit un seuil dans la plage de k où la courbe se stabilise "
                     "(plateau). Méthode réservée aux distributions à queue lourde."),
    "Gerstengarbe": ("Gerstengarbe",
                     "Test séquentiel : intersection de −H_k (décroissante) et H̃_k (croissante). Valide pour tout ξ."),
    "GPD Stabilité":("GPD Stabilité",
                     "ξ̂(u) et σ*(u) = σ(u)−ξu doivent être constants au-dessus du vrai seuil u*."),
    "MCDA":         ("MCDA consensus",
                     "Agrégation pondérée (KS, χ², RMSE) de tous les candidats u* via entropie de Shannon."),
    "PELT & Jenks": ("Détection de seuils multiples : PELT & Jenks Natural Breaks",
                     "PELT détecte les ruptures sur ln(charges) ; le nombre de ruptures dépend de la pénalité. "
                     "Jenks minimise la variance intra-classe sur ln(charges) pour trouver les coupures "
                     "naturelles. Les seuils détectés sont reconvertis en €."),
}

tabs = st.tabs(selected_methods + ["Synthèse"])
tab_map = {m: t for m, t in zip(selected_methods, tabs)}
synth_tab = tabs[-1]
retained: dict[str, dict] = {}

if "MRL" in tab_map:
    with tab_map["MRL"]:
        title, hint = TAB_DESC["MRL"]
        fig_mrl = mrl.plot(computed["mrl_df"], u_selected)
        _chart_card(title, hint, fig_mrl)
        _how_to_read("MRL")

        st.markdown("<div style='margin-top:10px'></div>", unsafe_allow_html=True)
        fig_qq = mrl.qq_plot(excesses, xi_hat, sigma_hat)
        _chart_card(
            "QQ plot des excès : ajustement GPD",
            f"Excès au-dessus de u* = {u_selected/1e3:.0f} k€ comparés à la GPD "
            f"ajustée (ξ̂ = {xi_hat:.3f}, σ̂ = {sigma_hat/1e3:.0f} k€). "
            f"Points alignés sur la diagonale ⇒ bon ajustement à la loi GPD.",
            fig_qq,
        )
        retained["MRL"] = _seuil_retenu_input("MRL", u_selected)

if "Hill" in tab_map:
    with tab_map["Hill"]:
        title, hint = TAB_DESC["Hill"]
        fig_hill = hill.plot(computed["hill_df"], u_selected)
        _chart_card(title, hint, fig_hill)
        _how_to_read("Hill")
        st.markdown(
            '<div style="font-size:11.5px;color:#5C657A;">'
            'Rappel : Hill valide uniquement pour ξ > 0 (queue lourde Fréchet).</div>',
            unsafe_allow_html=True,
        )
        retained["Hill"] = _seuil_retenu_input("Hill", u_selected)

if "Gerstengarbe" in tab_map:
    with tab_map["Gerstengarbe"]:
        title, hint = TAB_DESC["Gerstengarbe"]
        gers_res = computed["gers_res"]
        fig_gers = gerstengarbe.plot(gers_res)
        _chart_card(title, hint, fig_gers)
        _how_to_read("Gerstengarbe")
        retained["Gerstengarbe"] = _seuil_retenu_input(
            "Gerstengarbe", gers_res.get("u_star", u_selected))

if "GPD Stabilité" in tab_map:
    with tab_map["GPD Stabilité"]:
        title, hint = TAB_DESC["GPD Stabilité"]
        fig_gpd = gpd_stability.plot(computed["gpd_df"], u_selected)
        _chart_card(title, hint, fig_gpd)
        _how_to_read("GPD Stabilité")
        retained["GPD Stabilité"] = _seuil_retenu_input("GPD Stabilité", u_selected)

if "MCDA" in tab_map:
    with tab_map["MCDA"]:
        title, hint = TAB_DESC["MCDA"]
        mcda_res = computed["mcda_res"]
        if mcda_res:
            fig_mcda = mcda.plot(mcda_res, u_selected)
            _chart_card(title, hint, fig_mcda)
            _how_to_read("MCDA")
            W = mcda_res["W"]
            E = mcda_res["E"]
            for crit, w, e in zip(mcda_res["criteria"], W, E):
                st.markdown(
                    f'<div style="font-size:11px;color:#5C657A;margin-bottom:2px;">'
                    f'<b>{crit}</b>  w = {w:.3f}  E = {e:.4f}</div>',
                    unsafe_allow_html=True,
                )
            retained["MCDA"] = _seuil_retenu_input("MCDA", mcda_res["u_star"])
        else:
            st.warning("MCDA : pas assez de candidats valides.")

if "PELT & Jenks" in tab_map:
    with tab_map["PELT & Jenks"]:
        title, hint = TAB_DESC["PELT & Jenks"]
        _scale = st.radio(
            "Échelle de détection", ["Log", "Brut"], horizontal=True, key="mt_scale",
            help="Log = ratios / queue de distribution (recommandé pour sinistres graves). "
                 "Brut = écarts absolus en €.",
        )
        use_log = (_scale == "Log")
        _unit = "ln(charges)" if use_log else "charges (€)"

        _pc1, _pc2 = st.columns(2)
        pen     = _pc1.slider("Pénalité PELT", 0.5, 20.0, 5.0, step=0.5, key="pelt_pen",
                               help=f"Pénalité de PELT sur {_unit}. Pénalité basse ⇒ "
                                    "plus de ruptures ; pénalité haute ⇒ moins de ruptures.")
        jenks_k = _pc2.slider("Classes Jenks (k)", 2, 6, 4, key="jenks_k",
                               help=f"Nombre de classes sur {_unit} : k−1 seuils naturels seront tracés.")
        with st.spinner("Détection de seuils multiples…"):
            mt_res = compute_multi_threshold(charges.tobytes(), pen, jenks_k, use_log)
        fig_mt = multi_threshold.plot(mt_res, u_selected)
        _chart_card(title, hint, fig_mt)
        _how_to_read("PELT & Jenks")
        _pelt_u    = mt_res["pelt_u"]
        _jenks_v   = mt_res["jenks_vals"]
        if _pelt_u or _jenks_v:
            st.markdown(
                '<div style="font-size:11.5px;color:#5C657A;margin-top:4px;">'
                '<b>Seuils détectés</b></div>',
                unsafe_allow_html=True,
            )
            _cols = st.columns(max(len(_pelt_u) + len(_jenks_v), 1))
            _ci = 0
            for i, pu in enumerate(_pelt_u):
                pct = float(np.mean(charges <= pu)) * 100
                _cols[_ci].metric(f"PELT {i+1}", f"{pu/1e3:.0f} k€", f"P{pct:.0f}")
                _ci += 1
            for i, jv in enumerate(_jenks_v):
                pct = float(np.mean(charges <= jv)) * 100
                _cols[_ci].metric(f"Jenks {i+1}", f"{jv/1e3:.0f} k€", f"P{pct:.0f}")
                _ci += 1
        # Détection multi-seuils : on recopie directement les seuils détectés
        # (pilotés par la pénalité pour PELT, par k pour Jenks) : pas de saisie.
        retained["PELT"]  = {"mode": "Seuils détectés", "values": list(_pelt_u)}
        retained["Jenks"] = {"mode": "Seuils détectés", "values": list(_jenks_v)}
        st.caption("Les seuils détectés ci-dessus sont repris automatiquement dans la synthèse.")


# ─── Onglet Synthèse ──────────────────────────────────────────────────────────
with synth_tab:
    st.markdown(
        '<div class="card-title">Synthèse des seuils retenus</div>'
        '<div class="card-hint">Récapitulatif des seuils saisis pour chaque méthode, '
        'avec les paramètres de l\'analyse. Exportable en PDF.</div>',
        unsafe_allow_html=True,
    )
    st.markdown("<div style='margin-top:10px'></div>", unsafe_allow_html=True)

    # ── Métadonnées de l'analyse ──────────────────────────────────────────────
    _excl = []
    if excl_nan:  _excl.append("valeurs manquantes (NaN)")
    if excl_neg:  _excl.append("valeurs négatives (< 0)")
    if excl_zero: _excl.append("valeurs nulles (= 0)")
    _excl_str = ", ".join(_excl) if _excl else "Aucune"

    if applied_filters:
        _filt_str = " ; ".join(
            f"{k} = {', '.join(map(str, v))}" for k, v in applied_filters.items())
    else:
        _filt_str = "Aucun"

    meta = {"Base": uploaded.name}
    if sheet_name is not None:
        meta["Feuille"] = sheet_name
    meta["Variable analysée"] = col_selected
    meta["Observations"]      = f"{n:,} / {n_total:,} après filtres"
    meta["Filtres appliqués"] = _filt_str
    meta["Intervalle analysé"] = range_note or "Complet"
    meta["Exclusions"]        = _excl_str
    meta["Date"]              = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")

    _meta_html = "".join(
        f'<div style="font-size:12px;color:#5C657A;margin-bottom:3px;">'
        f'<b style="color:#0B1B36;">{k} :</b> {v}</div>'
        for k, v in meta.items()
    )
    st.markdown(_meta_html, unsafe_allow_html=True)
    st.markdown("<div style='margin-top:10px'></div>", unsafe_allow_html=True)

    # ── Tableau récapitulatif ─────────────────────────────────────────────────
    # "PELT & Jenks" est éclaté en deux lignes (deux méthodes distinctes).
    rows = []
    for m in selected_methods:
        sub_methods = ("PELT", "Jenks") if m == "PELT & Jenks" else (m,)
        for sm in sub_methods:
            r = retained.get(sm)
            if r is not None:
                rows.append((sm, r["mode"], _retenu_str(r)))
    if rows:
        st.table(pd.DataFrame(
            rows, columns=["Méthode", "Type", "Seuil(s) retenu(s)"]
        ).set_index("Méthode"))

        _stem = uploaded.name.rsplit(".", 1)[0]
        st.download_button(
            "⬇  Télécharger la synthèse (PDF)",
            data=_build_synthese_pdf(meta, rows),
            file_name=f"synthese_seuils_{_stem}.pdf",
            mime="application/pdf",
        )
    else:
        st.info("Aucun seuil retenu saisi pour les méthodes sélectionnées.")

