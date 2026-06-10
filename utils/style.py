"""Shared visual constants and figure layout for all method plots."""
import plotly.graph_objects as go

NAVY   = "#1E3A8A"
PURPLE = "#7227A0"
VIOLET = "#9C2F8E"
ORANGE = "#E36120"
GRIS   = "#9099AC"
VERT   = "#137A45"


def apply_layout(fig: go.Figure, height: int = 420,
                 margin_t: int = 40, margin_r: int = 20) -> None:
    fig.update_layout(
        legend=dict(orientation="h", y=-0.18, x=0.5, xanchor="center"),
        plot_bgcolor="#FCFDFE", paper_bgcolor="#FCFDFE",
        margin=dict(t=margin_t, b=70, l=60, r=margin_r),
        height=height,
        font=dict(family="Inter, sans-serif", size=12, color="#0B1B36"),
    )
    fig.update_xaxes(gridcolor="#EEF0F5")
    fig.update_yaxes(gridcolor="#EEF0F5")
