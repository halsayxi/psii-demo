from __future__ import annotations

from typing import Iterable

import plotly.graph_objects as go


COLORS = [
    "#1b4d89",
    "#c23b22",
    "#2e7d50",
    "#8f5aa3",
    "#d28f22",
    "#2f7d8c",
    "#6f5f4b",
    "#9c3d54",
]


def distribution_figure(
    option_codes: list[str],
    option_labels: list[str],
    human_probs: Iterable[float],
    method_series: dict[str, Iterable[float]],
) -> go.Figure:
    x_labels = [f"{code}. {label}" for code, label in zip(option_codes, option_labels)]
    human_values = list(human_probs)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x_labels,
            y=human_values,
            mode="lines+markers",
            name="<b>Human</b>",
            line=dict(color="#111827", width=5, dash="dash"),
            marker=dict(size=10),
            hovertemplate="%{x}<br>Probability: %{y:.3f}<extra>%{fullData.name}</extra>",
        )
    )

    for idx, (name, probs) in enumerate(method_series.items()):
        values = list(probs)
        is_PSII = name == "PSII"
        color = COLORS[idx % len(COLORS)]
        fig.add_trace(
            go.Scatter(
                x=x_labels,
                y=values,
                mode="lines+markers",
                name=f"<b>{name}</b>" if is_PSII else name,
                line=dict(color=color, width=5 if is_PSII else 3),
                marker=dict(size=10 if is_PSII else 8),
                hovertemplate="%{x}<br>Probability: %{y:.3f}<extra>%{fullData.name}</extra>",
            )
        )

    fig.update_layout(
        height=500,
        margin=dict(l=20, r=20, t=56, b=80),
        yaxis=dict(
            title=dict(text="Selection probability", font=dict(color="#172033", size=14)),
            tickfont=dict(color="#172033", size=12),
            range=[0, 1],
            gridcolor="#e8edf3",
            linecolor="#98a7b7",
            zerolinecolor="#c8d3df",
        ),
        xaxis=dict(
            title=dict(text="Answer option", font=dict(color="#172033", size=14)),
            tickfont=dict(color="#172033", size=12),
            tickangle=-25,
            automargin=True,
            linecolor="#98a7b7",
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.04,
            xanchor="left",
            x=0,
            bgcolor="#ffffff",
            bordercolor="#b9c6d3",
            borderwidth=1,
            font=dict(color="#172033", size=13),
            itemclick="toggle",
            itemdoubleclick="toggleothers",
        ),
        plot_bgcolor="#fbfcfe",
        paper_bgcolor="#ffffff",
        font=dict(
            family="Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, sans-serif",
            color="#172033",
            size=13,
        ),
        hoverlabel=dict(
            bgcolor="#ffffff",
            bordercolor="#98a7b7",
            font=dict(color="#172033", size=13),
        ),
        hovermode="x unified",
    )
    return fig
