import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from scipy.stats import gaussian_kde
from dash import Dash, html, dcc, callback, Output, Input

from tools.plot_helper import MONTH_ORDER, plot_pie, plot_dual_axis_bar_line

# ── Data ──────────────────────────────────────────────────────────────
df = pd.read_csv("dataset/processed_data.csv")
df["y"] = df["target"].map({1: "yes", 0: "no"})

ALL_JOBS      = sorted(df["job"].dropna().unique())
ALL_EDUCATION = sorted(df["education"].dropna().unique())
ALL_MARITAL   = sorted(df["marital"].dropna().unique())
TOTAL_RECORDS = len(df)

# ── Design tokens ─────────────────────────────────────────────────────
FONT      = "Inter, Arial, sans-serif"
BG        = "#080D18"
SURFACE   = "#0C1220"
SURFACE2  = "#111827"
BORDER    = "rgba(255,255,255,0.07)"
ACCENT    = "#3B82F6"
TEXT      = "#F1F5F9"
MUTED     = "#64748B"
SUCCESS   = "#22C55E"

YES_COLOR = "#4C78A8"   # data color — subscribed
NO_COLOR  = "#E45756"   # data color — not subscribed

MEDALS = [
    ("🥇", "#F59E0B"),
    ("🥈", "#94A3B8"),
    ("🥉", "#B45309"),
]

CARD = {
    "backgroundColor": SURFACE,
    "borderRadius":    "16px",
    "padding":         "24px",
    "border":          f"1px solid {BORDER}",
    "boxShadow":       "0 4px 20px rgba(0,0,0,0.35)",
}


# ── Layout helpers ────────────────────────────────────────────────────

def kpi_card(label, value, color):
    return html.Div(className="kpi-card", style={
        **CARD,
        "position": "relative",
        "overflow": "hidden",
        "paddingLeft": "22px",
    }, children=[
        html.Div(style={
            "position": "absolute", "left": 0,
            "top": "16px", "bottom": "16px", "width": "3px",
            "background": f"linear-gradient(180deg, {color}, {color}40)",
            "borderRadius": "2px",
        }),
        html.P(label, style={
            "color": MUTED, "margin": "0 0 10px",
            "fontSize": "10px", "fontWeight": "600",
            "textTransform": "uppercase", "letterSpacing": "1.2px",
        }),
        html.H2(value, style={
            "color": color, "margin": 0,
            "fontSize": "32px", "fontWeight": "800",
            "letterSpacing": "-1.5px", "lineHeight": "1",
        }),
    ])


def section_title(text, color=ACCENT):
    return html.Div(style={
        "display": "flex", "alignItems": "center",
        "gap": "12px", "margin": "44px 0 20px",
    }, children=[
        html.Div(style={
            "width": "3px", "height": "20px",
            "background": f"linear-gradient(180deg, {color}, {color}55)",
            "borderRadius": "2px", "flexShrink": 0,
        }),
        html.H3(text, style={
            "color": TEXT, "margin": 0,
            "fontSize": "15px", "fontWeight": "700",
            "letterSpacing": "-0.2px",
        }),
    ])


def group_container(label, border_color, *children):
    """Visually groups related charts with a labelled border."""
    return html.Div(style={
        "border": f"1px solid {border_color}28",
        "borderRadius": "20px",
        "padding": "20px",
        "marginBottom": "24px",
        "background": f"linear-gradient(135deg, {border_color}06, transparent)",
    }, children=[
        html.Div(style={
            "display": "flex", "alignItems": "center",
            "gap": "8px", "marginBottom": "14px",
        }, children=[
            html.Div(style={
                "width": "6px", "height": "6px", "borderRadius": "50%",
                "background": border_color, "flexShrink": 0,
            }),
            html.P(label, style={
                "color": border_color, "margin": 0,
                "fontSize": "10px", "fontWeight": "700",
                "textTransform": "uppercase", "letterSpacing": "1.5px",
            }),
        ]),
        *children,
    ])


def dark_theme(fig):
    """Apply polished dark theme to any plotly figure."""
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=FONT, color=TEXT, size=12),
        margin=dict(l=44, r=44, t=52, b=44),
        title=dict(x=0.5, font=dict(size=14, family=FONT, color=TEXT)),
        width=None, height=None,
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            borderwidth=0,
            font=dict(size=11, family=FONT),
        ),
        hoverlabel=dict(
            bgcolor=SURFACE2,
            bordercolor=BORDER,
            font=dict(family=FONT, size=12, color=TEXT),
        ),
    )
    fig.update_xaxes(
        gridcolor="rgba(255,255,255,0.04)",
        zerolinecolor="rgba(255,255,255,0.05)",
        tickfont=dict(size=11, family=FONT),
        title_font=dict(size=12, family=FONT),
    )
    fig.update_yaxes(
        gridcolor="rgba(255,255,255,0.04)",
        zerolinecolor="rgba(255,255,255,0.05)",
        tickfont=dict(size=11, family=FONT),
        title_font=dict(size=12, family=FONT),
    )
    return fig


def chart_card(*children, height="380px", extra_style=None):
    """A styled chart card with hover glow via CSS."""
    return html.Div(className="chart-card", style={
        **CARD, **(extra_style or {}),
    }, children=[
        dcc.Loading(
            children=list(children),
            type="dot",
            color=ACCENT,
            delay_show=120,
        )
    ])


def graph(fig, gid, height="380px"):
    return dcc.Graph(
        id=gid, figure=dark_theme(fig),
        config={"displayModeBar": False},
        style={"height": height},
    )


def filter_df(jobs, educations, maritals):
    d = df.copy()
    if jobs:       d = d[d["job"].isin(jobs)]
    if educations: d = d[d["education"].isin(educations)]
    if maritals:   d = d[d["marital"].isin(maritals)]
    return d


# ── Color fixing ──────────────────────────────────────────────────────

def apply_yes_no_colors(fig):
    for trace in fig.data:
        if isinstance(trace, go.Pie):
            trace.marker.colors = [YES_COLOR if l == "yes" else NO_COLOR for l in trace.labels]
        else:
            name = getattr(trace, "name", "")
            c = YES_COLOR if name == "yes" else (NO_COLOR if name == "no" else None)
            if c:
                trace.marker.color = c
                if hasattr(trace, "line") and trace.line:
                    trace.line.color = c
    return fig


# ── Grouped bar with % hover + peak annotations ───────────────────────

def make_grouped_bar_pct(data, cat_col, cat_label, title, horizontal=False, medals=False):
    d = data[data[cat_col] != "unknown"].copy()
    grp = d.groupby([cat_col, "y"]).size().reset_index(name="count")
    grp["pct"] = grp["count"] / grp.groupby(cat_col)["count"].transform("sum") * 100

    cat_order = (
        grp.groupby(cat_col)["count"].sum()
        .sort_values(ascending=horizontal)
        .index.tolist()
    )

    fig = go.Figure()
    for val, color, display in [("no", NO_COLOR, "Not Subscribed"), ("yes", YES_COLOR, "Subscribed")]:
        g = grp[grp["y"] == val].set_index(cat_col).reindex(cat_order).reset_index()
        hover = (
            f"<b>%{{{'y' if horizontal else 'x'}}}</b><br>"
            f"{display}: <b>%{{{'x' if horizontal else 'y'}:,}}</b><br>"
            f"<i>%{{customdata[0]:.1f}}% of this {cat_label.lower()}</i>"
            "<extra></extra>"
        )
        kwargs = dict(name=display, marker_color=color,
                      customdata=g[["pct"]].values, hovertemplate=hover)
        if horizontal:
            fig.add_bar(y=g[cat_col], x=g["count"], orientation="h", **kwargs)
        else:
            fig.add_bar(x=g[cat_col], y=g["count"], **kwargs)

    axis_kw = dict(categoryorder="array", categoryarray=cat_order)
    if horizontal:
        fig.update_layout(xaxis_title="Count", yaxis=dict(title=cat_label, **axis_kw))
    else:
        fig.update_layout(yaxis_title="Count", xaxis=dict(title=cat_label, **axis_kw))

    fig.update_layout(barmode="group", title=title, legend=dict(title="Subscribed"))

    # Peak annotations
    yes_grp = grp[grp["y"] == "yes"]
    if not yes_grp.empty:
        top_rows = yes_grp.nlargest(3 if medals else 1, "pct")
        for i, (_, row) in enumerate(top_rows.iterrows()):
            emoji, color = MEDALS[i] if medals else ("★", "#F59E0B")
            shared = dict(
                text=f"{emoji} <b>{row['pct']:.1f}%</b>",
                showarrow=False,
                font=dict(color=color, size=12, family=FONT),
                bgcolor="rgba(8,13,24,0.88)",
                bordercolor=color, borderwidth=1, borderpad=5,
            )
            if horizontal:
                fig.add_annotation(y=row[cat_col], x=row["count"],
                                   xanchor="left", yanchor="middle", xshift=10, **shared)
            else:
                fig.add_annotation(x=row[cat_col], y=row["count"],
                                   xanchor="center", yanchor="bottom", yshift=10, **shared)
    return fig


# ── Histogram + marginal box ──────────────────────────────────────────

def make_hist_box(data, col, col_label, group_val, color, group_label, cap_pct=None):
    values = data[data["y"] == group_val][col].dropna()
    if cap_pct:
        values = values[values <= values.quantile(cap_pct)]

    mean_val, median_val, n = values.mean(), values.median(), len(values)
    bin_w = (values.max() - values.min()) / 25
    kde_x = np.linspace(values.min(), values.max(), 400)
    kde_y = gaussian_kde(values)(kde_x) * n * bin_w

    fig = px.histogram(
        pd.DataFrame({col_label: values}), x=col_label, nbins=25,
        marginal="box", color_discrete_sequence=[color],
        title=f"{col_label} — {group_label} (n={n:,})", opacity=0.72,
    )
    fig.add_scatter(x=kde_x, y=kde_y, mode="lines", name="KDE",
                    line=dict(color=color, width=2.5), showlegend=False)
    for val, dash, ann_color, pos in [
        (mean_val,   "dash", TEXT,  "top right"),
        (median_val, "dot",  MUTED, "top left"),
    ]:
        fig.add_vline(x=val, line_dash=dash, line_color=ann_color, line_width=1.5,
                      annotation_text=f"{'Mean' if dash=='dash' else 'Median'} {val:,.1f}",
                      annotation_font=dict(color=ann_color, size=11, family=FONT),
                      annotation_position=pos)
    return fig


# ── Chart builders ────────────────────────────────────────────────────

def make_subscription_pie(data):
    fig = plot_pie(data["y"], title="Subscription Outcome", x_label="Subscribed", hole=0.55)
    return apply_yes_no_colors(fig)

def make_poutcome_bar(data):
    return make_grouped_bar_pct(data, "poutcome", "Previous Outcome",
                                "Previous Outcome vs Subscription  (Strongest Predictor)")

def make_month_dual(data):
    fig = plot_dual_axis_bar_line(data["month"], data["y"],
                                  title="Monthly Contacts vs Subscription Rate")
    monthly = (
        data.groupby("month")
        .agg(contacts=("y", "size"), subs=("y", lambda s: (s == "yes").sum()))
        .reset_index()
    )
    monthly["rate"] = monthly["subs"] / monthly["contacts"] * 100
    for i, (_, row) in enumerate(monthly.nlargest(3, "rate").iterrows()):
        emoji, color = MEDALS[i]
        fig.add_annotation(
            x=row["month"], y=row["rate"], yref="y2",
            text=f"{emoji} <b>{row['rate']:.1f}%</b>",
            showarrow=True, arrowhead=2,
            arrowcolor=color, arrowwidth=1.5, ax=0, ay=-44,
            xanchor="center",
            font=dict(color=color, size=12, family=FONT),
            bgcolor="rgba(8,13,24,0.88)",
            bordercolor=color, borderwidth=1, borderpad=5,
        )
    return fig

def make_job_bar(data):
    return make_grouped_bar_pct(data, "job", "Job", "Job Type vs Subscription",
                                horizontal=True, medals=True)

def make_education_bar(data):
    return make_grouped_bar_pct(data, "education", "Education",
                                "Education Level vs Subscription")

def make_contact_bar(data):
    return make_grouped_bar_pct(data, "contact", "Contact Method",
                                "Contact Method vs Subscription")

def make_housing_bar(data):
    return make_grouped_bar_pct(data, "housing", "Housing Loan",
                                "Housing Loan vs Subscription")

def make_loan_bar(data):
    return make_grouped_bar_pct(data, "loan", "Personal Loan",
                                "Personal Loan vs Subscription")

def make_age_yes(data):
    return make_hist_box(data, "age", "Age", "yes", YES_COLOR, "Subscribed")

def make_age_no(data):
    return make_hist_box(data, "age", "Age", "no", NO_COLOR, "Not Subscribed")

def make_balance_yes(data):
    return make_hist_box(data, "balance", "Balance (€)", "yes", YES_COLOR, "Subscribed", cap_pct=0.99)

def make_balance_no(data):
    return make_hist_box(data, "balance", "Balance (€)", "no", NO_COLOR, "Not Subscribed", cap_pct=0.99)


# ── App ───────────────────────────────────────────────────────────────

app = Dash(
    __name__,
    external_stylesheets=[
        "https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap"
    ],
)
app.title = "Bank Marketing Intelligence"

GRID2 = {"display": "grid", "gridTemplateColumns": "1fr 1fr",      "gap": "18px", "marginBottom": "18px"}
GRID3 = {"display": "grid", "gridTemplateColumns": "1fr 1fr 1fr", "gap": "18px"}


def filter_pill_row(row_id, label, options, accent=ACCENT):
    """One labelled row of pill-toggle buttons backed by a dcc.Checklist."""
    dot_color = accent
    return html.Div(style={"marginBottom": "18px"}, children=[
        html.Div(className="filter-row-label", children=[
            html.Div(style={
                "width": "5px", "height": "5px", "borderRadius": "50%",
                "background": dot_color, "flexShrink": 0,
            }),
            html.Span(label, style={
                "color": MUTED, "fontSize": "10px",
                "fontWeight": "700", "letterSpacing": "1.5px",
            }),
        ]),
        dcc.Checklist(
            id=row_id,
            className="pill-checklist",
            options=[{"label": f" {v.replace('-', '\u2011').title()}", "value": v}
                     for v in options],
            value=[],
            inline=True,
            inputStyle={"position": "absolute", "opacity": "0", "width": "0", "height": "0"},
        ),
    ])

# ── Layout ────────────────────────────────────────────────────────────

app.layout = html.Div(style={
    "backgroundColor": BG,
    "minHeight": "100vh",
    "fontFamily": FONT,
    "color": TEXT,
}, children=[

    # ── Gradient top bar
    html.Div(style={
        "height": "3px",
        "background": f"linear-gradient(90deg, {ACCENT}, #8B5CF6 50%, {ACCENT})",
    }),

    # ── Page body
    html.Div(style={"padding": "32px 40px"}, children=[

        # ── Filters ─────────────────────────────────────────────────
        html.Div(style={**CARD, "marginBottom": "28px"}, children=[
            # Top bar: label + status + reset
            html.Div(style={
                "display": "flex", "alignItems": "center",
                "justifyContent": "space-between", "marginBottom": "22px",
            }, children=[
                html.Div(style={"display": "flex", "alignItems": "center", "gap": "16px"}, children=[
                    html.Div(style={"display": "flex", "alignItems": "center", "gap": "8px"}, children=[
                        html.Div(style={
                            "width": "6px", "height": "6px",
                            "borderRadius": "50%", "background": ACCENT,
                        }),
                        html.Span("FILTERS", style={
                            "color": MUTED, "fontSize": "10px",
                            "fontWeight": "700", "letterSpacing": "1.8px",
                        }),
                    ]),
                    # Live record-count badge
                    html.Div(id="filter-status", style={
                        "background": f"rgba(59,130,246,0.10)",
                        "border": f"1px solid rgba(59,130,246,0.22)",
                        "color": ACCENT, "borderRadius": "999px",
                        "padding": "3px 12px", "fontSize": "11px", "fontWeight": "600",
                    }),
                ]),
                html.Button("✕  Reset All", id="btn-reset", n_clicks=0, className="btn-reset"),
            ]),

            # Pill rows
            filter_pill_row("filter-marital",   "MARITAL STATUS", ALL_MARITAL,   accent="#8B5CF6"),
            filter_pill_row("filter-education", "EDUCATION",      ALL_EDUCATION, accent="#06B6D4"),
            html.Div(style={"marginBottom": 0},  # job row — no bottom margin needed
                     children=filter_pill_row("filter-job", "JOB TYPE", ALL_JOBS, accent=ACCENT).children),
        ]),

        # ── KPIs ────────────────────────────────────────────────────
        html.Div(id="kpi-row", style={
            "display": "grid", "gridTemplateColumns": "repeat(5, 1fr)",
            "gap": "16px", "marginBottom": "8px",
        }),

        # ── SECTION 1: Overview ──────────────────────────────────────
        section_title("Campaign Overview"),
        html.Div(style={"display": "grid", "gridTemplateColumns": "1fr 2fr",
                        "gap": "18px", "marginBottom": "18px"}, children=[
            chart_card(html.Div(id="chart-pie"),      height="380px"),
            chart_card(html.Div(id="chart-poutcome"), height="380px"),
        ]),

        # ── SECTION 2: Who Subscribes? ───────────────────────────────
        section_title("Who Subscribes?"),
        html.Div(style=GRID2, children=[
            chart_card(html.Div(id="chart-month"), height="380px"),
            chart_card(html.Div(id="chart-job"),   height="500px"),
        ]),
        html.Div(style=GRID2, children=[
            chart_card(html.Div(id="chart-education"), height="360px"),
            chart_card(html.Div(id="chart-contact"),   height="360px"),
        ]),
        html.Div(style={**GRID2, "marginBottom": "24px"}, children=[
            chart_card(html.Div(id="chart-housing"), height="340px"),
            chart_card(html.Div(id="chart-loan"),    height="340px"),
        ]),

        # ── SECTION 3: Demographics ──────────────────────────────────
        section_title("Demographics & Financial Patterns"),
        group_container("Age Distribution", YES_COLOR,
            html.Div(style={"marginBottom": "14px"},
                     children=chart_card(html.Div(id="chart-age-yes"), height="400px")),
            chart_card(html.Div(id="chart-age-no"), height="400px"),
        ),
        group_container("Account Balance Distribution", NO_COLOR,
            html.Div(style={"marginBottom": "14px"},
                     children=chart_card(html.Div(id="chart-balance-yes"), height="400px")),
            chart_card(html.Div(id="chart-balance-no"), height="400px"),
        ),

        # ── Footer ──────────────────────────────────────────────────
        html.Div(style={
            "borderTop": f"1px solid {BORDER}",
            "marginTop": "48px", "paddingTop": "24px",
            "display": "flex", "justifyContent": "space-between", "alignItems": "center",
        }, children=[
            html.P("Abdallah Mohamed · Ahmed Gamal · Ahmed Saleem",
                   style={"color": MUTED, "margin": 0, "fontSize": "12px"}),
            html.P("AI Track — Intake 46 · Alexandria Branch",
                   style={"color": MUTED, "margin": 0, "fontSize": "12px"}),
        ]),
    ]),
])


# ── Callbacks ─────────────────────────────────────────────────────────

@callback(
    Output("filter-job",       "value"),
    Output("filter-education", "value"),
    Output("filter-marital",   "value"),
    Input("btn-reset", "n_clicks"),
    prevent_initial_call=True,
)
def reset_filters(_):
    return None, None, None


@callback(
    Output("filter-status",   "children"),
    Output("kpi-row",         "children"),
    Output("chart-pie",       "children"),
    Output("chart-poutcome",  "children"),
    Output("chart-month",     "children"),
    Output("chart-job",       "children"),
    Output("chart-education", "children"),
    Output("chart-contact",   "children"),
    Output("chart-housing",   "children"),
    Output("chart-loan",      "children"),
    Output("chart-age-yes",   "children"),
    Output("chart-age-no",    "children"),
    Output("chart-balance-yes","children"),
    Output("chart-balance-no", "children"),
    Input("filter-job",       "value"),
    Input("filter-education", "value"),
    Input("filter-marital",   "value"),
)
def update_dashboard(jobs, educations, maritals):
    data  = filter_df(jobs, educations, maritals)
    total = len(data)
    subs  = int(data["target"].sum())
    avg_b = data["balance"].mean()
    avg_a = data["age"].mean()

    kpis = [
        kpi_card("Total Clients",   f"{total:,}",      ACCENT),
        kpi_card("Subscribers",     f"{subs:,}",       YES_COLOR),
        kpi_card("Non-Subscribers", f"{total - subs:,}", NO_COLOR),
        kpi_card("Avg Balance",     f"€{avg_b:,.0f}",  "#8B5CF6"),
        kpi_card("Avg Age",         f"{avg_a:.0f}",    "#06B6D4"),
    ]

    status = (f"{total:,} records" if total == TOTAL_RECORDS
              else f"{total:,} / {TOTAL_RECORDS:,} records")

    return (
        status,
        kpis,
        graph(make_subscription_pie(data),  "pie"),
        graph(make_poutcome_bar(data),       "poutcome"),
        graph(make_month_dual(data),         "month"),
        graph(make_job_bar(data),            "job",         height="500px"),
        graph(make_education_bar(data),      "education",   height="360px"),
        graph(make_contact_bar(data),        "contact",     height="360px"),
        graph(make_housing_bar(data),        "housing",     height="340px"),
        graph(make_loan_bar(data),           "loan",        height="340px"),
        graph(make_age_yes(data),            "age-yes",     height="400px"),
        graph(make_age_no(data),             "age-no",      height="400px"),
        graph(make_balance_yes(data),        "balance-yes", height="400px"),
        graph(make_balance_no(data),         "balance-no",  height="400px"),
    )


if __name__ == "__main__":
    app.run(debug=True, port=8050)
