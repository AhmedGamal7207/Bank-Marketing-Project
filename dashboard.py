import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from scipy.stats import gaussian_kde
from dash import Dash, html, dcc, callback, Output, Input, State, no_update
import re
import requests
import os
import google.generativeai as genai

from tools.plot_helper import MONTH_ORDER, plot_pie, plot_dual_axis_bar_line

# ── API config ───────────────────────────────────────────────────────
API_URL = "https://bank-marketing-project.onrender.com"

# ── Gemini LLM config ───────────────────────────────────────────────
GEMINI_KEY = os.environ.get("GEMINI_API_KEY", "")
genai.configure(api_key=GEMINI_KEY)
GEMINI_MODELS = ["gemini-2.5-flash", "gemini-2.0-flash", "gemini-2.0-flash-lite"]


def ask_gemini(prompt):
    """Try multiple Gemini models with fallback."""
    for model_name in GEMINI_MODELS:
        try:
            model = genai.GenerativeModel(model_name)
            resp = model.generate_content(prompt)
            return resp.text
        except Exception as e:
            if "429" in str(e) or "quota" in str(e).lower():
                continue
            raise
    raise Exception("All Gemini models are rate-limited. Please try again in a minute.")

SYSTEM_PROMPT = """You are a friendly and knowledgeable AI Banking Analyst assistant embedded in the Bank Marketing Intelligence Dashboard.

YOUR PERSONALITY:
- Warm, helpful, and approachable — like a smart colleague who loves data
- Start responses with a brief friendly acknowledgment
- Use clear structure: short intro, then bullet points or numbered lists for details
- Always back up claims with exact numbers and percentages from the data
- Keep answers focused (3-6 sentences for simple questions, more for complex ones)
- Use emojis sparingly to make responses feel engaging (1-2 per response max)
- If asked something outside scope, kindly suggest a related question you CAN answer

DATASET: UCI Bank Marketing — 45,211 clients from a Portuguese bank's telephone campaign.
Target: term deposit subscription (yes/no). Overall subscription rate: 11.7% (imbalanced dataset).

COLUMNS IN THE DATASET:
- age: client's age in years
- job: occupation type — admin, blue-collar, entrepreneur, housemaid, management, retired, self-employed, services, student, technician, unemployed, unknown
- marital: marital status — married, single, divorced
- education: education level — primary, secondary, tertiary, unknown
- default: has credit in default? (yes/no)
- balance: average yearly balance in euros
- housing: has a housing loan? (yes/no)
- loan: has a personal loan? (yes/no)
- contact: how they were contacted — cellular, telephone, unknown
- day: last contact day of the month
- month: last contact month (jan through dec)
- duration: last contact duration in seconds (note: this is known only AFTER the call, so it can't be used to predict before calling)
- campaign: number of contacts performed during this campaign for this client
- pdays: number of days since the client was last contacted from a previous campaign (-1 means they were never contacted before)
- previous: number of contacts performed before this campaign
- poutcome: outcome of the previous marketing campaign — success, failure, other, unknown
- y (target): did the client subscribe to a term deposit? (yes/no)

KEY FINDINGS FROM OUR ANALYSIS:

Strongest Predictors:
- Previous campaign success is the #1 predictor: clients who previously subscribed convert at 64.73% (vs 9.16% for unknown history)
- Students have 28.68% conversion, retirees 22.79%, while blue-collar workers only 6.93%

Demographics & Subscription:
- Single clients subscribe most (14.95%), followed by divorced (11.95%), then married (10.12%)
- Tertiary education: 15.01% conversion vs primary education: 8.63%
- Age follows a U-shape: both young (<30) and elderly (>60) subscribe more than middle-aged clients

Financial Factors:
- Clients WITHOUT housing loans subscribe at 16.70% vs only 7.70% with housing loans
- Clients WITHOUT personal loans: 12.66% vs 6.68% with personal loans
- Subscribers tend to have higher account balances on average

Campaign Insights:
- Cellular contact outperforms telephone
- High-conversion months: March, September, October, December — interestingly, these have FEWER calls but BETTER results
- May has the highest call volume but one of the lowest conversion rates
- Most clients are contacted 1-3 times during the campaign

FEATURE ENGINEERING WE PERFORMED:
- age_group: Young(<30), Adult(30-45), Senior(45-60), Elder(>60)
- balance_group: Negative/Zero, Low(1-1k), Medium(1k-5k), High(>5k)
- high_month: binary flag for high-conversion months (Mar, Sep, Oct, Dec)
- was_contacted_before: binary flag derived from pdays (-1 means never contacted)
- campaign: capped at 10 to handle outliers

OUR ML MODEL:
- We compared 5 models: Logistic Regression, Random Forest, Gradient Boosting, XGBoost, MLP Neural Network
- Winner: Gradient Boosting — achieved the highest Test AUC
- Preprocessing: StandardScaler for numeric features + OneHotEncoder for categorical features
- Validation: 5-fold StratifiedKFold cross-validation with GridSearchCV
- We tuned the decision threshold by maximizing F1 score (instead of using default 0.5)
- Most important features: previous contacts, was_contacted_before, balance, age, high_month, poutcome_success
- Model is deployed as a FastAPI REST API on Render with a /predict endpoint

OUR STRATEGIC RECOMMENDATIONS:
1. Re-engage previous subscribers — they convert at 64.73%
2. Focus on students and retirees with tailored campaigns
3. Reduce outreach to clients burdened with housing or personal loans
4. Shift campaign effort to high-converting months (Mar, Sep, Oct, Dec)
5. Prioritize cellular over telephone contact
6. Train call agents for longer, more engaging conversations

PROJECT TEAM: Abdallah Mohamed, Ahmed Gamal, Ahmed Saleem
ITI AI Track — Intake 46, Alexandria Branch
"""

# ── Data ──────────────────────────────────────────────────────────────
df = pd.read_csv("dataset/processed_data.csv")
df["y"] = df["target"].map({1: "yes", 0: "no"})

ALL_JOBS      = sorted(df["job"].dropna().unique())
ALL_EDUCATION = sorted(df["education"].dropna().unique())
ALL_MARITAL   = sorted(df["marital"].dropna().unique())
TOTAL_RECORDS = len(df)

# ── Design tokens ─────────────────────────────────────────────────────
FONT      = "Inter, Arial, sans-serif"
BG        = "#060B14"
SURFACE   = "#0B1121"
SURFACE2  = "#111827"
SURFACE3  = "#151D2E"
BORDER    = "rgba(255,255,255,0.06)"
ACCENT    = "#3B82F6"
ACCENT2   = "#8B5CF6"   # purple accent
ACCENT3   = "#06B6D4"   # cyan accent
TEXT      = "#F1F5F9"
MUTED     = "#64748B"
SUCCESS   = "#22C55E"
WARNING   = "#F59E0B"
DANGER    = "#EF4444"

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
    suppress_callback_exceptions=True,
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
            options=[{"label": " " + v.replace("-", "\u2011").title(), "value": v}
                     for v in options],
            value=[],
            inline=True,
            inputStyle={"position": "absolute", "opacity": "0", "width": "0", "height": "0"},
        ),
    ])


# ── Tab button helper ────────────────────────────────────────────────

def tab_button(label, tab_id, icon, color):
    return html.Button(
        id=f"tab-btn-{tab_id}",
        n_clicks=0,
        className="tab-btn",
        children=[
            html.Span(icon, style={"fontSize": "16px"}),
            html.Span(label, style={"fontWeight": "600", "fontSize": "13px"}),
        ],
        style={
            "display": "flex", "alignItems": "center", "gap": "8px",
            "padding": "10px 22px", "border": f"1px solid {BORDER}",
            "borderRadius": "12px", "cursor": "pointer",
            "backgroundColor": SURFACE, "color": MUTED,
            "fontFamily": FONT, "transition": "all 0.25s ease",
        },
    )


# ── Prediction form helpers ──────────────────────────────────────────

def form_dropdown(id, label, options, default=None):
    return html.Div(style={"flex": "1", "minWidth": "180px"}, children=[
        html.Label(label, style={
            "color": MUTED, "fontSize": "10px", "fontWeight": "700",
            "letterSpacing": "1.2px", "textTransform": "uppercase",
            "marginBottom": "6px", "display": "block",
        }),
        dcc.Dropdown(
            id=id,
            options=[{"label": v.replace("-", " ").title(), "value": v} for v in options],
            value=default or options[0],
            clearable=False,
            className="dark-dropdown",
        ),
    ])


def form_number(id, label, default, min_val=None, max_val=None):
    return html.Div(style={"flex": "1", "minWidth": "140px"}, children=[
        html.Label(label, style={
            "color": MUTED, "fontSize": "10px", "fontWeight": "700",
            "letterSpacing": "1.2px", "textTransform": "uppercase",
            "marginBottom": "6px", "display": "block",
        }),
        dcc.Input(
            id=id, type="number", value=default,
            min=min_val, max=max_val,
            style={
                "width": "100%", "padding": "10px 14px",
                "backgroundColor": SURFACE2, "border": f"1px solid {BORDER}",
                "borderRadius": "10px", "color": TEXT, "fontFamily": FONT,
                "fontSize": "14px", "outline": "none",
            },
        ),
    ])


# ── Analytics tab content ────────────────────────────────────────────

def analytics_tab():
    return html.Div([
        # ── Filters
        html.Div(style={**CARD, "marginBottom": "28px"}, children=[
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
                    html.Div(id="filter-status", style={
                        "background": f"rgba(59,130,246,0.10)",
                        "border": f"1px solid rgba(59,130,246,0.22)",
                        "color": ACCENT, "borderRadius": "999px",
                        "padding": "3px 12px", "fontSize": "11px", "fontWeight": "600",
                    }),
                ]),
                html.Button("Reset All", id="btn-reset", n_clicks=0, className="btn-reset"),
            ]),
            filter_pill_row("filter-marital",   "MARITAL STATUS", ALL_MARITAL,   accent=ACCENT2),
            filter_pill_row("filter-education", "EDUCATION",      ALL_EDUCATION, accent=ACCENT3),
            html.Div(style={"marginBottom": 0},
                     children=filter_pill_row("filter-job", "JOB TYPE", ALL_JOBS, accent=ACCENT).children),
        ]),

        # ── KPIs
        html.Div(id="kpi-row", style={
            "display": "grid", "gridTemplateColumns": "repeat(5, 1fr)",
            "gap": "16px", "marginBottom": "8px",
        }),

        # ── Overview
        section_title("Campaign Overview"),
        html.Div(style={"display": "grid", "gridTemplateColumns": "1fr 2fr",
                        "gap": "18px", "marginBottom": "18px"}, children=[
            chart_card(html.Div(id="chart-pie"),      height="380px"),
            chart_card(html.Div(id="chart-poutcome"), height="380px"),
        ]),

        # ── Who Subscribes?
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

        # ── Demographics
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
    ])


# ── Prediction tab content ───────────────────────────────────────────

def prediction_tab():
    return html.Div([
        # Header
        html.Div(style={**CARD, "marginBottom": "24px",
                        "background": f"linear-gradient(135deg, {SURFACE} 0%, rgba(59,130,246,0.08) 100%)"}, children=[
            html.Div(style={"display": "flex", "alignItems": "center", "gap": "12px", "marginBottom": "8px"}, children=[
                html.Span(style={"fontSize": "24px"}, children="\U0001F3AF"),
                html.H3("Client Subscription Predictor", style={
                    "color": TEXT, "margin": 0, "fontSize": "18px", "fontWeight": "700",
                }),
            ]),
            html.P("Enter client details below to predict whether they will subscribe to a term deposit.",
                   style={"color": MUTED, "margin": 0, "fontSize": "13px"}),
        ]),

        # Form
        html.Div(style={**CARD, "marginBottom": "24px"}, children=[
            section_title("Client Demographics", color=ACCENT2),
            html.Div(style={"display": "flex", "flexWrap": "wrap", "gap": "16px", "marginBottom": "24px"}, children=[
                form_number("pred-age", "Age", 35, 18, 100),
                form_dropdown("pred-job", "Job Type", ALL_JOBS, "management"),
                form_dropdown("pred-marital", "Marital Status", ALL_MARITAL, "married"),
                form_dropdown("pred-education", "Education", ALL_EDUCATION, "tertiary"),
            ]),

            section_title("Financial Information", color=ACCENT3),
            html.Div(style={"display": "flex", "flexWrap": "wrap", "gap": "16px", "marginBottom": "24px"}, children=[
                form_number("pred-balance", "Balance (EUR)", 1500),
                form_dropdown("pred-housing", "Housing Loan", ["yes", "no"], "no"),
                form_dropdown("pred-loan", "Personal Loan", ["yes", "no"], "no"),
            ]),

            section_title("Campaign Details", color=WARNING),
            html.Div(style={"display": "flex", "flexWrap": "wrap", "gap": "16px", "marginBottom": "24px"}, children=[
                form_dropdown("pred-contact", "Contact Method", ["cellular", "telephone", "unknown"], "cellular"),
                form_dropdown("pred-month", "Month", MONTH_ORDER, "mar"),
                form_number("pred-campaign", "Campaign Contacts", 1, 1, 50),
                form_number("pred-previous", "Previous Contacts", 0, 0, 50),
                form_dropdown("pred-poutcome", "Previous Outcome", ["unknown", "success", "failure", "other"], "unknown"),
                form_number("pred-pdays", "Days Since Last Contact", -1),
            ]),

            # Submit button
            html.Div(style={"display": "flex", "justifyContent": "center", "marginTop": "8px"}, children=[
                html.Button("Predict Subscription", id="btn-predict", n_clicks=0, style={
                    "padding": "14px 48px", "fontSize": "15px", "fontWeight": "700",
                    "fontFamily": FONT, "color": "#fff", "border": "none", "cursor": "pointer",
                    "borderRadius": "14px",
                    "background": f"linear-gradient(135deg, {ACCENT}, {ACCENT2})",
                    "boxShadow": f"0 4px 20px rgba(59,130,246,0.35)",
                    "transition": "all 0.3s ease", "letterSpacing": "0.5px",
                }),
            ]),
        ]),

        # Result area
        html.Div(id="prediction-result", style={"marginBottom": "24px"}),
    ])


# ── AI Assistant tab content ─────────────────────────────────────────

def assistant_tab():
    return html.Div([
        # Header
        html.Div(style={**CARD, "marginBottom": "24px",
                        "background": f"linear-gradient(135deg, {SURFACE} 0%, rgba(139,92,246,0.08) 100%)"}, children=[
            html.Div(style={"display": "flex", "alignItems": "center", "gap": "12px", "marginBottom": "8px"}, children=[
                html.Span(style={"fontSize": "24px"}, children="\U0001F916"),
                html.H3("AI Banking Analyst", style={
                    "color": TEXT, "margin": 0, "fontSize": "18px", "fontWeight": "700",
                }),
            ]),
            html.P("Ask questions about the Bank Marketing dataset, EDA insights, model performance, or campaign strategy.",
                   style={"color": MUTED, "margin": 0, "fontSize": "13px"}),
        ]),

        # Chat area
        html.Div(id="chat-history", style={
            **CARD, "minHeight": "300px", "maxHeight": "500px",
            "overflowY": "auto", "marginBottom": "16px",
        }, children=[
            html.Div(style={
                "display": "flex", "alignItems": "center", "gap": "10px",
                "padding": "14px 18px", "borderRadius": "12px",
                "background": f"rgba(139,92,246,0.08)", "border": f"1px solid rgba(139,92,246,0.15)",
            }, children=[
                html.Span("\U0001F916", style={"fontSize": "16px"}),
                html.P("Hello! I'm your AI Banking Analyst. Ask me anything about the dataset, EDA findings, or model results.",
                       style={"color": TEXT, "margin": 0, "fontSize": "13px"}),
            ]),
        ]),

        # Input
        html.Div(style={
            "display": "flex", "gap": "12px", "alignItems": "center",
        }, children=[
            dcc.Input(
                id="chat-input", type="text",
                placeholder="Ask about campaign performance, client segments, model accuracy...",
                className="chat-text-input",
                debounce=True,
            ),
            html.Button("Send", id="btn-chat", n_clicks=0, style={
                "padding": "16px 32px", "fontSize": "15px", "fontWeight": "700",
                "fontFamily": FONT, "color": "#fff", "border": "none", "cursor": "pointer",
                "borderRadius": "14px", "flexShrink": "0",
                "background": f"linear-gradient(135deg, {ACCENT2}, {ACCENT3})",
                "boxShadow": "0 4px 20px rgba(139,92,246,0.30)",
            }),
        ]),
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
        "background": f"linear-gradient(90deg, {ACCENT}, {ACCENT2} 50%, {ACCENT3})",
    }),

    # ── Header + Navigation
    html.Div(style={"padding": "28px 40px 0"}, children=[
        # Title row
        html.Div(style={
            "display": "flex", "alignItems": "center", "justifyContent": "space-between",
            "marginBottom": "24px",
        }, children=[
            html.Div(style={"display": "flex", "alignItems": "center", "gap": "14px"}, children=[
                html.Div(style={
                    "width": "40px", "height": "40px", "borderRadius": "12px",
                    "background": f"linear-gradient(135deg, {ACCENT}, {ACCENT2})",
                    "display": "flex", "alignItems": "center", "justifyContent": "center",
                    "fontSize": "20px", "boxShadow": f"0 4px 15px rgba(59,130,246,0.3)",
                }, children="\U0001F3E6"),
                html.Div([
                    html.H1("Bank Marketing Intelligence", style={
                        "color": TEXT, "margin": 0, "fontSize": "22px",
                        "fontWeight": "800", "letterSpacing": "-0.5px",
                    }),
                    html.P("Campaign Analytics & Prediction Platform", style={
                        "color": MUTED, "margin": 0, "fontSize": "12px",
                        "fontWeight": "500", "letterSpacing": "0.3px",
                    }),
                ]),
            ]),
            html.Div(style={
                "background": f"rgba(34,197,94,0.10)", "border": f"1px solid rgba(34,197,94,0.22)",
                "color": SUCCESS, "borderRadius": "999px",
                "padding": "5px 14px", "fontSize": "11px", "fontWeight": "600",
            }, children="45,211 records"),
        ]),

        # Tab buttons
        html.Div(style={
            "display": "flex", "gap": "10px", "marginBottom": "28px",
        }, children=[
            tab_button("Analytics",  "analytics",  "\U0001F4CA", ACCENT),
            tab_button("Prediction", "prediction", "\U0001F3AF", ACCENT2),
            tab_button("AI Assistant", "assistant", "\U0001F916", ACCENT3),
        ]),
    ]),

    # ── Hidden store for active tab
    dcc.Store(id="active-tab", data="analytics"),

    # ── Tab content area
    html.Div(id="tab-content", style={"padding": "0 40px 32px"}),

    # ── Footer
    html.Div(style={
        "borderTop": f"1px solid {BORDER}",
        "margin": "0 40px", "padding": "20px 0 28px",
        "display": "flex", "justifyContent": "space-between", "alignItems": "center",
    }, children=[
        html.P("Abdallah Mohamed  |  Ahmed Gamal  |  Ahmed Saleem",
               style={"color": MUTED, "margin": 0, "fontSize": "12px", "fontWeight": "500"}),
        html.P("AI Track — Intake 46 — Alexandria Branch",
               style={"color": MUTED, "margin": 0, "fontSize": "12px"}),
    ]),
])


# ── Callbacks ─────────────────────────────────────────────────────────

# ── Tab switching ────────────────────────────────────────────────────

@callback(
    Output("active-tab", "data"),
    Input("tab-btn-analytics",  "n_clicks"),
    Input("tab-btn-prediction", "n_clicks"),
    Input("tab-btn-assistant",  "n_clicks"),
    prevent_initial_call=True,
)
def switch_tab(n1, n2, n3):
    from dash import ctx
    btn = ctx.triggered_id
    if btn == "tab-btn-prediction": return "prediction"
    if btn == "tab-btn-assistant":  return "assistant"
    return "analytics"


@callback(
    Output("tab-content", "children"),
    Output("tab-btn-analytics",  "style"),
    Output("tab-btn-prediction", "style"),
    Output("tab-btn-assistant",  "style"),
    Input("active-tab", "data"),
)
def render_tab(tab):
    base = {
        "display": "flex", "alignItems": "center", "gap": "8px",
        "padding": "10px 22px", "border": f"1px solid {BORDER}",
        "borderRadius": "12px", "cursor": "pointer",
        "fontFamily": FONT, "transition": "all 0.25s ease",
    }
    inactive = {**base, "backgroundColor": SURFACE, "color": MUTED}

    colors = {"analytics": ACCENT, "prediction": ACCENT2, "assistant": ACCENT3}
    styles = []
    for t in ["analytics", "prediction", "assistant"]:
        if t == tab:
            styles.append({**base, "backgroundColor": f"{colors[t]}18",
                           "color": colors[t], "border": f"1px solid {colors[t]}44"})
        else:
            styles.append(inactive)

    tabs = {"analytics": analytics_tab, "prediction": prediction_tab, "assistant": assistant_tab}
    return tabs.get(tab, analytics_tab)(), *styles


# ── Analytics filters ────────────────────────────────────────────────

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
    Output("filter-status",    "children"),
    Output("kpi-row",          "children"),
    Output("chart-pie",        "children"),
    Output("chart-poutcome",   "children"),
    Output("chart-month",      "children"),
    Output("chart-job",        "children"),
    Output("chart-education",  "children"),
    Output("chart-contact",    "children"),
    Output("chart-housing",    "children"),
    Output("chart-loan",       "children"),
    Output("chart-age-yes",    "children"),
    Output("chart-age-no",     "children"),
    Output("chart-balance-yes","children"),
    Output("chart-balance-no", "children"),
    Input("filter-job",        "value"),
    Input("filter-education",  "value"),
    Input("filter-marital",    "value"),
)
def update_dashboard(jobs, educations, maritals):
    data  = filter_df(jobs, educations, maritals)
    total = len(data)
    subs  = int(data["target"].sum())
    avg_b = data["balance"].mean()
    avg_a = data["age"].mean()
    rate  = subs / total * 100 if total > 0 else 0

    kpis = [
        kpi_card("Total Clients",      f"{total:,}",         ACCENT),
        kpi_card("Subscribers",        f"{subs:,}",          YES_COLOR),
        kpi_card("Non-Subscribers",    f"{total - subs:,}",  NO_COLOR),
        kpi_card("Avg Balance",        f"\u20AC{avg_b:,.0f}", ACCENT2),
        kpi_card("Subscription Rate",  f"{rate:.1f}%",       ACCENT3),
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


# ── Prediction callback ─────────────────────────────────────────────

@callback(
    Output("prediction-result", "children"),
    Input("btn-predict", "n_clicks"),
    State("pred-age",      "value"),
    State("pred-job",      "value"),
    State("pred-marital",  "value"),
    State("pred-education","value"),
    State("pred-balance",  "value"),
    State("pred-housing",  "value"),
    State("pred-loan",     "value"),
    State("pred-contact",  "value"),
    State("pred-month",    "value"),
    State("pred-campaign", "value"),
    State("pred-previous", "value"),
    State("pred-poutcome", "value"),
    State("pred-pdays",    "value"),
    prevent_initial_call=True,
)
def run_prediction(n, age, job, marital, education, balance,
                   housing, loan, contact, month, campaign,
                   previous, poutcome, pdays):
    if not n:
        return no_update

    # Handle None/empty values with safe defaults
    age = int(age) if age is not None else 35
    balance = int(balance) if balance is not None else 0
    campaign = int(campaign) if campaign is not None else 1
    previous = int(previous) if previous is not None else 0
    pdays = int(pdays) if pdays is not None else -1

    payload = {
        "age": age, "job": job or "unknown", "marital": marital or "single",
        "education": education or "unknown",
        "balance": balance, "housing": housing or "no", "loan": loan or "no",
        "contact": contact or "cellular", "month": month or "jan",
        "campaign": campaign, "previous": previous,
        "poutcome": poutcome or "unknown", "pdays": pdays,
    }

    try:
        resp = requests.post(f"{API_URL}/predict", json=payload, timeout=30)
        resp.raise_for_status()
        result = resp.json()
    except requests.exceptions.ConnectionError:
        return html.Div(style={**CARD, "borderLeft": f"4px solid {WARNING}"}, children=[
            html.P("API is starting up... Please wait 30-60 seconds and try again.",
                   style={"color": WARNING, "margin": 0, "fontSize": "14px"}),
            html.P("(Render free tier spins down after inactivity)",
                   style={"color": MUTED, "margin": "8px 0 0", "fontSize": "12px"}),
        ])
    except Exception as e:
        return html.Div(style={**CARD, "borderLeft": f"4px solid {DANGER}"}, children=[
            html.P(f"Error: {str(e)}", style={"color": DANGER, "margin": 0, "fontSize": "14px"}),
        ])

    pred = result["prediction"]
    prob = result["probability"]
    label = result["label"]
    is_yes = pred == 1
    color = SUCCESS if is_yes else DANGER
    icon = "\u2705" if is_yes else "\u274C"
    pct = prob * 100 if is_yes else (1 - prob) * 100

    return html.Div(style={
        **CARD, "borderLeft": f"4px solid {color}",
        "background": f"linear-gradient(135deg, {SURFACE} 0%, {color}08 100%)",
    }, children=[
        html.Div(style={"display": "flex", "alignItems": "center", "gap": "16px",
                        "marginBottom": "16px"}, children=[
            html.Span(icon, style={"fontSize": "36px"}),
            html.Div([
                html.H3(label, style={"color": color, "margin": 0, "fontSize": "20px", "fontWeight": "700"}),
                html.P(f"Confidence: {pct:.1f}%", style={"color": MUTED, "margin": "4px 0 0", "fontSize": "13px"}),
            ]),
        ]),
        # Probability bar
        html.Div(style={"marginTop": "8px"}, children=[
            html.Div(style={
                "display": "flex", "justifyContent": "space-between",
                "marginBottom": "6px",
            }, children=[
                html.Span("Not Subscribe", style={"color": DANGER, "fontSize": "11px", "fontWeight": "600"}),
                html.Span("Subscribe", style={"color": SUCCESS, "fontSize": "11px", "fontWeight": "600"}),
            ]),
            html.Div(style={
                "width": "100%", "height": "10px", "borderRadius": "5px",
                "backgroundColor": f"{DANGER}30",
            }, children=[
                html.Div(style={
                    "width": f"{prob * 100:.1f}%", "height": "100%", "borderRadius": "5px",
                    "background": f"linear-gradient(90deg, {DANGER}, {WARNING}, {SUCCESS})" if prob > 0.5
                               else f"linear-gradient(90deg, {DANGER}, {WARNING})",
                    "transition": "width 0.5s ease",
                }),
            ]),
            html.P(f"Probability of subscribing: {prob * 100:.2f}%",
                   style={"color": MUTED, "margin": "8px 0 0", "fontSize": "12px", "textAlign": "center"}),
        ]),
    ])


# ── Chat callback (Gemini Flash) ─────────────────────────────────────

@callback(
    Output("chat-history", "children"),
    Output("chat-input",   "value"),
    Input("btn-chat",   "n_clicks"),
    Input("chat-input", "n_submit"),
    State("chat-input",   "value"),
    State("chat-history", "children"),
    prevent_initial_call=True,
)
def chat_send(n_click, n_submit, message, history):
    if not message or not message.strip():
        return no_update, no_update

    history = history or []

    # User message bubble
    history.append(html.Div(style={
        "display": "flex", "justifyContent": "flex-end", "marginTop": "12px",
    }, children=[
        html.Div(style={
            "padding": "12px 18px", "borderRadius": "14px 14px 4px 14px",
            "background": "rgba(59,130,246,0.15)", "border": "1px solid rgba(59,130,246,0.2)",
            "maxWidth": "70%",
        }, children=[
            html.P(message, style={"color": TEXT, "margin": 0, "fontSize": "13px"}),
        ]),
    ]))

    # Call Gemini with fallback
    try:
        bot_text = ask_gemini(SYSTEM_PROMPT + "\n\nUser question: " + message)
        is_error = False
    except Exception as e:
        bot_text = str(e)[:150]
        is_error = True

    # ── Markdown to Dash HTML parser ────────────────────────────
    def md_inline(text):
        """Convert inline markdown (**bold**, *italic*) to html spans."""
        parts = []
        # Split by **bold** first
        bold_split = re.split(r'\*\*(.+?)\*\*', text)
        for i, chunk in enumerate(bold_split):
            if i % 2 == 1:
                # Bold chunk
                parts.append(html.Span(chunk, style={
                    "fontWeight": "700", "color": "#93C5FD",
                }))
            else:
                # Check for *italic* inside non-bold
                italic_split = re.split(r'\*(.+?)\*', chunk)
                for j, sub in enumerate(italic_split):
                    if j % 2 == 1:
                        parts.append(html.Span(sub, style={
                            "fontStyle": "italic", "color": "#94A3B8",
                        }))
                    elif sub:
                        parts.append(sub)
        return parts if parts else [text]

    def md_line(line, base_style):
        """Convert a markdown line to a styled html.P with inline formatting."""
        return html.P(md_inline(line), style=base_style)

    # Bot response — parse markdown lines
    response_children = []
    if is_error:
        response_children.append(
            html.P(["Oops! " + bot_text], style={
                "color": WARNING, "margin": 0, "fontSize": "13px",
            })
        )
        response_children.append(
            html.P("Please try again in a moment.", style={
                "color": MUTED, "margin": "6px 0 0", "fontSize": "12px",
            })
        )
    else:
        for p in bot_text.split("\n"):
            line = p.strip()
            if not line:
                continue

            # Headings (### or ## or lines that are just **bold**)
            heading_match = re.match(r'^#{1,3}\s+(.+)$', line)
            bold_heading = re.match(r'^\*\*([^*]+)\*\*:?\s*$', line)
            if heading_match or bold_heading:
                clean = (heading_match.group(1) if heading_match
                         else bold_heading.group(1))
                response_children.append(
                    html.P(clean.strip(":"), style={
                        "color": ACCENT3, "margin": "12px 0 4px", "fontSize": "14px",
                        "fontWeight": "700", "letterSpacing": "0.2px",
                    })
                )
            # Bullet points (* or -)
            elif re.match(r'^[\*\-]\s+', line):
                content = re.sub(r'^[\*\-]\s+', '', line)
                response_children.append(html.Div(
                    style={"display": "flex", "gap": "8px", "margin": "4px 0 4px 4px"},
                    children=[
                        html.Span("\u2022", style={
                            "color": ACCENT, "fontSize": "14px",
                            "lineHeight": "1.6", "flexShrink": "0",
                        }),
                        html.Span(md_inline(content), style={
                            "color": TEXT, "fontSize": "13px", "lineHeight": "1.6",
                        }),
                    ],
                ))
            # Numbered items
            elif re.match(r'^\d+[\.\)]\s+', line):
                num_match = re.match(r'^(\d+[\.\)])\s+(.+)$', line)
                num_label = num_match.group(1) if num_match else ""
                content = num_match.group(2) if num_match else line
                response_children.append(html.Div(
                    style={"display": "flex", "gap": "8px", "margin": "4px 0 4px 4px"},
                    children=[
                        html.Span(num_label, style={
                            "color": ACCENT, "fontSize": "13px", "fontWeight": "700",
                            "lineHeight": "1.6", "flexShrink": "0",
                        }),
                        html.Span(md_inline(content), style={
                            "color": TEXT, "fontSize": "13px", "lineHeight": "1.6",
                        }),
                    ],
                ))
            # Regular paragraph
            else:
                response_children.append(md_line(line, {
                    "color": TEXT, "margin": "0 0 6px",
                    "fontSize": "13px", "lineHeight": "1.7",
                }))

    bubble_bg = "rgba(239,68,68,0.08)" if is_error else "rgba(139,92,246,0.08)"
    bubble_border = "rgba(239,68,68,0.2)" if is_error else "rgba(139,92,246,0.15)"

    history.append(html.Div(style={
        "display": "flex", "alignItems": "flex-start", "gap": "10px", "marginTop": "12px",
    }, children=[
        html.Span("\U0001F916", style={"fontSize": "16px", "marginTop": "2px"}),
        html.Div(style={
            "padding": "14px 18px", "borderRadius": "14px 14px 14px 4px",
            "background": bubble_bg, "border": f"1px solid {bubble_border}",
            "maxWidth": "75%",
        }, children=response_children),
    ]))

    return history, ""


server = app.server  # expose Flask server for gunicorn

if __name__ == "__main__":
    app.run(debug=True, port=8050)
