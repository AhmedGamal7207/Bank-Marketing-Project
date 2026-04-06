import pandas as pd
from dash import Dash, html, dcc, callback, Output, Input

from tools.plot_helper import (
    PLOT_FONT_FAMILY,
    plot_pie, plot_grouped_bar, plot_grouped_box,
    plot_dual_axis_bar_line, plot_correlation_heatmap,
)

# ── Data ─────────────────────────────────────────────────────────────
df = pd.read_csv("dataset/processed_data.csv")
# plot_helper functions that check for "yes"/"no" need a string target
df["y"] = df["target"].map({1: "yes", 0: "no"})

ALL_JOBS = sorted(df["job"].dropna().unique())
ALL_EDUCATION = sorted(df["education"].dropna().unique())
ALL_MARITAL = sorted(df["marital"].dropna().unique())

# ── Reusable helpers ─────────────────────────────────────────────────

COLORS = {
    "bg": "#0f1117",
    "card": "#1a1d26",
    "accent": "#636efa",
    "green": "#2ecc71",
    "red": "#e74c3c",
    "text": "#e0e0e0",
    "muted": "#8c8c8c",
}

CARD_STYLE = {
    "backgroundColor": COLORS["card"],
    "borderRadius": "14px",
    "padding": "24px",
    "boxShadow": "0 4px 24px rgba(0,0,0,.35)",
}


def big_number_card(title, value, subtitle=None, color=COLORS["accent"]):
    return html.Div(style=CARD_STYLE, children=[
        html.P(title, style={"color": COLORS["muted"], "margin": "0 0 6px", "fontSize": "13px"}),
        html.H2(value, style={"color": color, "margin": "0", "fontSize": "36px", "fontWeight": "700"}),
        html.P(subtitle, style={"color": COLORS["muted"], "margin": "6px 0 0", "fontSize": "12px"}) if subtitle else None,
    ])


def section_title(text):
    return html.H3(text, style={"color": COLORS["text"], "margin": "32px 0 12px", "fontWeight": "600"})


def dark_theme(fig):
    """Apply dashboard dark theme to any plotly figure from plot_helper."""
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=PLOT_FONT_FAMILY, color=COLORS["text"]),
        margin=dict(l=40, r=40, t=50, b=40),
        title=dict(x=0.5, font=dict(size=15)),
        width=None, height=None,  # let Dash control sizing
    )
    return fig


def graph(fig, id_name):
    return dcc.Graph(id=id_name, figure=dark_theme(fig), config={"displayModeBar": False},
                     style={"height": "400px"})


def filter_df(jobs, educations, maritals):
    filtered = df.copy()
    if jobs:
        filtered = filtered[filtered["job"].isin(jobs)]
    if educations:
        filtered = filtered[filtered["education"].isin(educations)]
    if maritals:
        filtered = filtered[filtered["marital"].isin(maritals)]
    return filtered


# ── Chart builders — all use plot_helper functions ───────────────────

def make_subscription_pie(data):
    # Q1 — overall success rate
    return plot_pie(data["y"], title="Subscription Outcome", hole=0.5)


def make_job_bar(data):
    # Q16 — subscription rate by job
    return plot_grouped_bar(
        data["job"], data["y"],
        title="Job Type vs Subscription",
        x_label="Job", legend_label="Subscribed",
        horizontal=True,
    )


def make_month_dual(data):
    # Q7 + Q22 — monthly contacts vs subscription rate
    return plot_dual_axis_bar_line(
        data["month"], data["y"],
        title="Monthly Contacts vs Subscription Rate",
    )


def make_education_bar(data):
    # Q18 — subscription by education
    return plot_grouped_bar(
        data["education"], data["y"],
        title="Education Level vs Subscription",
        x_label="Education", legend_label="Subscribed",
    )


def make_poutcome_bar(data):
    # Q23 — previous outcome (strongest predictor)
    return plot_grouped_bar(
        data["poutcome"], data["y"],
        title="Previous Outcome vs Subscription (Strongest Predictor)",
        x_label="Previous Outcome", legend_label="Subscribed",
    )


def make_housing_bar(data):
    # Q20 — housing loan impact
    return plot_grouped_bar(
        data["housing"], data["y"],
        title="Housing Loan vs Subscription",
        x_label="Housing Loan", legend_label="Subscribed",
    )


def make_loan_bar(data):
    # Q25 — personal loan impact
    return plot_grouped_bar(
        data["loan"], data["y"],
        title="Personal Loan vs Subscription",
        x_label="Personal Loan", legend_label="Subscribed",
    )


def make_contact_bar(data):
    # Q21 — contact method
    return plot_grouped_bar(
        data["contact"], data["y"],
        title="Contact Method vs Subscription",
        x_label="Contact Method", legend_label="Subscribed",
    )


def make_age_box(data):
    # Q24 — age distribution by subscription
    return plot_grouped_box(
        data["age"], data["y"],
        title="Age: Subscribers vs Non-subscribers",
        x_label="Subscribed", y_label="Age",
    )


def make_balance_box(data):
    # Q26 — balance by subscription
    return plot_grouped_box(
        data["balance"], data["y"],
        title="Balance: Subscribers vs Non-subscribers",
        x_label="Subscribed", y_label="Balance (€)",
    )


def make_correlation(data):
    # Q31 — correlation heatmap
    return plot_correlation_heatmap(
        data, title="Feature Correlation Heatmap",
        target_column="y", target_map={"yes": 1, "no": 0},
    )


# ── Layout ───────────────────────────────────────────────────────────

app = Dash(__name__)
app.title = "Bank Marketing Dashboard"

dropdown_style = {
    "backgroundColor": COLORS["card"],
    "color": COLORS["text"],
    "border": "none",
    "borderRadius": "8px",
    "fontSize": "13px",
}

app.layout = html.Div(style={
    "backgroundColor": COLORS["bg"],
    "minHeight": "100vh",
    "padding": "28px 36px",
    "fontFamily": PLOT_FONT_FAMILY,
    "color": COLORS["text"],
}, children=[

    # Header
    html.Div(style={"textAlign": "center", "marginBottom": "28px"}, children=[
        html.H1("Bank Marketing Campaign", style={"margin": "0", "fontSize": "32px", "fontWeight": "700"}),
        html.P("Term Deposit Subscription Analysis — 45 211 client records",
               style={"color": COLORS["muted"], "margin": "4px 0 0"}),
    ]),

    # Filters
    html.Div(style={
        **CARD_STYLE,
        "display": "grid",
        "gridTemplateColumns": "1fr 1fr 1fr",
        "gap": "16px",
        "marginBottom": "24px",
    }, children=[
        html.Div([
            html.Label("Job Type", style={"fontSize": "12px", "color": COLORS["muted"]}),
            dcc.Dropdown(id="filter-job", options=ALL_JOBS, multi=True,
                         placeholder="All jobs", style=dropdown_style),
        ]),
        html.Div([
            html.Label("Education", style={"fontSize": "12px", "color": COLORS["muted"]}),
            dcc.Dropdown(id="filter-education", options=ALL_EDUCATION, multi=True,
                         placeholder="All education levels", style=dropdown_style),
        ]),
        html.Div([
            html.Label("Marital Status", style={"fontSize": "12px", "color": COLORS["muted"]}),
            dcc.Dropdown(id="filter-marital", options=ALL_MARITAL, multi=True,
                         placeholder="All statuses", style=dropdown_style),
        ]),
    ]),

    # KPI row
    html.Div(id="kpi-row", style={
        "display": "grid",
        "gridTemplateColumns": "repeat(5, 1fr)",
        "gap": "16px",
        "marginBottom": "24px",
    }),

    # Row 1: Pie + Previous Outcome
    section_title("Campaign Overview"),
    html.Div(style={"display": "grid", "gridTemplateColumns": "1fr 2fr", "gap": "16px", "marginBottom": "24px"}, children=[
        html.Div(id="chart-pie", style=CARD_STYLE),
        html.Div(id="chart-poutcome", style=CARD_STYLE),
    ]),

    # Row 2: Month dual + Job bar
    section_title("Who Subscribes?"),
    html.Div(style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "16px", "marginBottom": "24px"}, children=[
        html.Div(id="chart-month", style=CARD_STYLE),
        html.Div(id="chart-job", style=CARD_STYLE),
    ]),

    # Row 3: Education + Contact
    html.Div(style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "16px", "marginBottom": "24px"}, children=[
        html.Div(id="chart-education", style=CARD_STYLE),
        html.Div(id="chart-contact", style=CARD_STYLE),
    ]),

    # Row 4: Housing + Personal Loan
    html.Div(style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "16px", "marginBottom": "24px"}, children=[
        html.Div(id="chart-housing", style=CARD_STYLE),
        html.Div(id="chart-loan", style=CARD_STYLE),
    ]),

    # Row 5: Age + Balance boxes
    section_title("Financial & Demographic Patterns"),
    html.Div(style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "16px", "marginBottom": "24px"}, children=[
        html.Div(id="chart-age", style=CARD_STYLE),
        html.Div(id="chart-balance", style=CARD_STYLE),
    ]),

    # Row 6: Correlation heatmap
    section_title("Feature Correlations"),
    html.Div(id="chart-corr", style={**CARD_STYLE, "marginBottom": "24px"}),

    # Footer
    html.Div(style={"textAlign": "center", "padding": "20px 0 8px", "color": COLORS["muted"], "fontSize": "12px"}, children=[
        html.P("Abdallah Mohamed · Ahmed Gamal · Ahmed Saleem | AI — Intake 46, Alexandria"),
    ]),
])


# ── Callbacks ────────────────────────────────────────────────────────

@callback(
    Output("kpi-row", "children"),
    Output("chart-pie", "children"),
    Output("chart-poutcome", "children"),
    Output("chart-month", "children"),
    Output("chart-job", "children"),
    Output("chart-education", "children"),
    Output("chart-contact", "children"),
    Output("chart-housing", "children"),
    Output("chart-loan", "children"),
    Output("chart-age", "children"),
    Output("chart-balance", "children"),
    Output("chart-corr", "children"),
    Input("filter-job", "value"),
    Input("filter-education", "value"),
    Input("filter-marital", "value"),
)
def update_dashboard(jobs, educations, maritals):
    data = filter_df(jobs, educations, maritals)
    total = len(data)
    subs = int(data["target"].sum())
    rate = subs / total * 100 if total else 0
    avg_balance = data["balance"].mean()
    avg_age = data["age"].mean()

    kpis = [
        big_number_card("Total Clients", f"{total:,}", "filtered records"),
        big_number_card("Subscribers", f"{subs:,}", f"{rate:.1f}% conversion", COLORS["green"]),
        big_number_card("Non-Subscribers", f"{total - subs:,}", f"{100 - rate:.1f}%", COLORS["red"]),
        big_number_card("Avg Balance", f"€{avg_balance:,.0f}", "account balance"),
        big_number_card("Avg Age", f"{avg_age:.0f}", "years old"),
    ]

    return (
        kpis,
        graph(make_subscription_pie(data), "pie"),
        graph(make_poutcome_bar(data), "poutcome"),
        graph(make_month_dual(data), "month"),
        graph(make_job_bar(data), "job"),
        graph(make_education_bar(data), "education"),
        graph(make_contact_bar(data), "contact"),
        graph(make_housing_bar(data), "housing"),
        graph(make_loan_bar(data), "loan"),
        graph(make_age_box(data), "age"),
        graph(make_balance_box(data), "balance"),
        graph(make_correlation(data), "corr"),
    )


if __name__ == "__main__":
    app.run(debug=True, port=8050)
