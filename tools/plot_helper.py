# Data Dealers
import numpy as np
import pandas as pd


# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import gaussian_kde


PLOTLY_TEMPLATE = "plotly_white"
PLOT_FONT_FAMILY = "Arial"
PLOT_COLOR_SEQUENCE = px.colors.qualitative.Safe


sns.set_theme(style="whitegrid", palette="Set2")


def plot_pie(
    column,
    title=None,
    x_label=None,
    y_label="Count",
    hole=0,
    show_percent=True,
    hover_data=None,
):
    """Create a pie or donut chart from a pandas Series.

    Parameters:
    column: pandas Series to plot.
    title: Chart title.
    x_label: Label for the category names in the legend and hover.
    y_label: Label for the counted values.
    hole: Set greater than 0 to create a donut chart.
    show_percent: Show percentages on chart slices.
    hover_data: Extra hover fields to pass to Plotly.
    """
    pie_df = (
        column.value_counts(dropna=False)
        .rename_axis(x_label or column.name or "Category")
        .reset_index(name=y_label)
    )

    fig = px.pie(
        pie_df,
        names=x_label or column.name or "Category",
        values=y_label,
        title=title or f"Distribution of {column.name or 'Category'}",
        hole=hole,
        color_discrete_sequence=PLOT_COLOR_SEQUENCE,
        hover_data=hover_data,
    )

    fig.update_traces(
        textinfo="percent+label" if show_percent else "label",
        textfont=dict(family=PLOT_FONT_FAMILY, size=13),
    )

    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        font=dict(family=PLOT_FONT_FAMILY, size=13),
        title=dict(x=0.5),
        legend_title_text=x_label or column.name or "Category",
    )

    return fig


def plot_distribution(
    column,
    title=None,
    x_label=None,
    y_label="Count",
    bins=None,
    marginal=None,
):
    """Create a distribution plot from a numeric pandas Series.

    Parameters:
    column: pandas Series to plot.
    title: Chart title.
    x_label: Label for the x-axis.
    y_label: Label for the y-axis.
    bins: Number of histogram bins.
    marginal: Optional marginal plot such as "box" or "violin".
    """
    clean_column = column.dropna()
    mean_value = clean_column.mean()
    median_value = clean_column.median()
    bin_edges = np.histogram_bin_edges(clean_column, bins=bins or "auto")
    bin_width = np.mean(np.diff(bin_edges))

    fig = px.histogram(
        x=clean_column,
        nbins=bins,
        marginal=marginal,
        title=title or f"Distribution of {column.name or 'Value'}",
        color_discrete_sequence=[PLOT_COLOR_SEQUENCE[0]],
    )

    kde_x = np.linspace(clean_column.min(), clean_column.max(), 500)
    kde_y = gaussian_kde(clean_column)(kde_x) * len(clean_column) * bin_width

    fig.add_trace(
        go.Scatter(
            x=kde_x,
            y=kde_y,
            mode="lines",
            name="KDE",
            line=dict(color=PLOT_COLOR_SEQUENCE[1], width=3),
        )
    )

    fig.add_shape(
        type="line",
        x0=mean_value,
        x1=mean_value,
        y0=0,
        y1=1,
        xref="x",
        yref="y domain",
        line=dict(color="#D62728", dash="dash", width=2),
    )

    fig.add_shape(
        type="line",
        x0=median_value,
        x1=median_value,
        y0=0,
        y1=1,
        xref="x",
        yref="y domain",
        line=dict(color="#2CA02C", dash="dot", width=2),
    )

    fig.add_annotation(
        x=mean_value,
        y=1.02,
        xref="x",
        yref="y domain",
        text=f"Mean: {mean_value:.2f}",
        showarrow=False,
        xanchor="left",
        xshift=10,
        font=dict(color="#D62728", family=PLOT_FONT_FAMILY, size=12),
    )

    fig.add_annotation(
        x=median_value,
        y=0.94,
        xref="x",
        yref="y domain",
        text=f"Median: {median_value:.2f}",
        showarrow=False,
        xanchor="right",
        xshift=-10,
        font=dict(color="#2CA02C", family=PLOT_FONT_FAMILY, size=12),
    )

    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        font=dict(family=PLOT_FONT_FAMILY, size=13),
        title=dict(x=0.5),
        xaxis_title=x_label or column.name or "Value",
        yaxis_title=y_label,
        bargap=0.08,
    )

    return fig


def plot_bar(
    column,
    title=None,
    x_label=None,
    y_label="Count",
    sort_by="values",
    ascending=False,
    horizontal=False,
):
    """Create a bar plot from a pandas Series.

    Parameters:
    column: pandas Series to plot.
    title: Chart title.
    x_label: Label for the category axis.
    y_label: Label for the count axis.
    sort_by: Sort by "values" or "field".
    ascending: Sort order.
    horizontal: Show horizontal bars if True.
    """
    bar_df = (
        column.value_counts(dropna=False)
        .rename_axis(column.name or "Category")
        .reset_index(name=y_label)
    )

    category_label = x_label or column.name or "Category"
    bar_df = bar_df.rename(columns={bar_df.columns[0]: category_label})

    if sort_by == "field":
        bar_df = bar_df.sort_values(by=category_label, ascending=ascending)
    else:
        bar_df = bar_df.sort_values(by=y_label, ascending=ascending)

    if horizontal:
        fig = px.bar(
            bar_df,
            x=y_label,
            y=category_label,
            orientation="h",
            title=title or f"Distribution of {column.name or 'Category'}",
            color_discrete_sequence=[PLOT_COLOR_SEQUENCE[0]],
        )
        fig.update_layout(
            xaxis_title=y_label,
            yaxis_title=category_label,
        )
    else:
        fig = px.bar(
            bar_df,
            x=category_label,
            y=y_label,
            title=title or f"Distribution of {column.name or 'Category'}",
            color_discrete_sequence=[PLOT_COLOR_SEQUENCE[0]],
        )
        fig.update_layout(
            xaxis_title=category_label,
            yaxis_title=y_label,
        )

    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        font=dict(family=PLOT_FONT_FAMILY, size=13),
        title=dict(x=0.5),
        showlegend=False,
    )

    return fig
