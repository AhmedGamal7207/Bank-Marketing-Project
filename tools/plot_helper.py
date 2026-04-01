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
MPL_COLOR_SEQUENCE = sns.color_palette("Set2")


sns.set_theme(style="whitegrid", palette="Set2")
MONTH_ORDER = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]


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
    category_count = pie_df.shape[0]
    size = max(500, min(900, 420 + category_count * 35))

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
        width=size,
        height=size,
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

    category_count = bar_df.shape[0]
    width = max(700, min(1600, 450 + category_count * 70))
    height = max(500, min(1400, 300 + category_count * 35)) if horizontal else 500

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
        width=width,
        height=height,
    )

    return fig


def plot_grouped_bar(
    x,
    group,
    title=None,
    x_label=None,
    y_label="Count",
    legend_label=None,
    sort_by="values",
    ascending=False,
    horizontal=False,
):
    """Create a grouped bar plot from two pandas Series.

    Parameters:
    x: pandas Series for the main categories.
    group: pandas Series for the grouped categories.
    title: Chart title.
    x_label: Label for the main category axis.
    y_label: Label for the count axis.
    legend_label: Label for the legend groups.
    sort_by: Sort by "values" or "field".
    ascending: Sort order.
    horizontal: Show horizontal bars if True.
    """
    plot_df = pd.DataFrame(
        {
            x_label or x.name or "Category": x,
            legend_label or group.name or "Group": group,
        }
    ).dropna()

    category_label = plot_df.columns[0]
    group_label = plot_df.columns[1]

    grouped_df = (
        plot_df.groupby([category_label, group_label], dropna=False)
        .size()
        .reset_index(name=y_label)
    )

    if sort_by == "field":
        category_order = sorted(grouped_df[category_label].unique(), reverse=not ascending)
    else:
        category_order = (
            grouped_df.groupby(category_label)[y_label]
            .sum()
            .sort_values(ascending=ascending)
            .index
            .tolist()
        )

    group_order = sorted(grouped_df[group_label].unique())
    width = max(700, len(category_order) * max(90, len(group_order) * 45))
    height = max(500, len(category_order) * 35) if horizontal else 500

    if horizontal:
        fig = px.bar(
            grouped_df,
            x=y_label,
            y=category_label,
            color=group_label,
            orientation="h",
            barmode="group",
            category_orders={category_label: category_order, group_label: group_order},
            title=title or f"{x.name or 'Category'} by {group.name or 'Group'}",
            color_discrete_sequence=PLOT_COLOR_SEQUENCE,
        )
        fig.update_layout(
            xaxis_title=y_label,
            yaxis_title=category_label,
        )
    else:
        fig = px.bar(
            grouped_df,
            x=category_label,
            y=y_label,
            color=group_label,
            barmode="group",
            category_orders={category_label: category_order, group_label: group_order},
            title=title or f"{x.name or 'Category'} by {group.name or 'Group'}",
            color_discrete_sequence=PLOT_COLOR_SEQUENCE,
        )
        fig.update_layout(
            xaxis_title=category_label,
            yaxis_title=y_label,
        )

    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        font=dict(family=PLOT_FONT_FAMILY, size=13),
        title=dict(x=0.5),
        legend_title_text=group_label,
        width=width,
        height=height,
    )

    return fig


def plot_grouped_box(
    values,
    group,
    title=None,
    x_label=None,
    y_label=None,
    sort_by="field",
    ascending=True,
    horizontal=False,
    points="outliers",
):
    """Create a grouped box plot from pandas Series.

    Parameters:
    values: pandas Series for the numeric values.
    group: pandas Series for the category groups.
    title: Chart title.
    x_label: Label for the category axis.
    y_label: Label for the numeric axis.
    sort_by: Sort by "field" or "values".
    ascending: Sort order.
    horizontal: Show horizontal boxes if True.
    points: Points to show such as "outliers", "all", or False.
    """
    plot_df = pd.DataFrame(
        {
            x_label or group.name or "Category": group,
            y_label or values.name or "Value": values,
        }
    ).dropna()

    category_label = plot_df.columns[0]
    value_label = plot_df.columns[1]

    if sort_by == "values":
        category_order = (
            plot_df.groupby(category_label)[value_label]
            .median()
            .sort_values(ascending=ascending)
            .index
            .tolist()
        )
    else:
        category_order = sorted(plot_df[category_label].unique(), reverse=not ascending)

    width = max(700, min(1600, 450 + len(category_order) * 85))
    height = max(500, min(1400, 300 + len(category_order) * 40)) if horizontal else 500

    if horizontal:
        fig = px.box(
            plot_df,
            x=value_label,
            y=category_label,
            points=points,
            orientation="h",
            category_orders={category_label: category_order},
            title=title or f"{value_label} by {category_label}",
            color=category_label,
            color_discrete_sequence=PLOT_COLOR_SEQUENCE,
        )
        fig.update_layout(
            xaxis_title=value_label,
            yaxis_title=category_label,
        )
    else:
        fig = px.box(
            plot_df,
            x=category_label,
            y=value_label,
            points=points,
            category_orders={category_label: category_order},
            title=title or f"{value_label} by {category_label}",
            color=category_label,
            color_discrete_sequence=PLOT_COLOR_SEQUENCE,
        )
        fig.update_layout(
            xaxis_title=category_label,
            yaxis_title=value_label,
        )

    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        font=dict(family=PLOT_FONT_FAMILY, size=13),
        title=dict(x=0.5),
        width=width,
        height=height,
    )
    fig.update_layout(showlegend=False)

    return fig


def plot_dual_axis_bar_line(
    month,
    target,
    title=None,
    x_label="Month",
    bar_label="Contacts",
    line_label="Subscription Rate (%)",
):
    """Create a dual-axis bar and line plot from a category and target Series.

    Parameters:
    month: pandas Series for month.
    target: pandas Series for target values.
    title: Chart title.
    x_label: Label for the x-axis.
    bar_label: Label for the bar axis.
    line_label: Label for the line axis.
    """
    plot_df = pd.DataFrame(
        {
            x_label: month,
            "target": target,
        }
    ).dropna()

    monthly_df = (
        plot_df.groupby(x_label)
        .agg(
            contacts=("target", "size"),
            subscriptions=("target", lambda s: s.astype(str).str.lower().eq("yes").sum()),
        )
        .reset_index()
    )
    monthly_df["subscription_rate"] = monthly_df["subscriptions"] / monthly_df["contacts"] * 100

    month_order = [m for m in MONTH_ORDER if m in monthly_df[x_label].astype(str).str.lower().tolist()]
    if month_order:
        monthly_df["_month_key"] = monthly_df[x_label].astype(str).str.lower()
        monthly_df[x_label] = pd.Categorical(monthly_df["_month_key"], categories=month_order, ordered=True)
        monthly_df = monthly_df.sort_values(x_label)
        monthly_df[x_label] = monthly_df[x_label].astype(str)
        monthly_df = monthly_df.drop(columns="_month_key")
    else:
        monthly_df = monthly_df.sort_values(x_label)

    fig = go.Figure()

    fig.add_bar(
        x=monthly_df[x_label],
        y=monthly_df["contacts"],
        name=bar_label,
        marker_color=PLOT_COLOR_SEQUENCE[0],
        yaxis="y",
    )

    fig.add_scatter(
        x=monthly_df[x_label],
        y=monthly_df["subscription_rate"],
        name=line_label,
        mode="lines+markers",
        marker=dict(color=PLOT_COLOR_SEQUENCE[1], size=8),
        line=dict(color=PLOT_COLOR_SEQUENCE[1], width=3),
        yaxis="y2",
    )

    width = max(700, min(1400, 450 + monthly_df.shape[0] * 75))

    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        font=dict(family=PLOT_FONT_FAMILY, size=13),
        title=dict(text=title or "Monthly Contacts and Subscription Rate", x=0.5),
        xaxis=dict(title=x_label),
        yaxis=dict(title=bar_label),
        yaxis2=dict(title=line_label, overlaying="y", side="right"),
        legend=dict(x=1.02, y=1),
        width=width,
        height=500,
    )

    return fig


def plot_rate_line(
    x,
    target,
    title=None,
    x_label=None,
    y_label="Subscription Rate (%)",
):
    """Create a line plot for target rate by a numeric or ordered feature.

    Parameters:
    x: pandas Series for the feature on x-axis.
    target: pandas Series for target values.
    title: Chart title.
    x_label: Label for the x-axis.
    y_label: Label for the y-axis.
    """
    plot_df = pd.DataFrame(
        {
            x_label or x.name or "Value": x,
            "target": target,
        }
    ).dropna()

    axis_label = plot_df.columns[0]
    rate_df = (
        plot_df.groupby(axis_label)
        .agg(
            total=("target", "size"),
            success=("target", lambda s: s.astype(str).str.lower().eq("yes").sum()),
        )
        .reset_index()
    )
    rate_df[y_label] = rate_df["success"] / rate_df["total"] * 100
    rate_df = rate_df.sort_values(axis_label)

    width = max(700, min(1400, 450 + rate_df.shape[0] * 60))

    fig = px.line(
        rate_df,
        x=axis_label,
        y=y_label,
        markers=True,
        title=title or f"{y_label} by {axis_label}",
    )

    fig.update_traces(
        line=dict(color=PLOT_COLOR_SEQUENCE[0], width=3),
        marker=dict(color=PLOT_COLOR_SEQUENCE[1], size=8),
    )

    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        font=dict(family=PLOT_FONT_FAMILY, size=13),
        title=dict(x=0.5),
        xaxis_title=axis_label,
        yaxis_title=y_label,
        width=width,
        height=500,
    )

    return fig


def plot_class_histogram(
    values,
    target,
    title=None,
    x_label=None,
    y_label="Probability Density",
    legend_label=None,
):
    """Create a filled density plot split by target classes.

    Parameters:
    values: pandas Series for numeric values.
    target: pandas Series for class labels.
    title: Chart title.
    x_label: Label for the x-axis.
    y_label: Label for the y-axis.
    legend_label: Label for the legend.
    """
    plot_df = pd.DataFrame(
        {
            x_label or values.name or "Value": values,
            legend_label or target.name or "Class": target,
        }
    ).dropna()

    value_label = plot_df.columns[0]
    class_label = plot_df.columns[1]
    classes = plot_df[class_label].dropna().unique()
    fig_width = max(8, min(14, 6 + len(classes)))

    fig, ax = plt.subplots(figsize=(fig_width, 5))

    for index, current_class in enumerate(classes):
        class_values = plot_df.loc[plot_df[class_label] == current_class, value_label]
        sns.kdeplot(
            x=class_values,
            fill=True,
            alpha=0.35,
            linewidth=1.6,
            label=str(current_class),
            color=MPL_COLOR_SEQUENCE[index % len(MPL_COLOR_SEQUENCE)],
            ax=ax,
        )

    ax.set_title(title or f"{value_label} Distribution by {class_label}", fontsize=14)
    ax.set_xlabel(value_label)
    ax.set_ylabel(y_label)
    ax.legend(title=class_label)
    plt.tight_layout()
    plt.show()
    

    return fig, ax


def plot_facet_grouped_box(
    values,
    category,
    target,
    title=None,
    x_label=None,
    y_label=None,
    legend_label=None,
    facet_label=None,
    facet_col_wrap=3,
    sort_by="field",
    ascending=True,
    points="outliers",
):
    """Create faceted box plots for a numeric feature by target across categories.

    Parameters:
    values: pandas Series for numeric values.
    category: pandas Series used for facet panels.
    target: pandas Series for target groups.
    title: Chart title.
    x_label: Label for the target axis.
    y_label: Label for the numeric axis.
    legend_label: Label for the target legend.
    facet_label: Label for the facet variable.
    facet_col_wrap: Number of facet columns per row.
    sort_by: Sort facets by "field" or "values".
    ascending: Sort order.
    points: Points to show such as "outliers", "all", or False.
    """
    plot_df = pd.DataFrame(
        {
            x_label or target.name or "Group": target,
            y_label or values.name or "Value": values,
            facet_label or category.name or "Category": category,
        }
    ).dropna()

    group_name = plot_df.columns[0]
    value_name = plot_df.columns[1]
    facet_name = plot_df.columns[2]

    if sort_by == "values":
        facet_order = (
            plot_df.groupby(facet_name)[value_name]
            .median()
            .sort_values(ascending=ascending)
            .index
            .tolist()
        )
    else:
        facet_order = sorted(plot_df[facet_name].unique(), reverse=not ascending)

    facet_count = len(facet_order)
    width = max(900, min(1800, facet_col_wrap * 340))
    rows = int(np.ceil(facet_count / facet_col_wrap))
    height = max(500, min(1600, rows * 320))

    fig = px.box(
        plot_df,
        x=group_name,
        y=value_name,
        color=group_name,
        facet_col=facet_name,
        facet_col_wrap=facet_col_wrap,
        points=points,
        category_orders={facet_name: facet_order},
        title=title or f"{value_name} by {group_name} across {facet_name}",
        color_discrete_sequence=PLOT_COLOR_SEQUENCE,
    )

    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        font=dict(family=PLOT_FONT_FAMILY, size=12),
        title=dict(x=0.5),
        xaxis_title=group_name,
        yaxis_title=value_name,
        legend_title_text=group_name,
        width=width,
        height=height,
    )

    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))

    return fig


def plot_correlation_heatmap(
    df,
    title=None,
    target_column=None,
    target_map=None,
):
    """Create a correlation heatmap for numeric columns in a dataframe.

    Parameters:
    df: pandas DataFrame to analyze.
    title: Chart title.
    target_column: Optional target column to encode and include.
    target_map: Optional mapping for the target column values.
    """
    corr_df = df.select_dtypes(include=np.number).copy()

    if target_column is not None and target_column in df.columns:
        encoded_target = df[target_column].map(target_map or {"yes": 1, "no": 0})
        corr_df[target_column] = encoded_target

    corr_matrix = corr_df.corr(numeric_only=True)
    size = max(700, min(1200, 350 + corr_matrix.shape[0] * 55))

    fig = px.imshow(
        corr_matrix,
        text_auto=".2f",
        aspect="auto",
        color_continuous_scale="RdBu",
        zmin=-1,
        zmax=1,
        title=title or "Correlation Heatmap",
    )

    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        font=dict(family=PLOT_FONT_FAMILY, size=13),
        title=dict(x=0.5),
        width=size,
        height=size,
        coloraxis_colorbar_title="Correlation",
    )

    return fig
