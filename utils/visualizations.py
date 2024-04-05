import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def bar_plot(data, column, title):
    """ Takes a dataframe, a column name string, and a plot title string.
    Returns go.Figure: resulting barplot as plotly Figure
    """
    counts = data[column].value_counts(sort=True)
    fig = px.bar(
        x=counts.to_list(),
        y=counts.index.to_list(),
        text_auto=True,
        title=title,
        color_discrete_sequence=px.colors.qualitative.Prism,
        labels={
            "x": column,
            "y": "Counts",
        },
    )
    fig.update_traces(
        textfont_size=12, textangle=0, textposition="outside", cliponaxis=False
    )

    return fig


def box_plots(data, features):
    """ Takes a data set and list of feature names (strings).
        Returns figure of boxplots, one for each variable.
    """
    fig = make_subplots(rows=4, cols=4, start_cell="top-left",
                        subplot_titles=features)
    row_position = 1
    col_position = 1

    for i in range(len(features)):

        if col_position == 5:
            row_position += 1
            col_position = 1

        fig.add_trace(go.Box(x=data[features[i]], name=''),
                      row=row_position, col=col_position)

        col_position += 1

    fig.update_traces(showlegend=False)

    return fig
