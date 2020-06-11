import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

colors = ['#1f77b4',
          '#ff7f0e',
          '#2ca02c',
          '#d62728',
          '#9467bd',
          '#8c564b',
          '#e377c2',
          '#7f7f7f',
          '#bcbd22',
          '#17becf']


def generate_figure2D(X, Y, label_list, pca_components=[0, 1],
                      samples_per_class=1000):

    fig = make_subplots(rows=1, cols=1)  # , column_widths=[0.8, 0.2])
    size = 6

    n_classes = len(label_list)
    for j in range(n_classes):
        X_class_j = X[Y[:, j] == 1]
        s = min(samples_per_class, len(X_class_j))
        x_coord = X_class_j[:s, pca_components[0]]
        y_coord = X_class_j[:s, pca_components[1]]

        fig.add_trace(
            go.Scatter(
                x=x_coord, y=y_coord, name=label_list[j], mode='markers',
                marker={'size': size, 'symbol': 'circle', 'color': colors[j]}
            ), row=1, col=1
        )

    components_dict = {0: 'First', 1: 'Second', 2: 'Third', 3: 'Fourth'}

    fig.update_layout(
        title="2D space (PCA)",
        xaxis_title=components_dict[pca_components[0]
                                    ] + " principal component (x)",
        yaxis_title=components_dict[pca_components[1]
                                    ] + " principal component (y)",
        clickmode='event+select', uirevision=True,
        margin={'l': 100, 'b': 0, 't': 40, 'r': 10},
        width=800,
        height=600,
    )
    return fig


def generate_figure_mel(mel_spec):
    figure = go.Figure(px.imshow(mel_spec.T, origin='lower'), layout=go.Layout(
        title=go.layout.Title(text="A Bar Chart")))
    figure.update_layout(
        title="Mel-spectrogram",
        xaxis_title="Time (hops)",
        yaxis_title="Mel filter index",
        margin={'l': 0, 'b': 0, 't': 40, 'r': 10},
        width=400,
        height=400
    )
    figure.layout.coloraxis.showscale = False
    figure.update_traces(dict(showscale=False,
                              coloraxis=None,  # colorscale='gray'
                              ), selector={'type': 'heatmap'})
    return figure


def generate_figure_training(epochs, val, loss):
    size = 6
    figure_training = make_subplots(rows=2, cols=1)
    figure_training.add_trace(
        go.Scatter(
            x=epochs, y=val, name='val', mode='markers',
            marker={'size': size, 'symbol': 'circle', 'color': colors[0]}
        ), row=1, col=1
    )
    figure_training.add_trace(
        go.Scatter(
            x=epochs, y=loss, name='loss', mode='markers',
            marker={'size': size, 'symbol': 'circle', 'color': colors[1]}
        ), row=2, col=1
    )

    figure_training.update_xaxes(title_text="epochs", row=2, col=1)
    figure_training.update_yaxes(title_text="Accuracy (%)", row=1, col=1)
    figure_training.update_yaxes(title_text="Loss", row=2, col=1)

    figure_training.update_layout(
        clickmode='event+select', uirevision=True,
        autosize=False,
        width=800,
        height=600,
    )
    return figure_training
