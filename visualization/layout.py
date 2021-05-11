from .figures import generate_figure2D
from .figures import generate_figure_mel
from .figures import generate_figure_training
from .figures import generate_figure_features
from .figures import generate_figure_metrics

from dcase_models.data.datasets import get_available_datasets
from dcase_models.data.features import get_available_features
from dcase_models.model.models import get_available_models
from dcase_models.util.files import load_json

import numpy as np

import dash_html_components as html
import dash_core_components as dcc
import dash_bootstrap_components as dbc
import dash_audio_components

# params = load_json(
#   os.path.join(os.path.dirname(__file__), 'parameters.json'))
params = load_json('parameters.json')
# params_dataset = params["datasets"][args.dataset]
params_features = params["features"]
# params_model = params['models'][args.model]

X_pca = []
Y = []
label_list = []
predictions = []
X_test = []
Y_test = []
X_pca_test = []
file_names_test = []

# Plot 2D graph
figure2D = generate_figure2D(X_pca, Y, label_list, pca_components=[
                             0, 1], samples_per_class=1000)
plot2D = dcc.Graph(id='plot2D', figure=figure2D,
                   style={"height": "100%", "width": "100%"})

# Plot mel-spectrogram
X = np.zeros((64, 64))
figure_mel = generate_figure_mel(X)
plot_mel = dcc.Graph(
    id="plot_mel",
    figure=figure_mel,
    style={"width": "90%", "display": "inline-block", 'float': 'left'}
)

# Audio controls
audio_player = dash_audio_components.DashAudioComponents(
    id='audio-player', src="", autoPlay=False, controls=True
)

# Component selector for PCA
options_pca = [{'label': 'Component '+str(j+1), 'value': j} for j in range(4)]
x_select = dcc.Dropdown(id='x_select', options=options_pca,
                        value=0, style={'width': '100%'})
y_select = dcc.Dropdown(id='y_select', options=options_pca,
                        value=1, style={'width': '100%'})

output_select = dcc.Dropdown(id='output_select', options=[],
                             value=1, style={'width': '100%'})

# Slider to select number of instances
slider_samples = html.Div(
    dcc.Slider(
        id='samples_per_class', min=1, max=500,
        step=1, value=10, vertical=False),
    style={'width': '100%'}
)

# Plot Figure training
figure_training = generate_figure_training([], [], [])
plot_training_acc = dcc.Graph(id="plot_training",
                              figure=figure_training)

# Inputs for features parameters

# Sequence time and hop time inputs
sequence_time_input = dbc.FormGroup(
    [
        dbc.Label("sequence time (s)", html_for="example-email-row", width=2),
        dbc.Col(
            dbc.Input(type="number", id="sequence_time",
                      placeholder="sequence_time",
                      value=params_features['sequence_time']),
            width=4
        ),
        dbc.Label("sequence hop time", html_for="example-email-row", width=2),
        dbc.Col(
            dbc.Input(type="number", id="sequence_hop_time",
                      placeholder="sequence_hop_time",
                      value=params_features['sequence_hop_time']),
            width=4
        ),
    ],
    row=True,
)

# Audio win and hop size inputs
audio_win_input = dbc.FormGroup(
    [
        dbc.Label("audio window", html_for="example-email-row", width=2),
        dbc.Col(dbc.Input(type="number", id="audio_win",
                          placeholder="audio_win",
                          value=params_features['audio_win']), width=4,),
        dbc.Label("audio hop", html_for="example-email-row", width=2),
        dbc.Col(dbc.Input(type="number", id="audio_hop",
                          placeholder="audio_hop",
                          value=params_features['audio_hop']), width=4,),
    ],
    row=True,
)

# FFT number of samples and sampling rate inputs
sr_input = dbc.FormGroup(
    [
        # dbc.Label("N FFT", html_for="example-email-row", width=2),
        # dbc.Col(dbc.Input(type="number", id="n_fft", placeholder="n_fft",
        #                  value=params_features['n_fft']), width=4,),
        dbc.Label("sampling rate", html_for="example-email-row", width=2),
        dbc.Col(dbc.Input(type="number", id="sr", placeholder="sr",
                          value=params_features['sr']), width=4,),
    ],
    row=True,
)


# Features selector
features_classes = get_available_features()
options_features = [{'label': name, 'value': value}
                    for value, name in enumerate(features_classes.keys())]
feature_selector = dbc.FormGroup(
    [
        dbc.Label("Feature", html_for="dropdown", width=2),
        dbc.Col(dcc.Dropdown(id="feature_name",
                             options=options_features), width=10),
    ],
    row=True,
)

# Specific parameters of features selected
specific_parameters = dbc.Textarea(
    id='specific_parameters', className="mb-3",
    placeholder="Specific features parameters"
)

# Button to start feature extraction
btn_extract_features = dbc.Button(
    "Extract Features", id="extract_features",
    color="primary", className="mr-1", disabled=False
)

# Feedback message of feature extraction
msg_features = dbc.Alert(
    "Messages about feature extractor",
    id="msg_features",
    is_open=False,
    duration=4000,
)

# Dataset parameters

# Dataset selector
datasets_classes = get_available_datasets()
options_datasets = [{'label': name, 'value': value}
                    for value, name in enumerate(datasets_classes.keys())]
dataset_selector = dbc.FormGroup(
    [
        dbc.Label("Dataset", html_for="dropdown", width=2),
        dbc.Col(dcc.Dropdown(id="dataset_name",
                             options=options_datasets), width=10),
    ],
    row=True,
)

# Dataset path input
dataset_path_input = dbc.FormGroup(
    [
        dbc.Label("Dataset path", html_for="example-email-row", width=2),
        dbc.Col(dbc.Input(type="text", id="dataset_path",
                          placeholder="dataset_path"), width=10,),
    ],
    row=True,
)

# Dataset audio and feature folder inputs
dataset_folders_input = dbc.FormGroup(
    [
        dbc.Label("Audio folder", html_for="example-email-row", width=2),
        dbc.Col(dbc.Input(type="text", id="audio_folder",
                          placeholder="audio_folder"), width=4,),
        dbc.Label("Features folder", html_for="example-email-row", width=2),
        dbc.Col(dbc.Input(type="text", id="features_folder",
                          placeholder="features_folder"), width=4,),
    ],
    row=True,
)

# Model parameters

# Model selector
models_classes = get_available_models()
options_models = [{'label': name, 'value': value}
                  for value, name in enumerate(models_classes.keys())]
model_selector = dbc.FormGroup(
    [
        dbc.Label("Model", html_for="dropdown", width=2),
        dbc.Col(
            dcc.Dropdown(id="model_name", options=options_models),
            width=10
        ),
    ],
    row=True,
)

# Normalizer selector
options_normalizer = [{'label': 'MinMax', 'value': 'minmax'},
                      {'label': 'Standard', 'value': 'standard'}]
normalizer_selector = dbc.FormGroup(
    [
        dbc.Label("Normalizer", html_for="dropdown", width=2),
        dbc.Col(dcc.Dropdown(id="normalizer",
                             options=options_normalizer,
                             value='minmax'),
                width=10),
    ],
    row=True,
)

# Specific parameters of selected model
model_parameters = dbc.Textarea(
    id='model_parameters', className="mb-3", placeholder="Model parameters")

# Model path input
model_path = dbc.FormGroup(
    [
        dbc.Label("Model path", html_for="dropdown", width=2),
        dbc.Col(dbc.Input(type="text", id="model_path",
                          placeholder="model_path"), width=10,),
    ],
    row=True,
)

# Message for feedback (model)
alert = dbc.Alert(
    "Hello! I am an auto-dismissing alert!",
    id="alert-auto",
    is_open=False,
    duration=4000,
)

# Checklist for pipeline
check_pipeline = dbc.FormGroup(
    [
        dbc.Label("Pipeline"),
        dbc.Checklist(
            options=[
                {"label": "Dataset downloaded",
                    "value": 'dataset', "disabled": True},
                {"label": "Features extracted",
                    "value": 'features', "disabled": True},
                {"label": "Model created", "value": 'model', "disabled": True},
            ],
            value=[],
            id="check_pipeline",
        ),
    ]
)

# Define Tab Model (1)
tab_model = html.Div([
    alert,
    msg_features,

    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Features Parameters"),
                dbc.CardBody([
                    dbc.Form([feature_selector, sequence_time_input,
                              audio_win_input, sr_input,
                              specific_parameters, btn_extract_features])
                ]
                ),
            ]),
            dbc.Card([
                dbc.CardHeader("Dataset Parameters"),
                dbc.CardBody([
                    dbc.Form([dataset_selector, dataset_path_input,
                              dataset_folders_input])
                ]
                ),
            ]),
        ],
            width=6
        ),
        dbc.Col(
            dbc.Card([
                dbc.CardHeader("Model parameters"),
                dbc.CardBody([
                    dbc.Form([model_selector, normalizer_selector,
                              model_parameters, model_path, check_pipeline]),
                    dbc.Button("Create Model", id="create_model",
                               color="primary", className="mr-1"),
                    dbc.Button("Load Model", id="load_model",
                               color="primary", className="mr-1"),
                    dbc.Button("Summary Model", id="summary_model",
                               color="primary", className="mr-1",
                               disabled=True)
                ]
                ),
            ]), width=6
        ),
    ]),
    dbc.Modal(
        [
            dbc.ModalHeader("Model Summary"),
            dbc.ModalBody("This is the content of the modal", id='modal_body'),
            dbc.ModalFooter(
                dbc.Button("Close", id="close_modal", className="ml-auto")
            ),
        ],
        id="modal",
    ),
    html.Div("", id='status_features', style={'display': 'none'}),
    html.Div("", id='end_features_extraction', style={'display': 'none'})
])

# Options of folds, empty to start
options_folds = []
fold_selector = dbc.FormGroup(
    [
        dbc.Label("Fold", html_for="dropdown", width=2),
        dbc.Col(dcc.Dropdown(id="fold_name", options=options_folds), width=10),
    ],
    row=True,
)

# Number of epochs and early stopping inputs
epochs_input = dbc.FormGroup(
    [
        dbc.Label("Number of epochs", html_for="example-email-row", width=2),
        dbc.Col(dbc.Input(type="number", id="epochs", placeholder="epochs",
                          value=params['train']['epochs']), width=4,),
        dbc.Label("Early stopping", html_for="example-email-row", width=2),
        dbc.Col(dbc.Input(type="number", id="early_stopping",
                          placeholder="early_stopping",
                          value=params['train']['early_stopping']), width=4,),
    ],
    row=True,
)

# Optimizer and learning rate inputs
options_optimizers = [{'value': 0, 'label': 'SGD'},
                      {'value': 1, 'label': 'RMSprop'},
                      {'value': 2, 'label': 'Adam'},
                      {'value': 3, 'label': 'Adadelta'},
                      {'value': 4, 'label': 'Adagrad'},
                      {'value': 5, 'label': 'Adamax'},
                      {'value': 6, 'label': 'Nadam'},
                      {'value': 7, 'label': 'Ftrl'}]
optimizer_input = dbc.FormGroup(
    [
        dbc.Label("Optimizer", html_for="example-email-row", width=2),
        dbc.Col(dcc.Dropdown(id="optimizer",
                             options=options_optimizers, value=2), width=4,),
        dbc.Label("Learning rate", html_for="example-email-row", width=2),
        dbc.Col(dbc.Input(type="number", id="learning_rate",
                          placeholder="learning_rate",
                          value=params['train']['learning_rate']), width=4,),
    ],
    row=True,
)

# Batch size and considered improvement inputs
batch_size_input = dbc.FormGroup(
    [
        dbc.Label("Batch Size", html_for="example-email-row", width=2),
        dbc.Col(dbc.Input(type="text", id="batch_size",
                          placeholder="batch_size",
                          value=params['train']['batch_size']), width=4,),
        dbc.Label("Considered Improvement",
                  html_for="example-email-row", width=2),
        dbc.Col(dbc.Input(type="number", id="considered_improvement",
                          placeholder="considered_improvement",
                          value=params['train']['considered_improvement']),
                width=4
                ),
    ],
    row=True,
)

# Feedback messages for training process
alert_train = dbc.Alert(
    "Hello! I am an auto-dismissing alert!",
    id="alert_train",
    is_open=False,
    duration=4000,
)
alert_train2 = dbc.Alert(
    "Hello! I am an auto-dismissing alert!",
    id="alert_train2",
    is_open=False,
    duration=4000,
)

# Define Tab Train (2)
tab_train = html.Div([
    alert_train,
    alert_train2,
    dbc.Row(
        [
            dbc.Col(html.Div([plot_training_acc]), width=6),
            dbc.Col([
                    dbc.Card([
                        # dbc.CardHeader("Train model"),
                        dbc.CardBody([
                            dbc.Form([fold_selector, epochs_input,
                                      optimizer_input, batch_size_input]),
                            dbc.Button("Train Model", id="train_model",
                                       color="primary", className="mr-1"),
                        ]
                        ),
                    ])
                    ], width=6
                    ),
        ],
        justify="start"
    ),
    dcc.Interval(
        id='interval-component',
        interval=5*1000,  # in milliseconds
        disabled=False,
        n_intervals=0
    ),
    html.Div("", id='status', style={'display': 'none'}),
    html.Div("", id='end_training', style={'display': 'none'})
])

# run_visualization = html.Div([dbc.Button("Run", id="run_visualization",
#                              color="primary", className="mr-1")])
# Define Tab Visualization (3)
tab_visualization = html.Div([
    dbc.Row(
        [
            dbc.Col(html.Div([plot2D]), width=8),
            dbc.Col(
                [
                    dbc.Row([plot_mel], align='center'),
                    dbc.Row([audio_player], align='center'),
                    # dbc.Row([run_visualization], align='center')
                ], width=3),
        ]
    ),
    dbc.Row(
        [
            dbc.Col(html.Div([slider_samples]), width=7, align="center"),
        ],
        justify="start"
    ),
    dbc.Row(
        [
            dbc.Col(html.Div([x_select]), width=2, align="center"),
            dbc.Col(html.Div([y_select]), width=2, align="center"),
            dbc.Col(html.Div([output_select]), width=2, align="center"),
            dbc.Col(html.Div([" "]), width=7, align="center"),
        ],
        justify="around"
    ),
])

figure_metrics = generate_figure_metrics([], [])
plot_metrics = dcc.Graph(
    id="figure_metrics",
    figure=figure_metrics,
    style={"width": "90%", "display": "inline-block", 'float': 'left'}
)
button_eval = html.Div([
    dbc.Button("Evaluate", id="run_evaluation",
               className="ml-auto", color="primary")
])

tab_evaluation = html.Div([
    dbc.Row(
        [
            dbc.Col(html.Div([button_eval]), width=2),
            dbc.Col(html.Div([""], id='results'), width=2),
        ]
    ),
    dbc.Row(
        [
            dbc.Col(html.Div([plot_metrics]), width=10)
        ]
    )
])
# tab_evaluation = html.Div([""], id='results')


X_feat = np.zeros((10, 128, 64))
Y_t = np.zeros((10, 10))
label_list = []*10
figure_features = generate_figure_features(X_feat, Y_t, label_list)
plot_features = dcc.Graph(id='plot_features', figure=figure_features,
                          style={"height": "100%", "width": "100%"})
# Audio controls
audio_player_demo = dash_audio_components.DashAudioComponents(
    id='audio-player-demo', src="", autoPlay=False, controls=True
)
btn_run_demo = dbc.Button("Get new predictions", id="btn_run_demo",
                          className="ml-auto", color="primary")

upload_file = dcc.Upload([
        'Drag and Drop or ',
        html.A('Select a File')
    ], style={
        'width': '100%',
        'height': '60px',
        'lineHeight': '60px',
        'borderWidth': '1px',
        'borderStyle': 'dashed',
        'borderRadius': '5px',
        'textAlign': 'center'
    }, id='upload-data')

# Define Tab Demo (4)
tab_demo = html.Div([
    dbc.Row(
        [
            dbc.Col(html.Div([upload_file]), width=5),
            dbc.Col(html.Div([""], id='demo_output'), width=5),
        ]
    ),
    dbc.Row(
        [
            dbc.Col(html.Div([btn_run_demo]), width=2),
            dbc.Col(html.Div([audio_player_demo]), width=3),
            dbc.Col(html.Div("", id='demo_file_label'), width=3)
        ]
    ),
    dbc.Row(
        [
            dbc.Col(html.Div([plot_features]), width=12),
        ]
    ),
])


# Define Tabs
tabs = dbc.Tabs(
    [
        dbc.Tab(tab_model, label="Model definition", tab_id='tab_model'),
        dbc.Tab(tab_train, label="Model training", tab_id='tab_train'),
        dbc.Tab(tab_visualization, label="Data visualization",
                tab_id='tab_visualization'),
        dbc.Tab(tab_evaluation, label="Model evaluation",
                tab_id='tab_evaluation'),
        dbc.Tab(tab_demo, label="Prediction visualization",
                tab_id='tab_demo'),
    ], id='tabs'
)

# Define layout
layout = html.Div([tabs])
