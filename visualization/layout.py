import sys
import os
import glob
import numpy as np
import argparse
import inspect

import dash
import dash_html_components as html
import dash_core_components as dcc
import dash_bootstrap_components as dbc
import dash_audio_components
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

sys.path.append('../')
from dcase_models.utils.files import load_json, mkdir_if_not_exists, load_training_log
from dcase_models.data.data_generator import *
from dcase_models.model.container import *
from dcase_models.model.models import *
from dcase_models.data.scaler import Scaler
from dcase_models.data.feature_extractor import *
from dcase_models.utils.misc import get_class_by_name, get_default_args_of_function
from dcase_models.utils.gui import encode_audio

from figures import *

params = load_json('../tests/parameters.json')
#params_dataset = params["datasets"][args.dataset]
params_features = params["features"]
#params_model = params['models'][args.model]

X_pca = []
Y = []
label_list = []

# Plot 2D graph
figure2D = generate_figure2D(X_pca, Y, label_list, pca_components=[0, 1], samples_per_class=1000)
plot2D = dcc.Graph(id='plot2D', figure=figure2D,
                   style={"height" : "100%", "width" : "100%"})

# Plot mel-spectrogram
X = np.zeros((128,64))
figure_mel = generate_figure_mel(X)
plot_mel = dcc.Graph(id="plot_mel", 
                     figure = figure_mel,
                     style={"width": "90%", "display": "inline-block",'float':'left'})

# Audio controls
audio_player = dash_audio_components.DashAudioComponents(id='audio-player', src="",
                                                         autoPlay=False, controls=True)

# Component selector for PCA
options_pca = [{'label':'Component '+str(j+1),'value':j} for j in range(4)]
x_select = dcc.Dropdown(id='x_select',options=options_pca,value=0,style={'width': '100%'})
y_select = dcc.Dropdown(id='y_select',options=options_pca,value=1,style={'width': '100%'})

# Slider to select number of instances
slider_samples = html.Div(dcc.Slider(id='samples_per_class',min=1,max=500, step=1,value=10,vertical=False),style={'width':'100%'})

# Plot Figure training
figure_training = generate_figure_training([],[],[])
plot_training_acc = dcc.Graph(id="plot_training", 
                     figure = figure_training,
                     ) #"display": "inline-block"style={"height": "100%", "width": "100%", 'float':'left'}

## Inputs for features parameters

# Sequence time and hop time inputs
sequence_time_input = dbc.FormGroup(
    [
        dbc.Label("sequence time (s)", html_for="example-email-row", width=2),
        dbc.Col(dbc.Input(type="number", id="sequence_time", placeholder="sequence_time", value=params_features['sequence_time']), width=4,),
        dbc.Label("sequence hop time", html_for="example-email-row", width=2),
        dbc.Col(dbc.Input(type="number", id="sequence_hop_time", placeholder="sequence_hop_time", value=params_features['sequence_hop_time']), width=4,),
    ],
    row=True,
)

# Audio win and hop size inputs
audio_win_input = dbc.FormGroup(
    [
        dbc.Label("audio window", html_for="example-email-row", width=2),
        dbc.Col(dbc.Input(type="number", id="audio_win", placeholder="audio_win", value=params_features['audio_win']), width=4,),
        dbc.Label("audio hop", html_for="example-email-row", width=2),
        dbc.Col(dbc.Input(type="number", id="audio_hop", placeholder="audio_hop", value=params_features['audio_hop']), width=4,),
    ],
    row=True,
)

# FFT number of samples and sampling rate inputs
n_fft_input = dbc.FormGroup(
    [
        dbc.Label("N FFT", html_for="example-email-row", width=2),
        dbc.Col(dbc.Input(type="number", id="n_fft", placeholder="n_fft", value=params_features['n_fft']), width=4,),
        dbc.Label("sampling rate", html_for="example-email-row", width=2),
        dbc.Col(dbc.Input(type="number", id="sr", placeholder="sr", value=params_features['sr']), width=4,),
    ],
    row=True,
)

from dcase_models.data import feature_extractor

# Features selector
features_classes = [m[0] for m in inspect.getmembers(feature_extractor, inspect.isclass) if m[1].__module__ == 'dcase_models.data.feature_extractor']
options_features = [{'label': name, 'value': value} for value, name in enumerate(features_classes)]
feature_selector = dbc.FormGroup(
    [
        dbc.Label("Feature", html_for="dropdown", width=2),
        dbc.Col(dcc.Dropdown(id="feature_name", options=options_features), width=10),
    ],
    row=True,
) 

# Specific parameters of features selected
specific_parameters = dbc.Textarea(id='specific_parameters' ,className="mb-3", placeholder="Specific features parameters")

# Button to start feature extraction
btn_extract_features = dbc.Button("Extract Features", id="extract_features", color="primary", className="mr-1", disabled=False)

# Feedback message of feature extraction
msg_features = dbc.Alert(
            "Messages about feature extractor",
            id="msg_features",
            is_open=False,
            duration=4000,
        )

## Dataset parameters
from dcase_models.data import data_generator as datasets

# Dataset selector
datasets_classes = [m[0] for m in inspect.getmembers(datasets, inspect.isclass) if m[1].__module__ == 'dcase_models.data.data_generator']
options_datasets = [{'label': name, 'value': value} for value, name in enumerate(datasets_classes)]
dataset_selector = dbc.FormGroup(
    [
        dbc.Label("Dataset", html_for="dropdown", width=2),
        dbc.Col(dcc.Dropdown(id="dataset_name", options=options_datasets), width=10),
    ],
    row=True,
) 

# Dataset path inpt
dataset_path_input = dbc.FormGroup(
    [
        dbc.Label("Dataset path", html_for="example-email-row", width=2),
        dbc.Col(dbc.Input(type="text", id="dataset_path", placeholder="dataset_path"), width=10,),
    ],
    row=True,
)

# Dataset audio and feature folder inputs
dataset_folders_input = dbc.FormGroup(
    [
        dbc.Label("Audio folder", html_for="example-email-row", width=2),
        dbc.Col(dbc.Input(type="text", id="audio_folder", placeholder="audio_folder"), width=4,),
        dbc.Label("Features folder", html_for="example-email-row", width=2),
        dbc.Col(dbc.Input(type="text", id="features_folder", placeholder="features_folder"), width=4,),
    ],
    row=True,
)

## Model parameters
from dcase_models.model import models

# Model selector
models_classes = [m[0] for m in inspect.getmembers(models, inspect.isclass) if m[1].__module__ == 'dcase_models.model.models']
options_models = [{'label': name, 'value': value} for value, name in enumerate(models_classes)]
model_selector = dbc.FormGroup(
    [
        dbc.Label("Model", html_for="dropdown", width=2),
        dbc.Col(dcc.Dropdown(id="model_name", options=options_models), width=10),
    ],
    row=True,
) 

# Normalizer selector
options_normalizer = [{'label': 'MinMax', 'value': 'minmax'}, 
                      {'label': 'Standard', 'value': 'standard'}]
normalizer_selector  = dbc.FormGroup(
    [
        dbc.Label("Normalizer", html_for="dropdown", width=2),
        dbc.Col(dcc.Dropdown(id="normalizer", options=options_normalizer, value='minmax'), width=10),
    ],
    row=True,
) 

# Specific parameters of selected model
model_parameters = dbc.Textarea(id='model_parameters' ,className="mb-3", placeholder="Model parameters")

# Model path input
model_path = dbc.FormGroup(
    [
        dbc.Label("Model path", html_for="dropdown", width=2),
        dbc.Col(dbc.Input(type="text", id="model_path", placeholder="model_path"), width=10,),
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
                {"label": "Dataset downloaded", "value": 'dataset', "disabled": True},
                {"label": "Features extracted", "value": 'features', "disabled": True},
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
                    dbc.Form([feature_selector, sequence_time_input, audio_win_input, 
                              n_fft_input, specific_parameters, btn_extract_features])    
                    ]
                ),
            ]), 
            dbc.Card([
                dbc.CardHeader("Dataset Parameters"),
                dbc.CardBody([
                    dbc.Form([dataset_selector, dataset_path_input, dataset_folders_input])
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
                    dbc.Form([model_selector, normalizer_selector, model_parameters, model_path, check_pipeline]),
                    dbc.Button("Create Model", id="create_model", color="primary", className="mr-1"),
                    dbc.Button("Load Model", id="load_model", color="primary", className="mr-1"),
                    dbc.Button("Summary Model", id="summary_model", color="primary", className="mr-1", disabled=True)
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
#options_folds = [{'label': name, 'value': value} for value, name in enumerate(data_generator.fold_list)]
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
        dbc.Col(dbc.Input(type="number", id="epochs", placeholder="epochs", value=params['train']['epochs']), width=4,),
        dbc.Label("Early stopping", html_for="example-email-row", width=2),
        dbc.Col(dbc.Input(type="number", id="early_stopping", placeholder="early_stopping", value=params['train']['early_stopping']), width=4,),
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
        dbc.Col(dcc.Dropdown(id="optimizer", options=options_optimizers, value=2), width=4,),
        dbc.Label("Learning rate", html_for="example-email-row", width=2),
        dbc.Col(dbc.Input(type="number", id="learning_rate", placeholder="learning_rate", value=params['train']['learning_rate']), width=4,),
    ],
    row=True,
)

# Batch size and considered improvement inputs
batch_size_input = dbc.FormGroup(
    [
        dbc.Label("Batch Size", html_for="example-email-row", width=2),
        dbc.Col(dbc.Input(type="text", id="batch_size", placeholder="batch_size", value=params['train']['batch_size']), width=4,),
        dbc.Label("Considered Improvement", html_for="example-email-row", width=2),
        dbc.Col(dbc.Input(type="number", id="considered_improvement", placeholder="considered_improvement", value=params['train']['considered_improvement']), width=4,),
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
            dbc.Col(html.Div([plot_training_acc ]),width=6),
            dbc.Col( [
                dbc.Card([
                    # dbc.CardHeader("Train model"),
                        dbc.CardBody([
                            dbc.Form([fold_selector, epochs_input, optimizer_input, batch_size_input]),
                            dbc.Button("Train Model", id="train_model", color="primary", className="mr-1"),
                            ]
                        ),
                    ])
            ], width=6
            ),

            #dbc.Col(html.Div([plot_mel,html.Br(),audio_player]), width=4),
        ],
        justify="start"
    ), 
    dcc.Interval(
        id='interval-component',
        interval=5*1000, # in milliseconds
        disabled=False,
        n_intervals=0
    ),
    html.Div("", id='status', style={'display': 'none'}),
    html.Div("", id='end_training', style={'display': 'none'})               
])

# Define Tab Visualization (3)    
tab_visualization = html.Div([
    dbc.Row(
        [
            dbc.Col(html.Div([plot2D]),width=8),
            dbc.Col( [dbc.Row([plot_mel], align='center'), dbc.Row([audio_player], align='center')], width=4),
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
            dbc.Col(html.Div([" "]), width=7, align="center"),
        ],
        justify="around"
    ),   
])

# Define Tab Evaluation (4)
tab_evaluation = dbc.Card(
    dbc.CardBody(
        [
            html.P("This is tab 2!", className="card-text"),
            dbc.Button("Don't click here", color="danger"),
        ]
    ),
    className="mt-3",
)

# Define Tabs
tabs = dbc.Tabs(
    [
        dbc.Tab(tab_model, label="Model", tab_id='tab_model'),
        dbc.Tab(tab_train, label="Train model", tab_id='tab_train'),
        dbc.Tab(tab_visualization, label="Visualize model", tab_id='tab_visualization'),
        dbc.Tab(tab_evaluation, label="Evaluate model", tab_id='tab_evaluation'),
    ], id='tabs'
)

# Define layout
layout = html.Div([ tabs ])