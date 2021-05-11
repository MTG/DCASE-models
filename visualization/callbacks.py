from dcase_models.data.datasets import get_available_datasets
from dcase_models.data.features import get_available_features
from dcase_models.model.models import get_available_models
from dcase_models.data.dataset_base import Dataset
from dcase_models.data.data_generator import DataGenerator
from dcase_models.data.feature_extractor import FeatureExtractor
from dcase_models.util.gui import encode_audio
from dcase_models.util.misc import get_default_args_of_function
from dcase_models.data.scaler import Scaler
from dcase_models.util.files import save_pickle, load_pickle
from dcase_models.util.files import mkdir_if_not_exists, load_training_log
from dcase_models.util.data import evaluation_setup

from .layout import params
from .layout import options_datasets, options_features
from .layout import options_models, options_optimizers
from .app import app
from .figures import generate_figure2D, generate_figure_mel
from .figures import generate_figure_training
# from .figures import generate_figure2D_eval
from .figures import generate_figure_features
from .figures import generate_figure_metrics

import os
import numpy as np
import ast
import soundfile as sf
from tensorflow.compat.v1 import get_default_graph
from sklearn.decomposition import PCA
import base64

import dash
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State


dataset = Dataset("")
feature_extractor = FeatureExtractor()
data_generator_train = DataGenerator(dataset, feature_extractor, [])
data_generator_val = DataGenerator(dataset, feature_extractor, [])
data_generator_test = DataGenerator(dataset, feature_extractor, [])

X_pca = np.zeros((1, 4))
X = np.zeros((1, 128, 64))
Y = np.zeros((1, 10))
file_names = []

graph = get_default_graph()


# VIS TAB

def conv_path(file_or_folder):
    return os.path.join(os.path.dirname(__file__), file_or_folder)


mkdir_if_not_exists(conv_path('models'))


@app.callback(
    [Output('plot_mel', 'figure'),
     Output('audio-player', 'overrideProps')],
    [Input('plot2D', 'selectedData')],
    [State('x_select', 'value'),
     State('y_select', 'value')])
def click_on_plot2d(clickData, x_select, y_select):
    if clickData is None:
        figure_mel = generate_figure_mel(X[0])
        return [figure_mel, {'autoPlay': False, 'src': ''}]
    else:
        point = np.array([clickData['points'][0]['x'],
                          clickData['points'][0]['y']])
        distances_to_data = np.sum(
            np.power(X_pca[:, [x_select, y_select]] - point, 2), axis=-1)
        min_distance_index = np.argmin(distances_to_data)
        audio_file = file_names[min_distance_index]
        audio_data, sr = sf.read(audio_file['file_original'])
        figure_mel = generate_figure_mel(X[min_distance_index])

        return [
            figure_mel, {
                'autoPlay': True, 'src': encode_audio(audio_data, sr)
            }
        ]


@app.callback(
    [Output('plot2D', 'figure')],
    [Input('samples_per_class', 'value'),
     Input('x_select', 'value'),
     Input('y_select', 'value'),
     Input("tabs", "active_tab")],
    [State('fold_name', 'value'),
     State('model_path', 'value'),
     State('dataset_name', 'value'),
     State('sr', 'value')]
)
# Input('output_select', 'value')],
def update_plot2D(samples_per_class, x_select, y_select,
                  active_tab, fold_ix, model_path, dataset_ix, sr):
    global X
    global X_pca
    global Y
    global file_names
    global feature_extractor
    print('start visualization')
    if (active_tab == 'tab_visualization'):
        fold_name = dataset.fold_list[fold_ix]
        exp_folder_fold = conv_path(os.path.join(model_path, fold_name))
        scaler_path = os.path.join(exp_folder_fold, 'scaler.pickle')
        scaler = load_pickle(scaler_path)

        dataset_name = options_datasets[dataset_ix]['label']
        params_dataset = params['datasets'][dataset_name]
        folds_train, folds_val, _ = evaluation_setup(
            fold_name, dataset.fold_list,
            params_dataset['evaluation_mode']
        )
        print(feature_extractor)
        data_gen_train = DataGenerator(
            dataset, feature_extractor, folds=folds_train,
            batch_size=params['train']['batch_size'],
            shuffle=True, train=False, scaler=scaler
        )
        X_list, Y_list = data_gen_train.get_data()
        file_names = data_gen_train.audio_file_list
        # file_names = data_gen_train.convert_features_path_to_audio_path(
        #     file_names, sr=sr)
        Xt = []
        Yt = []
        for j in range(len(X_list)):
            ix = int(len(X_list[j])/2) if len(X_list[j]) > 1 else 0
            Xj = np.expand_dims(X_list[j][ix], 0)
            Yj = np.expand_dims(Y_list[j][ix], 0)
            Xt.append(Xj)
            Yt.append(Yj)
        X = np.concatenate(Xt, axis=0)
        Yt = np.concatenate(Yt, axis=0)
        with graph.as_default():
            model_container.load_model_weights(exp_folder_fold)
            X_emb = model_container.get_intermediate_output(-2, X)
            # output_select

        pca = PCA(n_components=4)
        pca.fit(X_emb)
        X_pca = pca.transform(X_emb)

    print('pca', X_pca.shape, Yt.shape)
    figure2D = generate_figure2D(X_pca, Yt, dataset.label_list,
                                 pca_components=[x_select, y_select],
                                 samples_per_class=samples_per_class)
    return [figure2D]


@app.callback(
    [Output('output_select', 'options'),
     Output('output_select', 'value')],
    [Input("tabs", "active_tab")],
)
def update_output_select(active_tab):
    if (active_tab == 'tab_visualization'):
        layers = model_container.get_available_intermediate_outputs()
        options = [{'label': x, 'value': x} for x in layers]
        return [options, layers[-1]]
    return [[], '']

# MODEL TAB


@app.callback(
    Output("status_features", "children"),
    [Input('extract_features', 'n_clicks'),
     Input('end_features_extraction', 'children')],
    [State('status_features', 'children')]

)
def trigger_feature_extraction(n_clicks, end_features_extraction,
                               status_features):
    ctx = dash.callback_context
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    if button_id == 'end_features_extraction':
        return 'NOT_EXTRACTING'

    if n_clicks is not None:
        if status_features == 'EXTRACTING':
            return 'NOT_EXTRACTING'
        return 'EXTRACTING'
    else:
        return ''


@app.callback(
    [Output("msg_features", "is_open"),
     Output("msg_features", "children"),
     Output("msg_features", "color"),
     Output("end_features_extraction", "children")],
    [Input('status_features', 'children')],
    [State('feature_name', 'value'),
     State('sequence_time', 'value'),
     State('sequence_hop_time', 'value'),
     State('audio_hop', 'value'),
     State('audio_win', 'value'),
     State('sr', 'value'),
     State('specific_parameters', 'value'),
     State('dataset_path', 'value'),
     State('audio_folder', 'value'),
     State('features_folder', 'value'),
     State('dataset_name', 'value')]
)
def do_features_extraction(status_features, feature_ix, sequence_time,
                           sequence_hop_time, audio_hop,
                           audio_win, sr, specific_parameters,
                           dataset_path, audio_folder, features_folder,
                           dataset_ix):
    global feature_extractor
    if status_features != 'EXTRACTING':
        # return [False, '', 'success', 'True']
        raise dash.exceptions.PreventUpdate
    if feature_ix is None:
        return [True, 'Please select a Feature name', 'danger', 'True']
    if dataset_ix is None:
        return [True, 'Please select a dataset', 'danger', 'True']

    features_name = options_features[feature_ix]['label']
    dataset_name = options_datasets[dataset_ix]['label']

    feature_extractor_class = get_available_features()[features_name]

    specific_parameters = ast.literal_eval(specific_parameters)
    feature_extractor = feature_extractor_class(
        sequence_time=sequence_time,
        sequence_hop_time=sequence_hop_time,
        audio_win=audio_win,
        audio_hop=audio_hop,
        sr=sr, **specific_parameters
    )

    # get dataset class
    dataset_class = get_available_datasets()[dataset_name]

    dataset = dataset_class(dataset_path)
    if not dataset.check_if_downloaded():
        return [
            True,
            'Please download the dataset before doing feature extraction',
            'danger'
        ]

    print('Extracting features...')
    feature_extractor.extract(dataset)
    print('Done!')

    return [True, 'Features extracted', 'success', 'True']


@app.callback(
    Output("extract_features", "children"),
    [Input('interval-component', 'n_intervals')],
    [State("status_features", "children")]
)
def manage_button_features(n_intervals, status_features):
    if status_features == 'EXTRACTING':
        button_features = [dbc.Spinner(size="sm"), " Extracting Features..."]
    else:
        button_features = "Extract Features"
    return button_features


@app.callback(
    [Output('specific_parameters', 'value')],
    [Input('feature_name', 'value')]
)
def select_feature(feature_ix):
    if feature_ix is not None:
        features_name = options_features[feature_ix]['label']
        features_class = get_available_features()[features_name]
        default_arguments = get_default_args_of_function(
            features_class.__init__)
        delete = ['sequence_time', 'sequence_hop_time',
                  'audio_win', 'audio_hop', 'sr']
        for key in delete:
            default_arguments.pop(key)
        if features_name in params['features']:
            params_features = params['features'][features_name]
            for key in params_features.keys():
                default_arguments[key] = params_features[key]
        return [str(default_arguments)]
    else:
        return [""]


@app.callback(
    [Output('dataset_path', 'value'),
     Output('audio_folder', 'value'),
     Output('features_folder', 'value'),
     Output('fold_name', 'options')],
    [Input('dataset_name', 'value')]
)
def select_dataset(dataset_ix):
    print(dataset_ix)
    if dataset_ix is not None:
        dataset_name = options_datasets[dataset_ix]['label']
        params_dataset = params['datasets'][dataset_name]
        # get dataset class
        dataset_class = get_available_datasets()[dataset_name]
        # init data_generator
        dataset = dataset_class(params_dataset['dataset_path'])

        options_folds = [
            {'label': name, 'value': value}
            for value, name in enumerate(dataset.fold_list)
        ]
        return [params_dataset['dataset_path'],
                # params_dataset['audio_folder'],
                # params_dataset['feature_folder'],
                '', '',
                options_folds]
    else:
        return [""]*4


@app.callback(
    [Output('check_pipeline', 'value')],
    [Input('feature_name', 'value'),
     Input('sequence_time', 'value'),
     Input('sequence_hop_time', 'value'),
     Input('audio_hop', 'value'),
     Input('audio_win', 'value'),
     Input('sr', 'value'),
     Input('specific_parameters', 'value'),
     Input('dataset_path', 'value'),
     Input('audio_folder', 'value'),
     Input('features_folder', 'value'),
     Input('dataset_name', 'value'),
     Input("end_features_extraction", "children"),
     Input('status_features', 'children'),
     Input('model_parameters', 'value'),
     Input('model_path', 'value'),
     Input('model_name', 'value'),
     ]
)
def check_pipeline(feature_ix, sequence_time, sequence_hop_time, audio_hop,
                   audio_win, sr, specific_parameters,
                   dataset_path, audio_folder, features_folder, dataset_ix,
                   end_features_extraction, status_features,
                   model_parameters, model_path, model_ix):

    global model_container
    global feature_extractor
    global data_generator_train

    ctx = dash.callback_context
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]

    # if was trigger by end_features_extraction and
    # the features were already calculated
    if button_id == 'end_features_extraction' and \
       status_features == 'NOT_EXTRACTING':
        raise dash.exceptions.PreventUpdate

    feature_extractor = None
    if feature_ix is not None:
        feature_name = (options_features[feature_ix]['label']
                        if feature_ix is not None else "")
        feature_extractor_class = get_available_features()[feature_name]

        specific_parameters = ast.literal_eval(specific_parameters)
        feature_extractor = feature_extractor_class(
            sequence_time=sequence_time,
            sequence_hop_time=sequence_hop_time,
            audio_win=audio_win,
            audio_hop=audio_hop,
            sr=sr, **specific_parameters
        )
    checks = []
    if dataset_ix is not None:
        dataset_name = options_datasets[dataset_ix]['label']

        # get dataset class
        dataset_class = get_available_datasets()[dataset_name]
        dataset = dataset_class(dataset_path)
        if dataset.check_if_downloaded():
            checks.append('dataset')
        if feature_ix is not None:
            features_extracted = feature_extractor.check_if_extracted(dataset)

            if features_extracted:
                checks.append('features')

            if model_ix is not None:
                model_name = options_models[model_ix]['label']

                features_shape = feature_extractor.get_shape()
                n_frames_cnn = features_shape[1]
                n_freq_cnn = features_shape[2]

                n_classes = len(dataset.label_list)

                model_class = get_available_models()[model_name]

                model_parameters = ast.literal_eval(model_parameters)

                with graph.as_default():
                    model_container = model_class(model=None, model_path=None,
                                                  n_classes=n_classes,
                                                  n_frames_cnn=n_frames_cnn,
                                                  n_freq_cnn=n_freq_cnn,
                                                  **model_parameters)

                    if model_name == 'VGGish':
                        model_container.load_pretrained_model_weights()
                        model_container.fine_tuning(
                            -1, new_number_of_classes=n_classes,
                            new_activation='softmax', freeze_source_model=True
                        )

                    model_exists = model_container.check_if_model_exists(
                        conv_path(model_path)
                    )
                    if model_exists:
                        checks.append('model')

    return [checks]


@app.callback(
    [Output('model_parameters', 'value')],
    [Input('model_name', 'value')]
)
def select_model(model_ix):
    if model_ix is not None:
        model_name = options_models[model_ix]['label']
        model_class = get_available_models()[model_name]
        default_arguments = get_default_args_of_function(model_class.__init__)
        delete = ['model', 'model_path', 'n_classes', 'n_frames_cnn', 'n_freq_cnn', 'n_frames', 'n_freq', 'n_freqs']
        for key in delete:
            if key in default_arguments:
                default_arguments.pop(key)
        if model_name in params['models']:
            params_model = params['models'][model_name]['model_arguments']
            for key in params_model.keys():
                default_arguments[key] = params_model[key]

        return [str(default_arguments)]
    else:
        return [""]


@app.callback(
    [Output("alert-auto", "is_open"),
     Output("alert-auto", "children"),
     Output("alert-auto", "color"),
     Output("modal_body", "children")],
    [Input('create_model', 'n_clicks'),
     Input('load_model', 'n_clicks')],
    [State('model_name', 'value'),
     State('feature_name', 'value'),
     State('dataset_name', 'value'),
     State('model_parameters', 'value'),
     State('sequence_time', 'value'),
     State('sequence_hop_time', 'value'),
     State('audio_hop', 'value'),
     State('audio_win', 'value'),
     State('sr', 'value'),
     State('specific_parameters', 'value'),
     State('dataset_path', 'value'),
     State('audio_folder', 'value'),
     State('features_folder', 'value'),
     State('model_path', 'value')]
)
def create_model(n_clicks_create_model, n_clicks_load_model, model_ix,
                 feature_ix, dataset_ix, model_parameters,
                 sequence_time, sequence_hop_time, audio_hop,
                 audio_win, sr, specific_parameters, dataset_path,
                 audio_folder, features_folder, model_path):
    global model_container
    global feature_extractor
    global dataset

    ctx = dash.callback_context
    if (n_clicks_create_model is None) & (n_clicks_load_model is None):
        return [False, "", 'success', '']
    else:
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]

    if (button_id == 'create_model') | (button_id == 'load_model'):
        if model_ix is None:
            return [True, 'Please select a Model', 'danger', '']
        if feature_ix is None:
            return [True, 'Please select a Feature extractor', 'danger', '']
        if dataset_ix is None:
            return [True, 'Please select a Dataset', 'danger', '']

        model_name = options_models[model_ix]['label']
        feature_name = options_features[feature_ix]['label']
        dataset_name = options_datasets[dataset_ix]['label']

        feature_extractor_class = get_available_features()[feature_name]
        specific_parameters = ast.literal_eval(specific_parameters)
        feature_extractor = feature_extractor_class(
            sequence_time=sequence_time,
            sequence_hop_time=sequence_hop_time,
            audio_win=audio_win,
            audio_hop=audio_hop,
            sr=sr, **specific_parameters
        )

        features_shape = feature_extractor.get_shape()
        n_frames_cnn = features_shape[1]
        n_freq_cnn = features_shape[2]

        # get dataset class
        dataset_class = get_available_datasets()[dataset_name]
        # init data_generator
        kwargs = {}
        if dataset_name == 'URBAN_SED':
            kwargs = {'sequence_hop_time': sequence_hop_time}
        dataset = dataset_class(dataset_path, **kwargs)

        n_classes = len(dataset.label_list)

        model_class = get_available_models()[model_name]

        model_parameters = ast.literal_eval(model_parameters)
        if (button_id == 'create_model'):
            with graph.as_default():
                model_container = model_class(model=None, model_path=None,
                                              n_classes=n_classes,
                                              n_frames_cnn=n_frames_cnn,
                                              n_freq_cnn=n_freq_cnn,
                                              **model_parameters)

                model_container.model.summary()
                if model_name == 'VGGish':
                    model_container.load_pretrained_model_weights()
                    model_container.fine_tuning(
                        -1, new_number_of_classes=n_classes,
                        new_activation='softmax', freeze_source_model=True
                    )

                stringlist = []
                model_container.model.summary(
                    print_fn=lambda x: stringlist.append(x))
                summary = "\n".join(stringlist)

                mkdir_if_not_exists(conv_path(os.path.dirname(model_path)))
                mkdir_if_not_exists(conv_path(model_path))
                model_container.save_model_json(conv_path(model_path))

                return [True, 'Model created', 'success', summary]

        if (button_id == 'load_model'):
            with graph.as_default():
                model_container = model_class(
                    model=None, model_path=conv_path(model_path)
                )
                model_container.model.summary()
                stringlist = []
                model_container.model.summary(
                    print_fn=lambda x: stringlist.append(x))
                summary = "\n".join(stringlist)
            return [True, 'Model loaded', 'success', summary]
        # model_container.save_model_weights(model_path)

    return [False, "", 'success', '']


@app.callback(
    Output("modal", "is_open"),
    [Input("summary_model", "n_clicks"),
     Input("close_modal", "n_clicks")],
    [State("modal", "is_open")],
)
def toggle_modal(n1, n2, is_open):
    if n1 or n2:
        return not is_open
    return is_open


@app.callback(
    Output("summary_model", "disabled"),
    [Input("create_model", "n_clicks"),
     Input("load_model", "n_clicks")],
)
def enable_summary_model(n_clicks_create_model, n_clicks_load_model):
    if (n_clicks_create_model is None) & (n_clicks_load_model is None):
        return True
    return False


@app.callback(
    Output('model_path', 'value'),
    [Input('dataset_name', 'value'),
     Input('model_name', 'value')],
)
def create_model_path(dataset_ix, model_ix):
    if (model_ix is None) & (dataset_ix is None):
        return ''
    model_name = 'model_name'
    if model_ix is not None:
        model_name = options_models[model_ix]['label']
    dataset_name = 'dataset_name'
    if dataset_ix is not None:
        dataset_name = options_datasets[dataset_ix]['label']

    model_path = os.path.join('models', model_name, dataset_name)
    return model_path


@app.callback(
    Output("plot_training", "figure"),
    [Input("fold_name", "value"),
     Input('interval-component', 'n_intervals')],
    [State('model_path', 'value')]
)
# Input("tabs", "active_tab"),
# TRAINING
def update_figure_training(fold_ix, n_intervals, model_path):
    figure_training = generate_figure_training([], [], [])
    if fold_ix is not None:
        fold_name = dataset.fold_list[fold_ix]
        training_log = load_training_log(
            conv_path(os.path.join(model_path, fold_name)))
    else:
        training_log = load_training_log(conv_path(model_path))
    if (training_log is None):
        figure_training = generate_figure_training([], [], [])
    else:
        if (len(training_log) == 0):
            figure_training = generate_figure_training([], [], [])
        else:
            figure_training = generate_figure_training(
                training_log['epoch'],
                training_log['accuracy'],
                training_log['loss']
            )
    return figure_training


@app.callback(
    [Output("alert_train", "is_open"),
     Output("alert_train", "children"),
     Output("alert_train", "color"),
     Output("status", "children")],
    [Input("train_model", "n_clicks"),
     Input("end_training", "children")],
    [State('fold_name', 'value'),
     State('normalizer', 'value'),
     State('model_path', 'value'),
     State('epochs', 'value'),
     State('early_stopping', 'value'),
     State('optimizer', 'value'),
     State('learning_rate', 'value'),
     State('batch_size', 'value'),
     State('considered_improvement', 'value'),
     State("status", "children")]
)
def trigger_training(n_clicks, end_training, fold_ix, normalizer,
                     model_path, epochs, early_stopping,
                     optimizer_ix, learning_rate, batch_size,
                     considered_improvement, status):
    ctx = dash.callback_context
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]

    if button_id == 'end_training':
        return [False, "", 'success', 'NOT_TRAINING']

    if n_clicks is not None:
        if status == 'TRAINING':
            return [False, "", 'success', 'NOT_TRAINING']
        if fold_ix is None:
            return [True, 'Please select a Fold', 'danger', '']
        if optimizer_ix is None:
            return [True, 'Please select an Optimizer', 'danger', '']
        return [False, "", 'success', 'TRAINING']
    else:
        return [False, "", 'success', '']


@app.callback(
    [Output("alert_train2", "is_open"),
     Output("alert_train2", "children"),
     Output("alert_train2", "color"),
     Output("end_training", "children")],
    [Input("status", "children")],
    [State('fold_name', 'value'),
     State('normalizer', 'value'),
     State('model_path', 'value'),
     State('epochs', 'value'),
     State('early_stopping', 'value'),
     State('optimizer', 'value'),
     State('learning_rate', 'value'),
     State('batch_size', 'value'),
     State('considered_improvement', 'value'),
     State("train_model", "n_clicks"),
     State('dataset_name', 'value')]
)
def start_training(status, fold_ix, normalizer, model_path,
                   epochs, early_stopping, optimizer_ix, learning_rate,
                   batch_size, considered_improvement,
                   n_clicks_train, dataset_ix):
    global data_generator_train
    global data_generator_val

    if status == 'TRAINING':
        if fold_ix is None:
            return [True, 'Please select a Fold', 'danger', ""]
        if optimizer_ix is None:
            return [True, 'Please select an Optimizer', 'danger', ""]

        dataset_name = options_datasets[dataset_ix]['label']
        fold_name = dataset.fold_list[fold_ix]
        params_dataset = params['datasets'][dataset_name]
        optimizer = options_optimizers[optimizer_ix]['label']

        use_validate_set = True
        if dataset_name in ['TUTSoundEvents2017', 'ESC50', 'ESC10']:
            # When have less data, don't use validation set.
            use_validate_set = False

        folds_train, folds_val, _ = evaluation_setup(
            fold_name, dataset.fold_list,
            params_dataset['evaluation_mode'],
            use_validate_set=use_validate_set
        )
        data_generator_train = DataGenerator(
            dataset, feature_extractor, folds=folds_train,
            batch_size=params['train']['batch_size'],
            shuffle=True, train=True, scaler=None
        )

        scaler = Scaler(normalizer=normalizer)
        print('Fitting scaler ...')
        scaler.fit(data_generator_train)
        print('Done!')

        # Pass scaler to data_gen_train to be used when data
        # loading
        data_generator_train.set_scaler(scaler)

        data_generator_val = DataGenerator(
            dataset, feature_extractor, folds=folds_val,
            batch_size=batch_size,
            shuffle=False, train=False, scaler=scaler
        )

        exp_folder_fold = conv_path(os.path.join(model_path, fold_name))
        mkdir_if_not_exists(exp_folder_fold, parents=True)

        scaler_path = os.path.join(exp_folder_fold, 'scaler.pickle')
        save_pickle(scaler, scaler_path)

        train_arguments = {
            'epochs': epochs, 'early_stopping': early_stopping,
            'optimizer': optimizer, 'learning_rate': learning_rate,
            'batch_size': batch_size,
            'considered_improvement': considered_improvement
        }
        with graph.as_default():
            model_container.train(data_generator_train, data_generator_val,
                                  weights_path=exp_folder_fold,
                                  label_list=dataset.label_list,
                                  **train_arguments)
            model_container.load_model_weights(exp_folder_fold)
        return [True, "Model trained", 'success', 'True']

    else:
        raise dash.exceptions.PreventUpdate


@app.callback(
    Output("train_model", "children"),
    [Input('interval-component', 'n_intervals')],
    [State("status", "children")]
)
def manage_button_train(n_intervals, status):
    if status == 'TRAINING':
        button_train = [dbc.Spinner(size="sm"), " Training..."]
    else:
        button_train = "Train model"
    return button_train

@app.callback(
    [Output("results", "children"),
     Output("figure_metrics", "figure")],
    [Input("run_evaluation", "n_clicks")],
    [State('fold_name', 'value'),
     State('model_path', 'value')]
)
def evaluate_model(n_clicks, fold_ix, model_path):
    global X_test
    global X_pca_test
    global file_names_test
    global Y_test
    global predictions
    global data_generator_test
    print('Change tab evaluation')
    # if (active_tab == "tab_evaluation") and (fold_ix is not None):
    if (n_clicks is not None) and (fold_ix is not None):
        print('Start evaluation')
        fold_name = dataset.fold_list[fold_ix]
        exp_folder_fold = conv_path(os.path.join(model_path, fold_name))
        scaler_path = os.path.join(exp_folder_fold, 'scaler.pickle')
        scaler = load_pickle(scaler_path)

        data_generator_test = DataGenerator(
            dataset, feature_extractor, folds=[fold_name],
            batch_size=params['train']['batch_size'],
            shuffle=True, train=False, scaler=scaler
        )
        print('Loading data...')
        X_test, Y_test = data_generator_test.get_data()
        print('Done')
        print(len(X_test), len(Y_test))
        with graph.as_default():
            model_container.load_model_weights(exp_folder_fold)
            results = model_container.evaluate(
                (X_test, Y_test), label_list=dataset.label_list)
        results = results['classification'].results()

        accuracy = results['overall']['accuracy']
        class_wise = results['class_wise']
        metrics = []
        for label in dataset.label_list:
            metrics.append(class_wise[label]['accuracy']['accuracy'])
        print(metrics)
        figure_metrics = generate_figure_metrics(dataset.label_list, metrics)
        msg = "Accuracy in fold %s is %1.2f" % (fold_name, accuracy)
        return [msg, figure_metrics]

    return ['Pa']
    # raise dash.exceptions.PreventUpdate


@app.callback(
    [Output('plot_features', 'figure'),
     Output('audio-player-demo', 'overrideProps'),
     Output('demo_file_label', 'children')],
    [Input("btn_run_demo", "n_clicks"),
     Input('upload-data', 'contents')],
    [State('fold_name', 'value'),
     State('model_path', 'value'),
     State('upload-data', 'filename'),
     State('upload-data', 'last_modified'),
     State('sr', 'value')]
)
# Input("tabs", "active_tab"),
def generate_demo(n_clicks, list_of_contents, fold_ix,
                  model_path, list_of_names,
                  list_of_dates, sr):
    print('generate demo')
    ctx = dash.callback_context
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    print(button_id, n_clicks)
    if (n_clicks is not None) & (button_id == 'btn_run_demo'):
        fold_name = dataset.fold_list[fold_ix]
        exp_folder_fold = conv_path(os.path.join(model_path, fold_name))
        scaler_path = os.path.join(exp_folder_fold, 'scaler.pickle')
        scaler = load_pickle(scaler_path)

        data_generator_test = DataGenerator(
            dataset, feature_extractor, folds=[fold_name],
            batch_size=params['train']['batch_size'],
            shuffle=True, train=False, scaler=scaler
        )

        n_files = len(data_generator_test.audio_file_list)
        ix = np.random.randint(n_files)

        fold_name = dataset.fold_list[fold_ix]
        exp_folder_fold = conv_path(os.path.join(model_path, fold_name))

        X_features, Y_file = data_generator_test.get_data_from_file(ix)

        with graph.as_default():
            model_container.load_model_weights(exp_folder_fold)
            Y_features = model_container.model.predict(X_features)

        fig_demo = generate_figure_features(
            X_features, Y_features, dataset.label_list)

        audio_file = data_generator_test.audio_file_list[ix]
        audio_data, sr = sf.read(audio_file['file_original'])

        class_ix = np.argmax(Y_file[0])
        file_label = dataset.label_list[class_ix]

        return [
            fig_demo,
            {'autoPlay': False, 'src': encode_audio(audio_data, sr)},
            'ground-truth: %s' % file_label
        ]

    if button_id == 'upload-data':
        fold_name = dataset.fold_list[fold_ix]
        exp_folder_fold = conv_path(os.path.join(model_path, fold_name))
        scaler_path = os.path.join(exp_folder_fold, 'scaler.pickle')
        scaler = load_pickle(scaler_path)

        filename = conv_path('upload.wav')
        data = list_of_contents.encode("utf8").split(b";base64,")[1]
        with open(filename, "wb") as fp:
            fp.write(base64.decodebytes(data))

        X_feat = feature_extractor.calculate(filename)
        X_feat = scaler.transform(X_feat)
        with graph.as_default():
            Y_t = model_container.model.predict(X_feat)

        label_list = dataset.label_list
        figure_features = generate_figure_features(X_feat, Y_t, label_list)
        return [
            figure_features,
            {'autoPlay': False, 'src': list_of_contents}, ""
        ]

    X_feat = np.zeros((10, 128, 64))
    Y_t = np.zeros((10, 10))
    label_list = []*10
    figure_features = generate_figure_features(X_feat, Y_t, label_list)

    return [
        figure_features,
        {'autoPlay': False, 'src': ""}, ""
    ]
