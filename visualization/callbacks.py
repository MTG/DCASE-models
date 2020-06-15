# import sys
# sys.path.append('../')
from dcase_models.data.datasets import get_available_datasets
from dcase_models.data.features import get_available_features
from dcase_models.model.models import get_available_models
from dcase_models.data.data_generator import DataGenerator
from dcase_models.data.feature_extractor import FeatureExtractor
from dcase_models.utils.gui import encode_audio
from dcase_models.utils.misc import get_default_args_of_function
from dcase_models.data.scaler import Scaler
from dcase_models.utils.files import save_pickle, load_pickle
from dcase_models.utils.files import mkdir_if_not_exists, load_training_log

from layout import params
from layout import options_datasets, options_features
from layout import options_models, options_optimizers
from app import app
from figures import generate_figure2D, generate_figure_mel
from figures import generate_figure_training
from figures import generate_figure2D_eval
from figures import generate_figure_features

import os
import numpy as np
import ast
import soundfile as sf
from tensorflow import get_default_graph
from sklearn.decomposition import PCA

import dash
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State


data_generator = DataGenerator('')
feature_extractor = FeatureExtractor()
X_pca = np.zeros((1, 4))
X = np.zeros((1, 128, 64))
Y = np.zeros((1, 10))
file_names = []

graph = get_default_graph()

# VIS TAB


@app.callback(
    [Output('plot_mel', 'figure'), Output('audio-player', 'overrideProps')],
    [Input('plot2D', 'selectedData')],
    [State('x_select', 'value'), State('y_select', 'value')])
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
        audio_file = data_generator.convert_features_path_to_audio_path(
            file_names[min_distance_index])
        audio_data, sr = sf.read(audio_file)
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
     State('model_path', 'value')]
)
def update_plot2D(samples_per_class, x_select, y_select,
                  active_tab, fold_ix, model_path):
    global X
    global X_pca
    global Y
    global file_names
    ctx = dash.callback_context
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]

    if (button_id == 'tabs') & (active_tab == 'tab_visualization'):
        if len(data_generator.data) == 0:
            data_generator.load_data()
        fold_name = data_generator.fold_list[fold_ix]
        X, Y, file_names = data_generator.get_one_example_per_file(fold_name)

        exp_folder_fold = os.path.join(model_path, fold_name)
        scaler_path = os.path.join(exp_folder_fold, 'scaler.pickle')
        scaler = load_pickle(scaler_path)
        X = scaler.transform(X)

        with graph.as_default():
            model_container.load_model_weights(exp_folder_fold)
            model_embeddings = model_container.cut_network(-2)
            X_emb = model_embeddings.predict(X)

        pca = PCA(n_components=4)
        pca.fit(X_emb)
        X_pca = pca.transform(X_emb)
        print(X_pca.shape)

    figure2D = generate_figure2D(X_pca, Y, data_generator.label_list,
                                 pca_components=[x_select, y_select],
                                 samples_per_class=samples_per_class)
    return [figure2D]


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
    print('trigger by ', button_id)
    print(n_clicks)
    if button_id == 'end_features_extraction':
        print('change status to NOT_EXTRACTING')
        return 'NOT_EXTRACTING'

    if n_clicks is not None:
        if status_features == 'EXTRACTING':
            print('change status to NOT_EXTRACTING')
            return 'NOT_EXTRACTING'
        print('change status to EXTRACTING')
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
     State('n_fft', 'value'),
     State('sr', 'value'),
     State('specific_parameters', 'value'),
     State('dataset_path', 'value'),
     State('audio_folder', 'value'),
     State('features_folder', 'value'),
     State('dataset_name', 'value')]
)
def do_features_extraction(status_features, feature_ix, sequence_time,
                           sequence_hop_time, audio_hop,
                           audio_win, n_fft, sr, specific_parameters,
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

    print('parameters', specific_parameters)
    specific_parameters = ast.literal_eval(specific_parameters)
    feature_extractor = feature_extractor_class(
        sequence_time=sequence_time,
        sequence_hop_time=sequence_hop_time,
        audio_win=audio_win,
        audio_hop=audio_hop,
        n_fft=n_fft,
        sr=sr, **specific_parameters
    )

    # get dataset class
    data_generator_class = get_available_datasets()[dataset_name]

    data_generator = data_generator_class(dataset_path, feature_extractor)
    if not data_generator.check_if_dataset_was_downloaded():
        return [
            True,
            'Please download the dataset before doing feature extraction',
            'danger'
        ]

    print('Extracting features...')
    data_generator.extract_features()
    print('Done!')

 #   folders_list = data_generator.get_folder_lists()
 #   for audio_features_paths in folders_list:
 #       print('Extracting features from folder: ',
 #             audio_features_paths['audio'])
 #       response = feature_extractor.extract(
 #           audio_features_paths['audio'], audio_features_paths['features'])
  #      if response is None:
  #          print('Features already were calculated, continue...')
  #      print('Done!')

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
                  'audio_win', 'audio_hop', 'n_fft', 'sr']
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
    if dataset_ix is not None:
        dataset_name = options_datasets[dataset_ix]['label']
        params_dataset = params['datasets'][dataset_name]
        # get dataset class
        data_generator_class = get_available_datasets()[dataset_name]
        # init data_generator
        data_generator = data_generator_class(params_dataset['dataset_path'])

        options_folds = [
            {'label': name, 'value': value}
            for value, name in enumerate(data_generator.fold_list)
        ]
        return [params_dataset['dataset_path'],
                params_dataset['audio_folder'],
                params_dataset['feature_folder'],
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
     Input('n_fft', 'value'),
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
     Input('create_model', 'n_clicks')
     ]
)
def check_pipeline(feature_ix, sequence_time, sequence_hop_time, audio_hop,
                   audio_win, n_fft, sr, specific_parameters,
                   dataset_path, audio_folder, features_folder, dataset_ix,
                   end_features_extraction, status_features,
                   model_parameters, model_path, model_ix, create_model):

    global model_container
    global feature_extractor

    ctx = dash.callback_context
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    print('trigger by ', button_id)
    # if was trigger by end_features_extraction and
    # the feautres were already calculated
    if button_id == 'end_features_extraction' and \
       status_features == 'NOT_EXTRACTING':
        raise dash.exceptions.PreventUpdate

    feature_extractor = None
    if feature_ix is not None:
        feature_name = (options_features[feature_ix]['label']
                        if feature_ix is not None else "")
        feature_extractor_class = get_available_features()[feature_name]
        print('parameters', specific_parameters)
        specific_parameters = ast.literal_eval(specific_parameters)
        feature_extractor = feature_extractor_class(
            sequence_time=sequence_time,
            sequence_hop_time=sequence_hop_time,
            audio_win=audio_win,
            audio_hop=audio_hop,
            n_fft=n_fft,
            sr=sr, **specific_parameters
        )
    checks = []
    if dataset_ix is not None:
        dataset_name = options_datasets[dataset_ix]['label']

        # get dataset class
        data_generator_class = get_available_datasets()[dataset_name]
        data_generator = data_generator_class(dataset_path, feature_extractor)
        if data_generator.check_if_dataset_was_downloaded():
            checks.append('dataset')

        if feature_ix is not None:
            features_extracted = data_generator.check_if_features_extracted()

           # folders_list = data_generator.get_folder_lists()
           # features_extracted = True
           # for audio_features_paths in folders_list:
           #     features_path = os.path.join(
           #         audio_features_paths['features'], feature_name)
           #     print('Checking features from folder: ', features_path)
           #     if not feature_extractor.check_features_folder(features_path):
           #         features_extracted = False
           #         break

            if features_extracted:
                checks.append('features')

            if model_ix is not None:
                model_name = options_models[model_ix]['label']

                features_example = feature_extractor.calculate_features(
                    '../tests/audio/40722-8-0-7.wav')
                n_frames_cnn = features_example.shape[1]
                n_freq_cnn = features_example.shape[2]

                n_classes = len(data_generator.label_list)

                model_class = get_available_models()[model_name]
                print(model_class, model_name)

                model_parameters = ast.literal_eval(model_parameters)
                print(model_parameters)
                with graph.as_default():
                    print(n_classes, n_frames_cnn, n_freq_cnn)
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

                    if model_container.check_if_model_exists(model_path):
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
        delete = ['model', 'model_path', 'n_classes', 'n_frames_cnn', 'n_freq_cnn']
        for key in delete:
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
     State('n_fft', 'value'),
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
                 audio_win, n_fft, sr, specific_parameters, dataset_path,
                 audio_folder, features_folder, model_path):
    global model_container
    global data_generator

    ctx = dash.callback_context
    if (n_clicks_create_model is None) & (n_clicks_load_model is None):
        return [False, "", 'success', '']
    else:
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]
        print(button_id)

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
        print(specific_parameters)
        specific_parameters = ast.literal_eval(specific_parameters)
        print(specific_parameters)
        feature_extractor = feature_extractor_class(
            sequence_time=sequence_time,
            sequence_hop_time=sequence_hop_time,
            audio_win=audio_win,
            audio_hop=audio_hop,
            n_fft=n_fft,
            sr=sr, **specific_parameters
        )

        features_example = feature_extractor.calculate_features(
            '../tests/audio/40722-8-0-7.wav')
        n_frames_cnn = features_example.shape[1]
        n_freq_cnn = features_example.shape[2]
        print(features_example.shape)

        # get dataset class
        data_generator_class = get_available_datasets()[dataset_name]
        print(data_generator_class)
        # init data_generator
        kwargs = {}
        if dataset_name == 'URBAN_SED':
            kwargs = {'sequence_hop_time': sequence_hop_time}
        data_generator = data_generator_class(dataset_path, feature_extractor, **kwargs)

        n_classes = len(data_generator.label_list)

        model_class = get_available_models()[model_name]
        print(model_class, model_name)

        model_parameters = ast.literal_eval(model_parameters)
        print(model_parameters)
        if (button_id == 'create_model'):
            with graph.as_default():
                model_container = model_class(model=None, model_path=None,
                                              n_classes=n_classes,
                                              n_frames_cnn=n_frames_cnn,
                                              n_freq_cnn=n_freq_cnn,
                                              **model_parameters)
                print('create model summary')
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

                mkdir_if_not_exists(os.path.dirname(model_path))
                mkdir_if_not_exists(model_path)
                model_container.save_model_json(model_path)

                return [True, 'Model created', 'success', summary]

        if (button_id == 'load_model'):
            with graph.as_default():
                model_container = model_class(model=None, model_path=model_path)
                print('loaded model summary')
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

    model_path = os.path.join('../tests', model_name, dataset_name)
    return model_path


@app.callback(
    Output("plot_training", "figure"),
    [Input("tabs", "active_tab"),
     Input("fold_name", "value"),
     Input('interval-component', 'n_intervals')],
    [State('model_path', 'value')]
)
# TRAINING
def update_figure_training(active_tab, fold_ix, n_intervals, model_path):
    if active_tab == "tab_train":
        if fold_ix is not None:
            fold_name = data_generator.fold_list[fold_ix]
            # print(model_path, fold_name)
            training_log = load_training_log(
                os.path.join(model_path, fold_name))
        else:
            training_log = load_training_log(model_path)
        # print(training_log)
        if (training_log is None):
            figure_training = generate_figure_training([], [], [])
        else:
            if (len(training_log) == 0):
                figure_training = generate_figure_training([], [], [])
            else:
                figure_training = generate_figure_training(
                    training_log['epoch'],
                    training_log['Acc'],
                    training_log['loss']
                )
        return figure_training

    figure_training = generate_figure_training([], [], [])
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
    print('trigger by ', button_id)
    print(n_clicks)
    if button_id == 'end_training':
        print('change status to NOT_TRAINING')
        return [False, "", 'success', 'NOT_TRAINING']

    if n_clicks is not None:
        if status == 'TRAINING':
            print('change status to NOT_TRAINING')
            return [False, "", 'success', 'NOT_TRAINING']
        if fold_ix is None:
            return [True, 'Please select a Fold', 'danger', '']
        if optimizer_ix is None:
            return [True, 'Please select an Optimizer', 'danger', '']
        print('change status to TRAINING')
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
     State("train_model", "n_clicks")]
)
def start_training(status, fold_ix, normalizer, model_path,
                   epochs, early_stopping, optimizer_ix, learning_rate,
                   batch_size, considered_improvement, n_clicks_train):
    print(status)
    if status == 'TRAINING':
        if fold_ix is None:
            return [True, 'Please select a Fold', 'danger', ""]
        if optimizer_ix is None:
            return [True, 'Please select an Optimizer', 'danger', ""]

        fold_name = data_generator.fold_list[fold_ix]
        optimizer = options_optimizers[optimizer_ix]['label']

        print('Loading data... ')
        data_generator.load_data()
        X_train, Y_train, X_val, Y_val = data_generator.get_data_for_training(
            fold_name)

        scaler = Scaler(normalizer=normalizer)
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        X_val = scaler.transform(X_val)
        exp_folder_fold = os.path.join(model_path, fold_name)
        mkdir_if_not_exists(exp_folder_fold)

        scaler_path = os.path.join(exp_folder_fold, 'scaler.pickle')
        save_pickle(scaler, scaler_path)

        train_arguments = {
            'epochs': epochs, 'early_stopping': early_stopping,
            'optimizer': optimizer, 'learning_rate': learning_rate,
            'batch_size': batch_size,
            'considered_improvement': considered_improvement
        }
        with graph.as_default():
            model_container.train(X_train, Y_train, X_val, Y_val,
                                  weights_path=exp_folder_fold,
                                  **train_arguments)
            model_container.load_model_weights(exp_folder_fold)
        return [True, "Model trained", 'success', 'True']

    else:
        raise dash.exceptions.PreventUpdate
        # return [False, "", 'success', ""]


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
    [Output("plot2D_eval", "figure"),
    Output("accuracy", "children")],
    [Input("tabs", "active_tab")],
    [State('fold_name', 'value'),
    State('model_path', 'value')]
)
def evaluate_model(active_tab, fold_ix, model_path):
    global X_test
    global X_pca_test
    global file_names_test
    global Y_test
    global predictions
    if (active_tab == "tab_evaluation") and (fold_ix is not None):
        # load data if needed
        if len(data_generator.data) == 0:
            data_generator.load_data()
        fold_name = data_generator.fold_list[fold_ix]
        X, Y, file_names = data_generator.get_one_example_per_file(fold_name)

        exp_folder_fold = os.path.join(model_path, fold_name)
        scaler_path = os.path.join(exp_folder_fold, 'scaler.pickle')
        scaler = load_pickle(scaler_path)
        X = scaler.transform(X)

        with graph.as_default():
            model_container.load_model_weights(exp_folder_fold)
            model_embeddings = model_container.cut_network(-2)
            X_emb = model_embeddings.predict(X)

        pca = PCA(n_components=4)
        pca.fit(X_emb)

        X_test, Y_test = data_generator.get_data_for_testing(fold_name)
        print(np.amin(X_test[0]), np.amax(X_test[0]))
        print(np.amin(data_generator.data[fold_name]['X'][0]), np.amax(data_generator.data[fold_name]['X'][0]))
        X_test = scaler.transform(X_test)
        print(np.amin(X_test[0]), np.amax(X_test[0]))
        print(np.amin(data_generator.data[fold_name]['X'][0]), np.amax(data_generator.data[fold_name]['X'][0]))
        with graph.as_default():
            results = model_container.evaluate(X_test, Y_test)

        X_test, Y_test, file_names_test = data_generator.get_one_example_per_file(fold_name, train=False)
        X_test = scaler.transform(X_test)

        with graph.as_default():
            X_emb = model_embeddings.predict(X_test)
        #   predictions = model_container.model.predict(X_test)

        predictions = np.zeros_like(Y_test)
        for j in range(len(predictions)):
            #print('results', results['predictions'][j].shape)
            predictions[j] = np.mean(results['predictions'][j], axis=0)
            Y_test[j] = results['annotations'][j][0]

        X_pca_test = pca.transform(X_emb)
        print(X_emb.shape)
        print(X_pca.shape)
        figure2d = generate_figure2D_eval(X_pca_test, predictions, Y_test, data_generator.label_list)
        return [figure2d , "Accuracy in fold %s is %f" % (fold_name, results['accuracy'])]

    raise dash.exceptions.PreventUpdate


@app.callback(
    [Output('plot_mel_eval', 'figure'),
    Output('audio-player-eval', 'overrideProps'),
    Output('predicted', 'children')],
    [Input('plot2D_eval', 'selectedData')])
def click_on_plot2d_eval(clickData):
    if clickData is None:
        figure_mel = generate_figure_mel(X_test[0])
        return [figure_mel, {'autoPlay': False, 'src': ''}, ""]
    else:
        point = np.array([clickData['points'][0]['x'],
                          clickData['points'][0]['y']])
        distances_to_data = np.sum(
            np.power(X_pca_test[:, [0, 1]] - point, 2), axis=-1)
        min_distance_index = np.argmin(distances_to_data)
        audio_file = data_generator.convert_features_path_to_audio_path(
            file_names_test[min_distance_index])
        audio_data, sr = sf.read(audio_file)
        figure_mel = generate_figure_mel(X_test[min_distance_index])


        class_ix = np.argmax(Y_test[min_distance_index])
        pred_ix = np.argmax(predictions[min_distance_index])
        predicted_text = "%s predicted as %s" % (data_generator.label_list[class_ix],
                                                 data_generator.label_list[pred_ix])
        return [
            figure_mel, 
            {'autoPlay': True, 'src': encode_audio(audio_data, sr)},
            predicted_text
        ]

import base64
@app.callback(
    [Output('plot_features', 'figure'),
    Output('audio-player-demo', 'overrideProps'),
    Output('demo_file_label', 'children')],
    [Input("tabs", "active_tab"),
    Input("btn_run_demo", "n_clicks"),
    Input('upload-data', 'contents')],
    [State('fold_name', 'value'),
    State('model_path', 'value'),
    State('plot2D_eval', 'selectedData'),
    State('upload-data', 'filename'),
    State('upload-data', 'last_modified')])
def generate_demo(active_tab, n_clicks, list_of_contents, fold_ix, model_path, selectedData,
                 list_of_names, list_of_dates):
    ctx = dash.callback_context
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]

    if button_id == 'upload-data':
        fold_name = data_generator.fold_list[fold_ix]
        exp_folder_fold = os.path.join(model_path, fold_name)
        scaler_path = os.path.join(exp_folder_fold, 'scaler.pickle')
        scaler = load_pickle(scaler_path)

        filename = os.path.join('./', 'upload.wav')
        data = list_of_contents.encode("utf8").split(b";base64,")[1]
        with open(filename, "wb") as fp:
            fp.write(base64.decodebytes(data))

        X_feat = feature_extractor.calculate_features(filename)
        X_feat = scaler.transform(X_feat)
        with graph.as_default():
            Y_t = model_container.model.predict(X_feat)

        label_list= data_generator.label_list
        figure_features = generate_figure_features(X_feat, Y_t, label_list)
        return [figure_features, {'autoPlay': False, 'src': list_of_contents}, ""]

    if active_tab == 'tab_demo':
        fold_name = data_generator.fold_list[fold_ix]
        if (button_id == 'tabs') and (selectedData is not None):
            point = np.array([selectedData['points'][0]['x'],
                            selectedData['points'][0]['y']])
            distances_to_data = np.sum(
                np.power(X_pca_test[:, [0, 1]] - point, 2), axis=-1)
            ix = np.argmin(distances_to_data)
        else:
            ix = np.random.randint(len(data_generator.data[fold_name]['X']))

        
        exp_folder_fold = os.path.join(model_path, fold_name)
        scaler_path = os.path.join(exp_folder_fold, 'scaler.pickle')
        scaler = load_pickle(scaler_path)

        X_features = data_generator.data[fold_name]['X'][ix]
        X_features = scaler.transform(X_features)

        with graph.as_default():
            model_container.load_model_weights(exp_folder_fold)
            Y_features = model_container.model.predict(X_features)

        print('X', X_features.shape)
        fig_demo =  generate_figure_features(X_features, Y_features, data_generator.label_list)

        features_file = data_generator.file_lists[fold_name][ix]
        audio_file = data_generator.convert_features_path_to_audio_path(features_file)
        audio_data, sr = sf.read(audio_file)

        Y_file = data_generator.data[fold_name]['Y'][ix][0]
        class_ix = np.argmax(Y_file)
        file_label = data_generator.label_list[class_ix]
        return [fig_demo, {'autoPlay': False, 'src': encode_audio(audio_data, sr)}, file_label]

    raise dash.exceptions.PreventUpdate
