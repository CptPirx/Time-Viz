__doc__ = """
The Time-Viz class. 
"""

import meta

from data_tf import get_dataset_numpy, get_dataset

from TABL.tabl_models_tf import TABL_model
from TABL.tabl_run_tf import run_model
from TABL.tabl_layers_tf import BL, TABL, MinMax

from Full.Full_model import Full_model

from TSNE.parametric_tSNE.core import Parametric_tSNE

import datetime
import os

import numpy as np

from tensorflow import keras


class TimeVisualiser(object):
    def __init__(self, transformer_dict, override,
                 horizon=meta.horizon, window=meta.window, output_dim=meta.output_dim):
        """
        Time-Viz initialisation.

        :param transformer_dict: the dictionary of models to run
        :param override: dictionary of run flags
        :param horizon: data horizon
        :param window: data window
        :param output_dim: the output dimensionality
        """
        self.train_x, self.train_y, self.test_x, self.test_y = get_dataset_numpy(horizon)
        self.training_dataset, self.test_dataset, self.true_labels = get_dataset(horizon)

        self.classifier_model = None
        self.full_model = None
        self.tsne_model = None

        self.horizon = horizon
        self.window = window
        self.output_dim = output_dim

        self.override = override
        self.transformer_dict = transformer_dict

        save_path = meta.model_path_template.format(model_tag=self.transformer_dict['tag'])

        self.tsne_path = meta.models_path + save_path \
                         + '_Precompute-{pre}'.format(pre=self.override['Precompute']) \
                         + '_arch-{arch}'.format(arch=self.transformer_dict['architecture']) \
                         + '_metric-{metric}'.format(metric=self.override['Metric']) \
                         + '_out-{dim}'.format(dim=meta.last_tensor) \
                         + '.h5'

        if override['Random init']:
            self.full_path = meta.models_path + 'full_' \
                             + save_path \
                             + '_Precompute-{pre}'.format(pre=self.override['Precompute']) \
                             + '_arch-{arch}'.format(arch=self.transformer_dict['architecture']) \
                             + '_metric-{metric}'.format(metric=self.override['Metric']) \
                             + '_class-{classifier}'.format(classifier=self.override['Classifier']) \
                             + '_out-{dim}'.format(dim=meta.last_tensor) \
                             + '_random-init' \
                             + '.h5'
        else:
            self.full_path = meta.models_path + 'full_' \
                             + save_path \
                             + '_Precompute-{pre}'.format(pre=self.override['Precompute']) \
                             + '_arch-{arch}'.format(arch=self.transformer_dict['architecture']) \
                             + '_metric-{metric}'.format(metric=self.override['Metric']) \
                             + '_class-{classifier}'.format(classifier=self.override['Classifier']) \
                             + '_out-{dim}'.format(dim=meta.last_tensor) \
                             + '.h5'

        self.classifier_path = meta.models_path + '{classifier}_model_out-{dim}.h5' \
            .format(classifier=self.override['Classifier'],
                    dim=meta.last_tensor)
        self.betas_path = meta.betas_path + 'training_betas_perplexity-' \
                          + str(self.transformer_dict['perplexity']) \
                          + '.npy'

    def create_tabl_model(self):
        """
        Creates the TABL model

        :return:
        """
        projection_regularizer = None
        projection_constraint = keras.constraints.max_norm(3.0, axis=0)
        attention_regularizer = None
        attention_constraint = keras.constraints.max_norm(5.0, axis=1)

        self.classifier_model = TABL_model(meta.template, meta.dropout, projection_regularizer, projection_constraint,
                                           attention_regularizer, attention_constraint)

    def create_lstm_model(self):
        """
        Creates the LSTM model

        :return:
        """
        input = keras.layers.Input(shape=(meta.input_dim, meta.window))
        x = keras.layers.LSTM(units=128, dropout=0.1, return_sequences=True)(input)
        x = keras.layers.LSTM(units=60, dropout=0.1)(x)
        output = keras.layers.Dense(units=3, activation='softmax')(x)

        model = keras.models.Model(input, output)
        model.compile(optimizer='adam', loss=keras.losses.categorical_crossentropy, metrics=['acc'])

        self.classifier_model = model

    def create_dense_model(self):
        """
        Creates the Dense model

        :return:
        """
        input = keras.layers.Input(shape=(meta.input_dim, meta.window))
        x = keras.layers.Flatten()(input)
        x = keras.layers.Dense(units=2000, activation='relu')(x)
        x = keras.layers.Dropout(rate=0.1)(x)
        x = keras.layers.Dense(units=1000, activation='relu')(x)
        x = keras.layers.Dropout(rate=0.1)(x)
        x = keras.layers.Dense(units=500, activation='relu')(x)
        x = keras.layers.Dropout(rate=0.1)(x)
        x = keras.layers.Dense(units=60, activation='relu')(x)
        x = keras.layers.Dropout(rate=0.1)(x)
        output = keras.layers.Dense(units=3, activation='softmax')(x)

        model = keras.models.Model(input, output)
        model.compile(optimizer='adam', loss=keras.losses.categorical_crossentropy, metrics=['acc'])

        self.classifier_model = model

    def create_cnn_model(self):
        """
        Creates the CNN model

        :return:
        """
        input = keras.layers.Input(shape=(meta.input_dim, meta.window))
        x = keras.layers.Conv1D(filters=8, kernel_size=3, activation='relu', padding='causal')(input)
        # x = keras.layers.MaxPooling1D(pool_size=2)(x)
        x = keras.layers.Conv1D(filters=16, kernel_size=3, activation='relu', padding='causal')(x)
        # x = keras.layers.MaxPooling1D(pool_size=2)(x)
        x = keras.layers.Conv1D(filters=32, kernel_size=3, activation='relu', padding='causal')(x)
        x = keras.layers.MaxPooling1D(pool_size=2)(x)
        x = keras.layers.Flatten()(x)
        x = keras.layers.Dense(units=60, activation='relu')(x)
        output = keras.layers.Dense(units=3, activation='softmax')(x)

        model = keras.models.Model(input, output)
        model.compile(optimizer='adam', loss=keras.losses.categorical_crossentropy, metrics=['acc'])

        self.classifier_model = model

    def train_classifier(self):
        """
        Trains the classifiers

        :return:
        """
        self.classifier_model.summary()

        self.classifier_model = run_model(model=self.classifier_model,
                                          mode=self.override['Classifier'],
                                          training_dataset=self.training_dataset,
                                          test_dataset=self.test_dataset,
                                          true_labels=self.true_labels)

        self.classifier_model.save(self.classifier_path)

    def get_intermediate_data(self):
        """
        Gets the data from the second to last layer of the classifier

        :return:
        """
        if self.override['Classifier'] == 'TABL':
            reshape = self.classifier_model.layers[-3].output
        else:
            reshape = self.classifier_model.layers[-2].output
            reshape = keras.layers.Reshape((60, 1))(reshape)

        intermediate_layer_model = keras.models.Model(inputs=self.classifier_model.input,
                                                      outputs=reshape)

        intermediate_output_train = intermediate_layer_model.predict(self.train_x)
        intermediate_output_test = intermediate_layer_model.predict(self.test_x)

        if intermediate_output_train.ndim > 2:
            intermediate_output_train = np.squeeze(intermediate_output_train)

        if intermediate_output_test.ndim > 2:
            intermediate_output_test = np.squeeze(intermediate_output_test)

        return intermediate_output_train, intermediate_output_test, intermediate_layer_model

    def create_tsne_model(self, training_betas):
        """
        Crates the parametric t-SNE network

        :param training_betas: training betas for calculating the target arrays
        :return:
        """
        # Define optimizer
        optimizer = keras.optimizers.Adam(learning_rate=0.01)
        # Create the model
        self.tsne_model = Parametric_tSNE(num_inputs=meta.last_tensor,
                                          num_outputs=self.output_dim,
                                          perplexities=self.transformer_dict['perplexity'],
                                          training_betas=training_betas,
                                          alpha=self.output_dim - 1.0,
                                          do_pretrain=self.override['Pretrain'],
                                          batch_size=meta.tsne_batch_size,
                                          seed=54321,
                                          precompute=self.override['Precompute'],
                                          optimizer=optimizer,
                                          all_layers=self.transformer_dict['architecture'],
                                          metric=self.override['Metric'])

    def train_tsne(self, train_data):
        """
        Trains the parametric t-SNE network

        :param train_data: training data
        :return:
        """
        self.tsne_model.fit(train_data, epochs=meta.tsne_epochs, verbose=True)
        print('{time}: Saving model {model_path}'.format(time=datetime.datetime.now(), model_path=self.tsne_path))
        self.tsne_model.save_model(self.tsne_path)

    def create_full_model(self, short_classifier, training_betas):
        """
        Create the connected model

        :param short_classifier: model of the classifier cut to intermediate layer
        :param training_betas: training betas for t-SNE
        :return:
        """
        # Define optimizer
        optimizer = keras.optimizers.Adam(learning_rate=0.01)
        # Create the model
        self.full_model = Full_model(num_inputs=meta.input_dim,
                                     perplexities=self.transformer_dict['perplexity'],
                                     short_tabl=short_classifier,
                                     trained_tsne=self.tsne_model,
                                     training_betas=training_betas,
                                     alpha=self.output_dim - 1.0,
                                     all_layers=self.transformer_dict['architecture'],
                                     batch_size=meta.tsne_batch_size,
                                     precompute=self.override['Precompute'],
                                     optimizer=optimizer,
                                     random_init=self.override['Random init'],
                                     metric=self.override['Metric'])

    def train_full(self):
        """
        Traing the connected model

        :return:
        """
        self.full_model.fit(self.train_x, epochs=meta.full_epochs, verbose=True)
        print('{time}: Saving model {model_path}'.format(time=datetime.datetime.now(), model_path=self.full_path))
        self.full_model.save_model(self.full_path)

    def run_model(self):
        """
        Run the Time-Viz model

        :return: the trained Time-Viz model
        :return: data
        """
        print('Starting run for perplexity {perp} and architecture {arch}'
              .format(perp=self.transformer_dict['perplexity'],
                      arch=self.transformer_dict['architecture']))

        if self.override['Classifier'] == 'TABL':
            # Create or load a TABL model
            if not self.override['TABL'] and os.path.exists(self.classifier_path):
                print('Loading TABL model')
                self.classifier_model = keras.models.load_model(self.classifier_path,
                                                                custom_objects={'BL': BL,
                                                                                'TABL': TABL,
                                                                                'MaxNorm': keras.constraints.max_norm})
            else:
                print('Training TABL model')
                self.create_tabl_model()
                self.train_classifier()
        elif self.override['Classifier'] == 'LSTM':
            # Create or load a lstm model
            if not self.override['LSTM'] and os.path.exists(self.classifier_path):
                print('Loading LSTM model')
                self.classifier_model = keras.models.load_model(self.classifier_path)
            else:
                print('Training LSTM model')
                self.create_lstm_model()
                self.train_classifier()
        elif self.override['Classifier'] == 'Dense':
            # Create or load a dense model
            if not self.override['Dense'] and os.path.exists(self.classifier_path):
                print('Loading Dense model')
                self.classifier_model = keras.models.load_model(self.classifier_path)
            else:
                print('Training Dense model')
                self.create_dense_model()
                self.train_classifier()
        elif self.override['Classifier'] == 'CNN':
            # Create or load a cnn model
            if not self.override['CNN'] and os.path.exists(self.classifier_path):
                print('Loading CNN model')
                self.classifier_model = keras.models.load_model(self.classifier_path)
            else:
                print('Training CNN model')
                self.create_cnn_model()
                self.train_classifier()

        # Get the results from second to last layer of a classifier
        train_data, test_data, short_classifier = self.get_intermediate_data()

        if os.path.exists(self.betas_path) and not self.override['Betas']:
            print('Loading training betas for perplexity {perp}'.format(perp=self.transformer_dict['perplexity']))
            training_betas = np.load(self.betas_path)
        else:
            training_betas = None

        # Create or load a TSNE model
        if not self.override['TSNE'] and os.path.exists(self.tsne_path):
            print('Loading TSNE model')
            self.create_tsne_model(training_betas)
            self.tsne_model = self.tsne_model.restore_model(self.tsne_path,
                                                            num_perplexities=self.transformer_dict['perplexity'])
        else:
            print('Training TSNE model')
            self.create_tsne_model(training_betas)
            self.train_tsne(train_data)

        # Now check if we want to finetune or not
        if self.override['Finetune']:
            # Create or load the full model
            if not self.override['Full'] and os.path.exists(self.full_path):
                print('Loading finetuned model')
                self.create_full_model(short_classifier=short_classifier,
                                       training_betas=training_betas)
                self.full_model = self.full_model.restore_model(self.full_path,
                                                                num_perplexities=self.transformer_dict['perplexity'])
                print()
            else:
                print('Finetuning model')
                self.create_full_model(short_classifier=short_classifier,
                                       training_betas=training_betas)
                self.train_full()
                print()
            return self.full_model, self.train_x, train_data, self.test_x, test_data, self.train_y, self.test_y
        else:
            return self.tsne_model, self.train_x, train_data, self.test_x, test_data, self.train_y, self.test_y
