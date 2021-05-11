#!/usr/bin/python
from __future__ import division  # Python 2 users only
from __future__ import print_function

__doc__ = """ 
Module for building a parametric tSNE model. 
Trains a neural network on input data. 
One can then transform other data based on this model
Main reference:
van der Maaten, L. (2009). Learning a parametric embedding by preserving local structure. RBM, 500(500), 26.
See README.md for others
"""

import datetime
import functools
import meta
import os
import time

import numpy as np
import tensorflow as tf
from tensorflow.keras import models, layers

from TSNE.parametric_tSNE.core import precompute_targets_euclidean
from TSNE.parametric_tSNE.core import kl_loss, calc_betas_loop, Generator

from TABL.tabl_layers_tf import BL, TABL
from TABL.tabl_models_tf import TABL_model

DEFAULT_EPS = 1e-7


class Full_model(object):
    def __init__(self,
                 num_inputs=meta.last_tensor,
                 perplexities=None,
                 short_tabl=None,
                 trained_tsne=None,
                 alpha=1.0,
                 optimizer='adam',
                 batch_size=64,
                 all_layers=None,
                 training_betas=None,
                 precompute=True,
                 random_init=False,
                 metric='Euclidean',
                 **kwargs):
        """

        """
        self.num_inputs = num_inputs

        if perplexities is not None and not isinstance(perplexities, (list, tuple, np.ndarray)):
            perplexities = np.array([perplexities])
        self.perplexities = perplexities

        self.num_perplexities = None
        if perplexities is not None:
            self.num_perplexities = len(np.array(perplexities))

        if training_betas is not None:
            self.training_betas = training_betas
        else:
            self.training_betas = None

        self.alpha = alpha
        self._optimizer = optimizer
        self._batch_size = batch_size
        self._loss_func = None
        self.precompute = precompute
        self.metric=metric

        # If random initialization build the model from scratch
        if random_init:
            print('Random initialization of the full model')
            projection_regularizer = None
            projection_constraint = tf.keras.constraints.max_norm(3.0, axis=0)
            attention_regularizer = None
            attention_constraint = tf.keras.constraints.max_norm(5.0, axis=1)

            classifier_model = TABL_model(meta.template, meta.dropout, projection_regularizer, projection_constraint,
                                    attention_regularizer, attention_constraint)

            if all_layers is None:
                all_layer_sizes = [500, 500, 2000]
            elif all_layers == 'single':
                pass
            else:
                all_layer_sizes = all_layers

            if all_layers == 'single':
                all_layers = [layers.Dense(meta.output_dim,
                                           input_shape=(meta.last_tensor,),
                                           activation='linear',
                                           kernel_initializer='glorot_uniform')]
            else:
                all_layers = [layers.Dense(all_layer_sizes[0],
                                           input_shape=(meta.last_tensor,),
                                           activation='sigmoid',
                                           kernel_initializer='glorot_uniform'), layers.Dropout(rate=0.1)]

                i = 1
                for lsize in all_layer_sizes[1:]:
                    cur_layer = layers.Dense(lsize,
                                             activation='sigmoid',
                                             kernel_initializer='glorot_uniform')
                    all_layers.append(cur_layer)
                    all_layers.append(layers.Dropout(rate=0.1))
                    i += 1

                all_layers.append(layers.Dense(meta.output_dim,
                                               activation='linear',
                                               kernel_initializer='glorot_uniform'))

                tsne_model = models.Sequential(all_layers)

                squeeze_in = classifier_model.layers[-3].output
                squeezed = tf.keras.layers.Lambda(lambda x: tf.keras.backend.squeeze(x, 2))(squeeze_in)

                self.model = models.Model(classifier_model.input,
                                          tsne_model(squeezed))
        else:
            squeeze_in = short_tabl.output
            squeezed = tf.keras.layers.Lambda(lambda x: tf.keras.backend.squeeze(x, 2))(squeeze_in)

            self.model = models.Model(inputs=short_tabl.input,
                                      outputs=trained_tsne.model(squeezed))

    @staticmethod
    def calc_training_betas(training_data, perplexities, beta_batch_size=3000):
        """
        Calculate beta values (gaussian kernel widths) used for training the model
        For memory reasons, only uses beta_batch_size points at a time.
        Parameters
        ----------
        training_data : 2d array_like, (N, D)
        perplexities : float or ndarray-like, (P,)
        beta_batch_size : int, optional
            Only use `beta_batch_size` points to calculate beta values. This is
            for speed and memory reasons. Data must be well-shuffled for this to be effective,
            betas will be calculated based on regular batches of this size
            # TODO K-NN or something would probably be better rather than just batches
        Returns
        -------
        betas : 2D array_like (N,P)
        """
        assert perplexities is not None, "Must provide desired perplexit(y/ies) if training beta values"
        num_pts = len(training_data)
        if not isinstance(perplexities, (list, tuple, np.ndarray)):
            perplexities = np.array([perplexities])
        num_perplexities = len(perplexities)
        training_betas = np.zeros([num_pts, num_perplexities])

        # To calculate betas, only use `beta_batch_size` points at a time
        cur_start = 0
        cur_end = min(cur_start + beta_batch_size, num_pts)
        while cur_start < num_pts:
            cur_training_data = training_data[cur_start:cur_end, :]

            for pind, curperp in enumerate(perplexities):
                cur_training_betas, cur_P, cur_Hs = calc_betas_loop(cur_training_data, curperp)
                training_betas[cur_start:cur_end, pind] = cur_training_betas

            cur_start += beta_batch_size
            cur_end = min(cur_start + beta_batch_size, num_pts)

        # Save the array
        np.save(file=meta.results_path + '/training_betas', arr=training_betas)

        return training_betas

    def _init_loss_func(self):
        """Initialize loss function based on parameters fed to constructor
        Necessary to do this so we can save/load the model using Keras, since
        the loss function is a custom object"""
        kl_loss_func = functools.partial(kl_loss, alpha=self.alpha,
                                         batch_size=self._batch_size, num_perplexities=self.num_perplexities)
        kl_loss_func.__name__ = 'KL-Divergence'
        self._loss_func = kl_loss_func

    @staticmethod
    def get_num_perplexities(training_betas, num_perplexities):
        if training_betas is None and num_perplexities is None:
            return None

        if training_betas is None:
            return num_perplexities
        elif training_betas is not None and num_perplexities is None:
            return training_betas.shape[1]
        else:
            if len(training_betas.shape) == 1:
                assert num_perplexities == 1, "Mismatch between input training betas and num_perplexities"
            else:
                assert training_betas.shape[1] == num_perplexities
            return num_perplexities

    def fit(self, training_data, training_betas=None, epochs=10, verbose=0):
        """
        Train the neural network model using provided `training_data`
        Parameters
        ----------
        training_data : 2d array_like (N, D)
            Data on which to train the tSNE model
        training_betas : 2d array_like (N,P), optional
            Widths for gaussian kernels. If `None` (the usual case), they will be calculated based on
            `training_data` and self.perplexities. One can also provide them here explicitly.
        epochs: int, optional
        verbose: int, optional
            Default 0. Verbosity level. Passed to Keras fit method
        Returns
        -------
        None. Model trained in place
        """
        assert training_data.shape[
                   1] == self.num_inputs, "Input training data must be same shape as training `num_inputs`"

        if self.training_betas is None:
            training_betas = self.calc_training_betas(training_data, self.perplexities)
            self.training_betas = training_betas
        else:
            self.num_perplexities = self.get_num_perplexities(training_betas, self.num_perplexities)

        # Early stopping callback
        earlystop_callback = tf.keras.callbacks.EarlyStopping(
            monitor='loss',
            min_delta=0.001,
            patience=6
        )

        # Reduce learning rate on plateau callback
        learning_rate_reduction_callback = tf.keras.callbacks.ReduceLROnPlateau(monitor='loss',
                                                                                patience=3,
                                                                                verbose=1,
                                                                                factor=0.5,
                                                                                min_lr=0.00001)

        # Tensorboard callback
        logs_path = "E:\\source\\repos\\Time-vis\\mproject\\Results\\Logs"

        tboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logs_path,
                                                         histogram_freq=5)

        batches_per_epoch = int(training_data.shape[0] // self._batch_size)

        self._init_loss_func()
        self.model.compile(optimizer=self._optimizer,
                           loss=self._loss_func)
        self.model.summary()

        if not self.precompute:
            if verbose:
                print(
                    '{time}: Beginning training on {epochs} epochs'.format(time=datetime.datetime.now(), epochs=epochs))

            train_generator = Generator(x_set=training_data,
                                        betas=self.training_betas,
                                        batch_size=self._batch_size,
                                        precompute=self.precompute)

            # self.model.fit_generator(train_generator, batches_per_epoch, epochs,
            #                          verbose=verbose,
            #                          callbacks=[earlystop_callback,
            #                                     learning_rate_reduction_callback])

            self.model.fit(x=train_generator,
                           verbose=verbose,
                           epochs=epochs,
                           steps_per_epoch=batches_per_epoch,
                           callbacks=[earlystop_callback,
                                      learning_rate_reduction_callback,
                                      tboard_callback],
                           use_multiprocessing=False,
                           workers=10)
        else:
            if verbose:
                print(
                    '{time}: Beginning training on {epochs} epochs'.format(time=datetime.datetime.now(), epochs=epochs))

            precomputed_path = meta.precomputed_path + '/' + self.metric + '/{batch_size}/'.format(batch_size=meta.tsne_batch_size)

            isdir = os.path.isdir(precomputed_path)

            if not isdir:
                print('Precomputed targets not found, commencing calculation of batch size {size}'.format(
                    size=self._batch_size))
                start = time.time()
                precompute_targets_euclidean(training_data=training_data,
                                             betas=self.training_betas,
                                             batch_size=meta.tsne_batch_size,
                                             data_path=precomputed_path)
                end = time.time()
                print('Precomputed targets trained in {time}'.format(time=end - start))

            train_generator = Generator(x_set=training_data,
                                        betas=self.training_betas,
                                        batch_size=self._batch_size,
                                        precompute=self.precompute,
                                        precomputed_path=precomputed_path,
                                        mode=meta.precompute_mode)

            self.model.fit(x=train_generator,
                           verbose=verbose,
                           epochs=epochs,
                           # steps_per_epoch=batches_per_epoch,
                           callbacks=[earlystop_callback,
                                      learning_rate_reduction_callback,
                                      tboard_callback],
                           use_multiprocessing=False,
                           workers=10)

        if verbose:
            print('{time}: Finished training on {epochs} epochs'.format(time=datetime.datetime.now(), epochs=epochs))

    def transform(self, test_data):
        """Transform the `test_data`. Must have the same second dimension as training data
        Parameters
        ----------
            test_data : 2d array_like (M, num_inputs)
                Data to transform using training model
        Returns
        -------
            predicted_data: 2d array_like (M, num_outputs)
        """
        assert self.model is not None, "Must train the model before transforming!"
        assert test_data.shape[1] == self.num_inputs, "Input test data must be same shape as training `num_inputs`"
        return self.model.predict(test_data)

    def save_model(self, model_path):
        """Save the underlying model to `model_path` using Keras"""
        return self.model.save(model_path)

    def restore_model(self, model_path, training_betas=None, num_perplexities=None):
        """Restore the underlying model from `model_path`"""
        if not self._loss_func:
            # Have to initialize this to load the model
            self.num_perplexities = self.get_num_perplexities(training_betas, num_perplexities)
            self._init_loss_func()

        cust_objects = {self._loss_func.__name__: self._loss_func,
                        'BL': BL,
                        'TABL': TABL,
                        'MaxNorm': tf.keras.constraints.max_norm}
        self.model = models.load_model(model_path, custom_objects=cust_objects)

        return self
