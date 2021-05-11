__doc__ = """
Manages the run of the full model.
"""

import os

# Tensorflow logging level
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Allow keras memory growth
import tensorflow as tf

physical_devices = tf.config.list_physical_devices('GPU')
try:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
    # Invalid device or cannot modify virtual devices once initialized.
    pass

import meta

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import csv
from tqdm import tqdm

from Source.Full.TimeVisualiser import TimeVisualiser
from TABL.tabl_layers_tf import TABL
from TABL.tabl_run_tf import run_model as run_tabl_experiment

import matplotlib
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import interactive
interactive(True)
plt.rcParams.update({'figure.max_open_warning': 0})

from sklearn.decomposition import PCA
from sklearn.manifold import trustworthiness
from sklearn.neighbors import KNeighborsClassifier
from mpl_toolkits.mplot3d import Axes3D
from tensorflow import keras

plt.style.use('ggplot')
matplotlib.use('Agg')


def plot_2D_scatter(data, color_palette, alpha=0.5):
    """
    Plot 2D scatter plot of the data
    :param data: np array
        The data to plot
    :param color_palette: matplotlib color palette
        Color palette to use
    :param alpha: matplotlib alpha
        Transparency
    :return: In place
    """
    data = data[:7500, :]

    num_clusters = len(set(data[:, 2]))
    for ci in range(num_clusters):
        cur_plot_rows = data[data[:, 2] == ci]
        cur_color = color_palette[ci]

        if ci == 0:
            label = 'Upward'
        if ci == 1:
            label = 'Stationary'
        if ci == 2:
            label = 'Downward'

        plt.scatter(cur_plot_rows[:, 0], cur_plot_rows[:, 1],
                    color=cur_color, label=label, alpha=alpha, s=10)
        plt.autoscale(enable=True)


def plot_3D_scatter(data, color_palette, alpha=0.5, elevation=10, azimut=0):
    """
    Plot 3D scatter plot of the data
    :param data: np array
        The data to plot
    :param color_palette: matplotlib color palette
        Color palette to use
    :param alpha: matplotlib alpha
        Transparency
    :return: In place
    """
    data = data[:2000, :]
    num_clusters = len(set(data[:, 3]))

    fig = plt.figure()
    ax = Axes3D(fig)

    ax.view_init(elev=elevation, azim=azimut)

    for ci in range(num_clusters):
        cur_plot_rows = data[data[:, 3] == ci]
        cur_color = color_palette[ci]
        ax.scatter(cur_plot_rows[:, 0], cur_plot_rows[:, 1], cur_plot_rows[:, 2],
                   color=cur_color, label=ci, alpha=alpha)


def visualise(output_dim, results):
    """
    Visualisation method
    :param output_dim: int
        The visualisation's dimension count
    :param results: dictionary
        Dictionary of all tested models and their respective results
    :return:
    """
    # Setup visualisation
    color_palette = sns.color_palette("hls", 3)

    # Test data visualisation
    figure_template = meta.results_path + 'Test data visualisation.pdf'
    pdf_obj = PdfPages(figure_template)
    for result in results:
        # Scatter plot of test data
        if output_dim == 2:
            plt.figure()
            plot_2D_scatter(result['Test data'], color_palette, alpha=0.5)
            if pdf_obj:
                plt.savefig(pdf_obj, format='pdf')
        else:
            for elev in range(0, 90, 20):
                for azi in range(0, 360, 20):
                    plt.figure()
                    plot_3D_scatter(result['Test data'], color_palette, alpha=0.5, elevation=elev, azimut=azi)
                    if pdf_obj:
                        plt.savefig(pdf_obj, format='pdf')
                        plt.close()

        plt.title('{label} test data visualisation'.format(label=result['label']))
        plt.legend()

    if pdf_obj:
        pdf_obj.close()
    else:
        plt.show()

    # Train data visualisation
    figure_template = meta.results_path + 'Train data visualisation.pdf'
    pdf_obj = PdfPages(figure_template)
    for result in results:
        plt.figure()
        # Scatter plot of test data
        if output_dim == 2:
            plot_2D_scatter(result['Train data'], color_palette, alpha=0.5)
        else:
            plot_3D_scatter(result['Train data'], color_palette, alpha=0.5)

        plt.title('Train data transformation')

        if pdf_obj:
            plt.savefig(pdf_obj, format='pdf')
    if pdf_obj:
        pdf_obj.close()
    else:
        plt.show()


def evaluate(plot_pca, output_dim, train_data, train_data_after_tabl, train_targets, test_data_after_tabl,
             test_data_full_model, test_targets, transformer_list):
    """
    Evaluate the models performance
    :param plot_pca: bool
        Whether to plot PCA in comparison
    :param output_dim: int
        The visualisation's dimension count
    :param train_data: np array
        Training data
    :param test_data_after_tabl: np array[n_samples, 60]
        Test data
    :param test_data_full_model: np array[n_samples, 40, 10]
        Test data for the full models -> TABL input
    :param test_targets: np array
        Targets to use in trustworthiness score
    :param transformer_list: list of dictionaries
        List of transformers to test
    :return: list of dictionaries
        List of results for each model
    """
    # A dictionary of all results to visualise
    results = []

    if plot_pca:
        pca_transformer = PCA(n_components=output_dim)
        # pca_train_data = np.ravel(train_data)
        pca_transformer.fit(train_data_after_tabl)
        transformer_list.append({'label': 'PCA', 'tag': 'PCA', 'architecture': None, 'transformer': pca_transformer,
                                 'perplexity': None})

    for transformer_dict in transformer_list:
        transformer = transformer_dict['transformer']
        tag = transformer_dict['tag']
        label = transformer_dict['label']

        if label == 'PCA' or not meta.override['Finetune']:
            train_data = train_data_after_tabl
            test_res = transformer.transform(test_data_after_tabl)
            train_res = transformer.transform(train_data)
        else:
            test_res = transformer.transform(test_data_full_model)
            train_res = transformer.transform(train_data)

        train_plot_data = np.column_stack((train_res, train_targets))
        test_plot_data = np.column_stack((test_res, test_targets))

        print('Evaluating for train data')
        # Measure trustworthiness
        trust_list = []
        trust_loop = 10000
        print('Measuring trustworthiness')
        for i in tqdm(range(0, len(train_data_after_tabl - trust_loop), trust_loop)):
            trust = trustworthiness(train_data_after_tabl[i:i + trust_loop], train_res[i:i + trust_loop], n_neighbors=12)
            trust_list.append(trust)
        trust = np.mean(trust_list)

        if train_targets.ndim > 1:
            train_targets = np.argmax(train_targets, axis=1)

        print('Measuring K-NN score with {num} neighbours'.format(num=3))
        knn = KNeighborsClassifier(n_neighbors=3).fit(train_res, train_targets)
        knn_score = knn.score(train_res, train_targets)

        print('Trustworthiness of {model} is {trust} %'.format(trust=trust * 100,
                                                               model=label))
        print('K-NN score of {model} is {score} %'.format(score=knn_score,
                                                          model=label))

        print('Evaluating for test data')
        # Measure trustworthiness
        trust_list = []
        trust_loop = 10000
        print('Measuring trustworthiness')
        for i in tqdm(range(0, len(test_data_after_tabl - trust_loop), trust_loop)):
            trust = trustworthiness(test_data_after_tabl[i:i + trust_loop], test_res[i:i + trust_loop], n_neighbors=12)
            trust_list.append(trust)
        trust = np.mean(trust_list)

        if train_targets.ndim > 1:
            train_targets = np.argmax(train_targets, axis=1)

        print('Measuring K-NN score with {num} neighbours'.format(num=3))
        knn = KNeighborsClassifier(n_neighbors=3).fit(train_res, train_targets)
        knn_score = knn.score(test_res, test_targets)

        print('Trustworthiness of {model} is {trust} %'.format(trust=trust * 100,
                                                               model=label))
        print('K-NN score of {model} is {score} %'.format(score=knn_score,
                                                          model=label))

        transformer_dict['Trust'] = trust
        transformer_dict['KNN score'] = knn_score
        transformer_dict['Test data'] = test_plot_data
        transformer_dict['Train data'] = train_plot_data

        results.append(transformer_dict)

        print()

    # Save the results
    csv_columns = ['label', 'perplexity', 'architecture', 'Trust', 'KNN score']

    print_results = []
    for item in results:
        cut_result = {new_key: item[new_key] for new_key in csv_columns}
        print_results.append(cut_result)

    with open(meta.results_path + 'test_results.csv', 'a+') as file:
        writer = csv.DictWriter(file, fieldnames=csv_columns)
        # writer.writeheader()
        for data in print_results:
            writer.writerow(data)

    return results


def check_degradation(time_visualiser):
    print('Cheking the classification degradation')
    trained_tabl = time_visualiser.classifier_model

    # TABL layer parameters
    projection_regularizer = None
    projection_constraint = keras.constraints.max_norm(3.0, axis=0)
    attention_regularizer = None
    attention_constraint = keras.constraints.max_norm(5.0, axis=1)

    # Create the model from finetuned TABL and fresh classifier layer
    x = time_visualiser.full_model.model.layers[-3].output
    x = TABL(meta.template[-1], projection_regularizer, projection_constraint,
             attention_regularizer, attention_constraint)(x)
    output = keras.layers.Activation('softmax', name='softmax')(x)

    finetuned_tabl = keras.Model(time_visualiser.full_model.model.input,
                                 output)

    # Freeze all layers except the classifier
    for layer in finetuned_tabl.layers[:-2]:
        layer.trainable = False

    optimizer = keras.optimizers.Adam(0.01, beta_1=0.9, beta_2=0.999)

    finetuned_tabl.compile(optimizer, 'categorical_crossentropy', ['acc'])
    finetuned_tabl.summary()

    run_tabl_experiment(model=finetuned_tabl,
                        mode='TABL')


def main():
    """
    Create the network model and run its training.
    """
    # The run's parameters
    # Load or override the models/support data, + pretrain setting
    plot_pca = False
    tabl_experiment = False

    # What transformations to use
    transformer_list = meta.transformer_list

    for tlist in transformer_list:
        # Create the model
        time_visualiser = TimeVisualiser(transformer_dict=tlist,
                                         override=meta.override,
                                         output_dim=meta.output_dim)

        # Do the thing
        tsne_model, train_data_full_model, train_data_after_tabl, test_data_full_model, \
        test_data_after_tabl, train_targets, test_targets = time_visualiser.run_model()

        tlist['transformer'] = tsne_model

        if tabl_experiment:
            # Do the TABL experiment
            check_degradation(time_visualiser)

    # Change the targets to 1 value
    test_targets = np.argmax(test_targets, axis=1)

    if not tabl_experiment:
        # Evaluate data
        results = evaluate(plot_pca=plot_pca,
                           output_dim=meta.output_dim,
                           train_data=train_data_full_model,
                           train_data_after_tabl=train_data_after_tabl,
                           train_targets=train_targets,
                           test_data_after_tabl=test_data_after_tabl,
                           test_data_full_model=test_data_full_model,
                           test_targets=test_targets,
                           transformer_list=transformer_list)

        # Visualise data
        visualise(output_dim=meta.output_dim,
                  results=results)


if __name__ == '__main__':
    main()
