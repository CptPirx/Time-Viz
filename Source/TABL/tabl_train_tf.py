import numpy as np
from sklearn.metrics import precision_recall_fscore_support
import meta
from tensorflow import keras
from datetime import datetime
from sklearn.utils import class_weight


def lob_evaluator(model, test_dataset, true_labels):
    """
    Evaluates the model trained on a single split
    :param true_labels: test laebls in np array
    :param test_dataset: TF test dataset
    :param model: model to evaluate
    :return: the complete metrics for this run
    """
    predicted_labels = model.predict(test_dataset,
                                     # batch_size=meta.batch_size,
                                     verbose=0)
    predicted_labels = predicted_labels.argmax(axis=1)

    precision, recall, f1, _ = precision_recall_fscore_support(true_labels, predicted_labels, average=None)
    precision_avg, recall_avg, f1_avg, _ = precision_recall_fscore_support(true_labels, predicted_labels,
                                                                           average='macro')

    metrics = {'accuracy': np.sum(true_labels == predicted_labels) / len(true_labels), 'precision': precision,
               'recall': recall, 'f1': f1, 'precision_avg': precision_avg, 'recall_avg': recall_avg,
               'f1_avg': f1_avg}

    return metrics


def train_evaluate(model,
                   train_dataset,
                   test_dataset,
                   true_labels,
                   evaluator=lob_evaluator,
                   n_runs=5,
                   train_epochs=meta.tabl_epochs):
    """
    Trains and evaluates a model for using an anchored walk-forward setup
    :param train_dataset: train TF dataset
    :param test_dataset: test TF dataset
    :param true_labels: test labels in np array
    :param model: model to train
    :param evaluator: function to use for evaluating the model (please refer to lob.model_utils.epoch_trainer() )
    :param train_epochs: number of epochs for training the model
    :return: metrics from all the splits
    """
    # Early stopping callback
    earlystop_callback = keras.callbacks.EarlyStopping(
        monitor='loss',
        min_delta=0.0001,
        patience=6
    )

    # Reduce learning rate on plateau callback
    learning_rate_reduction_callback = keras.callbacks.ReduceLROnPlateau(monitor='loss',
                                                                         patience=4,
                                                                         verbose=1,
                                                                         factor=0.5,
                                                                         min_lr=0.0001)

    results = []
    models = []

    # Class weights to 'balance' the dataset
    class_weights = class_weight.compute_class_weight('balanced', np.unique(true_labels), true_labels)
    class_weights = class_weights
    class_weights_dict = dict(enumerate(class_weights))
    print(class_weights)

    for i in range(0, n_runs):
        print("Evaluating for run: ", i)

        # Tensorboard callback
        # log_dir = "C:\\Users\\au614889\\source\\repos\\time-project\\logs\\tabl\\fit\\" + datetime.now().strftime(
        #     "%m_%d-%H_%M_%S")
        log_dir = "D:\\Projects\\Time-vis\\mproject\\logs\\tabl\\fit\\" + datetime.now().strftime("%m_%d-%H_%M_%S\\" +
                                                                                                  str(i))

        # Model checkpoints callback
        checkpoints_callback = keras.callbacks.ModelCheckpoint(log_dir, period=10)

        # Tensorboard callback
        tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=2)

        current_model = model

        model.fit(train_dataset,
                  epochs=train_epochs,
                  class_weight=class_weights_dict,
                  callbacks=[#tensorboard_callback,
                             earlystop_callback,
                             learning_rate_reduction_callback,
                             checkpoints_callback],
                  verbose=1,
                  use_multiprocessing=True,
                  workers=10)

        test_results = evaluator(current_model, test_dataset, true_labels)
        print(test_results)
        results.append(test_results)
        models.append(current_model)

    return results, models[-1]


def get_average_metrics(results):
    """
    Averages the metrics from all splits

    :param results: metrics from all splits
    :return: average metrics
    """
    precision, recall, f1 = [], [], []
    acc = []
    for x in results:
        acc.append(x['accuracy'])
        precision.append(x['precision_avg'])
        recall.append(x['recall_avg'])
        f1.append(x['f1_avg'])

    print("Accuracy = ", np.mean(acc))
    print("Precision = ", np.mean(precision))
    print("Recall = ", np.mean(recall))
    print("F1 = ", np.mean(f1))

    return acc, precision, recall, f1
