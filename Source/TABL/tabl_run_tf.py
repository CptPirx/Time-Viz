from TABL.tabl_train_tf import train_evaluate, get_average_metrics
from tensorflow import keras
from TABL.tabl_models_tf import TABL_model
import meta


def run_model(model, mode, training_dataset, test_dataset, true_labels, train_epochs=meta.tabl_epochs):
    """
    Method that runs the passed model and prints the end metrics.

    :param true_labels: test labels in np array
    :param test_dataset: test TF dataset
    :param training_dataset: training TF dataset
    :param model: the model to train and evaluate
    :param mode: string name of the run
    :param train_epochs: number of epochs to train the model for
    :return: the trained model
    """

    results1, trained_model = train_evaluate(model,
                                             train_epochs=train_epochs,
                                             n_runs=meta.num_runs,
                                             train_dataset=training_dataset,
                                             test_dataset=test_dataset,
                                             true_labels=true_labels)

    print("----------")
    print("Mode: ", mode)
    metrics_1 = get_average_metrics(results1)
    print(metrics_1)

    return trained_model


def main():
    """
    Create the network model and run its training.
    """
    # get Bilinear model
    projection_regularizer = None
    projection_constraint = keras.constraints.max_norm(3.0, axis=0)
    attention_regularizer = None
    attention_constraint = keras.constraints.max_norm(5.0, axis=1)

    model = TABL_model(meta.template, meta.dropout, projection_regularizer, projection_constraint,
                       attention_regularizer, attention_constraint, loss=None)
    model.summary()

    run_model(model, 'TABL')


if __name__ == '__main__':
    main()
