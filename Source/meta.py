__doc__ = """
Hold different values common for the project.
"""

# Data & results paths
train_path = "E:/source/repos/Time-vis/mproject/Data/deepFold_train.mat"
test_path = "E:/source/repos/Time-vis/mproject/Data/deepFold_test.mat"
results_path = "E:/source/repos/Time-vis/mproject/Results/"

betas_path = "E:/source/repos/Time-vis/mproject/Data/Betas/"
models_path = "E:/source/repos/Time-vis/mproject/Results/Models/"
precomputed_path = "E:/source/repos/Time-vis/mproject/Data/Precomputed/"

# Data parameters
horizon = 0
window = 10
input_dim = 40

# TSNE training parameters
tsne_epochs = 200
tsne_batch_size = 2048
pretrain_batch_size = 4096
last_tensor = 60
output_dim = 2
precompute_mode = 'single'

# TABL training parameters
tabl_epochs = 200
tabl_batch_size = 1024
num_runs = 1
dropout = 0.1
template = [[input_dim, 10], [60, 10], [120, 5], [60, 1], [3, 1]]

# Full model training parameters
full_epochs = 100

# Saving parameters
model_path_template = '{model_tag}'

# Run parameters
override = {
    # Which classifier to run [TABL, LSTM, CNN, Dense]
    'Classifier': 'TABL',
    # Retrain or load the classifier: False -> load, True -> train and overwrite
    'CNN': False,
    'Dense': False,
    'LSTM': False,
    'TABL': False,
    # Retrain or load the parametric t-sNE network: False -> load, True -> train and overwrite
    'TSNE': False,
    # Overwrite or load beta for t-SNE
    'Betas': False,
    # Distance metric for t-SNE. Tested 'Euclidean'
    'Metric': 'Euclidean',
    # Overwirte or load the full model
    'Full': False,
    # Randomly initalize werights -> untrained classifier
    'Random init': True,
    # Pretrain the t-SNE network
    'Pretrain': True,
    # Use precomputed targets for t-SNE or calculate on the fly
    'Precompute': True,
    # Finetune the model
    'Finetune': True
}

# List of models to evaluate
transformer_list = [
    {'label': 'Classifier-{classifier}'.format(classifier=override['Classifier']),
     'tag': 'tSNE-{dim}D_perp100'.format(dim=output_dim),
     # t-SNE perplexity
     'perplexity': 100,
     # Placeholder
     'transformer': None,
     # Model's architecture
     'architecture': [500, 500, 2000]}
]
