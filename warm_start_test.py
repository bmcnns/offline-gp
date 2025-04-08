experiments = [
    {
        "dataset_name": "Minimal-Hopper-Expert-v5",
        "independent_runs": 10,
        "methods": ['MLP', 'Operon', 'PySR'],
        "save_path": "models",
    },
    #{
    #    "dataset_name": "Minimal-Walker2d-Expert-v5",
    #    "independent_runs": 10,
    #    "methods": ['MLP', 'Operon'],
    #    "save_path": "models",
    #},
    #{
    #    "dataset_name": "Minimal-Swimmer-Expert-v5",
    #    "independent_runs": 10,
    #    "methods": ['MLP', 'Operon'],
    #    "save_path": "models",
    #},
    #{
    #    "dataset_name": "Minimal-HalfCheetah-Expert-v5",
    #    "independent_runs": 10,
    #    "methods": ['MLP', 'Operon'],
    #    "save_path": "models",
    #},
    #{
    #    "dataset_name": "Hopper-Expert-v5",
    #    "independent_runs": 10,
    #    "methods": ['Operon'],
    #    "save_path": "models",
    #},
    #{
    #    "dataset_name": "Walker2d-Expert-v5",
    #    "independent_runs": 10,
    #    "methods": ['MLP', 'Operon'],
    #    "save_path": "models",
    #},
    #{
    #    "dataset_name": "Swimmer-Expert-v5",
    #    "independent_runs": 10,
    #    "methods": ['MLP', 'Operon'],
    #    "save_path": "models",
    #},
    #{
    #    "dataset_name": "HalfCheetah-Expert-v5",
    #    "independent_runs": 10,
    #    "methods": ['MLP', 'Operon'],
    #    "save_path": "models",
    #},
]

import methods.operon.train as operon

def experiment_runner(experiments):
    for experiment in experiments:
        for method in experiment['methods']:
            print(f"Starting experiment using {method} on {experiment['dataset_name']}")
            
            if method == 'MLP':
                parameters = {
                    'max_iter': 1000,
                    'hidden_layer_sizes': (256, 256),
                    'activation': 'relu',
                    'solver': 'adam',
                    'learning_rate_init': 3e-4,
                    'learning_rate': 'constant',
                }
                runner = mlp.train
            elif method == 'PySR':
                parameters = {
                    'maxsize': 50,
                    'niterations': 1000,
                    'binary_operators': ['+', '*'],
                    'unary_operators': ['cos','exp','sin','inv(x) = 1/x'],
                    'extra_sympy_mappings': {"inv": lambda x: 1 / x},
                    'elementwise_loss': "loss(prediction, target) = (prediction - target)^2"
                }
                runner = pysr.train
            elif method == 'Operon':
                parameters = {
                    'allowed_symbols': 'add,sub,mul,div,fmin,fmax,pow,ceil,floor,exp,cos,sin,constant,variable',
                    'offspring_generator': 'basic',
                    'reinserter': 'keep-best',
                    'n_threads': 18,
                    'optimizer_iterations': 1000,
                    'objectives': ['mse', 'length'],
                    'tournament_size': 3,
                }
                runner = operon.train
            elif method == 'ParetoGP':
                parameters = {
                    'num_generations': 2,
                    'num_cascades': 3,
                    'population_size': 100,
                    'max_complexity': 400,
                    'archive_tournament_size': 3,
                    'population_tournament_size': 5
                }
                runner = paretogp.train
            else:
                raise Exception("Unknown machine learning / symbolic regression method requested")
            
            runner(dataset_name=experiment['dataset_name'],
                   num_independent_runs=experiment['independent_runs'],
                   save_dir=f"{experiment['save_path']}/{method}/{experiment['dataset_name']}",
                   hyperparameters=parameters)

experiment_runner(experiments)
