experiments = [
    {
        "mode": "eval",
        "dataset_name": "Hopper-Expert-v5",
        "model_folder": "models",
        "video_folder": "videos",
        "env_name": "Hopper-v5",
        "methods": ['PySR'],
        "seeds": [738491,204583,991627,458720,174392,618305,837154,265009,781463,549128]
    }
]

#import methods.paretogp.train as paretogp
#import methods.mlp.train as mlp
import methods.pysr as pysr
import methods.operon as operon

def experiment_runner(experiments):
    for experiment in experiments:
        for method in experiment['methods']:
            

            if experiment['mode'] == "train" and method == 'MLP':
                parameters = {
                    'max_iter': 1000,
                    'hidden_layer_sizes': (256, 256),
                    'activation': 'relu',
                    'solver': 'adam',
                    'learning_rate_init': 3e-4,
                    'learning_rate': 'constant',
                }
                runner = mlp.train
            elif experiment['mode'] == "train" and method == 'PySR':
                parameters = {
                    'maxsize': 50,
                    'niterations': 1000,
                    'binary_operators': ['+', '*'],
                    'unary_operators': ['cos','exp','sin','inv(x) = 1/x'],
                    'extra_sympy_mappings': {"inv": lambda x: 1 / x},
                    'elementwise_loss': "loss(prediction, target) = (prediction - target)^2"
                }
                runner = pysr.train
            elif experiment['mode'] == "train" and method == 'Operon':
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
            elif experiment['mode'] == "train" and method == 'ParetoGP':
                parameters = {
                    'num_generations': 2,
                    'num_cascades': 3,
                    'population_size': 100,
                    'max_complexity': 400,
                    'archive_tournament_size': 3,
                    'population_tournament_size': 5
                }
                runner = paretogp.train
            elif experiment['mode'] == "eval" and method == 'MLP':
                pass
                #evaluator = mlp.evaluate
            elif experiment['mode'] == "eval" and method == 'PySR':
                evaluator = pysr.evaluate
            elif experiment['mode'] == "eval" and method == 'Operon':
                evaluator = operon.evaluate
            elif experiment['mode'] == "eval" and method == 'ParetoGP':
                #evaluator = paretogp.evaluate
                pass
            else:
                raise Exception("Unknown machine learning / symbolic regression method requested")

            if experiment['mode'] == "train":
                print(f"Starting experiment using {method} on {experiment['dataset_name']}")
                
                runner(dataset_name=experiment['dataset_name'],
                       num_independent_runs=experiment['independent_runs'],
                       save_dir=f"{experiment['save_path']}/{method}/{experiment['dataset_name']}",
                       hyperparameters=parameters)
            else:
                print(f"Evaluating {method} on {experiment['dataset_name']}")
                
                evaluator(dataset_name=experiment['dataset_name'],
                          model_folder=experiment['model_folder'],
                          video_folder=experiment['video_folder'],
                          env_name=experiment['env_name'],
                          seeds=experiment['seeds'])

experiment_runner(experiments)
