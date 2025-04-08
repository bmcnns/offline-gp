from pysr import PySRRegressor
from dataset import load_dataset
from sklearn.preprocessing import StandardScaler
import os
import pickle

def train(dataset_name, num_independent_runs, save_dir, hyperparameters):

    for run in range(num_independent_runs):
        X, y = load_dataset(dataset_name)
        
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        generations = hyperparameters['niterations']
        del hyperparameters['niterations']
        model = PySRRegressor(**hyperparameters, niterations=1, warm_start=True)

        os.makedirs(save_dir, exist_ok=True)
        
        for generation in range(generations):
            model.fit(X, y)

            # Extract best loss per generation
            eqns = model.equations_
    
            with open(f'{save_dir}/results.csv', 'a') as f:
                for action in range(len(eqns)):
                    loss = eqns[action]['loss'].min()
                    print(f"Dataset {dataset_name},Run {run},Generation {generation+1},Action {action+1},Best {loss}\n")
                    f.write(f"{dataset_name},{run},{generation+1},{action+1},{loss}\n")
        
        with open(f'{save_dir}/run_{run}.pkl', 'wb') as f:
            pickle.dump((model, scaler), f)

   