import minari
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from dataset import load_dataset
import numpy as np
import pickle
import os

def train(dataset_name, num_independent_runs, save_dir, hyperparameters, starting_generation=0):
    X, y = load_dataset(dataset_name)
    
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    for run in range(starting_generation, num_independent_runs):
        model = MLPRegressor(**hyperparameters)
        
        model.fit(X, y)
        
        os.makedirs(save_dir, exist_ok=True)

        with open(f'{save_dir}/results.csv', 'a') as f:
            for epoch, loss in enumerate(model.loss_curve_):
                f.write(f"{dataset_name},{run},{epoch},{loss}\n")
                
        with open(f'{save_dir}/model_{run}.pkl', 'wb') as f:
            pickle.dump((model, scaler), f)

        
