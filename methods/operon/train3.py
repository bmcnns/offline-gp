from dataset import load_dataset
from sklearn.preprocessing import StandardScaler
from pyoperon.sklearn import SymbolicRegressor
import os
import pickle

def train(dataset_name, num_independent_runs, save_dir, hyperparameters):
    X, y = load_dataset(dataset_name)
    
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    for run in range(num_independent_runs):
        models = []
        for action in range(y.shape[1]):
            y_i = y[:, action]
        
            model = SymbolicRegressor(**hyperparameters)
        
            model.fit(X, y_i)
            models.append(model)
        
        os.makedirs(save_dir, exist_ok=True)
        with open(f'{save_dir}/model_{run}.pkl', 'wb') as f:
            pickle.dump((models, scaler), f)
