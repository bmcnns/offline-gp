from brycegp.model import SymbolicRegressor
from dataset import load_dataset
from sklearn.preprocessing import StandardScaler
import pickle
import os

def train(dataset_name, num_independent_runs, save_dir, hyperparameters):
    X, y = load_dataset(dataset_name)    
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    for run in range(num_independent_runs):
        results = []
        
        for action in range(y.shape[1]):
            model = SymbolicRegressor(**hyperparameters)
            
            model.fit(X, y[:, action])
            results.append(model)

        os.makedirs(save_dir, exist_ok=True)
        with open(f'{save_dir}/run_{run}.pkl', 'wb') as f:
            pickle.dump((*results, scaler), f)