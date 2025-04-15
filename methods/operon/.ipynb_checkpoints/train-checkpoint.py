from dataset import load_dataset
from sklearn.preprocessing import StandardScaler
from pyoperon.sklearn import SymbolicRegressor
import os
import pickle
import random
from sklearn.metrics import mean_squared_error

def train(dataset_name, num_independent_runs, save_dir, hyperparameters):
    X, y = load_dataset(dataset_name)
    
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    for run in range(num_independent_runs):
        models = []
        for action in range(y.shape[1]):
            y_i = y[:, action]

            start = random.randint(0, 1000000 - 1000)
            end = start + 1000
            X_subset = X[start:end]
            y_subset = y_i[start:end]
            
            parameters = {
                'allowed_symbols': 'add,sub,mul,div,fmin,fmax,pow,ceil,floor,exp,cos,sin,constant,variable',
                'offspring_generator': 'basic',
                'reinserter': 'keep-best',
                'n_threads': 16,
                'optimizer_iterations': 1,
                'objectives': ['mse'],
                'tournament_size': 3,
                'generations': 100,
                'warm_start': True,
            }
            
            model = SymbolicRegressor(**parameters)

            for generation in range(1):
                model.fit(X_subset, y_subset)
                preds = model.predict(X_subset)
                print(mean_squared_error(preds, y_subset))
                print(model.get_model_string(model.model_))
                
            models.append(model)
        
        os.makedirs(save_dir, exist_ok=True)
        with open(f'{save_dir}/model_{run}.pkl', 'wb') as f:
            pickle.dump((models, scaler), f)
