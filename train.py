from pysr import PySRRegressor
from dataset import load_dataset
import os
import pickle
from sklearn.preprocessing import StandardScaler

def train(dataset_name, num_independent_runs, save_path):
    X, y = load_dataset(dataset_name)
    
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    for run in range(num_independent_runs):
        model = PySRRegressor(
            maxsize=30,
            niterations=40,
            binary_operators=["+", "*"],
            unary_operators=[
                "cos",
                "exp",
                "sin",
                "inv(x) = 1/x",
            ],
            extra_sympy_mappings={"inv": lambda x: 1 / x},
            elementwise_loss="loss(prediction, target) = (prediction - target)^2",
        )
    
        model.fit(X, y)
    
        os.makedirs(save_path, exist_ok=True)
    
        with open(f'{save_path}/model_{run+1}.pkl', 'wb') as f:
            pickle.dump((model, scaler), f)