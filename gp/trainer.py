import minari
import pickle
import argparse

from gp.preprocessing import create_offline_dataset
from sklearn.model_selection import train_test_split
from pyoperon.sklearn import SymbolicRegressor
from gp.model import ImitationLearner

from sklearn.preprocessing import StandardScaler

"""
Example: python trainer.py --dataset_name "HalfCheetah-Expert-v2" --num_actions 6 --epochs 1
"""

def train(dataset_name, num_actions, epochs, num_threads):
    # Loads the dataset
    offline_data = create_offline_dataset(
        minari.load_dataset(dataset_name),
    ).dataset

    X, y, _, _, _ = offline_data.tensors

    regressors = {}
    histories = {}
    scalers = {}

    # Stores targets for each action in y_i
    for action in range(num_actions):
        y_i = y[:, action]

        scaler = StandardScaler()

        print(f"Beginning to fit action {action}...")

        reg = SymbolicRegressor(
            allowed_symbols='add,sub,mul,aq,sin,constant,variable',
            offspring_generator='basic',
            reinserter='keep-best',
            n_threads=num_threads,
            optimizer_iterations=epochs,
            objectives=['mse'],
            tournament_size=3
        )
        
        X = scaler.fit_transform(X)

        reg.fit(X, y_i)
        
        print("Finishing fit.")

        with open(f"models/gp/{dataset_name}-ImitationLearner-Action{action}.pkl", 'wb') as f:
            pickle.dump((reg, histories, scaler), f)
        
        histories[f"action{action}"] = [t['objective_values'] for t in reg.pareto_front_]
        regressors[f"action{action}"] = reg
        scalers[f"action{action}"] = scaler
        
    model = ImitationLearner(regressors, histories, scalers)

    with open(f'models/gp/{dataset_name}-ImitationLearner.pkl', 'wb') as f:
        pickle.dump(model, f)


