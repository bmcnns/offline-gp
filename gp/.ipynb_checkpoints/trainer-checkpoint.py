import minari
import pickle
import argparse

from gp.preprocessing import create_offline_dataset
from sklearn.model_selection import train_test_split
from pyoperon.sklearn import SymbolicRegressor
from gp.model import ImitationLearner

"""
Example: python trainer.py --dataset_name "HalfCheetah-Expert-v2" --num_actions 6 --epochs 1
"""

def train(dataset_name, num_actions, epochs, num_threads):
    # Loads the dataset
    offline_data = create_offline_dataset(
        minari.load_dataset(dataset_name),
    ).dataset

    X, y, _, _, _ = offline_data.tensors

    regressors = []
    histories = []

    # Stores targets for each action in y_i
    for action in range(num_actions):
        y_i = y[:, action]

        # 80/20 train, test split.
        X_train, X_test, y_train, y_test = train_test_split(X, y_i, test_size=0.2)

        print(f"Beginning to fit action {action}...")

        reg = SymbolicRegressor(
            allowed_symbols='add,sub,mul,aq,sin,constant,variable',
            offspring_generator='basic',
            reinserter='keep-best',
            n_threads=num_threads,
            optimizer_iterations=epochs,
            objectives=['mse', 'length'],
            tournament_size=3
        )
        
        reg.fit(X_train, y_train)
        
        print("Finishing fit.")

        with open(f"models/{dataset_name}-ImitationLearner-Action{action}.pkl", 'wb') as f:
            pickle.dump((reg, histories), f)
        
        histories += [t['objective_values'] for t in reg.pareto_front_]
        regressors.append(reg)
        
    model = ImitationLearner(regressors, histories)

    with open(f'models/{dataset_name}-ImitationLearner.pkl', 'wb') as f:
        pickle.dump(model, f)


