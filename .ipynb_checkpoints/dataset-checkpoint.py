import minari
import numpy as np

def load_dataset(dataset_name):
    dataset = minari.load_dataset(dataset_name)

    observations = []
    actions = []
    for episode in dataset:
        observations.append(episode.observations[:-1])
        actions.append(episode.actions)
        
    X = np.concatenate(observations, axis=0)
    y = np.concatenate(actions, axis=0)
    return X, y

def load_dataset_with_timesteps(dataset_name):
    dataset = minari.load_dataset(dataset_name)

    observations = []
    actions = []
    for episode in dataset:
        obs = episode.observations[:-1]
        act = episode.actions

        timesteps = np.arange(len(obs)).reshape(-1, 1)  # shape (T, 1)
        obs_with_timestep = np.concatenate([timesteps, obs], axis=-1)

        observations.append(obs_with_timestep)
        actions.append(act)
        
    X = np.concatenate(observations, axis=0)
    y = np.concatenate(actions, axis=0)
    return X, y