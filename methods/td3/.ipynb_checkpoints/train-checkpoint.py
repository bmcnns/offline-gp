import numpy as np
import gymnasium as gym
import argparse
import os

from stable_baselines3 import TD3
from stable_baselines3.common.noise import NormalActionNoise

def train(env_name="HalfCheetah-v5", num_timesteps=1e6):
    env = gym.make(env_name, render_mode="rgb_array")

    # The noise objects for TD3
    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

    model = TD3("MlpPolicy", env, action_noise=action_noise, verbose=1)
    model.learn(total_timesteps=num_timesteps, progress_bar=True)
    model.save(f"./models/td3/{env_name}")
