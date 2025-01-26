import numpy as np
import gymnasium as gym
from minari import DataCollector
from minari.serialization import serialize_space

from stable_baselines3 import TD3

def evaluate(model_path, env_name, seed, num_timesteps):
    """Runs policy for X episodes"""
    
    eval_env = gym.make(env_name)

    model = TD3.load(model_path, env=eval_env)
    
    eval_env = DataCollector(eval_env, record_infos=False)

    avg_reward = 0.
    t = 0
    num_episodes = 0

    while t < num_timesteps:
        state, terminated, truncated = eval_env.reset(seed=(seed+t)), False, False
        state = state[0]
        while not (terminated or truncated):
            action, _states = model.predict(state, deterministic=True)
            state, reward, terminated, truncated, _ = eval_env.step(action)
            t += 1

            avg_reward += reward

            if t >= num_timesteps:
                break

        num_episodes += 1

    avg_reward /= num_episodes

    print("---------------------------------------")
    print(f"Evaluation over {num_timesteps} timesteps: {avg_reward:.3f}")
    print("---------------------------------------")

    save_dataset(eval_env, env_name)

def save_dataset(env, env_name):
    """Creates a Minari dataset from the environment"""

    envname_without_version = env_name.split('-')[0]
    env_version_number = env_name.split('-')[1]

    dataset = env.create_dataset(
        dataset_id=f"{envname_without_version}-Expert-{env_version_number}",
        algorithm_name="TD3",
        code_permalink="https://github.com/sfujim/TD3",
        author="Bryce MacInnis",
        author_email="Bryce.MacInnis@dal.ca"
    )

def evaluate_and_save_dataset(model_path, env_name, seed, num_timesteps=1e6):
    """Evaluates a TD3 policy and saves the results as a Minari dataset"""
    evaluate(model_path, env_name, seed, num_timesteps)
