import numpy as np
import torch
import gymnasium as gym
from minari import DataCollector
from minari.serialization import serialize_space
from td3.TD3 import TD3

def evaluate(policy, env_name, seed, num_timesteps):
    """Runs policy for X episodes"""
    
    eval_env = gym.make(env_name)
    
    eval_env = DataCollector(eval_env, record_infos=False)

    avg_reward = 0.
    t = 0
    num_episodes = 0

    while t < num_timesteps:
        state, terminated, truncated = eval_env.reset(seed=(seed+t)), False, False
        state = state[0]
        while not (terminated or truncated):
            action = policy.select_action(np.array(state))
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

def evaluate_and_save_dataset(model_path, env_name, seed,
        num_timesteps=1e6, discount=0.99, tau=0.005,
        policy_noise=0.2, noise_clip=0.5, policy_freq=2):
    """Evaluates a TD3 policy and saves the results as a Minari dataset"""

    env = gym.make(env_name)

    env.action_space.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    kwargs = {
        "state_dim": env.observation_space.shape[0],
        "action_dim": env.action_space.shape[0],
        "max_action": float(env.action_space.high[0]),
        "discount": discount,
        "tau": tau,
        "policy_noise": policy_noise,
        "noise_clip": noise_clip,
        "policy_freq": policy_freq
    }

    policy = TD3(**kwargs)
    policy.load(model_path)

    evaluate(policy, env_name, seed, num_timesteps)
