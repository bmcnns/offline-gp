import numpy as np
import torch
import gymnasium as gym
import argparse
import os

import td3.utils
from td3.TD3 import TD3

# Runs policy for X episodes and returns average reward
# A fixed seed is used for the eval environment
def eval_policy(policy, env_name, seed, eval_episodes=10):
    eval_env = gym.make(env_name)

    avg_reward = 0.
    for _ in range(eval_episodes):
        state, terminated, truncated = eval_env.reset(seed=(seed+100)), False, False
        state = state[0]
        while not (terminated or truncated):
            action = policy.select_action(np.array(state))
            state, reward, terminated, truncated, _ = eval_env.step(action)
            avg_reward += reward

    avg_reward /= eval_episodes

    print("---------------------------------------")
    print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
    print("---------------------------------------")
    return avg_reward


def train(env_name="HalfCheetah-v5", seed=0, start_timesteps=25e3, eval_freq=5e3, max_timesteps=1e6, expl_noise=0.1, batch_size=256, discount=0.99, tau=0.005, policy_noise=0.2, noise_clip=0.5, policy_freq=2):

    file_name = f"TD3_{env_name}_{seed}"
    print("---------------------------------------")
    print(f"Policy: TD3, Env: {env_name}, Seed: {seed}")
    print("---------------------------------------")
 
    if not os.path.exists("./models/td3/"):
        os.makedirs("./models/td3")

    env = gym.make(env_name)

    # Set seeds
    env.action_space.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0] 
    max_action = float(env.action_space.high[0])

    kwargs = {
        "state_dim": state_dim,
        "action_dim": action_dim,
        "max_action": max_action,
        "discount": discount,
        "tau": tau,
        "policy_noise": policy_noise * max_action,
        "noise_clip": noise_clip * max_action,
        "policy_freq": policy_freq
    }

    policy = TD3(**kwargs)

    replay_buffer = td3.utils.ReplayBuffer(state_dim, action_dim)

    # Evaluate untrained policy
    evaluations = [eval_policy(policy, env_name, seed)]

    state, terminated, truncated = env.reset(seed=seed), False, False
    state = state[0]
    episode_reward = 0
    episode_timesteps = 0
    episode_num = 0

    for t in range(int(max_timesteps)):
		
        episode_timesteps += 1

        # Select action randomly or according to policy
        if t < start_timesteps:
            action = env.action_space.sample()
        else:
            action = (policy.select_action(np.array(state)) + np.random.normal(0, max_action * expl_noise, size=action_dim)).clip(-max_action, max_action)

            # Perform action
            next_state, reward, terminated, truncated, _ = env.step(action) 
            done_bool = float((terminated or truncated)) if episode_timesteps < env._max_episode_steps else 0

            # Store data in replay buffer
            replay_buffer.add(state, action, next_state, reward, done_bool)

            state = next_state
            episode_reward += reward

            # Train agent after collecting sufficient data
            policy.train(replay_buffer, batch_size)

            if terminated or truncated: 
                # +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
                print(f"Total T: {t+1} Episode Num: {episode_num+1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f}")

                # Reset environment
                state, terminated, truncated = env.reset(), False, False
                state = state[0]
                episode_reward = 0
                episode_timesteps = 0
                episode_num += 1 

            # Evaluate episode
            if (t + 1) % eval_freq == 0:
                evaluations.append(eval_policy(policy, env_name, seed))
                policy.save(f"./models/td3/{file_name}")
