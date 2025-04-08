import numpy as np
import torch
import gymnasium as gym
from minari import DataCollector
from minari.serialization import serialize_space
import TD3

def evaluate(policy, env_name, seed, eval_episodes=10):
    """Runs policy for X episodes"""
    
    eval_env = gym.make(env_name)
    
	eval_env = DataCollector(eval_env, record_infos=False)
	print(f"Observation Space: {eval_env.observation_space}")
	print(f"Action Space: {eval_env.action_space}")

	eval_env.seed(seed + 100)

	avg_reward = 0.
	for _ in range(eval_episodes):
		state, _ = eval_env.reset(), False
		state = state[0]
		terminated = False
		truncated = False
		while not (terminated or truncated):
			action = policy.select_action(np.array(state))
			state, reward, terminated, truncated, _ = eval_env.step(action)
			avg_reward += reward

	avg_reward /= eval_episodes

	print("---------------------------------------")
	print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
	print("---------------------------------------")

    save_dataset(eval_env, env_name)

def save_dataset(env, env_name):
    """Creates a Minari dataset from the environment"""

    envname_without_version = env_name.split('-')[0]
    env_version_number = env_name.split('-')[1]

	dataset = eval_env.create_dataset(
        dataset_id=f"{envname_without_version}-Expert-{env_version_number}",
        algorithm_name="TD3",
        code_permalink="https://github.com/sfujim/TD3",
        author="Bryce MacInnis",
        author_email="Bryce.MacInnis@dal.ca"
    )

def output_success():
    print("SUCCESS MOFO")

def evaluate_and_save_dataset(model_path, env, seed,
                 num_observations, num_actions, eval_episodes=1000,
                 policy_noise=0.2, noise_clip=0.5, policy_freq=2):
    """Evaluates a TD3 policy and saves the results as a Minari dataset"""

    kwargs = {
		"state_dim": num_observations,
		"action_dim": num_actions,
		"max_action": 1.0,
        "policy_noise": policy_noise,
        "noise_clip": noise_clip,
        "policy_freq": policy_freq
	}
    
	policy = TD3.TD3(**kwargs)
	policy.load(model_path)

	evaluate(policy, env, seed, eval_episodes)



	
