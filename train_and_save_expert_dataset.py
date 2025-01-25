from td3.make_minari_dataset import evaluate_and_save_dataset
import td3.train as TD3

import random

environments = ['Hopper-v5']
experiments = [ (environment, random.randint(0, 2**32-1)) for environment in environments ]

# Train TD3 against the following environments
for environment, seed in experiments:
    TD3.train(env_name=environment, seed=seed, start_timesteps=1e3, max_timesteps=1101, eval_freq=1e2)

# Create offline datasets using the TD3 policy
for environment, seed in experiments:
    evaluate_and_save_dataset(
        env_name=environment,
        model_path=f"models/td3/TD3_{environment}_{seed}",
        seed=seed+100,
        num_timesteps=100)
