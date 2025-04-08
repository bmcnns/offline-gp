from td3.make_minari_dataset import evaluate_and_save_dataset
import td3.train as TD3

import random

environments = ['HalfCheetah-v5']
experiments = [ (environment, random.randint(0, 2**32-1)) for environment in environments ]

# Train TD3 against the following environments
#for environment, _ in experiments:
#   TD3.train(env_name=environment, num_timesteps=1e6)

# Create offline datasets using the TD3 policy
for environment, seed in experiments:
    evaluate_and_save_dataset(
        env_name=environment,
        model_path=f"models/td3/{environment}",
        seed=seed,
        num_timesteps=1e6)
