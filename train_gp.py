from gp.trainer import train
from tqdm import tqdm

if __name__ == '__main__':
    experiments = [
        {
            "dataset_name": "Hopper-v2",
            "num_actions": 3,
            "epochs": 5,
            "num_threads": 18
        },
        {
            "dataset_name": "Walker2d-v2",
            "num_actions": 6,
            "epochs": 5,
            "num_threads": 18
        },
        {
            "dataset_name": "HalfCheetah-Expert-v2",
            "num_actions": 6,
            "epochs": 5,
            "num_threads": 18
        }
    ]

    for experiment in tqdm(experiments):
        print(f"Starting experiment: {experiment["dataset_name"]}")

        train(experiment["dataset_name"],
              experiment["num_actions"],
              experiment["epochs"],
              experiment["num_threads"])

        print(f"Finished experiment... ")
