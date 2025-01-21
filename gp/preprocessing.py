import torch
from torch import device
from torch.utils.data import TensorDataset, DataLoader

def create_offline_dataset(dataset, batch_size=256, shuffle=True):
    states, actions, rewards, next_states, next_actions = [], [], [], [], []

    for episode in dataset:
        num_steps = len(episode.observations) - 1
        for i in range(num_steps - 1):
            states.append(torch.tensor(episode.observations[i], dtype=torch.float32))
            actions.append(torch.tensor(episode.actions[i], dtype=torch.float32))
            rewards.append(episode.rewards[i])
            next_states.append(torch.tensor(episode.observations[i + 1], dtype=torch.float32))
            next_actions.append(torch.tensor(episode.actions[i + 1], dtype=torch.float32))

    # Convert lists to tensors in one go (much faster than doing it in loops)
    states = torch.stack(states)
    actions = torch.stack(actions)
    rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1)
    next_states = torch.stack(next_actions)
    next_actions = torch.stack(next_actions)

    # Create TensorDataset
    tensor_dataset = TensorDataset(states, actions, rewards, next_states, next_actions)

    # Return DataLoader
    return DataLoader(tensor_dataset, batch_size=batch_size, shuffle=shuffle)
