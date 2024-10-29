# utils.py
import torch
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import gymnasium as gym
from gymnasium.spaces import Box, MultiBinary

def process_observation(obs):
    """
    Convert observation to a flat vector format.
    Handles both dictionary and numpy array formats for flexibility.
    """
    if isinstance(obs, dict):
        # If observation is a dictionary, flatten and concatenate numeric values
        obs_vector = np.concatenate([
            np.array(value).flatten()
            for value in obs.values()
            if isinstance(value, (np.ndarray, list)) or np.issubdtype(type(value), np.number)
        ])
    elif isinstance(obs, np.ndarray):
        # If observation is already a numpy array, flatten directly
        obs_vector = obs.flatten()
    else:
        raise TypeError("Unsupported observation type. Expected dict or numpy.ndarray.")

    return obs_vector

def plot_training_progress(reward_history, environment_name):
    """
    Plot cumulative reward over episodes to visualize agent training progress.
    Parameters:
    - reward_history: List of cumulative rewards per episode
    - environment_name: Name of the environment for labeling
    """
    plt.figure(figsize=(10, 6))
    plt.plot(reward_history, label=f"{environment_name} Cumulative Reward")
    plt.xlabel("Episodes")
    plt.ylabel("Cumulative Reward")
    plt.title(f"Training Performance for {environment_name}")
    plt.legend()
    plt.grid()
    plt.show()



def save_model(agent, filename):
    """
    Save the weights of the actor and critic networks.
    Parameters:
    - agent: Trained DDPG agent with actor and critic networks
    - filename: File path to save the model
    """
    torch.save({
        'actor_state_dict': agent.actor.state_dict(),
        'critic_state_dict': agent.critic.state_dict(),
    }, filename)
    print(f"Model saved to {filename}")


def load_model(agent, filename):
    """
    Load a model from a specified file to the agent.
    Parameters:
    - agent: Agent instance to load the model into
    - filename: Name of the file to load the model from
    """
    agent.load_model(filename)
    print(f"Model loaded from {filename}")
