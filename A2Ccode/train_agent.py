# train_agent.py

import gymnasium as gym
from stable_baselines3 import A2C
from stable_baselines3.common.env_util import make_vec_env
import torch
import numpy as np
import matplotlib.pyplot as plt
from gym2op_env import Gym2OpEnv  # Import your environment
from base_agent import BaseAgent  # Import the BaseAgent class

class FlattenedActionWrapper(gym.ActionWrapper):
    def __init__(self, env):
        super(FlattenedActionWrapper, self).__init__(env)
        # Flatten only the action space
        self.action_space = gym.spaces.flatten_space(env.action_space)
        # The observation space remains as defined in Gym2OpEnv
        self.observation_space = env.observation_space

    def action(self, action):
        # Unflatten the action back into the original Dict action space
        action_unflattened = gym.spaces.unflatten(self.env.action_space, action)
        return action_unflattened

# Function to create a new environment instance for each parallel worker
def make_env():
    def _init():
        env = Gym2OpEnv()
        env = FlattenedActionWrapper(env)  # Wrap the environment with the FlattenedActionWrapper
        return env
    return _init

def train_a2c_agent():
    # Create a vectorized environment for parallel processing to improve training efficiency
    vec_env = make_vec_env(make_env(), n_envs=4)  # Creates 4 parallel environments

    # Initialize A2C agent with policy and optimizer settings
    model = A2C(
        "MultiInputPolicy",
        vec_env,
        verbose=1,
        learning_rate=0.01,
        gamma=0.99,  # Discount factor for long-term rewards
        n_steps=5,  # Number of steps to run for each environment per update
        gae_lambda=0.95,  # GAE lambda parameter
        ent_coef=0.01,  # Entropy coefficient for exploration
        vf_coef=0.5,  # Value function coefficient
        max_grad_norm=0.5,  # Gradient clipping
        use_rms_prop=True,
        seed=42,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

    # Train the model
    print("Training A2C model...")
    model.learn(total_timesteps=100000)  # Adjust total timesteps as needed

    # Save the model for future use
    model.save("a2c_grid2op_baseline")
    print("Model saved as 'a2c_grid2op_baseline'.")

    # Evaluate the trained agent
    base_agent = BaseAgent(vec_env)  # Initialize BaseAgent with the environment
    evaluate_agent_performance(base_agent, model)

def evaluate_agent_performance(agent, model, num_episodes=100):
    # Evaluate agent performance and plot rewards
    env = FlattenedActionWrapper(Gym2OpEnv())  # Create an instance of the environment for evaluation
    episode_rewards = agent.evaluate(model, env, num_episodes)

    # Plot the episode rewards with a smooth line and no marker
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, num_episodes + 1), episode_rewards)
    plt.title('Agent Performance Over Episodes')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    train_a2c_agent()
