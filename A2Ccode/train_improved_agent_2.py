# train_improved_agent_2.py

from improved_agent_2 import ImprovedAgent2
from stable_baselines3 import A2C
from stable_baselines3.common.env_util import make_vec_env
import torch
import matplotlib.pyplot as plt
import gymnasium as gym
from gym2op_env import Gym2OpEnv
#from FlattenedActionWrapper import FlattenedActionWrapper  # Assuming FlattenedActionWrapper is defined separately

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

def make_env():
    def _init():
        env = Gym2OpEnv()
        env = FlattenedActionWrapper(env)
        return env
    return _init

def train_a2c_improved_agent():
    # Create vectorized environment for parallel processing
    vec_env = make_vec_env(make_env(), n_envs=4)

    # Initialize A2C agent with modified settings, if needed
    model = A2C(
        "MultiInputPolicy",
        vec_env,
        verbose=1,
        learning_rate=0.0007,
        gamma=0.99,
        n_steps=5,
        gae_lambda=0.95,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        use_rms_prop=True,
        seed=42,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

    # Train the model
    print("Training A2C model with ImprovedAgent2...")
    model.learn(total_timesteps=100000)  # Adjust total timesteps as needed

    # Save the model for future use
    model.save("a2c_grid2op_improved_agent_2")
    print("Model saved as 'a2c_grid2op_improved_agent_2'.")

    # Evaluate the improved agent
    improved_agent = ImprovedAgent2(vec_env)
    evaluate_improved_agent(improved_agent, model)

def evaluate_improved_agent(agent, model, num_episodes=100):
    # Create the evaluation environment instance
    env = FlattenedActionWrapper(Gym2OpEnv())  # Make sure Gym2OpEnv() is correctly defined and imported

    # Pass the environment to the agent's evaluate method
    episode_rewards = agent.evaluate(model, env, num_episodes)

    # Plot the episode rewards
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, num_episodes + 1), episode_rewards)
    plt.title('Improved Agent 2 Performance Over Episodes')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    train_a2c_improved_agent()
