# improved_agent_3.py

from base_agent import BaseAgent
import numpy as np

class ImprovedAgent3(BaseAgent):
    def __init__(self, env):
        super().__init__(env)
        # Define high-risk actions that should be penalized
        self.high_risk_penalty_actions = []
        # Any additional initialization specific to ImprovedAgent3

    def reward_shaping(self, reward, action=None):
        # Reward shaping to encourage stable grid operation
        print("Shaping reward...")

        # Example shaping: Clipping reward to avoid extreme values and amplify positives
        shaped_reward = np.clip(reward, -10, 10)

        # Apply penalties for high-risk actions if any were taken
        if action in self.high_risk_penalty_actions:
            shaped_reward -= 5  # Apply a penalty for high-risk actions

        # Amplify positive rewards for stable operations
        if reward > 0:
            shaped_reward *= 1.5
        elif reward < 0:
            shaped_reward *= 0.7

        return shaped_reward

    def train(self):
        # Implement training logic with reward shaping
        obs, info = self.env.reset()
        done = False

        while not done:
            action = self.env.action_space.sample()  # Placeholder for actual action selection
            obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated

            # Apply reward shaping
            shaped_reward = self.reward_shaping(reward, action)
            print(f"Original Reward: {reward}, Shaped Reward: {shaped_reward}")

            # Use shaped_reward in the training logic (e.g., updating agent)
            # ... (agent update logic goes here)

class ImprovedAgent3(BaseAgent):
    # Existing initialization and other methods here

    def evaluate(self, model, env, num_episodes=5):
        episode_rewards = []

        for ep in range(num_episodes):
            obs, info = env.reset()  # Make sure `env` is passed correctly
            done = False
            episode_reward = 0

            while not done:
                action, _states = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                episode_reward += reward

            episode_rewards.append(episode_reward)

        return episode_rewards


