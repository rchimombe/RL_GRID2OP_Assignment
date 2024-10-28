# improved_agent_3.py

import train_agent
import yaml
from gym2op_env import Gym2OpEnv
from base_agent import BaseAgent
import numpy as np

class ImprovedAgent3(BaseAgent):
    def __init__(self, env, config_path="config.yaml"):
        super().__init__(env)
        self.config = self.load_config(config_path)  # Load configuration settings

    def load_config(self, config_path):
        """Load configuration settings from config.yaml for ImprovedAgent3."""
        with open(config_path, "r") as file:
            config = yaml.safe_load(file)
        return config

    def train(self):
        """Call train_agent to train this agent using its own environment and config."""
        train_agent.train_agent(self, self.config)

    def evaluate(self, model, num_episodes=5):
        """Evaluate the agent over a set number of episodes."""
        return super().evaluate(model, num_episodes)

    def reward_shaping(self, reward, action=None):
        """Apply custom reward shaping based on configuration."""
        # Load reward shaping parameters from the configuration
        max_reward = self.config["reward_shaping"].get("max_reward", 10)
        min_reward = self.config["reward_shaping"].get("min_reward", -10)
        amplify_positive = self.config["reward_shaping"].get("amplify_positive", 1.5)
        reduce_negative = self.config["reward_shaping"].get("reduce_negative", 0.7)
        penalty_for_high_risk_actions = self.config["action_restriction"].get("penalty_for_high_risk_actions", 5)

        # Clip the reward to specified max/min range
        shaped_reward = np.clip(reward, min_reward, max_reward)

        # Amplify positive rewards and reduce negative rewards based on configuration
        if shaped_reward > 0:
            shaped_reward *= amplify_positive
        elif shaped_reward < 0:
            shaped_reward *= reduce_negative

        # Apply penalty for high-risk actions if the action parameter is provided and deemed high risk
        if action and self.is_high_risk_action(action):
            shaped_reward -= penalty_for_high_risk_actions

        return shaped_reward

    def is_high_risk_action(self, action):
        """Identify high-risk actions based on critical substations and lines."""
        critical_substations = self.config["action_restriction"]["critical_substations"]
        critical_lines = self.config["action_restriction"]["critical_lines"]

        # Determine if the action disconnects a critical substation or line
        return (
            (action.substation_id in critical_substations and action.is_disconnect_substation()) or
            (action.line_id in critical_lines and action.is_disconnect_line())
        )
if __name__ == "__main__":
    env = Gym2OpEnv(config_path="config.yaml")
    agent = ImprovedAgent3(env)
    agent.train()  # This will call train_agent with ImprovedAgent3's configuration
