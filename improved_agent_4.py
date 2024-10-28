# improved_agent_4.py

import train_agent
import yaml
from gym2op_env import Gym2OpEnv
from base_agent import BaseAgent
import numpy as np


class ImprovedAgent4(BaseAgent):
    def __init__(self, env, config_path="config.yaml"):
        super().__init__(env)
        self.config = self.load_config(config_path)  # Load configuration settings

    def load_config(self, config_path):
        """Load configuration settings from config.yaml for ImprovedAgent4."""
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
        """Custom reward shaping logic specific to ImprovedAgent4."""
        # Example reward shaping logic
        shaped_reward = min(max(reward, -10), 10)  # Clip rewards between -10 and 10
        if action and self.is_high_risk_action(action):
            shaped_reward -= self.config["action_restriction"]["penalty_for_high_risk_actions"]
        return shaped_reward

    def get_restricted_actions(self, obs):
        """Get allowed actions based on critical substation and load-based restrictions."""
        restricted_actions = []
        load_threshold = self.config["action_restriction"]["load_threshold"]
        critical_substations = self.config["action_restriction"]["critical_substations"]
        critical_lines = self.config["action_restriction"]["critical_lines"]

        for action in self.env.action_space:
            # Only allow actions on critical substations/lines above the load threshold
            if action.substation_id in critical_substations or action.line_id in critical_lines:
                current_load = obs["line_loadings"][action.line_id]
                if current_load >= load_threshold and not self.is_high_risk_action(action):
                    restricted_actions.append(action)
        
        return restricted_actions if restricted_actions else self.env.action_space

    def is_high_risk_action(self, action):
        """Identify high-risk actions based on critical substations and lines."""
        critical_substations = self.config["action_restriction"]["critical_substations"]
        critical_lines = self.config["action_restriction"]["critical_lines"]

        return (
            (action.substation_id in critical_substations and action.is_disconnect_substation()) or
            (action.line_id in critical_lines and action.is_disconnect_line())
        )

if __name__ == "__main__":
    env = Gym2OpEnv(config_path="config.yaml")
    agent = ImprovedAgent4(env)
    agent.train()  # This will call train_agent with ImprovedAgent4's configuration
