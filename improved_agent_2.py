# improved_agent_2.py

import train_agent
import yaml
from gym2op_env import Gym2OpEnv
from base_agent import BaseAgent
import numpy as np


class ImprovedAgent2(BaseAgent):
    def __init__(self, env, config_path="config.yaml"):
        super().__init__(env)
        self.config = self.load_config(config_path)  # Load configuration settings

    def load_config(self, config_path):
        """Load configuration settings from config.yaml for ImprovedAgent2."""
        with open(config_path, "r") as file:
            config = yaml.safe_load(file)
        return config

    def train(self):
        """Call train_agent to train this agent using its own environment and config."""
        train_agent.train_agent(self, self.config)

    def evaluate(self, model, num_episodes=5):
        """Evaluate the agent over a set number of episodes."""
        return super().evaluate(model, num_episodes)

    def get_restricted_actions(self, obs):
        """Define custom action restrictions based on critical substations, lines, and load."""
        restricted_actions = []
        
        # Load parameters from config
        load_threshold = self.config["action_restriction"]["load_threshold"]
        critical_substations = self.config["action_restriction"]["critical_substations"]
        critical_lines = self.config["action_restriction"]["critical_lines"]

        for action in self.env.action_space:
            # Only allow actions on critical substations/lines and if load exceeds threshold
            if action.substation_id in critical_substations or action.line_id in critical_lines:
                current_load = obs["line_loadings"][action.line_id]
                if current_load >= load_threshold and not self.is_high_risk_action(action):
                    restricted_actions.append(action)

        return restricted_actions if restricted_actions else self.env.action_space

    def is_high_risk_action(self, action):
        """Identify high-risk actions based on critical substations and lines."""
        critical_substations = self.config["action_restriction"]["critical_substations"]
        critical_lines = self.config["action_restriction"]["critical_lines"]

        # Determine if the action disconnects a critical substation or line
        return (
            (action.substation_id in critical_substations and action.is_disconnect_substation()) or
            (action.line_id in critical_lines and action.is_disconnect_line())
        )

    def choose_action(self, obs):
        """Choose an action from the restricted action space based on the current observation."""
        restricted_actions = self.get_restricted_actions(obs)
        # Randomly select an action from the restricted set for simplicity
        action = np.random.choice(restricted_actions)
        return action

if __name__ == "__main__":
    env = Gym2OpEnv(config_path="config.yaml")
    agent = ImprovedAgent2(env)
    agent.train()  # This will call train_agent with ImprovedAgent2's configuration

