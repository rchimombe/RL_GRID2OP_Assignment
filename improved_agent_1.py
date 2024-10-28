# improved_agent_1.py

import train_agent
import yaml
from gym2op_env import Gym2OpEnv
from base_agent import BaseAgent
import numpy as np


class ImprovedAgent1(BaseAgent):
    def __init__(self, env, config_path="config.yaml"):
        super().__init__(env)
        self.config = self.load_config(config_path)  # Load configuration settings

    def load_config(self, config_path):
        """Load configuration settings from config.yaml for ImprovedAgent1."""
        with open(config_path, "r") as file:
            config = yaml.safe_load(file)
        return config

    def train(self):
        """Call train_agent to train this agent using its own environment and config."""
        train_agent.train_agent(self, self.config)

    def evaluate(self, model, num_episodes=5):
        """Evaluate the agent over a set number of episodes."""
        return super().evaluate(model, num_episodes)

    def preprocess_observation(self, observation):
        """Custom observation processing specific to ImprovedAgent1."""
        # Normalize critical features like line loadings, generator outputs, and load demands
        line_loadings = observation["line_loadings"]
        generator_outputs = observation["generator_outputs"]
        load_demands = observation["load_demands"]
        
        # Normalize each observation feature by dividing by the maximum observed value
        normalized_line_loadings = line_loadings / max(line_loadings) if max(line_loadings) > 0 else line_loadings
        normalized_generator_outputs = generator_outputs / max(generator_outputs) if max(generator_outputs) > 0 else generator_outputs
        normalized_load_demands = load_demands / max(load_demands) if max(load_demands) > 0 else load_demands
        
        # Return the processed observation with normalized values
        return {
            "line_loadings": normalized_line_loadings,
            "generator_outputs": normalized_generator_outputs,
            "load_demands": normalized_load_demands,
            "topology": observation["topology"]  # Pass topology unchanged
        }


if __name__ == "__main__":
    env = Gym2OpEnv(config_path="config.yaml")
    agent = ImprovedAgent1(env)
    agent.train()  # This will call train_agent with ImprovedAgent1's configuration
