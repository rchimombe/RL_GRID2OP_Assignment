# DDGP_gym2op_improved_2.py

from DDPG_gym2op_env import Gym2OpEnv
import numpy as np
from gymnasium.spaces import MultiBinary
import gymnasium as gym


class ImprovedEnv2(Gym2OpEnv):
    def setup_actions(self):
        """ Constrain action space to high-impact areas and safety limits. """
        print("Setting up constrained action space...")

        original_action_space = self._g2op_env.action_space
        constrained_space = {}

        # High-impact substations and lines
        for substation in original_action_space.get_attr("substations"):
            if substation.impact > 0.8:  # Hypothetical threshold for high-impact
                constrained_space[substation.name] = original_action_space.get_attr(substation.name)

        # Limit actions to lines with high load
        for line in original_action_space.get_attr("lines"):
            load = line.get_attr("rho")  # Load on the line
            if load > 0.8:  # Threshold of 80% load
                constrained_space[line.name] = original_action_space.get_attr(line.name)

        # Safety constraints - restrict critical components from being disconnected
        critical_components = ["critical_substation", "critical_line"]
        for component in critical_components:
            constrained_space[component] = MultiBinary(1)  # Treat these as binary, on/off

        self.action_space = gym.spaces.Dict(constrained_space)
