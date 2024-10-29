# DDGP_gym2op_improved_1.py

from DDPG_gym2op_env import Gym2OpEnv
import gymnasium as gym

class ImprovedEnv1(Gym2OpEnv):
    def setup_observations(self):
        """ Modify observation space to include only essential grid stability indicators. """
        print("Setting up observation space with key grid stability indicators...")

        # Directly access attributes within the observation space if available
        original_obs = self._g2op_env.observation_space

        # Use getattr to handle cases where attributes might be missing
        obs_space = {
            "line_loadings": getattr(original_obs, "rho", None),         # Line loadings (rho)
            "generator_outputs": getattr(original_obs, "prod_p", None),  # Generator outputs (prod_p)
            "load_demands": getattr(original_obs, "load_p", None),       # Load demands (load_p)
            "topology": getattr(original_obs, "topo_vect", None)         # Topology configuration (topo_vect)
        }

        # Filter out any None values (if some attributes are missing)
        obs_space = {k: v for k, v in obs_space.items() if v is not None}

        # Ensure that obs_space is compatible with gym's Dict space
        self.observation_space = gym.spaces.Dict(obs_space)
