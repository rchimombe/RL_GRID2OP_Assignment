# gym2op_env.py

import gymnasium as gym
import yaml
import grid2op
from grid2op import gym_compat
from grid2op.Parameters import Parameters
from grid2op.Action import PlayableAction
from grid2op.Observation import CompleteObservation
from grid2op.Reward import L2RPNReward, N1Reward, CombinedScaledReward
from lightsim2grid import LightSimBackend
import numpy as np

class Gym2OpEnv(gym.Env):
    def __init__(self, config_path="config.yaml"):
        super().__init__()

        # Load configurations from config.yaml
        with open(config_path, "r") as file:
            config = yaml.safe_load(file)

        # Backend and environment setup based on config.yaml
        self._backend = LightSimBackend() if config["environment"]["backend"] == "LightSimBackend" else None
        self._env_name = config["environment"]["name"]
        
        # Set action and observation classes
        action_class = PlayableAction
        observation_class = CompleteObservation
        reward_class = CombinedScaledReward

        # Configure environment parameters
        p = Parameters()
        p.MAX_SUB_CHANGED = config["environment"]["max_sub_changed"]
        p.MAX_LINE_STATUS_CHANGED = config["environment"]["max_line_status_changed"]

        # Initialize Grid2Op environment with loaded configurations
        self._g2op_env = grid2op.make(
            self._env_name, backend=self._backend, test=False,
            action_class=action_class, observation_class=observation_class,
            reward_class=reward_class, param=p
        )

        # Composite Reward Setup
        cr = self._g2op_env.get_reward_instance()
        cr.addReward("N1", N1Reward(), 1.0)
        cr.addReward("L2RPN", L2RPNReward(), 1.0)
        cr.initialize(self._g2op_env)

        # Apply FlattenedActionWrapper if specified in config
        if config["environment"].get("use_flattened_action_wrapper", False):
            from FlattenedActionWrapper import FlattenedActionWrapper
            self._gym_env = FlattenedActionWrapper(gym_compat.GymEnv(self._g2op_env))
        else:
            self._gym_env = gym_compat.GymEnv(self._g2op_env)

        # Setup observation and action spaces
        self.setup_observations()
        self.setup_actions()

        # Finalize observation and action spaces for the environment
        self.observation_space = self._gym_env.observation_space
        self.action_space = self._gym_env.action_space

    def setup_observations(self):
        # Focus on critical features in the observation space
        print("Setting up observation space...")
        
        # Modify observation space to focus only on:
        # - Line Loadings (rho)
        # - Generator Outputs (prod_p)
        # - Load Demands (load_p)
        # - Topology (topo_vect)
        
        original_obs = self._g2op_env.observation_space
        obs_space = {}

        # Restricting the observation space to critical features
        obs_space["line_loadings"] = original_obs.get_attr("rho")  # Line loadings (rho)
        obs_space["generator_outputs"] = original_obs.get_attr("prod_p")  # Generator outputs (prod_p)
        obs_space["load_demands"] = original_obs.get_attr("load_p")  # Load demands (load_p)
        obs_space["topology"] = original_obs.get_attr("topo_vect")  # Topology configuration (topo_vect)
        
        # Create a Dict observation space with restricted features
        self.observation_space = gym.spaces.Dict(obs_space)


    def setup_actions(self):
        # Define a combined restriction logic for the action space
        print("Setting up restricted action space...")

        # Full action space as provided by Gym2Op
        full_action_space = self._g2op_env.action_space

        # Critical substations and lines (identified by their IDs)
        critical_substations = [2, 4, 7]  # Example critical substation IDs
        critical_lines = [3, 8, 12]       # Example critical line IDs

        # Safety thresholds
        load_threshold = 0.8  # Only allow actions on lines above 80% load
        high_risk_penalty_actions = []  # To keep track of penalized actions if needed

        # Function to identify if an action is high risk
        def is_high_risk(action):
            # Define criteria for high-risk actions
            # Example: turning off critical substations or disconnecting lines above the threshold
            return (
                action.substation_id in critical_substations and action.is_disconnect_substation()
            ) or (
                action.line_id in critical_lines and action.is_disconnect_line()
            )

        # Filter actions based on combined restriction logic
        restricted_actions = []
        for action in full_action_space:
            # Check if action targets a critical substation or line
            if action.substation_id in critical_substations or action.line_id in critical_lines:
                # Check if line load meets the dynamic load-based restriction
                current_load = self._g2op_env.get_obs().rho[action.line_id]
                if current_load >= load_threshold:
                    # If the action is high risk, consider it for penalization
                    if is_high_risk(action):
                        high_risk_penalty_actions.append(action)
                    else:
                        restricted_actions.append(action)
            else:
                # Non-critical actions are ignored
                continue

        # Finalize the action space to include only restricted actions
        self.action_space = restricted_actions if restricted_actions else full_action_space

    def reward_shaping(self, reward, action=None):
        # Reward shaping to encourage stable grid operation
        print("Shaping reward...")

        # Example shaping: Clipping reward to avoid extreme values and amplify positives
        shaped_reward = np.clip(reward, -10, 10)

        # Apply penalties for high-risk actions if any were taken
        if action in high_risk_penalty_actions:
            shaped_reward -= 5  # Apply a penalty for high-risk actions

        # Amplify positive rewards for stable operations
        if reward > 0:
            shaped_reward *= 1.5
        elif reward < 0:
            shaped_reward *= 0.7

        return shaped_reward



    def reset(self, seed=None):
        obs, info = self._gym_env.reset(seed=seed, options=None)
        return self._filter_observation(obs), info  # Filter the observation to focus on critical features

    def step(self, action):
        obs, reward, terminated, truncated, info = self._gym_env.step(action)
        
        # Apply reward shaping
        shaped_reward = self.reward_shaping(reward)
        
        # Filter the observation to return only critical features
        filtered_obs = self._filter_observation(obs)
        
        return filtered_obs, shaped_reward, terminated, truncated, info

    def render(self):
        return self._gym_env.render()

    def _filter_observation(self, obs):
        """Filter observation to include only critical features."""
        return {
            "line_loadings": obs.rho,
            "generator_outputs": obs.prod_p,
            "load_demands": obs.load_p,
            "topology": obs.topo_vect
        }
