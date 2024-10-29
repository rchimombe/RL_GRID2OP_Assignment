# gym2op_env.py

import numpy as np
import gymnasium as gym
import grid2op
from grid2op import gym_compat
from grid2op.Parameters import Parameters
from grid2op.Action import PlayableAction
from grid2op.Observation import CompleteObservation
from grid2op.Reward import L2RPNReward, N1Reward, CombinedScaledReward
from lightsim2grid import LightSimBackend
from gymnasium.spaces import Box, Dict, MultiBinary


class Gym2OpEnv(gym.Env):
    def __init__(self):
        super().__init__()

        # Backend and environment name setup
        self._backend = LightSimBackend()
        self._env_name = "l2rpn_case14_sandbox"

        # Setup action, observation, and reward classes
        action_class = PlayableAction
        observation_class = CompleteObservation
        reward_class = CombinedScaledReward

        # Set environment parameters
        p = Parameters()
        p.MAX_SUB_CHANGED = 4
        p.MAX_LINE_STATUS_CHANGED = 4

        # Initialize Grid2Op environment with specified configurations
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

        # Wrapping Grid2Op in Gym-compatible environment
        self._gym_env = gym_compat.GymEnv(self._g2op_env)

        # Set up observation and action spaces
        self.setup_observations()
        self.action_space = self._gym_env.action_space
    
    def setup_observations(self):
        # Define custom observation space here
        print("WARNING: setup_observations is not fully implemented. Customize for specific needs.")

    def setup_actions(self):
        # Define custom action space here
        print("WARNING: setup_actions is not fully implemented. Customize for specific needs.")

    def reset(self, seed=None, options=None):
        obs = self._gym_env.reset(seed=seed)
        return self.process_observation(obs)

    def reset(self, seed=None, options=None):
        obs, _ = self._gym_env.reset(seed=seed, options=options)  # Unpack the tuple
        return self.process_observation(obs)

    def step(self, action):
        obs, reward, done, _, info = self._gym_env.step(action)  # Unpack step output
        return self.process_observation(obs), reward, done, info

    def process_observation(self, obs):
        """ Convert observation dictionary to vector format for agent """
        obs_vector = np.concatenate([
            np.array(value).flatten()
            for value in obs.values()
            if isinstance(value, (np.ndarray, list)) or np.issubdtype(type(value), np.number)
        ])
        return obs_vector


class FlattenComplexActionSpace(gym.ActionWrapper):
    def __init__(self, env):
        super(FlattenComplexActionSpace, self).__init__(env)
        
        self.flattened_low = []
        self.flattened_high = []
        self.action_components = []

        for key, space in env.action_space.spaces.items():
            if isinstance(space, Box):
                self.flattened_low.append(space.low.flatten())
                self.flattened_high.append(space.high.flatten())
                self.action_components.append((key, 'box', space.shape))
            elif isinstance(space, MultiBinary):
                self.flattened_low.append(np.zeros(space.n))
                self.flattened_high.append(np.ones(space.n))
                self.action_components.append((key, 'binary', (space.n,)))
            else:
                raise NotImplementedError(f"Action space component '{key}' is not supported.")

        self.flattened_low = np.concatenate(self.flattened_low)
        self.flattened_high = np.concatenate(self.flattened_high)
        self.action_space = Box(low=self.flattened_low, high=self.flattened_high, dtype=np.float32)

    def action(self, action):
        action = np.atleast_1d(np.array(action))
        split_actions = {}
        start = 0
        for key, component_type, shape in self.action_components:
            end = start + np.prod(shape)
            if component_type == 'box':
                split_actions[key] = action[start:end].reshape(shape)
            elif component_type == 'binary':
                split_actions[key] = (action[start:end] > 0.5).astype(int)
            start = end
        return split_actions
