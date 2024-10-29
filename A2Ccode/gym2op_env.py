# gym2op_env.py

import gymnasium as gym
import grid2op
from grid2op import gym_compat
from grid2op.Parameters import Parameters
from grid2op.Action import PlayableAction
from grid2op.Observation import CompleteObservation
from grid2op.Reward import L2RPNReward, N1Reward, CombinedScaledReward
from lightsim2grid import LightSimBackend

class Gym2OpEnv(gym.Env):
    def __init__(self):
        super().__init__()

        # Backend and environment name setup
        self._backend = LightSimBackend()
        self._env_name = "l2rpn_case14_sandbox"  # Environment name

        # Classes setup for actions, observations, and rewards
        action_class = PlayableAction
        observation_class = CompleteObservation
        reward_class = CombinedScaledReward  # Placeholder for composite rewards

        # Setting up the parameters for environment
        p = Parameters()
        p.MAX_SUB_CHANGED = 4
        p.MAX_LINE_STATUS_CHANGED = 4

        # Initializing Grid2Op environment with specified configurations
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

        # Setup observation and action spaces
        self.setup_observations()
        self.setup_actions()

        self.observation_space = self._gym_env.observation_space
        self.action_space = self._gym_env.action_space

    def setup_observations(self):
        # Define custom observation space here
        print("WARNING: setup_observations is not fully implemented. Customize for specific needs.")

    def setup_actions(self):
        # Define custom action space here
        print("WARNING: setup_actions is not fully implemented. Customize for specific needs.")

    def reset(self, seed=None, options=None):
        return self._gym_env.reset(seed=seed)


    def step(self, action):
        return self._gym_env.step(action)

    def render(self):
        return self._gym_env.render()
