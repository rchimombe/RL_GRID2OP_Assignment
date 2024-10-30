# DDGP_gym2op_improved_3.py

from DDPG_gym2op_env import Gym2OpEnv
import numpy as np

class ImprovedEnv3(Gym2OpEnv):
    def setup_reward(self):
        """ Apply custom reward shaping to align agent actions with stability goals. """
        print("Setting up custom reward shaping...")

        original_reward_function = self._g2op_env.reward_class

        class CustomReward(original_reward_function):
            def compute(self, action, done=False):
                reward = super().compute(action, done)

                # Reward clipping within range [-1, 1]
                reward = np.clip(reward, -1.0, 1.0)

                # Amplify positive rewards by a factor of 1.5 to reinforce stability actions
                if reward > 0:
                    reward *= 1.5

                # Reduce negative rewards by a factor of 0.7 to moderate penalties
                elif reward < 0:
                    reward *= 0.7

                # Apply additional penalties for high-risk actions
                high_risk_penalty = 0.5 if action in ["disconnect_critical", "disconnect_high_load"] else 0
                reward -= high_risk_penalty

                return reward

        # Assign the custom reward function
        self._g2op_env.reward_class = CustomReward
