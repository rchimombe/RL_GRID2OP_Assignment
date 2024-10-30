# improved_agent_2.py

from base_agent import BaseAgent
import numpy as np

class ImprovedAgent2(BaseAgent):
    def __init__(self, env):
        super().__init__(env)
        self.setup_actions()  # Set up restricted action space during initialization

    def setup_actions(self):
        print("Setting up restricted action space...")

        # Sampled action space as a workaround
        restricted_actions = []
        critical_substations = [2, 4, 7]
        critical_lines = [3, 8, 12]
        load_threshold = 0.8
        high_risk_penalty_actions = []

        def is_high_risk(action):
            return (
                (action.substation_id in critical_substations and action.is_disconnect_substation())
                or (action.line_id in critical_lines and action.is_disconnect_line())
            )

        for _ in range(100):  # Adjust sample size as needed
            action = self.env.action_space.sample()  # Get a random action
            # Use only actions that meet specific conditions
            if hasattr(action, 'substation_id') and hasattr(action, 'line_id'):
                if action.substation_id in critical_substations or action.line_id in critical_lines:
                    current_load = self.env.get_obs().rho[action.line_id]
                    if current_load >= load_threshold:
                        if is_high_risk(action):
                            high_risk_penalty_actions.append(action)
                        else:
                            restricted_actions.append(action)

        self.action_space = restricted_actions if restricted_actions else [self.env.action_space.sample()]
        print(f"Restricted action space size: {len(self.action_space)}")

    def modify_action(self, action):
        """
        Example action modification logic. Adjusts actions to fit within the restricted action space.
        """
        action_modified = np.clip(action, self.env.action_space.low, self.env.action_space.high)
        return action_modified

    def train(self):
        """
        Training logic that interacts with the restricted action space.
        """
        obs, info = self.env.reset()
        done = False

        while not done:
            # Sample an action from the restricted action space
            action = self.env.action_space.sample()  # Placeholder for actual action selection
            modified_action = self.modify_action(action)  # Modify the action
            obs, reward, terminated, truncated, info = self.env.step(modified_action)
            done = terminated or truncated

            # Logic that could use modified actions in training
            print(f"Modified Action: {modified_action}, Reward: {reward}")

def evaluate(self, model, env, num_episodes=5):
    episode_rewards = []

    for ep in range(num_episodes):
        obs, info = env.reset()  # Ensure `env` is correctly passed here
        done = False
        episode_reward = 0

        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_reward += reward

        episode_rewards.append(episode_reward)

    return episode_rewards


