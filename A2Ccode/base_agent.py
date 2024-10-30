# base_agent.py

class BaseAgent:
    def __init__(self, env):
        self.env = env
        # Initialize policy and any required model parameters for the base agent

    def train(self):
        # Placeholder for training logic
        pass

    def evaluate(self, model, env, num_episodes=5):
        # Evaluate agent performance over a set number of episodes
        episode_rewards = []

        for ep in range(num_episodes):
            obs, info = env.reset()
            done = False
            episode_reward = 0
            while not done:
                action, _states = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                episode_reward += reward

            episode_rewards.append(episode_reward)

        return episode_rewards
