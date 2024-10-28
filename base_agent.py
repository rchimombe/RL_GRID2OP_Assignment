# base_agent.py

class BaseAgent:
    def __init__(self, env):
        self.env = env  # Environment should be generic

    def train(self):
        """Generic train method. Override in derived classes for environment-specific logic."""
        obs, info = self.env.reset()
        done = False

        while not done:
            # Placeholder action selection (random sampling)
            action = self.env.action_space.sample()
            obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated

            # Log the step (without environment-specific handling)
            print(f"Action: {action}, Reward: {reward}")

    def evaluate(self, model, num_episodes=5):
        """Generic evaluation method. Override in derived classes for environment-specific logic."""
        episode_rewards = []
        
        for ep in range(num_episodes):
            obs, info = self.env.reset()
            done = False
            episode_reward = 0
            
            while not done:
                action, _states = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                episode_reward += reward

            episode_rewards.append(episode_reward)

        return episode_rewards
