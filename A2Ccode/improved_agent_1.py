# improved_agent_1.py

import numpy as np

class ImprovedAgent1:
    def __init__(self, env):
        self.env = env
        # Additional initialization if necessary

    def preprocess_observation(self, observation):
        # Extract features from observation
        # Assuming observation is a dictionary with keys
        rho = observation.get('rho', np.zeros((4,)))
        prod_p = observation.get('prod_p', np.zeros((10,)))
        load_p = observation.get('load_p', np.zeros((5,)))
        topo_vect = observation.get('topo_vect', np.zeros((15,)))

        # Flatten all arrays to 1D
        rho_flat = rho.flatten()
        prod_p_flat = prod_p.flatten()
        load_p_flat = load_p.flatten()
        topo_vect_flat = topo_vect.flatten()

        # Concatenate the flattened arrays
        critical_features = np.concatenate([rho_flat, prod_p_flat, load_p_flat, topo_vect_flat])

        # Normalize the features
        normalized_features = (critical_features - np.mean(critical_features)) / (np.std(critical_features) + 1e-8)

        return normalized_features.astype(np.float32)

    def evaluate(self, model, num_episodes=5):
        """
        Evaluate the agent using preprocessed observations.
        """
        episode_rewards = []

        for ep in range(num_episodes):
            obs = self.env.reset()
            done = False
            episode_reward = 0

            while not done:
                # Model predicts action based on preprocessed observation
                action, _states = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                episode_reward += reward

            episode_rewards.append(episode_reward)

        return episode_rewards
