import gymnasium as gym
from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import DummyVecEnv
import torch
import numpy as np
import matplotlib.pyplot as plt
from gym2op_env import Gym2OpEnv
from improved_agent_1 import ImprovedAgent1


# Custom action wrapper to flatten Dict action space
class FlattenedActionWrapper(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)

        # Initialize action slices
        self.action_slices = {}
        start = 0

        for key, space in env.action_space.spaces.items():
            # Handle gymnasium space types
            if isinstance(space, (gym.spaces.MultiBinary, gym.spaces.multi_binary.MultiBinary)):
                end = start + space.n
            elif isinstance(space, (gym.spaces.Box, gym.spaces.box.Box)):
                end = start + int(np.prod(space.shape))
            elif isinstance(space, (gym.spaces.Discrete, gym.spaces.discrete.Discrete)):
                end = start + 1
            else:
                raise NotImplementedError(f"Unsupported space type: {type(space)} for key: {key}")

            self.action_slices[key] = (key, start, end, space)
            start = end

        flat_dims = start
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(flat_dims,), dtype=np.float32)
        self.observation_space = env.observation_space

    def action(self, action):
        unflattened_action = {}
        for key, start, end, space in self.action_slices.values():
            sub_action = action[start:end]

            if isinstance(space, (gym.spaces.MultiBinary, gym.spaces.multi_binary.MultiBinary)):
                unflattened_action[key] = (sub_action > 0).astype(np.int8)
            elif isinstance(space, (gym.spaces.Box, gym.spaces.box.Box)):
                unflattened_action[key] = np.clip(sub_action, space.low, space.high).astype(space.dtype).reshape(space.shape)
            elif isinstance(space, (gym.spaces.Discrete, gym.spaces.discrete.Discrete)):
                unflattened_action[key] = int(np.clip(np.round(sub_action[0]), 0, space.n - 1))
            else:
                raise NotImplementedError(f"Unsupported space type: {type(space)} for key: {key}")

        return unflattened_action

class PreprocessedObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        sample_observation, _ = env.reset()
        sample_observation = self.observation(sample_observation)
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=sample_observation.shape,
            dtype=np.float32
        )

    def observation(self, observation):
        rho = observation.get('rho', np.zeros((4,)))
        prod_p = observation.get('prod_p', np.zeros((10,)))
        load_p = observation.get('load_p', np.zeros((5,)))
        topo_vect = observation.get('topo_vect', np.zeros((15,)))

        rho_flat = rho.flatten()
        prod_p_flat = prod_p.flatten()
        load_p_flat = load_p.flatten()
        topo_vect_flat = topo_vect.flatten()

        critical_features = np.concatenate([rho_flat, prod_p_flat, load_p_flat, topo_vect_flat])
        normalized_features = (critical_features - np.mean(critical_features)) / (np.std(critical_features) + 1e-8)

        return normalized_features.astype(np.float32)

def make_env():
    def _init():
        env = Gym2OpEnv()
        env = FlattenedActionWrapper(env)
        env = PreprocessedObservationWrapper(env)
        return env
    return _init

def train_a2c_agent():
    # Create a list of 4 environment creation functions
    env_fns = [make_env() for _ in range(4)]

    # Create the vectorized environment
    vec_env = DummyVecEnv(env_fns)

    model = A2C(
        "MlpPolicy",
        vec_env,
        verbose=1,
        learning_rate=0.0007,
        gamma=0.99,
        n_steps=5,
        gae_lambda=0.95,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        use_rms_prop=True,
        seed=42,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Training A2C model with ImprovedAgent1...")
    model.learn(total_timesteps=100000)
    model.save("a2c_grid2op_improved_agent_1")
    print("Model saved as 'a2c_grid2op_improved_agent_1'.")

    # Evaluate the trained agent
    evaluate_agent_performance(model)

def evaluate_agent_performance(model, num_episodes=5):
    # Create an instance of the environment for evaluation
    env = make_env()()
    agent = ImprovedAgent1(env)
    episode_rewards = agent.evaluate(model, num_episodes)

    # Plot the episode rewards
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, num_episodes + 1), episode_rewards, marker='o')
    plt.title('Improved Agent Performance Over Episodes')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    train_a2c_agent()
