# DDPG_train.py

import matplotlib.pyplot as plt
from DDPG_agent import DDPGAgent
import DDPG_gym2op_baseline as baseline_env
import DDPG_gym2op_improved_1 as improved_env_1
import DDPG_gym2op_improved_2 as improved_env_2
import DDPG_gym2op_improved_3 as improved_env_3
from utils import process_observation, plot_training_progress, save_model, load_model
from DDPG_gym2op_env import FlattenComplexActionSpace




def select_environment(env_name):
    """ Select environment based on user input """
    if env_name == 'baseline':
        return baseline_env.BaselineEnv()
    elif env_name == 'improved_1':
        return improved_env_1.ImprovedEnv1()
    elif env_name == 'improved_2':
        return improved_env_2.ImprovedEnv2()
    elif env_name == 'improved_3':
        return improved_env_3.ImprovedEnv3()
    else:
        raise ValueError("Unknown environment name")

def train_ddpg_agent(env, agent, num_episodes=100, batch_size=64):
    all_rewards = []
    for episode in range(num_episodes):
        obs = env.reset()  # Update: no unpacking, expect a single observation
        obs = process_observation(obs)
        episode_reward = 0
        done = False

        while not done:
            action_vector = agent.select_action(obs)
            next_obs, reward, done, info = env.step(action_vector)
            next_obs = process_observation(next_obs)
            agent.store_transition((obs, action_vector, next_obs, reward, done))
            obs = next_obs
            episode_reward += reward

            if len(agent.replay_buffer) > batch_size:
                agent.train(batch_size)

        all_rewards.append(episode_reward)
        print(f"Episode {episode + 1}, Reward: {episode_reward}")

    return all_rewards

def evaluate_agent(agent, env, num_episodes=5):
    total_reward = 0
    for _ in range(num_episodes):
        obs = env.reset()  # Update: no unpacking, expect a single observation
        obs = process_observation(obs)
        episode_reward = 0
        done = False

        while not done:
            action_vector = agent.select_action(obs)
            obs, reward, done, info = env.step(action_vector)
            obs = process_observation(obs)
            episode_reward += reward

        total_reward += episode_reward
    avg_reward = total_reward / num_episodes
    print(f"Average Evaluation Reward: {avg_reward}")
    return avg_reward

def plot_performance(rewards, env_name):
    plt.plot(rewards, label=env_name)
    plt.xlabel("Episodes")
    plt.ylabel("Cumulative Reward")
    plt.title(f"Training Performance for {env_name}")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    # Select environment
    env_name = input("Enter the environment to train (baseline/improved_1/improved_2/improved_3): ")
    env = select_environment(env_name)

    # Initialize the environment and agent
    wrapped_env = FlattenComplexActionSpace(env)
    state_dim = len(wrapped_env.reset())
    action_dim = wrapped_env.action_space.shape[0]
    max_action = float(wrapped_env.action_space.high[0])

    agent = DDPGAgent(state_dim=state_dim, action_dim=action_dim, max_action=max_action, env=wrapped_env)

    # Train and evaluate
    rewards = train_ddpg_agent(wrapped_env, agent, num_episodes=100)
    avg_reward = evaluate_agent(agent, wrapped_env, num_episodes=5)

 
 
    # Save the model after training
    save_model(agent, f"ddpg_agent_{env_name}.pt")
    
    
    
    # Plot performance
    plot_performance(rewards, env_name)
