# train_agent.py

import yaml
from stable_baselines3 import A2C
from stable_baselines3.common.env_util import make_vec_env
import torch
import matplotlib.pyplot as plt

def load_config(config_path="config.yaml"):
    """Load configuration settings from config.yaml."""
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config

def train_agent(agent_instance, config):
    """Train the agent using the A2C model with parameters from config."""
    vec_env = make_vec_env(lambda: agent_instance.env, n_envs=config["environment"]["n_envs"])
    model = A2C(
        "MlpPolicy",
        vec_env,
        verbose=1,
        learning_rate=config["agent"]["learning_rate"],
        gamma=config["agent"]["gamma"],
        n_steps=config["agent"]["n_steps"],
        gae_lambda=config["agent"]["gae_lambda"],
        ent_coef=config["agent"]["ent_coef"],
        vf_coef=config["agent"]["vf_coef"],
        max_grad_norm=config["agent"]["max_grad_norm"],
        use_rms_prop=config["agent"]["use_rms_prop"],
        seed=config["agent"]["seed"],
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

    # Train the model
    print(f"Training model with {agent_instance.__class__.__name__}...")
    model.learn(total_timesteps=config["training"]["total_timesteps"])

    # Save the trained model if specified
    if config["evaluation"]["save_model"]:
        model_save_path = f"{config['evaluation']['model_save_path']}/{agent_instance.__class__.__name__.lower()}"
        model.save(model_save_path)
        print(f"Model saved as '{model_save_path}'.")

    return model

def evaluate_agent(agent_instance, model, config):
    """Evaluate the agent and plot rewards."""
    num_episodes = config["evaluation"]["num_episodes"]
    episode_rewards = agent_instance.evaluate(model, num_episodes)

    # Plot the episode rewards
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, num_episodes + 1), episode_rewards, marker='o', label=agent_instance.__class__.__name__)
    plt.title('Agent Performance Over Episodes')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.legend()
    plt.grid(True)

    # Save the plot if specified
    if config["evaluation"].get("plot_save_path"):
        plot_save_path = f"{config['evaluation']['plot_save_path']}/performance_{agent_instance.__class__.__name__.lower()}.png"
        plt.savefig(plot_save_path)
        print(f"Plot saved as '{plot_save_path}'.")

    plt.show()
