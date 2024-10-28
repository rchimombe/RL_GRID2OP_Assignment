import wandb
import yaml

# Load the config file
try:
    with open("config.yaml", "r") as file:
        config = yaml.safe_load(file)
except FileNotFoundError:
    print("Error: config.yaml file not found.")
    config = None
except yaml.YAMLError as exc:
    print(f"Error parsing config.yaml: {exc}")
    config = None

# Initialize W&B if config is loaded and logging is enabled
if config and 'logging' in config and config['logging'].get('use_wandb'):
    wandb.init(project=config['logging']['project_name'], config=config)

# Example logging metrics in your training loop
if config and 'model' in config:
    for epoch in range(config['model']['num_epochs']):
        train_loss = 0.05 * epoch  # Replace with actual calculation
        val_loss = 0.04 * epoch  # Replace with actual calculation

        # Log metrics to W&B
        if config['logging'].get('use_wandb'):
            wandb.log({"train_loss": train_loss, "val_loss": val_loss, "epoch": epoch})
else:
    print("Error: Configuration for model training is missing.")

