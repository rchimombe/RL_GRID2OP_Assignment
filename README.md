# RL_GRID2OP_Assignment
Suggested Directory Structure for traing a DDPG
The proposed directory structure organizes each custom environment and agent in a modular way:
RL_GRID2OP_Assignment/
│
├── DDPG_train.py                     # Main script for selecting and training the environment
├── DDPG_agent.py                     # DDPG agent, including Actor and Critic
├── DDGP_gym2op_baseline.py           # Baseline environment setup
├── DDGP_gym2op_improved_1.py         # Improvement 1 environment setup (Observation Space Modification)
├── DDGP_gym2op_improved_2.py         # Improvement 2 environment setup (Action Space Constraints)
├── DDGP_gym2op_improved_3.py         # Improvement 3 environment setup (Reward Shaping)
└── utils.py                          # Utility functions (for processing observations, etc.)

Each environment will be selectable and trainable by DDPG_train.py, which will log and plot each environment's results.