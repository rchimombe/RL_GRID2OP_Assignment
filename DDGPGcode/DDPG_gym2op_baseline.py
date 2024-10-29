# DDGP_gym2op_baseline.py

from DDPG_gym2op_env import Gym2OpEnv

class BaselineEnv(Gym2OpEnv):
    def __init__(self):
        super().__init__()
        print("Using Baseline Environment with default observation and action spaces.")
