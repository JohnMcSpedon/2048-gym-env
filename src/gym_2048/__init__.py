from .env_2048 import Env2048, RewardConfig

import gymnasium

# Create a wrapper that provides default reward config
def make_env(**kwargs):
    """Create environment with default reward config if not provided."""
    if 'reward_config' not in kwargs:
        kwargs['reward_config'] = RewardConfig()
    return Env2048(**kwargs)

gymnasium.register(id="Gym2048-v0", entry_point=make_env)
